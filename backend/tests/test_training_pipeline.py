import inspect
import json
import os
import subprocess
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from schemas import ModelCardResponse
from scripts import train_model
from services.model_competition import CandidateConfig
from services.model_metadata import load_model_card
from services.split_service import build_split_manifest


def make_fixture_frame(size=30):
    models = ["Alpha One", "Alpha Two", "Beta One", "Gamma One"]
    return pd.DataFrame(
        {
            "price": np.arange(size, dtype=float) * 10_000 + 100_000,
            "mileage": np.arange(size, dtype=float) * 1_000,
            "displacement": np.full(size, 1.2),
            "seats": np.full(size, 5),
            "owner_count": np.ones(size),
            "year": np.arange(size) % 8 + 2015,
            "brand": ["Honda" if index % 2 else "Toyota" for index in range(size)],
            "model": [models[index % len(models)] for index in range(size)],
            "city": ["Pune" if index % 2 else "Delhi" for index in range(size)],
            "transmission": ["Manual"] * size,
            "fuel_type": ["Petrol"] * size,
            "vehicle_type": ["car"] * size,
            "color": ["Grey"] * size,
            "accident_history": ["unknown"] * size,
            "seller_type": ["Dealer"] * size,
            "drivetrain": ["FWD"] * size,
            "max_power_bhp": np.full(size, 90.0),
            "power_rpm": np.full(size, 6_000.0),
            "max_torque_nm": np.full(size, 110.0),
            "torque_rpm": np.full(size, 4_000.0),
            "length_mm": np.full(size, 4_000.0),
            "width_mm": np.full(size, 1_700.0),
            "height_mm": np.full(size, 1_500.0),
            "fuel_tank_liter": np.full(size, 40.0),
        }
    )


def candidate_result(name="tiny-tree", model_type="extra_trees"):
    config = CandidateConfig(
        name,
        model_type,
        {
            "n_estimators": 1,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "n_jobs": 1,
        }
        if model_type == "extra_trees"
        else {
            "hidden_dims": (4,),
            "dropout": 0.0,
            "learning_rate": 0.001,
            "max_epochs": 1,
            "early_stopping_patience": 1,
        },
        1,
    )
    return {
        "name": config.name,
        "model_type": config.model_type,
        "config": dict(config.params),
        "complexity": config.complexity,
        "collection_year": 2026,
        "candidate": config.to_dict(),
        "cv": {
            "acc_10_mean": 0.75,
            "acc_10_std": 0.01,
            "median_ape_mean": 0.08,
            "median_ape_std": 0.01,
            "r2_mean": 0.70,
            "r2_std": 0.02,
            "rmse_mean": 20_000.0,
            "rmse_std": 500.0,
        },
        "fold_metrics": [],
        "observed_train_indices": [],
        "observed_validation_indices": [],
    }


class SavingAdapter:
    def save(self, directory):
        model_path = Path(directory) / "winner.bin"
        model_path.write_bytes(b"winner-model")
        return {
            "model_type": "extra_trees",
            "target_transform": "log1p",
            "artifacts": {"bundle": model_path.name},
        }


def make_fixture_experiment():
    frame = make_fixture_frame(12)
    winner = candidate_result()
    metrics = {
        "mse": 100.0,
        "rmse": 10.0,
        "mae": 8.0,
        "r2": 0.8,
        "acc_10": 0.75,
        "acc_20": 1.0,
        "median_ape": 0.05,
        "rmsle": 0.04,
        "baseline_rmse": 25.0,
        "baseline_r2": -0.1,
    }
    return {
        "run_id": "run-fixture",
        "created_at": "2026-07-23T00:00:00+00:00",
        "frame": frame,
        "split_manifest": {
            "split_version": "3.0.0",
            "seed": 42,
            "n_splits": 2,
            "development": list(range(8)),
            "test": list(range(8, 12)),
            "folds": [
                {"fold": 0, "train": [0, 1, 2, 3], "validation": [4, 5, 6, 7]},
                {"fold": 1, "train": [4, 5, 6, 7], "validation": [0, 1, 2, 3]},
            ],
        },
        "leaderboard": [winner],
        "winner": winner,
        "fitted_model": SavingAdapter(),
        "refit": {
            "strategy": "full_development",
            "train_indices": list(range(8)),
            "validation_indices": [],
        },
        "holdout": {
            "metrics": metrics,
            "actual": frame.loc[range(8, 12), "price"].tolist(),
            "predictions": frame.loc[range(8, 12), "price"].tolist(),
        },
        "gate": {
            "quality_gate": "pass",
            "warnings": [],
            "thresholds": {"min_r2": 0.0, "min_acc_10": 0.5},
        },
        "seed": 42,
        "collection_year": 2026,
        "provenance": {"source_id": "unit-test"},
    }


def write_publishable_experiment(root, quality_gate="pass", artifact_path="winner.bin"):
    train_model.build_artifacts(make_fixture_experiment(), root)
    if quality_gate != "pass":
        metrics_path = root / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["quality_gate"] = quality_gate
        metrics_path.write_text(json.dumps(metrics, allow_nan=False), encoding="utf-8")
    if artifact_path != "winner.bin":
        manifest_path = root / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model_artifacts"]["bundle"] = artifact_path
        manifest["model_artifact_metadata"]["artifacts"]["bundle"] = artifact_path
        manifest_path.write_text(
            json.dumps(manifest, allow_nan=False), encoding="utf-8"
        )


def mark_owned_model_directory(path):
    (path / ".model-publication-owner.json").write_text(
        json.dumps(
            {
                "owner": "car-valuation-model-publication",
                "sentinel_version": 1,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


class TrainingPipelineTests(unittest.TestCase):
    def require_function(self, name):
        function = getattr(train_model, name, None)
        self.assertIsNotNone(function, f"scripts.train_model.{name} is required")
        return function

    def test_v3_artifact_set_contains_auditable_reports_and_saved_winner(self):
        build_artifacts = self.require_function("build_artifacts")
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "experiment"

            paths = build_artifacts(make_fixture_experiment(), output_dir)

            expected = {
                "model_manifest.json",
                "feature_config.json",
                "metrics.json",
                "leaderboard.json",
                "error_analysis.json",
                "model_card.json",
            }
            self.assertTrue(expected.issubset(path.name for path in paths))
            self.assertEqual((output_dir / "winner.bin").read_bytes(), b"winner-model")
            for name in expected:
                payload = json.loads(
                    (output_dir / name).read_text(encoding="utf-8"),
                    parse_constant=lambda value: self.fail(
                        f"non-strict JSON constant {value} in {name}"
                    ),
                )
                self.assertEqual(payload["artifact_version"], "3.0.0")

            leaderboard = json.loads(
                (output_dir / "leaderboard.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("test_metrics", leaderboard["candidates"][0])

    def test_smoke_loader_seam_receives_validated_saved_manifest(self):
        build_artifacts = self.require_function("build_artifacts")
        smoke_load_experiment = self.require_function("smoke_load_experiment")
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "experiment"
            build_artifacts(make_fixture_experiment(), output_dir)
            loader = Mock(return_value=object())

            loaded = smoke_load_experiment(output_dir, loader=loader)

            self.assertTrue(loaded)
            loaded_dir, manifest = loader.call_args.args
            self.assertEqual(loaded_dir, output_dir)
            self.assertEqual(manifest["model_type"], "extra_trees")
            self.assertEqual(manifest["model_artifacts"], {"bundle": "winner.bin"})

    def test_generated_model_card_supports_current_consumers_and_keeps_v3_evidence(self):
        build_artifacts = self.require_function("build_artifacts")
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "experiment"
            build_artifacts(make_fixture_experiment(), output_dir)

            try:
                card = load_model_card(output_dir / "model_card.json")
                response = ModelCardResponse.model_validate(card)
            except Exception as exc:
                self.fail(f"generated model card failed current consumers: {exc}")

            self.assertEqual(card["currency"], "INR")
            self.assertEqual(card["price_unit"], "INR")
            self.assertEqual(card["mileage_unit"], "km")
            self.assertEqual(card["sample_count"], 12)
            self.assertEqual(card["data_source"], {"source_id": "unit-test"})
            self.assertEqual(
                card["split"],
                {"train": 8, "validation": 0, "development": 8, "test": 4},
            )
            self.assertEqual(card["thresholds"], {"min_r2": 0.0, "min_acc_10": 0.5})
            self.assertEqual(card["quality_gate"], "pass")
            self.assertEqual(card["warnings"], [])
            self.assertEqual(card["test_metrics"]["rmse"], 10.0)
            self.assertEqual(response.model_version, "v3-run-fixture")
            self.assertIn("cv_selection", card)
            self.assertIn("independent_holdout", card)

    def test_experiment_validation_rejects_cross_report_disagreement(self):
        validate_experiment = self.require_function("_validate_experiment_directory")

        def add_better_ranked_candidate(payload):
            better = candidate_result("better-tree")
            better["cv"].update(
                acc_10_mean=0.99,
                median_ape_mean=0.01,
                r2_mean=0.99,
            )
            payload["candidates"].append(better)
            payload["candidate_count"] = len(payload["candidates"])

        mutations = {
            "feature version": (
                "feature_config.json",
                lambda payload: payload.update(feature_version="wrong"),
            ),
            "model version": (
                "metrics.json",
                lambda payload: payload.update(model_version="wrong"),
            ),
            "model type": (
                "metrics.json",
                lambda payload: payload.update(model_type="mlp"),
            ),
            "winner": (
                "leaderboard.json",
                lambda payload: payload.update(winner="not-the-winner"),
            ),
            "split indices": (
                "metrics.json",
                lambda payload: payload["split_indices"].update(test=[7]),
            ),
            "holdout metrics": (
                "model_card.json",
                lambda payload: payload["test_metrics"].update(rmse=999.0),
            ),
            "development baseline": (
                "error_analysis.json",
                lambda payload: payload.update(development_mean_baseline=999.0),
            ),
            "baseline evidence": (
                "metrics.json",
                lambda payload: payload["baseline_evidence"].update(
                    value_inr=999.0
                ),
            ),
            "nested error evidence": (
                "model_card.json",
                lambda payload: payload["error_analysis"].update(
                    development_mean_baseline=999.0
                ),
            ),
            "leaderboard scope": (
                "leaderboard.json",
                lambda payload: payload.update(selection_scope="outer_test"),
            ),
            "leaderboard rank": (
                "leaderboard.json",
                add_better_ranked_candidate,
            ),
            "artifact roles": (
                "model_manifest.json",
                lambda payload: payload["model_artifacts"].update(model="winner.bin"),
            ),
        }
        for label, (filename, mutate) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory) / "experiment"
                train_model.build_artifacts(make_fixture_experiment(), output_dir)
                artifact_path = output_dir / filename
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                mutate(payload)
                artifact_path.write_text(
                    json.dumps(payload, allow_nan=False), encoding="utf-8"
                )

                with self.assertRaises((TypeError, ValueError)):
                    validate_experiment(output_dir)

    def test_experiment_validation_recalculates_and_verifies_quality_gate(self):
        validate_experiment = self.require_function("_validate_experiment_directory")
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "experiment"
            train_model.build_artifacts(make_fixture_experiment(), output_dir)
            metrics_path = output_dir / "metrics.json"
            card_path = output_dir / "model_card.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            card = json.loads(card_path.read_text(encoding="utf-8"))
            metrics["test_metrics"]["acc_10"] = 0.1
            card["test_metrics"]["acc_10"] = 0.1
            card["independent_holdout"]["metrics"]["acc_10"] = 0.1
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            card_path.write_text(json.dumps(card), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "quality gate|gate"):
                validate_experiment(output_dir)

    def test_experiment_validation_rejects_coordinated_semantic_corruption(self):
        validate_experiment = self.require_function("_validate_experiment_directory")

        def mutate_both_split_reports(payloads, mutate):
            for report_name in ("model_manifest.json", "metrics.json"):
                mutate(payloads[report_name])

        def overlap_outer_split(payloads):
            def mutate(report):
                report["split_manifest"]["test"][0] = 0
                report["split_indices"]["test"][0] = 0

            mutate_both_split_reports(payloads, mutate)

        def corrupt_features(payloads):
            feature = payloads["feature_config.json"]
            feature["target_col"] = "sale_price"
            feature["feature_cols"] = list(reversed(feature["feature_cols"]))
            feature["numeric_features"] = ["mileage", "bogus_numeric"]
            feature["categorical_features"] = ["bogus_category"]

        def corrupt_subgroup_sources(payloads):
            for analysis in (
                payloads["error_analysis.json"],
                payloads["model_card.json"]["error_analysis"],
            ):
                analysis["price_quartiles"]["definition_source"] = "recorded_test"
                analysis["price_quartiles"]["boundary_source"] = "recorded_test"
                analysis["model_family_frequency"]["frequency_source"] = "recorded_test"
                analysis["full_model_seen_status"]["seen_set_source"] = "recorded_test"

        def mismatch_collection_year(payloads):
            for report_name in (
                "feature_config.json",
                "metrics.json",
                "leaderboard.json",
                "model_card.json",
            ):
                payloads[report_name]["collection_year"] = 2025

        def mismatch_seed(payloads):
            for report_name in (
                "metrics.json",
                "leaderboard.json",
                "model_card.json",
            ):
                payloads[report_name]["seed"] = 99

        def corrupt_fold(payloads):
            def mutate(report):
                report["split_manifest"]["folds"][0]["train"].append(4)
                report["split_indices"]["folds"][0]["train"].append(4)

            mutate_both_split_reports(payloads, mutate)

        def corrupt_refit(payloads):
            shortened = list(range(7))
            manifest = payloads["model_manifest.json"]
            metrics = payloads["metrics.json"]
            card = payloads["model_card.json"]
            manifest["refit_strategy"]["train_indices"] = shortened
            manifest["split_indices"]["train"] = shortened
            metrics["split_indices"]["train"] = shortened
            card["refit_strategy"]["train_indices"] = shortened
            card["split"]["train"] = len(shortened)

        def corrupt_split_row_identity(payloads):
            def mutate(report):
                report["split_manifest"]["development"][0] = 100
                report["split_indices"]["development"][0] = 100
                for fold in report["split_manifest"]["folds"]:
                    for role in ("train", "validation"):
                        fold[role] = [
                            100 if row_id == 0 else row_id
                            for row_id in fold[role]
                        ]
                report["split_indices"]["folds"] = report["split_manifest"][
                    "folds"
                ]
                report["split_indices"]["train"] = [
                    100 if row_id == 0 else row_id
                    for row_id in report["split_indices"]["train"]
                ]

            mutate_both_split_reports(payloads, mutate)
            manifest = payloads["model_manifest.json"]
            manifest["refit_strategy"]["train_indices"] = manifest[
                "split_indices"
            ]["train"]
            card = payloads["model_card.json"]
            card["refit_strategy"] = manifest["refit_strategy"]

        def corrupt_refit_family_strategy(payloads):
            train_ids = list(range(6))
            validation_ids = [6, 7]
            for report_name in ("model_manifest.json", "metrics.json"):
                report = payloads[report_name]
                report["split_indices"]["train"] = train_ids
                report["split_indices"]["validation"] = validation_ids
            refit = payloads["model_manifest.json"]["refit_strategy"]
            refit.update(
                strategy="deterministic_development_holdback",
                train_indices=train_ids,
                validation_indices=validation_ids,
            )
            card = payloads["model_card.json"]
            card["refit_strategy"] = refit
            card["split"].update(train=len(train_ids), validation=len(validation_ids))
            payloads["metrics.json"]["split"].update(
                train=len(train_ids), validation=len(validation_ids)
            )

        def corrupt_ranked_winner(payloads):
            worse = candidate_result("worse-tree")
            worse["cv"].update(
                acc_10_mean=0.50,
                median_ape_mean=0.20,
                r2_mean=0.10,
                rmse_mean=50_000.0,
            )
            leaderboard = payloads["leaderboard.json"]
            leaderboard["candidates"].append(worse)
            leaderboard["candidate_count"] = 2
            leaderboard["winner"] = worse["name"]
            manifest = payloads["model_manifest.json"]
            manifest["winner"] = worse
            manifest["counts"]["candidates"] = 2
            metrics = payloads["metrics.json"]
            metrics["winner"] = worse["name"]
            card = payloads["model_card.json"]
            card["winner"] = worse
            card["counts"]["candidates"] = 2
            card["cv_selection"].update(
                winner=worse["name"],
                winner_cv=worse["cv"],
                candidate_count=2,
            )

        def corrupt_target_transform(payloads):
            manifest = payloads["model_manifest.json"]
            manifest["target_transform"] = "identity"
            manifest["model_artifact_metadata"]["target_transform"] = "identity"
            payloads["feature_config.json"]["target_transform"] = "identity"
            payloads["model_card.json"]["target_transform"] = "identity"

        corruptions = {
            "overlapping development and test": overlap_outer_split,
            "bogus feature contract": corrupt_features,
            "test-derived subgroup sources": corrupt_subgroup_sources,
            "mismatched collection years": mismatch_collection_year,
            "mismatched seeds": mismatch_seed,
            "invalid fold": corrupt_fold,
            "invalid refit": corrupt_refit,
            "noncanonical split row identity": corrupt_split_row_identity,
            "model-family refit strategy": corrupt_refit_family_strategy,
            "coordinated wrong CV winner": corrupt_ranked_winner,
            "noncanonical target transform": corrupt_target_transform,
        }
        for label, corrupt in corruptions.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory) / "experiment"
                train_model.build_artifacts(make_fixture_experiment(), output_dir)
                payloads = {
                    name: json.loads(
                        (output_dir / name).read_text(encoding="utf-8")
                    )
                    for name in train_model.REQUIRED_REPORTS
                }
                corrupt(payloads)
                for name, payload in payloads.items():
                    (output_dir / name).write_text(
                        json.dumps(payload, allow_nan=False), encoding="utf-8"
                    )

                with self.assertRaises((TypeError, ValueError)):
                    validate_experiment(output_dir)

    def test_builtin_smoke_loader_rejects_unusable_family_artifacts(self):
        load_saved = self.require_function("_load_saved_model_for_smoke")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.bin").write_bytes(b"invalid")
            catboost = Mock(tree_count_=0)
            with patch.object(train_model, "CatBoostRegressor", return_value=catboost):
                with self.assertRaisesRegex(ValueError, "CatBoost|fitted|tree"):
                    load_saved(
                        root,
                        {
                            "model_type": "catboost",
                            "model_artifacts": {"model": "model.bin"},
                        },
                    )

            with patch.object(
                train_model.joblib,
                "load",
                return_value={"preprocessor": object(), "model": object()},
            ):
                with self.assertRaisesRegex(ValueError, "ExtraTrees|fitted|bundle"):
                    load_saved(
                        root,
                        {
                            "model_type": "extra_trees",
                            "model_artifacts": {"bundle": "model.bin"},
                        },
                    )

            (root / "preprocessor.bin").write_bytes(b"invalid")
            with (
                patch.object(train_model.joblib, "load", return_value=object()),
                patch.object(train_model.torch, "load", return_value={}),
            ):
                with self.assertRaisesRegex(ValueError, "MLP|checkpoint|fitted"):
                    load_saved(
                        root,
                        {
                            "model_type": "mlp",
                            "model_artifacts": {
                                "preprocessor": "preprocessor.bin",
                                "model": "model.bin",
                            },
                        },
                    )

    def test_artifact_builder_refuses_preexisting_uninitialized_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            output_dir.mkdir()
            marker = output_dir / "user-marker.bin"
            marker.write_bytes(b"do-not-overwrite")

            with self.assertRaises(FileExistsError):
                train_model.build_artifacts(make_fixture_experiment(), output_dir)

            self.assertEqual(marker.read_bytes(), b"do-not-overwrite")

    def test_quality_gate_is_fixed_strict_and_rejects_nonfinite_metrics(self):
        passing = train_model.assess_quality_gate(
            {"r2": 0.1, "acc_10": 0.5, "rmse": 9.99, "baseline_rmse": 10.0}
        )
        equal_baseline = train_model.assess_quality_gate(
            {"r2": 0.1, "acc_10": 0.5, "rmse": 10.0, "baseline_rmse": 10.0}
        )
        missing = train_model.assess_quality_gate({"r2": 0.1})
        nonfinite = train_model.assess_quality_gate(
            {"r2": np.nan, "acc_10": 0.9, "rmse": 1.0, "baseline_rmse": np.inf}
        )

        self.assertEqual(passing["quality_gate"], "pass")
        self.assertEqual(equal_baseline["quality_gate"], "fail")
        self.assertEqual(missing["quality_gate"], "fail")
        self.assertEqual(nonfinite["quality_gate"], "fail")
        self.assertEqual(passing["thresholds"], {"min_r2": 0.0, "min_acc_10": 0.5})
        with self.assertRaisesRegex(ValueError, "fixed"):
            train_model.assess_quality_gate(
                {"r2": 0.1, "acc_10": 0.1, "rmse": 1.0, "baseline_rmse": 2.0},
                {"min_r2": -1.0, "min_acc_10": 0.0},
            )

    def test_mutating_public_threshold_mapping_cannot_bypass_gate(self):
        with patch.dict(
            train_model.QUALITY_THRESHOLDS,
            {"min_r2": -1.0, "min_acc_10": 0.0},
            clear=True,
        ):
            result = train_model.assess_quality_gate(
                {"r2": 0.1, "acc_10": 0.1, "rmse": 1.0, "baseline_rmse": 2.0}
            )

        self.assertEqual(result["quality_gate"], "fail")
        self.assertEqual(result["thresholds"], {"min_r2": 0.0, "min_acc_10": 0.5})

    def test_run_competition_rejects_post_selection_result_fields(self):
        run_competition = self.require_function("run_competition")
        candidate = CandidateConfig(
            "tiny-tree",
            "extra_trees",
            {"n_estimators": 1, "min_samples_leaf": 1, "max_features": 1.0, "n_jobs": 1},
            1,
        )
        evaluated = {**candidate_result(), "test_metrics": {"acc_10": 1.0}}
        evaluator = Mock(return_value=evaluated)

        with self.assertRaisesRegex(ValueError, "unexpected|post-selection"):
            run_competition(
                make_fixture_frame(12),
                {"development": list(range(10)), "test": [10, 11], "folds": []},
                [candidate],
                seed=42,
                collection_year=2026,
                evaluator=evaluator,
            )

    def test_outer_test_value_changes_cannot_change_competition_or_winner(self):
        run_competition = self.require_function("run_competition")
        select_winner = self.require_function("select_winner")
        frame = make_fixture_frame(14)
        manifest = {
            "development": list(range(12)),
            "test": [12, 13],
            "folds": [],
        }
        altered = frame.copy(deep=True)
        altered.loc[manifest["test"], "price"] = [90_000_000.0, 99_000_000.0]
        altered.loc[manifest["test"], "mileage"] = [8_000_000.0, 9_000_000.0]
        altered.loc[manifest["test"], "model"] = ["Outer Secret A", "Outer Secret B"]
        candidates = [
            CandidateConfig(
                name,
                "extra_trees",
                {"n_estimators": 1, "min_samples_leaf": 1, "max_features": 1.0, "n_jobs": 1},
                complexity,
            )
            for name, complexity in (("alpha", 1), ("beta", 2))
        ]
        observed_indices = []

        def evaluator(development_frame, sealed_manifest, config, **kwargs):
            observed_indices.append(development_frame.index.tolist())
            checksum = float(
                development_frame["price"].sum()
                + development_frame["mileage"].sum()
            )
            result = candidate_result(config.name)
            result["config"] = dict(config.params)
            result["complexity"] = config.complexity
            result["candidate"] = config.to_dict()
            result["cv"]["rmse_mean"] = checksum
            return result

        first = run_competition(
            frame,
            manifest,
            candidates,
            seed=42,
            collection_year=2026,
            evaluator=evaluator,
        )
        second = run_competition(
            altered,
            manifest,
            candidates,
            seed=42,
            collection_year=2026,
            evaluator=evaluator,
        )

        self.assertEqual(first, second)
        self.assertEqual(select_winner(first)["name"], select_winner(second)["name"])
        self.assertEqual(
            observed_indices,
            [manifest["development"]] * (len(candidates) * 2),
        )

    def test_malicious_competition_evaluator_cannot_retrieve_test_manifest_value(self):
        run_competition = self.require_function("run_competition")
        candidate = CandidateConfig(
            "tiny-tree",
            "extra_trees",
            {"n_estimators": 1, "min_samples_leaf": 1, "max_features": 1.0, "n_jobs": 1},
            1,
        )
        accepted_results = []

        def malicious_evaluator(frame, manifest, config, **kwargs):
            manifest["test"]
            accepted_results.append(candidate_result())
            return accepted_results[-1]

        with self.assertRaisesRegex(RuntimeError, "sealed"):
            run_competition(
                make_fixture_frame(12),
                {"development": list(range(10)), "test": [10, 11], "folds": []},
                [candidate],
                seed=42,
                collection_year=2026,
                evaluator=malicious_evaluator,
            )

        self.assertEqual(accepted_results, [])

    def test_run_competition_rejects_nested_non_cv_evidence(self):
        run_competition = self.require_function("run_competition")
        candidate = CandidateConfig(
            "tiny-tree",
            "extra_trees",
            {"n_estimators": 1, "min_samples_leaf": 1, "max_features": 1.0, "n_jobs": 1},
            1,
        )
        evaluated = candidate_result()
        evaluated["cv"]["diagnostics"] = {
            "holdout_metrics": {"acc_10": 1.0},
            "subgroup": {"unseen_model": "outer-secret"},
        }

        with self.assertRaisesRegex(ValueError, "cv|unexpected|post-selection"):
            run_competition(
                make_fixture_frame(12),
                {"development": list(range(10)), "test": [10, 11], "folds": []},
                [candidate],
                seed=42,
                collection_year=2026,
                evaluator=Mock(return_value=evaluated),
            )

    def test_standard_candidate_evaluator_works_with_sealed_manifest(self):
        run_competition = self.require_function("run_competition")
        frame = make_fixture_frame(40)
        manifest = build_split_manifest(frame, seed=42)
        candidate = CandidateConfig(
            "tiny-tree",
            "extra_trees",
            {
                "n_estimators": 1,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "n_jobs": 1,
            },
            1,
        )

        leaderboard = run_competition(
            frame,
            manifest,
            [candidate],
            seed=42,
            collection_year=2026,
        )

        self.assertEqual(len(leaderboard), 1)
        self.assertEqual(leaderboard[0]["name"], "tiny-tree")
        self.assertEqual(
            set().union(*map(set, leaderboard[0]["observed_validation_indices"])),
            set(manifest["development"]),
        )
        self.assertTrue(
            set(manifest["test"]).isdisjoint(
                set().union(*map(set, leaderboard[0]["observed_train_indices"]))
            )
        )

    def test_artifact_builder_rejects_nested_candidate_holdout_evidence(self):
        build_artifacts = self.require_function("build_artifacts")
        experiment = make_fixture_experiment()
        experiment["leaderboard"][0]["cv"]["diagnostics"] = {
            "test_metrics": {"rmse": 0.0}
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "cv|unexpected|post-selection"):
                build_artifacts(experiment, Path(directory) / "experiment")

    def test_run_competition_resolves_patched_tiny_evaluator_at_call_time(self):
        run_competition = self.require_function("run_competition")
        candidate = CandidateConfig(
            "tiny-tree",
            "extra_trees",
            {"n_estimators": 1, "min_samples_leaf": 1, "max_features": 1.0, "n_jobs": 1},
            1,
        )
        with patch.object(
            train_model,
            "evaluate_candidate",
            return_value=candidate_result(),
        ) as tiny_evaluator:
            leaderboard = run_competition(
                make_fixture_frame(12),
                {"development": list(range(10)), "test": [10, 11], "folds": []},
                [candidate],
                seed=42,
                collection_year=2026,
            )

        self.assertEqual(leaderboard, [candidate_result()])
        tiny_evaluator.assert_called_once()

    def test_refit_is_deterministic_and_uses_only_development_rows(self):
        refit_winner = self.require_function("refit_winner")
        development = make_fixture_frame(20)
        development.index = pd.Index(range(100, 120))
        winner = candidate_result("tiny-mlp", "mlp")
        observed = []

        class RecordingAdapter:
            def fit(self, train_frame, validation_frame=None):
                observed.append(
                    (
                        train_frame.index.tolist(),
                        validation_frame.index.tolist(),
                    )
                )
                return self

        def factory(config, seed):
            self.assertEqual(config.model_type, "mlp")
            self.assertEqual(seed, 42)
            return RecordingAdapter()

        first = refit_winner(
            development,
            winner,
            seed=42,
            collection_year=2026,
            adapter_factory=factory,
        )
        second = refit_winner(
            development,
            winner,
            seed=42,
            collection_year=2026,
            adapter_factory=factory,
        )

        self.assertEqual(observed[0], observed[1])
        train_ids, validation_ids = observed[0]
        self.assertEqual(set(train_ids) | set(validation_ids), set(development.index))
        self.assertTrue(set(train_ids).isdisjoint(validation_ids))
        self.assertEqual(first.strategy, second.strategy)
        self.assertEqual(first.train_indices, train_ids)
        self.assertEqual(first.validation_indices, validation_ids)

    def test_extra_trees_refit_uses_all_development_rows(self):
        refit_winner = self.require_function("refit_winner")
        development = make_fixture_frame(12)
        adapter = Mock()
        adapter.fit.return_value = adapter

        result = refit_winner(
            development,
            candidate_result(),
            seed=42,
            collection_year=2026,
            adapter_factory=Mock(return_value=adapter),
        )

        fitted_frame, validation_frame = adapter.fit.call_args.args
        self.assertEqual(fitted_frame.index.tolist(), development.index.tolist())
        self.assertIsNone(validation_frame)
        self.assertEqual(result.strategy, "full_development")

    def test_refit_rejects_duplicate_or_noninteger_development_index(self):
        refit_winner = self.require_function("refit_winner")
        invalid_indices = (
            [1, 1, 2, 3],
            [True, 1, 2, 3],
            [0.0, 1.0, 2.0, 3.0],
            ["0", "1", "2", "3"],
        )
        for values in invalid_indices:
            with self.subTest(index=values):
                development = make_fixture_frame(4)
                development.index = pd.Index(values)
                factory = Mock()

                with self.assertRaises((TypeError, ValueError)):
                    refit_winner(
                        development,
                        candidate_result(),
                        seed=42,
                        collection_year=2026,
                        adapter_factory=factory,
                    )

                factory.assert_not_called()

    def test_error_analysis_uses_development_only_group_definitions_and_nulls(self):
        build_error_analysis = self.require_function("build_error_analysis")
        development = make_fixture_frame(3)
        development["model"] = ["Alpha One", "Alpha Two", "Beta One"]
        holdout = make_fixture_frame(4)
        holdout.index = pd.Index(range(10, 14))
        holdout["model"] = ["Alpha New", "Gamma One", "Beta One", "Delta One"]
        predictions = holdout["price"].to_numpy(dtype=float) + np.array(
            [1_000.0, -1_000.0, 2_000.0, -2_000.0]
        )
        development_mean = float(development["price"].mean())

        report = build_error_analysis(
            development,
            holdout,
            predictions,
            development_mean,
            rare_threshold=2,
        )

        family = report["model_family_frequency"]
        self.assertEqual(family["frequency_source"], "development_only")
        self.assertEqual(family["rare_threshold"], 2)
        self.assertEqual(family["development_frequencies"], {"Alpha": 2, "Beta": 1})
        family_counts = {group["label"]: group["count"] for group in family["groups"]}
        self.assertEqual(family_counts, {"common": 1, "rare": 3})

        seen = report["full_model_seen_status"]
        self.assertEqual(seen["seen_set_source"], "development_only")
        seen_counts = {group["label"]: group["count"] for group in seen["groups"]}
        self.assertEqual(seen_counts, {"seen": 1, "unseen": 3})
        self.assertEqual(report["development_mean_baseline"], development_mean)
        self.assertEqual(
            sum(group["count"] for group in report["price_quartiles"]["groups"]),
            len(holdout),
        )
        singleton = next(group for group in seen["groups"] if group["count"] == 1)
        self.assertIsNone(singleton["metrics"]["r2"])
        json.dumps(report, allow_nan=False)

    def test_price_quartile_boundaries_are_development_derived_and_stable(self):
        development = make_fixture_frame(12)
        development["price"] = [100.0] * 4 + [200.0] * 4 + [400.0] * 4
        first_holdout = make_fixture_frame(4)
        first_holdout["price"] = [50.0, 150.0, 300.0, 800.0]
        second_holdout = first_holdout.copy()
        second_holdout["price"] = [1.0, 2.0, 3.0, 4.0]

        first = train_model.build_error_analysis(
            development,
            first_holdout,
            first_holdout["price"].to_numpy(),
            float(development["price"].mean()),
        )
        second = train_model.build_error_analysis(
            development,
            second_holdout,
            second_holdout["price"].to_numpy(),
            float(development["price"].mean()),
        )

        first_quartiles = first["price_quartiles"]
        second_quartiles = second["price_quartiles"]
        self.assertEqual(first_quartiles["definition_source"], "development_only")
        self.assertEqual(
            first_quartiles["boundaries_inr"],
            second_quartiles["boundaries_inr"],
        )
        self.assertEqual(len(first_quartiles["boundaries_inr"]), 3)
        self.assertEqual(len(first_quartiles["groups"]), 4)
        json.dumps(first, allow_nan=False)

    def test_singleton_metrics_share_canonical_nextafter_accuracy_boundaries(self):
        singleton_metrics = self.require_function("_single_or_multi_metrics")
        actual = np.array([100.0])
        for threshold, field in ((0.10, "acc_10"), (0.20, "acc_20")):
            for factor in (
                np.nextafter(1.0 + threshold, np.inf),
                np.nextafter(np.nextafter(1.0 + threshold, np.inf), np.inf),
            ):
                with self.subTest(threshold=threshold, factor=factor):
                    predicted = actual * factor
                    canonical = train_model.calculate_metrics(
                        np.repeat(actual, 2),
                        np.repeat(predicted, 2),
                        np.repeat(actual, 2),
                    )
                    singleton = singleton_metrics(actual, predicted, 100.0)

                    self.assertEqual(singleton[field], canonical[field])
                    self.assertIsNone(singleton["r2"])
                    self.assertIsNone(singleton["baseline_r2"])

    def test_train_and_publish_uses_fixed_orchestration_order_and_one_holdout(self):
        train_and_publish = self.require_function("train_and_publish")
        events = []
        frame = make_fixture_frame(13)
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        manifest = {"development": list(range(10)), "test": [10, 11, 12], "folds": []}
        leaderboard = [candidate_result()]
        winner = leaderboard[0]
        fitted = object()
        holdout = {"metrics": {"r2": 0.5, "acc_10": 0.8, "rmse": 1.0, "baseline_rmse": 2.0}}
        gate = {"quality_gate": "pass", "warnings": [], "thresholds": train_model.QUALITY_THRESHOLDS}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_dir = root / "experiments" / "run"

            def load(path):
                events.append("load")
                return frame

            def split(received, seed):
                events.append("split")
                self.assertEqual(len(received), 13)
                self.assertEqual(received.index.tolist(), list(range(13)))
                return manifest

            def compete(*args, **kwargs):
                events.append("competition")
                return leaderboard

            def select(value):
                events.append("select")
                self.assertIs(value, leaderboard)
                return winner

            def refit(received, *args, **kwargs):
                events.append("refit")
                self.assertEqual(received.index.tolist(), manifest["development"])
                return fitted

            def evaluate(received_model, received, *args, **kwargs):
                events.append("evaluate")
                self.assertIs(received_model, fitted)
                self.assertEqual(received.index.tolist(), manifest["test"])
                return holdout

            def assess(metrics):
                events.append("gate")
                self.assertIs(metrics, holdout["metrics"])
                return gate

            def write(experiment, output_dir):
                events.append("write")
                self.assertEqual(output_dir, experiment_dir)
                self.assertEqual(
                    experiment["provenance"]["source_id"],
                    "normalized-training-csv",
                )
                self.assertEqual(
                    experiment["provenance"]["source_url"],
                    train_model.SOURCE_URL,
                )
                self.assertEqual(experiment["provenance"]["sha256"], "fixture-sha")
                return []

            def smoke(output_dir, loader=None):
                events.append("smoke")
                self.assertEqual(output_dir, experiment_dir)
                return True

            def publish(source, target):
                events.append("publish")
                self.assertEqual(source, experiment_dir)
                return True

            with (
                patch.object(train_model, "load_dataset", side_effect=load),
                patch.object(train_model, "build_split_manifest", side_effect=split),
                patch.object(train_model, "run_competition", side_effect=compete),
                patch.object(train_model, "select_winner", side_effect=select),
                patch.object(train_model, "refit_winner", side_effect=refit),
                patch.object(train_model, "evaluate_holdout", side_effect=evaluate) as evaluate_mock,
                patch.object(train_model, "assess_quality_gate", side_effect=assess),
                patch.object(train_model, "write_experiment_artifacts", side_effect=write),
                patch.object(train_model, "smoke_load_experiment", side_effect=smoke),
                patch.object(train_model, "publish_experiment", side_effect=publish),
                patch.object(train_model, "_new_run_id", return_value="run"),
            ):
                result = train_and_publish(
                    "fixture.csv",
                    models_dir=root / "models",
                    experiments_dir=root / "experiments",
                    seed=42,
                    collection_year=2026,
                    candidates=(object(),),
                    provenance={"sha256": "fixture-sha"},
                )
                retained_status = json.loads(
                    (experiment_dir / "run_status.json").read_text(encoding="utf-8")
                )
                self.assertEqual(retained_status["status"], "completed")
                self.assertEqual(retained_status["stage"], "complete")
                self.assertTrue(retained_status["published"])

        self.assertEqual(
            events,
            ["load", "split", "competition", "select", "refit", "evaluate", "gate", "write", "smoke", "publish"],
        )
        self.assertEqual(evaluate_mock.call_count, 1)
        self.assertTrue(result["published"])

    def test_stage_failures_retain_audit_directory_and_preserve_formal_model(self):
        train_and_publish = self.require_function("train_and_publish")
        frame = make_fixture_frame(12)
        manifest = {
            "development": list(range(10)),
            "test": [10, 11],
            "folds": [],
        }
        holdout = {
            "metrics": {
                "r2": 0.5,
                "acc_10": 0.8,
                "rmse": 1.0,
                "baseline_rmse": 2.0,
            }
        }
        gate = {
            "quality_gate": "pass",
            "warnings": [],
            "thresholds": train_model.QUALITY_THRESHOLDS,
        }
        failure_points = {
            "build_split_manifest": "build_split_manifest",
            "run_competition": "run_competition",
            "write_experiment_artifacts": "write_experiment_artifacts",
            "publish_experiment": "publish_experiment",
        }

        for expected_stage, failing_name in failure_points.items():
            with self.subTest(stage=expected_stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                formal = root / "models"
                formal.mkdir()
                marker = formal / "marker.bin"
                marker.write_bytes(b"formal-before")
                run_id = f"run-{expected_stage}"
                experiment_dir = root / "experiments" / run_id

                def write_artifacts(experiment, output_dir):
                    (output_dir / "artifact-marker.bin").write_bytes(b"partial")
                    return []

                patches = {
                    "load_dataset": Mock(return_value=frame),
                    "build_split_manifest": Mock(return_value=manifest),
                    "run_competition": Mock(return_value=[candidate_result()]),
                    "select_winner": Mock(return_value=candidate_result()),
                    "refit_winner": Mock(return_value=object()),
                    "evaluate_holdout": Mock(return_value=holdout),
                    "assess_quality_gate": Mock(return_value=gate),
                    "write_experiment_artifacts": Mock(side_effect=write_artifacts),
                    "smoke_load_experiment": Mock(return_value=True),
                    "publish_experiment": Mock(return_value=True),
                }
                patches[failing_name] = Mock(
                    side_effect=RuntimeError(f"failed at {expected_stage}")
                )

                with ExitStack() as stack:
                    for name, mocked in patches.items():
                        stack.enter_context(patch.object(train_model, name, mocked))
                    stack.enter_context(
                        patch.object(train_model, "_new_run_id", return_value=run_id)
                    )
                    with self.assertRaisesRegex(RuntimeError, expected_stage):
                        train_and_publish(
                            "fixture.csv",
                            models_dir=formal,
                            experiments_dir=root / "experiments",
                            collection_year=2026,
                            candidates=(object(),),
                        )

                self.assertTrue(experiment_dir.is_dir())
                failure = json.loads(
                    (experiment_dir / "failure.json").read_text(encoding="utf-8"),
                    parse_constant=lambda value: self.fail(
                        f"non-strict failure JSON constant: {value}"
                    ),
                )
                self.assertEqual(failure["stage"], expected_stage)
                self.assertEqual(failure["error_type"], "RuntimeError")
                self.assertIn(expected_stage, failure["message"])
                self.assertEqual(marker.read_bytes(), b"formal-before")

    def test_smoke_failure_preserves_formal_directory_and_experiment(self):
        train_and_publish = self.require_function("train_and_publish")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "models"
            formal.mkdir()
            marker = formal / "marker.bin"
            marker.write_bytes(b"formal-before")
            experiment_dir = root / "experiments" / "run"

            def write(experiment, output_dir):
                (output_dir / "audit.txt").write_text("kept", encoding="utf-8")
                return []

            with (
                patch.object(train_model, "load_dataset", return_value=make_fixture_frame(12)),
                patch.object(
                    train_model,
                    "build_split_manifest",
                    return_value={"development": list(range(10)), "test": [10, 11], "folds": []},
                ),
                patch.object(train_model, "run_competition", return_value=[candidate_result()]),
                patch.object(train_model, "select_winner", return_value=candidate_result()),
                patch.object(train_model, "refit_winner", return_value=object()),
                patch.object(
                    train_model,
                    "evaluate_holdout",
                    return_value={
                        "metrics": {"r2": 0.5, "acc_10": 0.8, "rmse": 1.0, "baseline_rmse": 2.0}
                    },
                ),
                patch.object(
                    train_model,
                    "assess_quality_gate",
                    return_value={
                        "quality_gate": "pass",
                        "warnings": [],
                        "thresholds": train_model.QUALITY_THRESHOLDS,
                    },
                ),
                patch.object(train_model, "write_experiment_artifacts", side_effect=write),
                patch.object(train_model, "smoke_load_experiment", side_effect=RuntimeError("broken model")),
                patch.object(train_model, "publish_experiment") as publish,
                patch.object(train_model, "_new_run_id", return_value="run"),
            ):
                result = train_and_publish(
                    "fixture.csv",
                    models_dir=formal,
                    experiments_dir=root / "experiments",
                    collection_year=2026,
                    candidates=(object(),),
                )

            self.assertFalse(result["published"])
            self.assertEqual(result["smoke_status"], "fail")
            self.assertEqual(marker.read_bytes(), b"formal-before")
            self.assertEqual((experiment_dir / "audit.txt").read_text(encoding="utf-8"), "kept")
            failure = json.loads(
                (experiment_dir / "failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["stage"], "smoke_load_experiment")
            publish.assert_not_called()

    def test_failed_gate_returns_false_and_preserves_marker_byte_for_byte(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = root / "failed"
            formal = root / "models"
            formal.mkdir()
            marker = formal / "marker.bin"
            original = bytes(range(256))
            marker.write_bytes(original)
            write_publishable_experiment(failed, quality_gate="fail")

            published = publish_experiment(failed, formal)

            self.assertFalse(published)
            self.assertEqual(marker.read_bytes(), original)
            self.assertTrue(failed.is_dir())

    def test_successful_publication_copies_and_preserves_experiment_directory(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            formal = root / "models"
            formal.mkdir()
            (formal / "old.bin").write_bytes(b"old")
            mark_owned_model_directory(formal)
            write_publishable_experiment(experiment)

            published = publish_experiment(experiment, formal)

            self.assertTrue(published)
            self.assertEqual((experiment / "winner.bin").read_bytes(), b"winner-model")
            self.assertEqual((formal / "winner.bin").read_bytes(), b"winner-model")
            self.assertFalse((formal / "old.bin").exists())
            self.assertTrue(all((experiment / name).is_file() for name in train_model.REQUIRED_REPORTS))

    def test_publication_rejects_artifact_path_outside_experiment(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            formal = root / "models"
            formal.mkdir()
            marker = formal / "marker.bin"
            marker.write_bytes(b"formal-before")
            mark_owned_model_directory(formal)
            write_publishable_experiment(experiment, artifact_path="../outside.bin")
            (root / "outside.bin").write_bytes(b"outside")

            with self.assertRaisesRegex(ValueError, "inside"):
                publish_experiment(experiment, formal)

            self.assertEqual(marker.read_bytes(), b"formal-before")
            self.assertTrue(experiment.is_dir())

    def test_publication_rejects_formal_directory_ancestor_of_experiment(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "models"
            formal.mkdir()
            formal_marker = formal / "formal-marker.bin"
            formal_bytes = bytes(range(64))
            formal_marker.write_bytes(formal_bytes)
            experiment = formal / "experiments" / "run"
            experiment.parent.mkdir()
            write_publishable_experiment(experiment)
            experiment_marker = experiment / "winner.bin"
            experiment_bytes = experiment_marker.read_bytes()

            with patch.object(
                train_model.shutil,
                "copytree",
                side_effect=ValueError("copy attempted before relationship rejection"),
            ) as copytree:
                with self.assertRaisesRegex(ValueError, "ancestor|descendant"):
                    publish_experiment(experiment, formal)

            copytree.assert_not_called()
            self.assertEqual(formal_marker.read_bytes(), formal_bytes)
            self.assertEqual(experiment_marker.read_bytes(), experiment_bytes)
            self.assertEqual(list(root.glob(".models.staging-*")), [])
            self.assertEqual(list(root.glob(".models.backup-*")), [])

    def test_publication_rejects_experiment_directory_ancestor_of_formal(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            write_publishable_experiment(experiment)
            experiment_marker = experiment / "winner.bin"
            experiment_bytes = experiment_marker.read_bytes()
            formal = experiment / "formal"
            formal.mkdir()
            formal_marker = formal / "formal-marker.bin"
            formal_bytes = bytes(reversed(range(64)))
            formal_marker.write_bytes(formal_bytes)

            with patch.object(
                train_model.shutil,
                "copytree",
                side_effect=ValueError("copy attempted before relationship rejection"),
            ) as copytree:
                with self.assertRaisesRegex(ValueError, "ancestor|descendant"):
                    publish_experiment(experiment, formal)

            copytree.assert_not_called()
            self.assertEqual(formal_marker.read_bytes(), formal_bytes)
            self.assertEqual(experiment_marker.read_bytes(), experiment_bytes)
            self.assertEqual(list(experiment.glob(".formal.staging-*")), [])
            self.assertEqual(list(experiment.glob(".formal.backup-*")), [])

    def test_publication_rechecks_mislabeled_passing_gate(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            formal = root / "models"
            formal.mkdir()
            marker = formal / "marker.bin"
            marker.write_bytes(b"formal-before")
            mark_owned_model_directory(formal)
            write_publishable_experiment(experiment)
            metrics_path = experiment / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["test_metrics"]["acc_10"] = 0.1
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

            published = publish_experiment(experiment, formal)

            self.assertFalse(published)
            self.assertEqual(marker.read_bytes(), b"formal-before")

    def test_atomic_publication_rolls_back_when_staging_swap_fails(self):
        publish_experiment = self.require_function("publish_experiment")
        rename_directory = self.require_function("_rename_directory")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            formal = root / "models"
            formal.mkdir()
            marker = formal / "marker.bin"
            marker.write_bytes(b"formal-before")
            mark_owned_model_directory(formal)
            write_publishable_experiment(experiment)

            def fail_staging_swap(source, target):
                if ".staging-" in Path(source).name:
                    raise OSError("simulated staging swap failure")
                return rename_directory(source, target)

            with patch.object(train_model, "_rename_directory", side_effect=fail_staging_swap):
                with self.assertRaisesRegex(OSError, "staging swap"):
                    publish_experiment(experiment, formal)

            self.assertEqual(marker.read_bytes(), b"formal-before")
            self.assertTrue(experiment.is_dir())
            self.assertEqual(list(root.glob(".models.backup-*")), [])
            self.assertEqual(list(root.glob(".models.staging-*")), [])

    def test_publication_cleans_staging_when_copied_artifact_validation_fails(self):
        publish_experiment = self.require_function("publish_experiment")
        validate_experiment = self.require_function("_validate_experiment_directory")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            formal = root / "models"
            formal.mkdir()
            marker = formal / "marker.bin"
            marker.write_bytes(b"formal-before")
            mark_owned_model_directory(formal)
            write_publishable_experiment(experiment)

            def fail_staging_validation(path):
                if ".staging-" in Path(path).name:
                    raise ValueError("simulated copied artifact failure")
                return validate_experiment(path)

            with patch.object(
                train_model,
                "_validate_experiment_directory",
                side_effect=fail_staging_validation,
            ):
                with self.assertRaisesRegex(ValueError, "copied artifact"):
                    publish_experiment(experiment, formal)

            self.assertEqual(marker.read_bytes(), b"formal-before")
            self.assertEqual(list(root.glob(".models.staging-*")), [])
            self.assertEqual(list(root.glob(".models.backup-*")), [])

    def test_publication_rechecks_gate_on_staged_copy_before_formal_swap(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            formal = root / "models"
            formal.mkdir()
            marker = formal / "marker.bin"
            marker.write_bytes(b"formal-before")
            mark_owned_model_directory(formal)
            write_publishable_experiment(experiment)
            real_copytree = train_model.shutil.copytree

            def copy_then_replace_gate(source, target, *args, **kwargs):
                result = real_copytree(source, target, *args, **kwargs)
                metrics_path = Path(target) / "metrics.json"
                card_path = Path(target) / "model_card.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                card = json.loads(card_path.read_text(encoding="utf-8"))
                failed_metrics = dict(metrics["test_metrics"])
                failed_metrics.update(r2=-0.1, acc_10=0.1, rmse=30.0)
                failed_gate = train_model.assess_quality_gate(failed_metrics)
                metrics.update(
                    test_metrics=failed_metrics,
                    quality_gate=failed_gate["quality_gate"],
                    warnings=failed_gate["warnings"],
                    thresholds=failed_gate["thresholds"],
                )
                card.update(
                    test_metrics=failed_metrics,
                    quality_gate=failed_gate["quality_gate"],
                    warnings=failed_gate["warnings"],
                    thresholds=failed_gate["thresholds"],
                )
                card["independent_holdout"].update(
                    metrics=failed_metrics,
                    quality_gate=failed_gate,
                )
                metrics_path.write_text(
                    json.dumps(metrics, allow_nan=False), encoding="utf-8"
                )
                card_path.write_text(
                    json.dumps(card, allow_nan=False), encoding="utf-8"
                )
                return result

            with patch.object(
                train_model.shutil,
                "copytree",
                side_effect=copy_then_replace_gate,
            ):
                with self.assertRaisesRegex(ValueError, "gate|passing"):
                    publish_experiment(experiment, formal)

            self.assertEqual(marker.read_bytes(), b"formal-before")
            self.assertEqual(list(root.glob(".models.staging-*")), [])
            self.assertEqual(list(root.glob(".models.backup-*")), [])
            self.assertFalse((root / ".models.publish.lock").exists())

    def test_publication_rejects_protected_and_broad_targets_before_mutation(self):
        publish_experiment = self.require_function("publish_experiment")
        source_path = Path(train_model.__file__).resolve()
        backend_root = source_path.parents[1]
        project_root = source_path.parents[2]
        targets = {
            "home": Path.home().resolve(),
            "filesystem anchor": Path(project_root.anchor).resolve(),
            "backend root": backend_root,
            "project root": project_root,
            "worktree container": project_root.parent,
            "workspace repository root": project_root.parents[1],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            write_publishable_experiment(experiment)
            for label, target in targets.items():
                with self.subTest(target=label):
                    with (
                        patch.object(
                            train_model.shutil,
                            "copytree",
                            side_effect=AssertionError("copy must not be attempted"),
                        ) as copytree,
                        patch.object(
                            train_model,
                            "_rename_directory",
                            side_effect=AssertionError("rename must not be attempted"),
                        ) as rename,
                        patch.object(
                            train_model.shutil,
                            "rmtree",
                            side_effect=AssertionError("delete must not be attempted"),
                        ) as rmtree,
                    ):
                        with self.assertRaisesRegex(ValueError, "protected|model directory"):
                            publish_experiment(experiment, target)
                        copytree.assert_not_called()
                        rename.assert_not_called()
                        rmtree.assert_not_called()

            broad = root / "uploads"
            broad.mkdir()
            marker = broad / "user-content.bin"
            marker.write_bytes(b"preserve-user-content")
            with self.assertRaisesRegex(ValueError, "owned|model directory"):
                publish_experiment(experiment, broad)
            self.assertEqual(marker.read_bytes(), b"preserve-user-content")
            self.assertEqual(list(root.glob(".uploads.staging-*")), [])
            self.assertEqual(list(root.glob(".uploads.backup-*")), [])
            self.assertFalse((root / ".uploads.publish.lock").exists())

    def test_protected_paths_include_real_home_and_root_ancestors(self):
        protected_paths = self.require_function("_protected_publication_paths")()
        home = Path.home().resolve()
        source_path = Path(train_model.__file__).resolve()
        backend_root = source_path.parents[1]
        project_root = source_path.parents[2]

        for path in {
            home,
            *home.parents,
            backend_root,
            *backend_root.parents,
            project_root,
            *project_root.parents,
            Path(project_root.anchor).resolve(),
        }:
            with self.subTest(path=path):
                self.assertIn(path, protected_paths)

    def test_simulated_users_directory_cannot_be_authorized_by_sentinel(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            simulated_home = root / "Users" / "person"
            simulated_home.mkdir(parents=True)
            users_dir = simulated_home.parent
            profile = users_dir / "profile.bin"
            profile_bytes = bytes(range(128))
            profile.write_bytes(profile_bytes)
            mark_owned_model_directory(users_dir)
            experiment = root / "experiment"
            write_publishable_experiment(experiment)

            with (
                patch.object(train_model.Path, "home", return_value=simulated_home),
                patch.object(train_model.shutil, "copytree") as copytree,
                patch.object(train_model, "_rename_directory") as rename,
                patch.object(train_model.shutil, "rmtree") as rmtree,
            ):
                with self.assertRaisesRegex(ValueError, "protected"):
                    publish_experiment(experiment, users_dir)

            copytree.assert_not_called()
            rename.assert_not_called()
            rmtree.assert_not_called()
            self.assertEqual(profile.read_bytes(), profile_bytes)

    def test_publication_rejects_dangling_directory_link_before_mutation(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            target = root / "removed-target"
            link = root / "dangling-models"
            target.mkdir()
            write_publishable_experiment(experiment)
            try:
                os.symlink(target, link, target_is_directory=True)
            except OSError as exc:
                if os.name != "nt":
                    self.skipTest(f"directory links are unavailable: {exc}")
                junction = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if junction.returncode != 0:
                    self.skipTest(
                        f"directory links are unavailable: {junction.stderr or exc}"
                    )
            target.rmdir()
            self.assertTrue(os.path.lexists(link))
            self.assertFalse(link.exists())

            try:
                with (
                    patch.object(train_model.shutil, "copytree") as copytree,
                    patch.object(train_model, "_rename_directory") as rename,
                    patch.object(train_model.shutil, "rmtree") as rmtree,
                ):
                    with self.assertRaisesRegex(ValueError, "dangling|link|junction"):
                        publish_experiment(experiment, link)
                copytree.assert_not_called()
                rename.assert_not_called()
                rmtree.assert_not_called()
            finally:
                if os.path.lexists(link):
                    os.rmdir(link)

    def test_publication_rejects_symlink_resolving_to_protected_directory(self):
        publish_experiment = self.require_function("publish_experiment")
        backend_root = Path(train_model.__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            link = root / "models-link"
            write_publishable_experiment(experiment)
            try:
                os.symlink(backend_root, link, target_is_directory=True)
            except OSError as exc:
                if os.name != "nt":
                    self.skipTest(f"directory symlinks are unavailable: {exc}")
                junction = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link), str(backend_root)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if junction.returncode != 0:
                    self.skipTest(
                        f"directory links are unavailable: {junction.stderr or exc}"
                    )

            try:
                with (
                    patch.object(train_model.shutil, "copytree") as copytree,
                    patch.object(train_model, "_rename_directory") as rename,
                    patch.object(train_model.shutil, "rmtree") as rmtree,
                ):
                    with self.assertRaisesRegex(ValueError, "protected|model directory"):
                        publish_experiment(experiment, link)
                copytree.assert_not_called()
                rename.assert_not_called()
                rmtree.assert_not_called()
            finally:
                if link.exists():
                    os.rmdir(link)

    def test_first_publication_installs_ownership_sentinel_for_future_updates(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            formal = root / "models"
            write_publishable_experiment(first)
            write_publishable_experiment(second)

            self.assertTrue(publish_experiment(first, formal))
            sentinel = formal / ".model-publication-owner.json"
            self.assertTrue(sentinel.is_file())
            formal_status = json.loads(
                (formal / "run_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(formal_status["status"], "completed")
            self.assertTrue(formal_status["published"])
            self.assertTrue(publish_experiment(second, formal))
            self.assertTrue(sentinel.is_file())
            self.assertEqual(list(root.glob(".models.staging-*")), [])
            self.assertEqual(list(root.glob(".models.backup-*")), [])
            self.assertFalse((root / ".models.publish.lock").exists())

    def test_publication_reresolves_target_after_lock_before_mutation(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            formal = root / "models"
            protected = Path(train_model.__file__).resolve().parents[1]
            write_publishable_experiment(experiment)

            with (
                patch.object(
                    train_model,
                    "_resolve_publication_target",
                    side_effect=[formal.resolve(), protected],
                ) as resolve_target,
                patch.object(
                    train_model.shutil,
                    "copytree",
                    side_effect=self.fail,
                ) as copytree,
            ):
                with self.assertRaisesRegex(ValueError, "protected|changed"):
                    publish_experiment(experiment, formal)

            self.assertEqual(resolve_target.call_count, 2)
            copytree.assert_not_called()
            self.assertFalse((root / ".models.publish.lock").exists())

    def test_publication_rejects_file_target_before_mutation(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            formal = root / "models"
            write_publishable_experiment(experiment)
            formal.write_bytes(b"not-a-directory")

            with (
                patch.object(train_model.shutil, "copytree") as copytree,
                patch.object(train_model, "_rename_directory") as rename,
                patch.object(train_model.shutil, "rmtree") as rmtree,
            ):
                with self.assertRaisesRegex(ValueError, "directory|file"):
                    publish_experiment(experiment, formal)
            copytree.assert_not_called()
            rename.assert_not_called()
            rmtree.assert_not_called()
            self.assertEqual(formal.read_bytes(), b"not-a-directory")

    def test_concurrent_publisher_fails_before_mutation_and_releases_lock(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            formal = root / "models"
            write_publishable_experiment(first)
            write_publishable_experiment(second)
            entered_copy = threading.Event()
            release_copy = threading.Event()
            real_copytree = train_model.shutil.copytree
            outcome = {}

            def blocking_copytree(source, target, *args, **kwargs):
                if threading.current_thread().name == "first-publisher":
                    entered_copy.set()
                    if not release_copy.wait(timeout=5):
                        raise TimeoutError("test did not release first publisher")
                return real_copytree(source, target, *args, **kwargs)

            def publish_first():
                try:
                    outcome["result"] = publish_experiment(first, formal)
                except Exception as exc:
                    outcome["error"] = exc

            with patch.object(train_model.shutil, "copytree", side_effect=blocking_copytree):
                worker = threading.Thread(target=publish_first, name="first-publisher")
                worker.start()
                self.assertTrue(entered_copy.wait(timeout=5))
                try:
                    with self.assertRaisesRegex(FileExistsError, "publication|lock"):
                        publish_experiment(second, formal)
                finally:
                    release_copy.set()
                    worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            self.assertNotIn("error", outcome)
            self.assertTrue(outcome.get("result"))
            self.assertTrue((formal / "winner.bin").is_file())
            self.assertTrue((formal / ".model-publication-owner.json").is_file())
            self.assertEqual(list(root.glob(".models.staging-*")), [])
            self.assertEqual(list(root.glob(".models.backup-*")), [])
            self.assertFalse((root / ".models.publish.lock").exists())

    def test_completed_lock_release_failure_is_recovered_by_next_publication(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            formal = root / "models"
            lock_dir = root / ".models.publish.lock"
            write_publishable_experiment(first)
            write_publishable_experiment(second)
            real_rmtree = train_model.shutil.rmtree

            def fail_lock_release(path, *args, **kwargs):
                if Path(path) == lock_dir:
                    raise PermissionError("simulated lock release failure")
                return real_rmtree(path, *args, **kwargs)

            with patch.object(
                train_model.shutil,
                "rmtree",
                side_effect=fail_lock_release,
            ):
                published = publish_experiment(first, formal)

            self.assertTrue(published)
            self.assertTrue((formal / "winner.bin").is_file())
            self.assertTrue(lock_dir.is_dir())
            lock_metadata = json.loads(
                (lock_dir / "lock.json").read_text(encoding="utf-8"),
                parse_constant=lambda value: self.fail(
                    f"non-strict lock JSON constant: {value}"
                ),
            )
            self.assertEqual(lock_metadata["state"], "completed")
            self.assertTrue(lock_metadata["owner"])
            self.assertTrue(lock_metadata["acquired_at"])
            self.assertTrue(lock_metadata["updated_at"])

            (formal / "first-marker.bin").write_bytes(b"first")
            self.assertTrue(publish_experiment(second, formal))
            self.assertFalse((formal / "first-marker.bin").exists())
            self.assertFalse(lock_dir.exists())

    def test_failed_terminal_lock_is_recovered_without_masking_original_error(self):
        publication_lock = self.require_function("_publication_lock")
        acquire_lock = self.require_function("_acquire_publication_lock")
        mark_lock = self.require_function("_mark_publication_lock")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "models"
            real_mark_lock = train_model._mark_publication_lock

            def fail_failed_state(lock, state, error=None):
                if state == "failed":
                    raise PermissionError("cannot update failed lock state")
                return real_mark_lock(lock, state, error)

            with patch.object(
                train_model,
                "_mark_publication_lock",
                side_effect=fail_failed_state,
            ):
                with self.assertRaisesRegex(RuntimeError, "original publication error"):
                    with publication_lock(formal):
                        raise RuntimeError("original publication error")

            self.assertFalse((root / ".models.publish.lock").exists())

            stale_lock = acquire_lock(formal)
            mark_lock(stale_lock, "failed", RuntimeError("recorded failure"))
            recovered_lock = acquire_lock(formal)
            self.assertNotEqual(recovered_lock.token, stale_lock.token)
            train_model._mark_publication_lock(recovered_lock, "completed")
            train_model._release_publication_lock(recovered_lock)
            self.assertFalse((root / ".models.publish.lock").exists())

    def test_post_swap_backup_cleanup_failure_keeps_success_and_recovery_copy(self):
        publish_experiment = self.require_function("publish_experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            formal = root / "models"
            write_publishable_experiment(first)
            write_publishable_experiment(second)
            self.assertTrue(publish_experiment(first, formal))
            (formal / "old-marker.bin").write_bytes(b"recoverable-old-model")
            real_rmtree = train_model.shutil.rmtree

            def fail_backup_cleanup(path, *args, **kwargs):
                if ".backup-" in Path(path).name:
                    raise OSError("simulated backup cleanup failure")
                return real_rmtree(path, *args, **kwargs)

            with patch.object(
                train_model.shutil,
                "rmtree",
                side_effect=fail_backup_cleanup,
            ):
                published = publish_experiment(second, formal)

            self.assertTrue(published)
            self.assertFalse((formal / "old-marker.bin").exists())
            backups = list(root.glob(".models.backup-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(
                (backups[0] / "old-marker.bin").read_bytes(),
                b"recoverable-old-model",
            )
            warning = json.loads(
                (second / "publication_warning.json").read_text(encoding="utf-8"),
                parse_constant=lambda value: self.fail(
                    f"non-strict warning JSON constant: {value}"
                ),
            )
            self.assertEqual(warning["warning_type"], "backup_cleanup_failed")
            self.assertEqual(warning["error_type"], "OSError")
            self.assertEqual(warning["backup_name"], backups[0].name)
            self.assertEqual(list(root.glob(".models.staging-*")), [])
            self.assertFalse((root / ".models.publish.lock").exists())

    def test_failed_publish_override_is_removed_from_api_and_cli(self):
        self.assertNotIn("allow_failed_publish", inspect.signature(train_model.train_and_publish).parameters)
        source = Path(train_model.__file__).read_text(encoding="utf-8")
        self.assertNotIn("--allow-failed-publish", source)


if __name__ == "__main__":
    unittest.main()
