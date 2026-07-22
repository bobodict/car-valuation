import inspect
import json
import tempfile
import unittest
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
            "development": list(range(8)),
            "test": list(range(8, 12)),
            "folds": [
                {"fold": 0, "train": [0, 1, 2, 3], "validation": [4, 5, 6, 7]}
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
        "error_analysis": {
            "development_mean_baseline": float(frame.loc[range(8), "price"].mean()),
            "price_quartiles": {"groups": []},
            "model_family_frequency": {"groups": []},
            "full_model_seen_status": {"groups": []},
        },
        "seed": 42,
        "collection_year": 2026,
        "provenance": {"source_id": "unit-test"},
    }


def write_publishable_experiment(root, quality_gate="pass", artifact_path="winner.bin"):
    root.mkdir()
    (root / "winner.bin").write_bytes(b"winner-model")
    manifest = {
        "artifact_version": "3.0.0",
        "model_type": "extra_trees",
        "model_artifacts": {"bundle": artifact_path},
    }
    metrics = {
        "artifact_version": "3.0.0",
        "quality_gate": quality_gate,
        "thresholds": {"min_r2": 0.0, "min_acc_10": 0.5},
        "test_metrics": {
            "r2": 0.8,
            "acc_10": 0.75,
            "rmse": 10.0,
            "baseline_rmse": 25.0,
        },
    }
    payloads = {
        "model_manifest.json": manifest,
        "feature_config.json": {"artifact_version": "3.0.0"},
        "metrics.json": metrics,
        "leaderboard.json": {"artifact_version": "3.0.0", "candidates": []},
        "error_analysis.json": {"artifact_version": "3.0.0"},
        "model_card.json": {"artifact_version": "3.0.0"},
    }
    for name, payload in payloads.items():
        (root / name).write_text(
            json.dumps(payload, allow_nan=False), encoding="utf-8"
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
                output_dir.mkdir(parents=True)
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

        self.assertEqual(
            events,
            ["load", "split", "competition", "select", "refit", "evaluate", "gate", "write", "smoke", "publish"],
        )
        self.assertEqual(evaluate_mock.call_count, 1)
        self.assertTrue(result["published"])

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
                output_dir.mkdir(parents=True)
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

    def test_failed_publish_override_is_removed_from_api_and_cli(self):
        self.assertNotIn("allow_failed_publish", inspect.signature(train_model.train_and_publish).parameters)
        source = Path(train_model.__file__).read_text(encoding="utf-8")
        self.assertNotIn("--allow-failed-publish", source)


if __name__ == "__main__":
    unittest.main()
