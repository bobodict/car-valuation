import json
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from services import model_competition
from services.feature_engineering import CATEGORICAL_FEATURES, MODEL_FEATURES
from services.model_competition import (
    CATBOOST_CONFIGS,
    EXTRA_TREES_CONFIGS,
    MLP_CONFIGS,
    CandidateConfig,
    calculate_metrics,
    candidate_sort_key,
    rank_candidates,
)


def candidate(
    name,
    acc_10,
    median_ape,
    r2,
    complexity,
    test_acc_10=0.5,
    config=None,
):
    return {
        "name": name,
        "cv": {
            "acc_10_mean": acc_10,
            "median_ape_mean": median_ape,
            "r2_mean": r2,
        },
        "complexity": complexity,
        "config": config or {},
        "test_metrics": {"acc_10": test_acc_10},
    }


def make_cv_fixture():
    row_ids = list(range(100, 112))
    frame = pd.DataFrame(
        {
            "price": np.linspace(100_000.0, 210_000.0, len(row_ids)),
            "year": [2020] * len(row_ids),
            "mileage": np.linspace(10_000.0, 21_000.0, len(row_ids)),
            "brand": ["Toyota", "Honda"] * 6,
            "model": [f"model-{index % 3}" for index in range(len(row_ids))],
        },
        index=row_ids,
    )
    development = row_ids[:10]
    folds = []
    for fold_number in range(5):
        validation = development[fold_number * 2 : fold_number * 2 + 2]
        folds.append(
            {
                "fold": fold_number,
                "train": [row_id for row_id in development if row_id not in validation],
                "validation": validation,
            }
        )
    manifest = {
        "development": development,
        "test": row_ids[10:],
        "n_splits": 5,
        "folds": folds,
    }
    return frame, manifest


def tiny_candidate_configs():
    return (
        CandidateConfig(
            "tiny-catboost",
            "catboost",
            {
                "depth": 2,
                "learning_rate": 0.1,
                "l2_leaf_reg": 3.0,
                "loss_function": "RMSE",
                "iterations": 6,
                "early_stopping_patience": 2,
            },
            1,
        ),
        CandidateConfig(
            "tiny-extra-trees",
            "extra_trees",
            {
                "n_estimators": 5,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "n_jobs": 1,
            },
            1,
        ),
        CandidateConfig(
            "tiny-mlp",
            "mlp",
            {
                "hidden_dims": (8,),
                "dropout": 0.0,
                "learning_rate": 0.01,
                "max_epochs": 4,
                "early_stopping_patience": 2,
            },
            1,
        ),
    )


class SealedTestManifest(Mapping):
    def __init__(self, values):
        self._values = values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("outer test value accessed")
        return self._values[key]


class ModelCompetitionTests(unittest.TestCase):
    def test_metrics_include_relative_and_log_errors(self):
        metrics = calculate_metrics(
            actual=np.array([100.0, 200.0]),
            predicted=np.array([110.0, 180.0]),
            baseline=np.array([150.0, 150.0]),
        )

        self.assertEqual(metrics["acc_10"], 1.0)
        self.assertEqual(metrics["acc_20"], 1.0)
        self.assertAlmostEqual(metrics["median_ape"], 0.1)
        expected_rmsle = np.sqrt(
            np.mean((np.log1p([110.0, 180.0]) - np.log1p([100.0, 200.0])) ** 2)
        )
        self.assertAlmostEqual(metrics["rmsle"], expected_rmsle)

    def test_acc_10_distinguishes_exact_threshold_from_decimal_outside(self):
        actual = np.array([100.0, 200.0])
        exact = calculate_metrics(actual, actual * 1.1, actual)
        outside = calculate_metrics(
            actual,
            actual * 1.100000000000001,
            actual,
        )

        self.assertEqual(exact["acc_10"], 1.0)
        self.assertEqual(outside["acc_10"], 0.0)

    def test_acc_20_distinguishes_exact_threshold_from_decimal_outside(self):
        actual = np.array([100.0, 200.0])
        exact = calculate_metrics(actual, actual * 1.2, actual)
        outside = calculate_metrics(
            actual,
            actual * 1.200000000000001,
            actual,
        )

        self.assertEqual(exact["acc_20"], 1.0)
        self.assertEqual(outside["acc_20"], 0.0)

    def test_relative_accuracy_counts_exact_bounds_across_magnitudes(self):
        actual = np.array(
            [3.0, 9.0, 43.0, 109.0, 0.1, 123.456, 1e-6, 1e9]
        )

        for threshold, metric_name in ((0.10, "acc_10"), (0.20, "acc_20")):
            for factor in (1.0 - threshold, 1.0 + threshold):
                with self.subTest(threshold=threshold, factor=factor):
                    metrics = calculate_metrics(
                        actual,
                        actual * factor,
                        actual,
                    )

                    self.assertEqual(metrics[metric_name], 1.0)

    def test_relative_accuracy_rejects_factors_outside_exact_bounds(self):
        actual = np.array(
            [3.0, 9.0, 43.0, 109.0, 0.1, 123.456, 1e-6, 1e9]
        )
        outside_cases = (
            ("acc_10", 0.899999999999999),
            ("acc_10", 1.100000000000001),
            ("acc_20", 0.799999999999999),
            ("acc_20", 1.200000000000001),
        )

        for metric_name, factor in outside_cases:
            with self.subTest(metric_name=metric_name, factor=factor):
                metrics = calculate_metrics(actual, actual * factor, actual)

                self.assertEqual(metrics[metric_name], 0.0)

    def test_relative_accuracy_preserves_zero_actual_fallback(self):
        metrics = calculate_metrics(
            actual=np.array([0.0, 0.0]),
            predicted=np.array([0.0, 1e-8]),
            baseline=np.array([0.0, 0.0]),
        )

        self.assertEqual(metrics["acc_10"], 0.5)
        self.assertEqual(metrics["acc_20"], 0.5)

    def test_metrics_reject_negative_price_series(self):
        valid = np.array([100.0, 200.0])

        for name in ("actual", "predicted", "baseline"):
            values = {
                "actual": valid.copy(),
                "predicted": valid.copy(),
                "baseline": valid.copy(),
            }
            values[name][0] = -1.0
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{name} must contain only nonnegative prices",
                ):
                    calculate_metrics(**values)

    def test_metrics_reject_sklearn_negative_one_rmsle_edge_case(self):
        with self.assertRaisesRegex(
            ValueError,
            "actual must contain only nonnegative prices",
        ):
            calculate_metrics(
                actual=np.array([-1.0, 0.0]),
                predicted=np.array([-1.0, 0.0]),
                baseline=np.array([0.0, 0.0]),
            )

    def test_metrics_reject_nonfinite_price_series(self):
        valid = np.array([100.0, 200.0])

        for name in ("actual", "predicted", "baseline"):
            for invalid in (np.nan, np.inf, -np.inf):
                values = {
                    "actual": valid.copy(),
                    "predicted": valid.copy(),
                    "baseline": valid.copy(),
                }
                values[name][0] = invalid
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{name} must contain only finite prices",
                    ):
                        calculate_metrics(**values)

    def test_metrics_reject_column_and_multioutput_arrays(self):
        valid = np.array([100.0, 200.0])
        invalid_shapes = (
            np.array([[100.0], [200.0]]),
            np.array([[100.0, 200.0], [300.0, 400.0]]),
        )

        for name in ("actual", "predicted", "baseline"):
            for invalid in invalid_shapes:
                values = {
                    "actual": valid,
                    "predicted": valid,
                    "baseline": valid,
                    name: invalid,
                }
                with self.subTest(name=name, shape=invalid.shape):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{name} must be one-dimensional",
                    ):
                        calculate_metrics(**values)

    def test_metrics_reject_scalar_and_empty_arrays(self):
        valid = np.array([100.0, 200.0])

        for name in ("actual", "predicted", "baseline"):
            for invalid, message in (
                (100.0, f"{name} must be one-dimensional"),
                (np.array([]), f"{name} must not be empty"),
            ):
                values = {
                    "actual": valid,
                    "predicted": valid,
                    "baseline": valid,
                    name: invalid,
                }
                with self.subTest(name=name, message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        calculate_metrics(**values)

    def test_metrics_require_equal_lengths_before_broadcasting(self):
        cases = (
            (
                np.array([100.0, 200.0, 300.0]),
                np.array([100.0, 200.0]),
                np.array([100.0, 200.0, 300.0]),
            ),
            (
                np.array([100.0, 200.0]),
                np.array([100.0]),
                np.array([100.0, 200.0]),
            ),
        )

        for actual, predicted, baseline in cases:
            lengths = (len(actual), len(predicted), len(baseline))
            with self.subTest(lengths=lengths):
                with self.assertRaisesRegex(
                    ValueError,
                    "actual, predicted, and baseline must have equal lengths",
                ):
                    calculate_metrics(actual, predicted, baseline)

    def test_metrics_require_at_least_two_samples(self):
        values = np.array([100.0])

        with self.assertRaisesRegex(
            ValueError,
            "metric inputs must contain at least two samples",
        ):
            calculate_metrics(values, values, values)

    def test_candidate_config_and_search_spaces_are_exact_and_bounded(self):
        config = CandidateConfig(
            name="cat-depth-6",
            model_type="catboost",
            params=CATBOOST_CONFIGS[0],
            complexity=2,
        )

        self.assertEqual(config.name, "cat-depth-6")
        self.assertEqual(config.model_type, "catboost")
        self.assertEqual(config.params, CATBOOST_CONFIGS[0])
        self.assertEqual(config.complexity, 2)
        self.assertEqual(
            CATBOOST_CONFIGS,
            (
                {
                    "depth": 6,
                    "learning_rate": 0.03,
                    "l2_leaf_reg": 3.0,
                    "loss_function": "RMSE",
                },
                {
                    "depth": 7,
                    "learning_rate": 0.03,
                    "l2_leaf_reg": 5.0,
                    "loss_function": "RMSE",
                },
                {
                    "depth": 8,
                    "learning_rate": 0.03,
                    "l2_leaf_reg": 8.0,
                    "loss_function": "RMSE",
                },
                {
                    "depth": 7,
                    "learning_rate": 0.05,
                    "l2_leaf_reg": 10.0,
                    "loss_function": "RMSE",
                },
            ),
        )
        self.assertEqual(
            EXTRA_TREES_CONFIGS,
            (
                {
                    "n_estimators": 600,
                    "min_samples_leaf": 1,
                    "max_features": 0.7,
                    "n_jobs": -1,
                },
                {
                    "n_estimators": 600,
                    "min_samples_leaf": 1,
                    "max_features": 1.0,
                    "n_jobs": -1,
                },
                {
                    "n_estimators": 600,
                    "min_samples_leaf": 2,
                    "max_features": 0.7,
                    "n_jobs": -1,
                },
                {
                    "n_estimators": 600,
                    "min_samples_leaf": 2,
                    "max_features": 1.0,
                    "n_jobs": -1,
                },
            ),
        )
        self.assertEqual(
            MLP_CONFIGS,
            (
                {
                    "hidden_dims": (128, 64),
                    "dropout": 0.0,
                    "learning_rate": 0.001,
                },
                {
                    "hidden_dims": (256, 128),
                    "dropout": 0.1,
                    "learning_rate": 0.001,
                },
                {
                    "hidden_dims": (128, 64, 32),
                    "dropout": 0.1,
                    "learning_rate": 0.0005,
                },
            ),
        )

    def test_future_training_bounds_are_explicit_and_exported(self):
        bounds = {
            "CATBOOST_MAX_ITERATIONS": getattr(
                model_competition, "CATBOOST_MAX_ITERATIONS", None
            ),
            "CATBOOST_EARLY_STOPPING_PATIENCE": getattr(
                model_competition, "CATBOOST_EARLY_STOPPING_PATIENCE", None
            ),
            "MLP_MAX_EPOCHS": getattr(
                model_competition, "MLP_MAX_EPOCHS", None
            ),
            "MLP_EARLY_STOPPING_PATIENCE": getattr(
                model_competition, "MLP_EARLY_STOPPING_PATIENCE", None
            ),
        }

        self.assertEqual(
            bounds,
            {
                "CATBOOST_MAX_ITERATIONS": 1500,
                "CATBOOST_EARLY_STOPPING_PATIENCE": 80,
                "MLP_MAX_EPOCHS": 400,
                "MLP_EARLY_STOPPING_PATIENCE": 40,
            },
        )

    def test_evaluate_candidate_uses_only_recorded_fold_rows(self):
        evaluate_candidate = getattr(
            model_competition,
            "evaluate_candidate",
            None,
        )
        self.assertIsNotNone(evaluate_candidate)
        frame, manifest = make_cv_fixture()
        config = CandidateConfig(
            "tiny-extra-trees",
            "extra_trees",
            {
                "n_estimators": 1,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "n_jobs": 1,
            },
            1,
        )
        observed_fits = []

        class RecordingAdapter:
            def fit(self, train_frame, validation_frame=None):
                observed_fits.append(
                    {
                        "train": train_frame.index.tolist(),
                        "validation": validation_frame.index.tolist(),
                    }
                )
                self.prediction = float(train_frame["price"].mean())
                return self

            def predict(self, prediction_frame):
                return np.full(len(prediction_frame), self.prediction)

            def save(self, directory):
                return {"model_type": "recording"}

        with patch.object(
            model_competition,
            "_create_adapter",
            create=True,
            side_effect=lambda candidate_config, seed: RecordingAdapter(),
        ):
            result = evaluate_candidate(frame, manifest, config, seed=42)

        self.assertEqual(len(result["fold_metrics"]), 5)
        for observed, expected in zip(
            result["observed_validation_indices"],
            manifest["folds"],
        ):
            self.assertEqual(observed, expected["validation"])
        self.assertEqual(
            observed_fits,
            [
                {
                    "train": fold["train"],
                    "validation": fold["validation"],
                }
                for fold in manifest["folds"]
            ],
        )
        self.assertNotIn("test_metrics", result)
        self.assertTrue(
            {
                "acc_10_mean",
                "acc_10_std",
                "median_ape_mean",
                "r2_mean",
                "rmse_mean",
                "baseline_rmse_mean",
            }.issubset(result["cv"])
        )

    def test_default_candidates_cover_all_model_families(self):
        default_candidates = getattr(
            model_competition,
            "default_candidates",
            None,
        )
        self.assertIsNotNone(default_candidates)
        self.assertEqual(
            {item.model_type for item in default_candidates()},
            {"catboost", "extra_trees", "mlp"},
        )

        first = default_candidates()
        second = default_candidates()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 11)
        self.assertEqual(len({item.name for item in first}), 11)
        self.assertTrue(all(item.complexity >= 0 for item in first))

    def test_evaluate_candidate_does_not_read_outer_test_value(self):
        frame, values = make_cv_fixture()
        manifest = SealedTestManifest(values)
        config = tiny_candidate_configs()[1]

        class NoOpAdapter:
            def fit(self, train_frame, validation_frame=None):
                self.prediction = float(train_frame["price"].mean())
                return self

            def predict(self, prediction_frame):
                return np.full(len(prediction_frame), self.prediction)

        try:
            with patch.object(
                model_competition,
                "_create_adapter",
                return_value=NoOpAdapter(),
            ):
                result = model_competition.evaluate_candidate(
                    frame,
                    manifest,
                    config,
                    seed=42,
                )
        except AssertionError as exc:
            self.fail(str(exc))

        self.assertEqual(len(result["fold_metrics"]), 5)

    def test_manifest_contract_requires_outer_test_membership(self):
        evaluate_candidate = getattr(
            model_competition,
            "evaluate_candidate",
            None,
        )
        self.assertIsNotNone(evaluate_candidate)
        frame, manifest = make_cv_fixture()
        del manifest["test"]
        config = tiny_candidate_configs()[1]

        class NoOpAdapter:
            def fit(self, train_frame, validation_frame=None):
                self.prediction = float(train_frame["price"].mean())
                return self

            def predict(self, prediction_frame):
                return np.full(len(prediction_frame), self.prediction)

        with patch.object(
            model_competition,
            "_create_adapter",
            create=True,
            return_value=NoOpAdapter(),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "manifest must include development, test, and folds",
            ):
                evaluate_candidate(frame, manifest, config, seed=42)

    def test_evaluate_candidate_rejects_seed_outside_uint32_range(self):
        frame, manifest = make_cv_fixture()
        config = tiny_candidate_configs()[1]

        for invalid_seed in (-1, 2**32):
            with self.subTest(seed=invalid_seed):
                with self.assertRaisesRegex(
                    ValueError,
                    r"seed must be an integer between 0 and 2\*\*32 - 1",
                ):
                    with patch.object(
                        model_competition,
                        "_create_adapter",
                        create=True,
                    ):
                        model_competition.evaluate_candidate(
                            frame,
                            manifest,
                            config,
                            seed=invalid_seed,
                        )

    def test_evaluate_candidate_validates_only_development_prices_clearly(self):
        frame, manifest = make_cv_fixture()
        frame["price"] = frame["price"].astype(object)
        frame.loc[manifest["development"][0], "price"] = "not-a-price"
        frame.loc[manifest["test"], "price"] = "sealed-test-value"
        config = tiny_candidate_configs()[1]

        with self.assertRaisesRegex(
            TypeError,
            "development prices must be a one-dimensional numeric array",
        ):
            with patch.object(
                model_competition,
                "_create_adapter",
                create=True,
            ):
                model_competition.evaluate_candidate(
                    frame,
                    manifest,
                    config,
                    seed=42,
                )

    def test_real_candidate_adapters_fit_predict_and_save_with_tiny_budgets(self):
        create_adapter = getattr(model_competition, "_create_adapter", None)
        self.assertIsNotNone(create_adapter)
        frame, manifest = make_cv_fixture()
        fold = manifest["folds"][0]
        train_frame = model_competition._model_frame(
            frame.loc[fold["train"]],
            2026,
        )
        validation_frame = model_competition._model_frame(
            frame.loc[fold["validation"]],
            2026,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for config in tiny_candidate_configs():
                with self.subTest(model_type=config.model_type):
                    adapter = create_adapter(config, seed=42)
                    fitted = adapter.fit(train_frame, validation_frame)
                    predictions = adapter.predict(
                        validation_frame.loc[:, MODEL_FEATURES]
                    )
                    artifact_dir = root / config.model_type
                    metadata = adapter.save(artifact_dir)

                    self.assertIs(fitted, adapter)
                    self.assertEqual(predictions.shape, (len(validation_frame),))
                    self.assertTrue(np.all(np.isfinite(predictions)))
                    self.assertTrue(np.all(predictions >= 0))
                    self.assertEqual(metadata["model_type"], config.model_type)
                    self.assertEqual(metadata["target_transform"], "log1p")
                    self.assertTrue(metadata["artifacts"])
                    for relative_path in metadata["artifacts"].values():
                        self.assertTrue((artifact_dir / relative_path).is_file())
                    json.dumps(metadata, allow_nan=False)

    def test_fold_preprocessors_fit_numeric_and_categories_on_train_only(self):
        create_adapter = getattr(model_competition, "_create_adapter", None)
        self.assertIsNotNone(create_adapter)
        frame, manifest = make_cv_fixture()
        fold = manifest["folds"][0]
        train_source = frame.loc[fold["train"]].copy()
        validation_source = frame.loc[fold["validation"]].copy()
        train_source.loc[fold["train"][0], "mileage"] = np.nan
        validation_source["mileage"] = 9_999_999.0
        validation_source["brand"] = "validation-only-brand"
        expected_median = float(train_source["mileage"].median())
        train_frame = model_competition._model_frame(train_source, 2026)
        validation_frame = model_competition._model_frame(validation_source, 2026)

        for config in tiny_candidate_configs()[1:]:
            with self.subTest(model_type=config.model_type):
                adapter = create_adapter(config, seed=42)
                adapter.fit(train_frame, validation_frame)
                numeric_pipeline = adapter.preprocessor.named_transformers_["numeric"]
                categorical_pipeline = adapter.preprocessor.named_transformers_[
                    "categorical"
                ]
                encoder = categorical_pipeline.named_steps["one_hot"]
                brand_index = CATEGORICAL_FEATURES.index("brand")

                self.assertEqual(
                    numeric_pipeline.named_steps["imputer"].statistics_[0],
                    expected_median,
                )
                self.assertNotIn(
                    "validation-only-brand",
                    encoder.categories_[brand_index],
                )

    def test_catboost_uses_native_categorical_feature_indexes(self):
        create_adapter = getattr(model_competition, "_create_adapter", None)
        self.assertIsNotNone(create_adapter)
        frame, manifest = make_cv_fixture()
        fold = manifest["folds"][0]
        train_frame = model_competition._model_frame(frame.loc[fold["train"]], 2026)
        validation_frame = model_competition._model_frame(
            frame.loc[fold["validation"]],
            2026,
        )
        adapter = create_adapter(tiny_candidate_configs()[0], seed=42)

        adapter.fit(train_frame, validation_frame)

        self.assertEqual(
            adapter.model.get_cat_feature_indices(),
            list(
                range(
                    len(MODEL_FEATURES) - len(CATEGORICAL_FEATURES),
                    len(MODEL_FEATURES),
                )
            ),
        )
        self.assertIsInstance(adapter.best_iteration, int)
        self.assertGreaterEqual(adapter.best_iteration, 0)

    def test_mlp_training_is_deterministic_for_fixed_seed(self):
        create_adapter = getattr(model_competition, "_create_adapter", None)
        self.assertIsNotNone(create_adapter)
        frame, manifest = make_cv_fixture()
        fold = manifest["folds"][0]
        train_frame = model_competition._model_frame(frame.loc[fold["train"]], 2026)
        validation_frame = model_competition._model_frame(
            frame.loc[fold["validation"]],
            2026,
        )

        first = create_adapter(tiny_candidate_configs()[2], seed=42)
        second = create_adapter(tiny_candidate_configs()[2], seed=42)
        first.fit(train_frame, validation_frame)
        second.fit(train_frame, validation_frame)

        np.testing.assert_allclose(
            first.predict(validation_frame.loc[:, MODEL_FEATURES]),
            second.predict(validation_frame.loc[:, MODEL_FEATURES]),
            rtol=0,
            atol=0,
        )
        self.assertEqual(first.best_epoch, second.best_epoch)

    def test_adapter_factory_rejects_invalid_family_params_and_budgets(self):
        create_adapter = getattr(model_competition, "_create_adapter", None)
        self.assertIsNotNone(create_adapter)
        cases = (
            (
                CandidateConfig("bad-family", "linear", {}, 0),
                "unsupported model_type: linear",
            ),
            (
                CandidateConfig(
                    "bad-cat-loss",
                    "catboost",
                    {"loss_function": "MAE"},
                    0,
                ),
                "catboost loss_function must be RMSE",
            ),
            (
                CandidateConfig(
                    "cat-over-budget",
                    "catboost",
                    {"loss_function": "RMSE", "iterations": 1501},
                    0,
                ),
                "catboost iterations must be between 1 and 1500",
            ),
            (
                CandidateConfig(
                    "mlp-over-budget",
                    "mlp",
                    {
                        "hidden_dims": (8,),
                        "dropout": 0.0,
                        "learning_rate": 0.01,
                        "max_epochs": 401,
                    },
                    0,
                ),
                "mlp max_epochs must be between 1 and 400",
            ),
            (
                CandidateConfig(
                    "unknown-param",
                    "extra_trees",
                    {"n_estimators": 5, "not_a_parameter": True},
                    0,
                ),
                "unsupported extra_trees parameters: not_a_parameter",
            ),
        )

        for config, message in cases:
            with self.subTest(name=config.name):
                with self.assertRaisesRegex(ValueError, message):
                    create_adapter(config, seed=42)

    def test_legacy_training_script_reuses_canonical_mlp(self):
        from scripts import train_model

        canonical_model = getattr(model_competition, "MLPRegressor", None)
        canonical_trainer = getattr(model_competition, "fit_mlp_network", None)
        self.assertIsNotNone(canonical_model)
        self.assertIsNotNone(canonical_trainer)
        self.assertIs(train_model.MLPRegressor, canonical_model)
        self.assertIs(train_model.fit_mlp_network, canonical_trainer)

    def test_checked_in_search_space_mappings_are_immutable(self):
        for configs in (CATBOOST_CONFIGS, EXTRA_TREES_CONFIGS, MLP_CONFIGS):
            with self.subTest(configs=configs):
                config = configs[0]
                key = next(iter(config))
                original = config[key]
                replacement = object()
                try:
                    with self.assertRaises(TypeError):
                        config[key] = replacement
                finally:
                    if config[key] is replacement:
                        config[key] = original

    def test_candidate_config_defensively_freezes_nested_params(self):
        source = {
            "hidden_dims": [128, 64],
            "optimizer": {"learning_rate": 0.001},
        }
        config = CandidateConfig(
            name="mlp-128-64",
            model_type="mlp",
            params=source,
            complexity=2,
        )

        source["hidden_dims"].append(32)
        source["optimizer"]["learning_rate"] = 0.5
        source["dropout"] = 0.1

        self.assertIsInstance(config.params, Mapping)
        self.assertEqual(config.params["hidden_dims"], (128, 64))
        self.assertEqual(config.params["optimizer"]["learning_rate"], 0.001)
        self.assertNotIn("dropout", config.params)
        with self.assertRaises(TypeError):
            config.params["dropout"] = 0.1
        with self.assertRaises(TypeError):
            config.params["optimizer"]["learning_rate"] = 0.5
        with self.assertRaises(FrozenInstanceError):
            config.params = {}

    def test_candidate_config_freezes_arrays_sets_and_bytearrays(self):
        hidden_dims = np.array([[128, 64], [32, 16]])
        labels = {"stable", "cpu"}
        signature = bytearray(b"v1")
        source = {
            "nested": {
                "hidden_dims": hidden_dims,
                "labels": labels,
                "signature": signature,
            }
        }
        config = CandidateConfig(
            name="nested-values",
            model_type="mlp",
            params=source,
            complexity=2,
        )

        hidden_dims[0, 0] = 999
        labels.add("changed")
        signature[0] = ord("x")

        nested = config.params["nested"]
        self.assertIsInstance(nested["hidden_dims"], tuple)
        self.assertEqual(nested["hidden_dims"], ((128, 64), (32, 16)))
        self.assertEqual(nested["labels"], frozenset({"stable", "cpu"}))
        self.assertEqual(nested["signature"], b"v1")
        with self.assertRaises(TypeError):
            nested["hidden_dims"][0][0] = 999
        with self.assertRaises(AttributeError):
            nested["labels"].add("changed")

    def test_candidate_config_rejects_unsupported_custom_values(self):
        class MutableCustomValue:
            def __init__(self):
                self.values = []

        with self.assertRaisesRegex(
            TypeError,
            "unsupported config value type: MutableCustomValue",
        ):
            CandidateConfig(
                name="custom-value",
                model_type="mlp",
                params={"custom": MutableCustomValue()},
                complexity=1,
            )

    def test_candidate_config_validates_identity_params_and_complexity(self):
        valid = {
            "name": "candidate",
            "model_type": "mlp",
            "params": {},
            "complexity": 1,
        }
        cases = (
            ("name", 1, TypeError, "name must be a nonempty string"),
            ("name", " ", ValueError, "name must be a nonempty string"),
            (
                "model_type",
                None,
                TypeError,
                "model_type must be a nonempty string",
            ),
            (
                "model_type",
                "",
                ValueError,
                "model_type must be a nonempty string",
            ),
            ("params", [], TypeError, "params must be a mapping"),
            (
                "complexity",
                True,
                TypeError,
                "complexity must be a nonnegative integer",
            ),
            (
                "complexity",
                1.5,
                TypeError,
                "complexity must be a nonnegative integer",
            ),
            (
                "complexity",
                -1,
                ValueError,
                "complexity must be nonnegative",
            ),
        )

        for field, invalid, exception, message in cases:
            values = {**valid, field: invalid}
            with self.subTest(field=field, invalid=invalid):
                with self.assertRaisesRegex(exception, message):
                    CandidateConfig(**values)

    def test_candidate_config_requires_recursive_nonempty_string_keys(self):
        cases = (
            ({1: "value"}, TypeError),
            ({"": "value"}, ValueError),
            ({"nested": {" ": "value"}}, ValueError),
        )

        for params, exception in cases:
            with self.subTest(params=params):
                with self.assertRaisesRegex(
                    exception,
                    "config keys must be nonempty strings",
                ):
                    CandidateConfig("candidate", "mlp", params, 1)

    def test_candidate_config_rejects_nonfinite_params(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "config float values must be finite",
                ):
                    CandidateConfig(
                        "candidate",
                        "mlp",
                        {"learning_rate": invalid},
                        1,
                    )

    def test_candidate_config_to_dict_is_detached_and_json_safe(self):
        config = CandidateConfig(
            name="mlp-canonical",
            model_type="mlp",
            params={
                "hidden_dims": np.array([128, 64]),
                "nested": {
                    "labels": {"stable", "cpu"},
                    "pairs": {(2, 1), (1, 2)},
                },
                "signature": bytearray(b"v1"),
            },
            complexity=np.int64(2),
        )

        serialized = config.to_dict()

        self.assertEqual(
            serialized,
            {
                "name": "mlp-canonical",
                "model_type": "mlp",
                "params": {
                    "hidden_dims": [128, 64],
                    "nested": {
                        "labels": ["cpu", "stable"],
                        "pairs": [[1, 2], [2, 1]],
                    },
                    "signature": {"__bytes_hex__": "7631"},
                },
                "complexity": 2,
            },
        )
        self.assertEqual(
            json.loads(json.dumps(serialized, allow_nan=False)),
            serialized,
        )
        serialized["params"]["hidden_dims"].append(32)
        serialized["params"]["nested"]["labels"].append("changed")
        self.assertEqual(config.params["hidden_dims"], (128, 64))
        self.assertEqual(
            config.params["nested"]["labels"],
            frozenset({"stable", "cpu"}),
        )

    def test_metadata_sort_key_reuses_canonical_config_serialization(self):
        config = CandidateConfig(
            "canonical",
            "extra_trees",
            {"labels": {"zeta", "alpha"}},
            1,
        )
        result = candidate("same", 0.5, 0.1, 0.8, 1, config=config.params)

        self.assertEqual(
            model_competition._metadata_sort_key(result)[1],
            '{"labels":["alpha","zeta"]}',
        )

    def test_candidate_sort_key_uses_cv_metrics_then_complexity(self):
        result = candidate("cat", 0.55, 0.12, 0.8, 2)

        self.assertEqual(candidate_sort_key(result), (-0.55, 0.12, -0.8, 2))

    def test_rank_candidates_rejects_empty_results(self):
        with self.assertRaisesRegex(
            ValueError,
            "results must contain at least one candidate",
        ):
            rank_candidates([])

    def test_rank_candidates_rejects_missing_or_invalid_cv_data(self):
        valid = candidate("valid", 0.5, 0.1, 0.8, 1)
        cases = (
            (
                {key: value for key, value in valid.items() if key != "cv"},
                ValueError,
                "candidate 0 must include cv",
            ),
            (
                {**valid, "cv": []},
                TypeError,
                "candidate 0 cv must be a mapping",
            ),
        )

        for result, exception, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(exception, message):
                    rank_candidates([result])

        for field in ("acc_10_mean", "median_ape_mean", "r2_mean"):
            invalid = candidate("invalid", 0.5, 0.1, 0.8, 1)
            del invalid["cv"][field]
            with self.subTest(missing=field):
                with self.assertRaisesRegex(
                    ValueError,
                    f"candidate 0 cv must include {field}",
                ):
                    rank_candidates([invalid])

    def test_rank_candidates_rejects_bool_and_nonnumeric_cv_metrics(self):
        for field in ("acc_10_mean", "median_ape_mean", "r2_mean"):
            for invalid_value in (True, "0.5"):
                result = candidate("invalid", 0.5, 0.1, 0.8, 1)
                result["cv"][field] = invalid_value
                with self.subTest(field=field, invalid_value=invalid_value):
                    with self.assertRaisesRegex(
                        TypeError,
                        f"candidate 0 cv.{field} must be a real number",
                    ):
                        rank_candidates([result])

    def test_rank_candidates_rejects_nonfinite_cv_metrics(self):
        for field in ("acc_10_mean", "median_ape_mean", "r2_mean"):
            for invalid_value in (np.nan, np.inf, -np.inf):
                result = candidate("invalid", 0.5, 0.1, 0.8, 1)
                result["cv"][field] = invalid_value
                with self.subTest(field=field, invalid_value=invalid_value):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"candidate 0 cv.{field} must be finite",
                    ):
                        rank_candidates([result])

    def test_rank_candidates_rejects_invalid_metric_ranges(self):
        cases = (
            ("acc_10_mean", -0.01, "must be between 0 and 1"),
            ("acc_10_mean", 1.01, "must be between 0 and 1"),
            ("median_ape_mean", -0.01, "must be nonnegative"),
        )

        for field, invalid_value, message in cases:
            result = candidate("invalid", 0.5, 0.1, 0.8, 1)
            result["cv"][field] = invalid_value
            with self.subTest(field=field, invalid_value=invalid_value):
                with self.assertRaisesRegex(
                    ValueError,
                    f"candidate 0 cv.{field} {message}",
                ):
                    rank_candidates([result])

    def test_rank_candidates_accepts_finite_negative_r2(self):
        result = candidate("valid", 0.5, 0.1, -2.0, 1)

        self.assertIs(rank_candidates([result]), result)

    def test_rank_candidates_rejects_invalid_complexity(self):
        cases = (
            ({}, ValueError, "candidate 0 must include complexity"),
            (
                {"complexity": True},
                TypeError,
                "candidate 0 complexity must be a nonnegative integer",
            ),
            (
                {"complexity": 1.5},
                TypeError,
                "candidate 0 complexity must be a nonnegative integer",
            ),
            (
                {"complexity": -1},
                ValueError,
                "candidate 0 complexity must be nonnegative",
            ),
        )

        for replacement, exception, message in cases:
            result = candidate("invalid", 0.5, 0.1, 0.8, 1)
            result.pop("complexity")
            result.update(replacement)
            with self.subTest(message=message):
                with self.assertRaisesRegex(exception, message):
                    rank_candidates([result])

    def test_rank_candidates_validates_results_outside_tie_window(self):
        valid = candidate("valid", 0.9, 0.1, 0.8, 1)
        invalid = candidate("invalid", 0.1, np.nan, 0.8, 1)

        for results in ([valid, invalid], [invalid, valid]):
            with self.subTest(order=[result["name"] for result in results]):
                with self.assertRaisesRegex(
                    ValueError,
                    "cv.median_ape_mean must be finite",
                ):
                    rank_candidates(results)

    def test_rank_candidates_ignores_test_metrics(self):
        cat = candidate("cat", 0.55, 0.12, 0.8, 2, test_acc_10=0.0)
        tree = candidate("tree", 0.40, 0.10, 0.9, 1, test_acc_10=1.0)
        cat["test_metrics"] = object()
        tree["test_metrics"] = {"acc_10": np.nan, "invalid": True}

        winner = rank_candidates([cat, tree])

        self.assertEqual(winner["name"], "cat")

    def test_acc_10_difference_of_exactly_point_zero_one_is_a_tie(self):
        winner = rank_candidates(
            [
                candidate("higher-accuracy", 0.51, 0.20, 0.7, 2),
                candidate("better-relative-error", 0.50, 0.10, 0.6, 2),
            ]
        )

        self.assertEqual(winner["name"], "better-relative-error")

    def test_decimal_acc_10_difference_of_point_zero_one_is_a_tie(self):
        winner = rank_candidates(
            [
                candidate("higher-accuracy", 0.10, 0.20, 0.7, 2),
                candidate("better-relative-error", 0.09, 0.10, 0.6, 2),
            ]
        )

        self.assertEqual(winner["name"], "better-relative-error")

    def test_decimal_acc_10_difference_outside_point_zero_one_is_not_a_tie(self):
        winner = rank_candidates(
            [
                candidate("higher-accuracy", 0.10, 0.20, 0.7, 2),
                candidate(
                    "better-relative-error",
                    0.089999999999999,
                    0.10,
                    0.9,
                    1,
                ),
            ]
        )

        self.assertEqual(winner["name"], "higher-accuracy")

    def test_acc_10_difference_above_point_zero_one_is_not_a_tie(self):
        winner = rank_candidates(
            [
                candidate("higher-accuracy", 0.51, 0.20, 0.7, 2),
                candidate("better-relative-error", 0.4999, 0.10, 0.9, 1),
            ]
        )

        self.assertEqual(winner["name"], "higher-accuracy")

    def test_rank_candidates_is_deterministic_when_cv_results_are_equal(self):
        alpha = candidate("alpha", 0.5, 0.1, 0.8, 1, test_acc_10=0.0)
        beta = candidate("beta", 0.5, 0.1, 0.8, 1, test_acc_10=1.0)

        self.assertEqual(rank_candidates([beta, alpha])["name"], "alpha")
        self.assertEqual(rank_candidates([alpha, beta])["name"], "alpha")


if __name__ == "__main__":
    unittest.main()
