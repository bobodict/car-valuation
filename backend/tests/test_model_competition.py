import unittest
from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import numpy as np

from services import model_competition
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

    def test_acc_10_counts_mathematically_exact_threshold_errors(self):
        actual = np.array([100.0, 200.0])
        predicted = actual * np.array([1.1, 1.10001])

        metrics = calculate_metrics(actual, predicted, actual)

        self.assertEqual(metrics["acc_10"], 0.5)

    def test_acc_20_counts_mathematically_exact_threshold_errors(self):
        actual = np.array([7.0, 43.0])
        predicted = actual * np.array([1.2, 1.20001])

        metrics = calculate_metrics(actual, predicted, actual)

        self.assertEqual(metrics["acc_20"], 0.5)

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

    def test_candidate_sort_key_uses_cv_metrics_then_complexity(self):
        result = candidate("cat", 0.55, 0.12, 0.8, 2)

        self.assertEqual(candidate_sort_key(result), (-0.55, 0.12, -0.8, 2))

    def test_rank_candidates_ignores_test_metrics(self):
        winner = rank_candidates(
            [
                candidate("cat", 0.55, 0.12, 0.8, 2, test_acc_10=0.0),
                candidate("tree", 0.40, 0.10, 0.9, 1, test_acc_10=1.0),
            ]
        )

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
