"""Auditable metrics, search spaces, and ranking for model competition."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)


ACC_10_TIE_WINDOW = 0.01
CATBOOST_MAX_ITERATIONS = 1500
CATBOOST_EARLY_STOPPING_PATIENCE = 80
MLP_MAX_EPOCHS = 400
MLP_EARLY_STOPPING_PATIENCE = 40


def _freeze_value(value):
    if isinstance(value, Mapping):
        return _immutable_mapping(value)
    if isinstance(value, np.ndarray):
        return _freeze_value(value.tolist())
    if isinstance(value, np.generic):
        return _freeze_value(value.item())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    raise TypeError(f"unsupported config value type: {type(value).__name__}")


def _immutable_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _freeze_value(value) for key, value in values.items()}
    )


CATBOOST_CONFIGS = (
    _immutable_mapping(
        {
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "loss_function": "RMSE",
        }
    ),
    _immutable_mapping(
        {
            "depth": 7,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,
            "loss_function": "RMSE",
        }
    ),
    _immutable_mapping(
        {
            "depth": 8,
            "learning_rate": 0.03,
            "l2_leaf_reg": 8.0,
            "loss_function": "RMSE",
        }
    ),
    _immutable_mapping(
        {
            "depth": 7,
            "learning_rate": 0.05,
            "l2_leaf_reg": 10.0,
            "loss_function": "RMSE",
        }
    ),
)
EXTRA_TREES_CONFIGS = tuple(
    _immutable_mapping(
        {
            "n_estimators": 600,
            "min_samples_leaf": leaf,
            "max_features": features,
            "n_jobs": -1,
        }
    )
    for leaf in (1, 2)
    for features in (0.7, 1.0)
)
MLP_CONFIGS = (
    _immutable_mapping(
        {"hidden_dims": (128, 64), "dropout": 0.0, "learning_rate": 0.001}
    ),
    _immutable_mapping(
        {"hidden_dims": (256, 128), "dropout": 0.1, "learning_rate": 0.001}
    ),
    _immutable_mapping(
        {
            "hidden_dims": (128, 64, 32),
            "dropout": 0.1,
            "learning_rate": 0.0005,
        }
    ),
)


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    model_type: str
    params: Mapping[str, Any]
    complexity: int

    def __post_init__(self):
        object.__setattr__(self, "params", _immutable_mapping(self.params))


def _relative_accuracy_hits(
    actual: np.ndarray,
    predicted: np.ndarray,
    threshold: float,
) -> np.ndarray:
    nonzero_actual = actual != 0
    ratio = np.zeros_like(predicted, dtype=float)
    np.divide(predicted, actual, out=ratio, where=nonzero_actual)
    hits = (ratio >= 1.0 - threshold) & (ratio <= 1.0 + threshold)
    hits[~nonzero_actual] = (
        np.abs(predicted[~nonzero_actual]) / 1e-8 <= threshold
    )
    return hits


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    relative_error = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-8)

    return {
        "mse": float(mean_squared_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "acc_10": float(
            np.mean(_relative_accuracy_hits(actual, predicted, 0.10))
        ),
        "acc_20": float(
            np.mean(_relative_accuracy_hits(actual, predicted, 0.20))
        ),
        "median_ape": float(np.median(relative_error)),
        "rmsle": float(mean_squared_log_error(actual, predicted) ** 0.5),
        "baseline_rmse": float(mean_squared_error(actual, baseline) ** 0.5),
        "baseline_r2": float(r2_score(actual, baseline)),
    }


def candidate_sort_key(result: dict) -> tuple:
    cv = result["cv"]
    return (
        -cv["acc_10_mean"],
        cv["median_ape_mean"],
        -cv["r2_mean"],
        result["complexity"],
    )


def _metadata_sort_key(result: dict) -> tuple[str, str]:
    config = json.dumps(
        result.get("config", {}),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return str(result.get("name", "")), config


def rank_candidates(results: list[dict]) -> dict:
    decimal_scores = [
        Decimal(str(result["cv"]["acc_10_mean"])) for result in results
    ]
    best_acc_10 = max(decimal_scores)
    tie_window = Decimal(str(ACC_10_TIE_WINDOW))
    tied = [
        result
        for result, score in zip(results, decimal_scores)
        if best_acc_10 - score <= tie_window
    ]
    return min(
        tied,
        key=lambda result: (
            *candidate_sort_key(result)[1:],
            *_metadata_sort_key(result),
        ),
    )
