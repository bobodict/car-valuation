"""Auditable metrics, search spaces, and ranking for model competition."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral, Real
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
    if type(value) is float and not np.isfinite(value):
        raise ValueError("config float values must be finite")
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    raise TypeError(f"unsupported config value type: {type(value).__name__}")


def _validate_config_key(key: Any) -> None:
    if not isinstance(key, str):
        raise TypeError("config keys must be nonempty strings")
    if not key.strip():
        raise ValueError("config keys must be nonempty strings")


def _immutable_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = {}
    for key, value in values.items():
        _validate_config_key(key)
        frozen[key] = _freeze_value(value)
    return MappingProxyType(frozen)


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
        for field, value in (("name", self.name), ("model_type", self.model_type)):
            if not isinstance(value, str):
                raise TypeError(f"{field} must be a nonempty string")
            if not value.strip():
                raise ValueError(f"{field} must be a nonempty string")
        if not isinstance(self.params, Mapping):
            raise TypeError("params must be a mapping")
        if isinstance(self.complexity, bool) or not isinstance(
            self.complexity, Integral
        ):
            raise TypeError("complexity must be a nonnegative integer")
        if self.complexity < 0:
            raise ValueError("complexity must be nonnegative")
        object.__setattr__(self, "params", _immutable_mapping(self.params))
        object.__setattr__(self, "complexity", int(self.complexity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "params": _json_safe_value(self.params),
            "complexity": self.complexity,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, CandidateConfig):
        return value.to_dict()
    if isinstance(value, Mapping):
        for key in value:
            _validate_config_key(key)
        return {
            key: _json_safe_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_safe_value(item) for item in value]
        return sorted(items, key=_canonical_json)
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if type(value) is float and not np.isfinite(value):
        raise ValueError("config float values must be finite")
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"unsupported config value type: {type(value).__name__}")


def _relative_accuracy_hits(
    actual: np.ndarray,
    predicted: np.ndarray,
    threshold: float,
) -> np.ndarray:
    nonzero_actual = actual != 0
    ratio = np.zeros_like(predicted, dtype=float)
    np.divide(predicted, actual, out=ratio, where=nonzero_actual)
    # Allow only one ULP for the multiplication/division rounding step.
    lower = np.nextafter(1.0 - threshold, -np.inf)
    upper = np.nextafter(1.0 + threshold, np.inf)
    hits = (ratio >= lower) & (ratio <= upper)
    hits[~nonzero_actual] = (
        np.abs(predicted[~nonzero_actual]) / 1e-8 <= threshold
    )
    return hits


def _validate_price_series(name: str, values: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a one-dimensional numeric array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite prices")
    if np.any(array < 0):
        raise ValueError(f"{name} must contain only nonnegative prices")
    return array


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    actual = _validate_price_series("actual", actual)
    predicted = _validate_price_series("predicted", predicted)
    baseline = _validate_price_series("baseline", baseline)
    if not (len(actual) == len(predicted) == len(baseline)):
        raise ValueError(
            "actual, predicted, and baseline must have equal lengths"
        )
    if len(actual) < 2:
        raise ValueError("metric inputs must contain at least two samples")
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
    config = _canonical_json(_json_safe_value(result.get("config", {})))
    return str(result.get("name", "")), config


def _validate_cv_metric(
    candidate_index: int,
    cv: Mapping[str, Any],
    field: str,
) -> Decimal:
    if field not in cv:
        raise ValueError(f"candidate {candidate_index} cv must include {field}")
    value = cv[field]
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise TypeError(
            f"candidate {candidate_index} cv.{field} must be a real number"
        )
    decimal_value = Decimal(str(value))
    if not decimal_value.is_finite():
        raise ValueError(f"candidate {candidate_index} cv.{field} must be finite")
    if field == "acc_10_mean" and not Decimal(0) <= decimal_value <= Decimal(1):
        raise ValueError(
            f"candidate {candidate_index} cv.{field} must be between 0 and 1"
        )
    if field == "median_ape_mean" and decimal_value < 0:
        raise ValueError(
            f"candidate {candidate_index} cv.{field} must be nonnegative"
        )
    return decimal_value


def _validate_candidate_result(candidate_index: int, result: Any) -> Decimal:
    if not isinstance(result, Mapping):
        raise TypeError(f"candidate {candidate_index} must be a mapping")
    if "cv" not in result:
        raise ValueError(f"candidate {candidate_index} must include cv")
    cv = result["cv"]
    if not isinstance(cv, Mapping):
        raise TypeError(f"candidate {candidate_index} cv must be a mapping")
    acc_10 = _validate_cv_metric(candidate_index, cv, "acc_10_mean")
    _validate_cv_metric(candidate_index, cv, "median_ape_mean")
    _validate_cv_metric(candidate_index, cv, "r2_mean")
    if "complexity" not in result:
        raise ValueError(f"candidate {candidate_index} must include complexity")
    complexity = result["complexity"]
    if isinstance(complexity, bool) or not isinstance(complexity, Integral):
        raise TypeError(
            f"candidate {candidate_index} complexity must be a nonnegative integer"
        )
    if complexity < 0:
        raise ValueError(
            f"candidate {candidate_index} complexity must be nonnegative"
        )
    return acc_10


def rank_candidates(results: list[dict]) -> dict:
    if not results:
        raise ValueError("results must contain at least one candidate")
    decimal_scores = [
        _validate_candidate_result(index, result)
        for index, result in enumerate(results)
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
