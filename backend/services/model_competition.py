"""Auditable metrics, search spaces, and ranking for model competition."""

import copy
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from services.feature_engineering import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    enrich_features,
    inverse_target,
    transform_target,
)


ACC_10_TIE_WINDOW = 0.01
CATBOOST_MAX_ITERATIONS = 1500
CATBOOST_EARLY_STOPPING_PATIENCE = 80
MLP_MAX_EPOCHS = 400
MLP_EARLY_STOPPING_PATIENCE = 40
_MLP_TRAINING_LOCK = threading.Lock()


class CandidateAdapter(Protocol):
    def fit(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame | None = None,
    ): ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...

    def save(self, directory: Path) -> dict: ...


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


def default_candidates() -> tuple[CandidateConfig, ...]:
    candidates = []
    for index, params in enumerate(CATBOOST_CONFIGS, start=1):
        depth = int(params["depth"])
        candidates.append(
            CandidateConfig(
                name=f"catboost-{index}-depth-{depth}",
                model_type="catboost",
                params=params,
                complexity=(2**depth) * CATBOOST_MAX_ITERATIONS,
            )
        )
    for index, params in enumerate(EXTRA_TREES_CONFIGS, start=1):
        candidates.append(
            CandidateConfig(
                name=f"extra-trees-{index}",
                model_type="extra_trees",
                params=params,
                complexity=int(
                    params["n_estimators"]
                    * params["max_features"]
                    / params["min_samples_leaf"]
                ),
            )
        )
    for index, params in enumerate(MLP_CONFIGS, start=1):
        candidates.append(
            CandidateConfig(
                name=f"mlp-{index}-{'-'.join(map(str, params['hidden_dims']))}",
                model_type="mlp",
                params=params,
                complexity=sum(params["hidden_dims"]),
            )
        )
    return tuple(candidates)


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


def _validated_row_ids(name: str, values: Any) -> list[int]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list of integer row IDs")
    row_ids = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a list of integer row IDs")
        row_ids.append(int(value))
    if not row_ids:
        raise ValueError(f"{name} must not be empty")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"{name} must not contain duplicate row IDs")
    return row_ids


def _validated_evaluation_folds(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    config: CandidateConfig,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not frame.index.is_unique or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (Integral, np.integer))
        for value in frame.index
    ):
        raise ValueError("frame must have a unique integer index")
    if "price" not in frame.columns:
        raise ValueError("frame must contain a price column")
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    if not isinstance(config, CandidateConfig):
        raise TypeError("config must be a CandidateConfig")
    if config.model_type not in {"catboost", "extra_trees", "mlp"}:
        raise ValueError(f"unsupported model_type: {config.model_type}")
    _normalized_seed(seed)
    manifest_keys = set(iter(manifest))
    required_keys = {"development", "test", "folds"}
    if not required_keys.issubset(manifest_keys):
        raise ValueError("manifest must include development, test, and folds")

    development = _validated_row_ids("manifest development", manifest["development"])
    development_set = set(development)
    missing_development = development_set.difference(map(int, frame.index))
    if missing_development:
        raise ValueError("manifest development row IDs must exist in frame")
    _validate_price_series(
        "development prices",
        frame.loc[development, "price"],
    )
    folds_value = manifest["folds"]
    if not isinstance(folds_value, (list, tuple)) or not folds_value:
        raise ValueError("manifest folds must be a nonempty list")
    if "n_splits" in manifest_keys:
        n_splits = manifest["n_splits"]
        if (
            isinstance(n_splits, bool)
            or not isinstance(n_splits, Integral)
            or int(n_splits) != len(folds_value)
        ):
            raise ValueError("manifest n_splits must match the number of folds")

    folds = []
    validation_counts = {row_id: 0 for row_id in development}
    for fold_index, fold in enumerate(folds_value):
        if not isinstance(fold, Mapping):
            raise TypeError(f"manifest fold {fold_index} must be a mapping")
        if "train" not in fold or "validation" not in fold:
            raise ValueError(
                f"manifest fold {fold_index} must include train and validation"
            )
        train_ids = _validated_row_ids(
            f"manifest fold {fold_index} train",
            fold["train"],
        )
        validation_ids = _validated_row_ids(
            f"manifest fold {fold_index} validation",
            fold["validation"],
        )
        if len(validation_ids) < 2:
            raise ValueError(
                f"manifest fold {fold_index} validation must contain at least two rows"
            )
        train_set = set(train_ids)
        validation_set = set(validation_ids)
        if train_set.intersection(validation_set):
            raise ValueError(
                f"manifest fold {fold_index} train and validation must be disjoint"
            )
        if train_set.union(validation_set) != development_set:
            raise ValueError(
                f"manifest fold {fold_index} must partition development row IDs"
            )
        for row_id in validation_ids:
            validation_counts[row_id] += 1
        folds.append((train_ids, validation_ids))

    if any(count != 1 for count in validation_counts.values()):
        raise ValueError(
            "manifest validation folds must partition development exactly once"
        )
    return folds


def _model_frame(source: pd.DataFrame, collection_year: int) -> pd.DataFrame:
    enriched = enrich_features(source, collection_year=collection_year)
    model_frame = enriched.loc[:, MODEL_FEATURES].copy()
    model_frame["price"] = source["price"].astype(float)
    return model_frame


def _normalized_seed(seed: int) -> int:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or not 0 <= int(seed) <= 2**32 - 1
    ):
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    return int(seed)


def _normalized_collection_year(collection_year: Any) -> int:
    if (
        isinstance(collection_year, bool)
        or not isinstance(collection_year, Integral)
        or not 1980 <= int(collection_year) <= 2100
    ):
        raise ValueError(
            "collection_year must be an integer between 1980 and 2100"
        )
    return int(collection_year)


def _validate_parameter_names(
    family: str,
    params: Mapping[str, Any],
    allowed: set[str],
) -> None:
    unsupported = sorted(set(params).difference(allowed))
    if unsupported:
        raise ValueError(
            f"unsupported {family} parameters: {', '.join(unsupported)}"
        )


def _bounded_positive_integer(
    family: str,
    name: str,
    value: Any,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 1 <= int(value) <= maximum
    ):
        raise ValueError(
            f"{family} {name} must be between 1 and {maximum}"
        )
    return int(value)


def _positive_integer(family: str, name: str, value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{family} {name} must be a positive integer")
    return int(value)


def _finite_positive_real(family: str, name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{family} {name} must be finite and positive")
    return float(value)


def _finite_nonnegative_real(family: str, name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{family} {name} must be finite and nonnegative")
    return float(value)


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("adapter frame must be a pandas DataFrame")
    missing = [column for column in MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(
            f"adapter frame missing model features: {', '.join(missing)}"
        )
    features = frame.loc[:, MODEL_FEATURES].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
    for column in CATEGORICAL_FEATURES:
        normalized = (
            features[column].astype("string").fillna("unknown").str.strip()
        )
        features[column] = normalized.mask(
            normalized.eq(""),
            "unknown",
        ).astype(str)
    return features


def _transformed_target(frame: pd.DataFrame) -> np.ndarray:
    if "price" not in frame.columns:
        raise ValueError("adapter training frame must contain a price column")
    prices = _validate_price_series("price", frame["price"])
    return transform_target(prices)


def _require_validation_frame(
    validation_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    if validation_frame is None:
        raise ValueError("validation_frame is required for early stopping")
    return validation_frame


def _inr_predictions(transformed: Any) -> np.ndarray:
    values = np.asarray(transformed, dtype=float).reshape(-1)
    max_log_price = np.log(np.finfo(float).max)
    bounded = np.clip(values, 0.0, max_log_price)
    return np.maximum(inverse_target(bounded), 0.0)


def _build_fold_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median", keep_empty_features=True),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="unknown",
                    keep_empty_features=True,
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=2,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            (
                "categorical",
                categorical_pipeline,
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        previous = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(previous, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = hidden_dim
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)


def _tensor(values: Any) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def _validated_mlp_array(
    name: str,
    values: Any,
    *,
    dimensions: int,
) -> np.ndarray:
    message = (
        f"{name} must be a nonempty finite {dimensions}-D numeric array"
    )
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if (
        array.ndim != dimensions
        or array.size == 0
        or any(length == 0 for length in array.shape)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(message)
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(normalized)):
        raise ValueError(message)
    return normalized


def _validated_hidden_dims(hidden_dims: Any) -> tuple[int, ...]:
    if (
        not isinstance(hidden_dims, (tuple, list))
        or not hidden_dims
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
            for value in hidden_dims
        )
    ):
        raise ValueError(
            "mlp hidden_dims must be a nonempty tuple or list of positive integers"
        )
    return tuple(int(value) for value in hidden_dims)


def _validated_dropout(dropout: Any) -> float:
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, Real)
        or not np.isfinite(float(dropout))
        or not 0 <= float(dropout) < 1
    ):
        raise ValueError("mlp dropout must be finite and between 0 and 1")
    return float(dropout)


def _validated_learning_rate(learning_rate: Any) -> float:
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, Real)
        or not np.isfinite(float(learning_rate))
        or float(learning_rate) <= 0
    ):
        raise ValueError("mlp learning_rate must be finite and positive")
    return float(learning_rate)


def _fit_mlp_network_with_seed(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    hidden_dims: tuple[int, ...],
    dropout: float,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[MLPRegressor, float, int]:
    torch.random.default_generator.manual_seed(seed)
    model = MLPRegressor(
        train_features.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    train_tensor = _tensor(train_features)
    train_target_tensor = _tensor(train_targets).reshape(-1, 1)
    validation_tensor = _tensor(validation_features)
    validation_target_tensor = _tensor(validation_targets).reshape(-1, 1)
    best_state = None
    best_validation_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(train_tensor), train_target_tensor)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("mlp training loss must be finite")
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(
                    model(validation_tensor),
                    validation_target_tensor,
                ).item()
            )
        if not np.isfinite(validation_loss):
            raise RuntimeError("mlp validation loss must be finite")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None or best_epoch < 0 or not np.isfinite(
        best_validation_loss
    ):
        raise RuntimeError("mlp training did not produce a valid best state")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_validation_loss, best_epoch


def fit_mlp_network(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    hidden_dims: tuple[int, ...] | list[int] = (128, 64),
    dropout: float = 0.0,
    learning_rate: float = 0.001,
    max_epochs: int = MLP_MAX_EPOCHS,
    patience: int = MLP_EARLY_STOPPING_PATIENCE,
    seed: int = 42,
) -> tuple[MLPRegressor, float, int]:
    seed = _normalized_seed(seed)
    train_features = _validated_mlp_array(
        "train_features",
        train_features,
        dimensions=2,
    )
    validation_features = _validated_mlp_array(
        "validation_features",
        validation_features,
        dimensions=2,
    )
    train_targets = _validated_mlp_array(
        "train_targets",
        train_targets,
        dimensions=1,
    )
    validation_targets = _validated_mlp_array(
        "validation_targets",
        validation_targets,
        dimensions=1,
    )
    if train_features.shape[0] != train_targets.shape[0]:
        raise ValueError("train feature and target row counts must match")
    if validation_features.shape[0] != validation_targets.shape[0]:
        raise ValueError("validation feature and target row counts must match")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("train and validation feature widths must match")
    hidden_dims = _validated_hidden_dims(hidden_dims)
    dropout = _validated_dropout(dropout)
    learning_rate = _validated_learning_rate(learning_rate)
    max_epochs = _bounded_positive_integer(
        "mlp",
        "max_epochs",
        max_epochs,
        MLP_MAX_EPOCHS,
    )
    patience = _bounded_positive_integer(
        "mlp",
        "early_stopping_patience",
        patience,
        MLP_EARLY_STOPPING_PATIENCE,
    )
    with _MLP_TRAINING_LOCK, torch.random.fork_rng(devices=[]):
        return _fit_mlp_network_with_seed(
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            hidden_dims=hidden_dims,
            dropout=dropout,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
        )


class CatBoostCandidateAdapter:
    def __init__(self, config: CandidateConfig, seed: int):
        params = dict(config.params)
        _validate_parameter_names(
            "catboost",
            params,
            {
                "depth",
                "learning_rate",
                "l2_leaf_reg",
                "loss_function",
                "iterations",
                "early_stopping_patience",
            },
        )
        loss_function = params.pop("loss_function", "RMSE")
        if loss_function != "RMSE":
            raise ValueError("catboost loss_function must be RMSE")
        if "depth" in params:
            params["depth"] = _bounded_positive_integer(
                "catboost",
                "depth",
                params["depth"],
                16,
            )
        if "learning_rate" in params:
            params["learning_rate"] = _finite_positive_real(
                "catboost",
                "learning_rate",
                params["learning_rate"],
            )
        if "l2_leaf_reg" in params:
            params["l2_leaf_reg"] = _finite_nonnegative_real(
                "catboost",
                "l2_leaf_reg",
                params["l2_leaf_reg"],
            )
        self.iterations = _bounded_positive_integer(
            "catboost",
            "iterations",
            params.pop("iterations", CATBOOST_MAX_ITERATIONS),
            CATBOOST_MAX_ITERATIONS,
        )
        self.patience = _bounded_positive_integer(
            "catboost",
            "early_stopping_patience",
            params.pop(
                "early_stopping_patience",
                CATBOOST_EARLY_STOPPING_PATIENCE,
            ),
            CATBOOST_EARLY_STOPPING_PATIENCE,
        )
        self.config = config
        self.seed = _normalized_seed(seed)
        self.categorical_feature_indices = [
            MODEL_FEATURES.index(column) for column in CATEGORICAL_FEATURES
        ]
        self.model = CatBoostRegressor(
            **params,
            loss_function="RMSE",
            iterations=self.iterations,
            random_seed=self.seed,
            verbose=False,
            allow_writing_files=False,
            cat_features=self.categorical_feature_indices,
        )
        self.best_iteration = -1
        self._is_fitted = False

    def fit(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame | None = None,
    ):
        self._is_fitted = False
        validation_frame = _require_validation_frame(validation_frame)
        train_features = _feature_frame(train_frame)
        validation_features = _feature_frame(validation_frame)
        self.model.fit(
            train_features,
            _transformed_target(train_frame),
            eval_set=(
                validation_features,
                _transformed_target(validation_frame),
            ),
            use_best_model=True,
            early_stopping_rounds=self.patience,
            verbose=False,
        )
        self.best_iteration = int(self.model.get_best_iteration())
        self._is_fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "catboost adapter must be fitted before prediction"
            )
        return _inr_predictions(self.model.predict(_feature_frame(frame)))

    def save(self, directory: Path) -> dict:
        if not self._is_fitted:
            raise RuntimeError("catboost adapter must be fitted before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "catboost.cbm"
        self.model.save_model(model_path)
        return {
            "model_type": "catboost",
            "candidate": self.config.to_dict(),
            "target_transform": "log1p",
            "best_iteration": self.best_iteration,
            "categorical_feature_indices": self.categorical_feature_indices,
            "artifacts": {"model": model_path.name},
        }


class ExtraTreesCandidateAdapter:
    def __init__(self, config: CandidateConfig, seed: int):
        params = dict(config.params)
        _validate_parameter_names(
            "extra_trees",
            params,
            {
                "n_estimators",
                "min_samples_leaf",
                "max_features",
                "n_jobs",
            },
        )
        for name in ("n_estimators", "min_samples_leaf"):
            if name in params:
                params[name] = _positive_integer(
                    "extra_trees",
                    name,
                    params[name],
                )
        if "max_features" in params:
            max_features = params["max_features"]
            if (
                isinstance(max_features, bool)
                or not isinstance(max_features, Real)
                or not np.isfinite(float(max_features))
                or not 0 < float(max_features) <= 1
            ):
                raise ValueError(
                    "extra_trees max_features must be finite and between 0 and 1"
                )
            params["max_features"] = float(max_features)
        if "n_jobs" in params:
            n_jobs = params["n_jobs"]
            if (
                isinstance(n_jobs, bool)
                or not isinstance(n_jobs, Integral)
                or int(n_jobs) == 0
            ):
                raise ValueError("extra_trees n_jobs must be a nonzero integer")
            params["n_jobs"] = int(n_jobs)
        self.config = config
        self.seed = _normalized_seed(seed)
        self.preprocessor = _build_fold_preprocessor(scale_numeric=False)
        self.model = ExtraTreesRegressor(
            **params,
            random_state=self.seed,
        )
        self._is_fitted = False

    def fit(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame | None = None,
    ):
        self._is_fitted = False
        train_features = self.preprocessor.fit_transform(
            _feature_frame(train_frame)
        )
        self.model.fit(train_features, _transformed_target(train_frame))
        self._is_fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "extra_trees adapter must be fitted before prediction"
            )
        features = self.preprocessor.transform(_feature_frame(frame))
        return _inr_predictions(self.model.predict(features))

    def save(self, directory: Path) -> dict:
        if not self._is_fitted:
            raise RuntimeError("extra_trees adapter must be fitted before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        bundle_path = directory / "extra_trees.joblib"
        joblib.dump(
            {
                "preprocessor": self.preprocessor,
                "model": self.model,
                "target_transform": "log1p",
            },
            bundle_path,
        )
        return {
            "model_type": "extra_trees",
            "candidate": self.config.to_dict(),
            "target_transform": "log1p",
            "artifacts": {"bundle": bundle_path.name},
        }


class MLPCandidateAdapter:
    def __init__(self, config: CandidateConfig, seed: int):
        params = dict(config.params)
        _validate_parameter_names(
            "mlp",
            params,
            {
                "hidden_dims",
                "dropout",
                "learning_rate",
                "max_epochs",
                "early_stopping_patience",
            },
        )
        hidden_dims = _validated_hidden_dims(
            params.pop("hidden_dims", (128, 64))
        )
        dropout = _validated_dropout(params.pop("dropout", 0.0))
        learning_rate = _validated_learning_rate(
            params.pop("learning_rate", 0.001)
        )
        self.max_epochs = _bounded_positive_integer(
            "mlp",
            "max_epochs",
            params.pop("max_epochs", MLP_MAX_EPOCHS),
            MLP_MAX_EPOCHS,
        )
        self.patience = _bounded_positive_integer(
            "mlp",
            "early_stopping_patience",
            params.pop(
                "early_stopping_patience",
                MLP_EARLY_STOPPING_PATIENCE,
            ),
            MLP_EARLY_STOPPING_PATIENCE,
        )
        self.config = config
        self.seed = _normalized_seed(seed)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.preprocessor = _build_fold_preprocessor(scale_numeric=True)
        self.model = None
        self.best_validation_loss = float("inf")
        self.best_epoch = -1

    def fit(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame | None = None,
    ):
        validation_frame = _require_validation_frame(validation_frame)
        train_features = self.preprocessor.fit_transform(
            _feature_frame(train_frame)
        )
        validation_features = self.preprocessor.transform(
            _feature_frame(validation_frame)
        )
        self.model, self.best_validation_loss, self.best_epoch = fit_mlp_network(
            train_features,
            _transformed_target(train_frame),
            validation_features,
            _transformed_target(validation_frame),
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            patience=self.patience,
            seed=self.seed,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("mlp adapter must be fitted before prediction")
        features = self.preprocessor.transform(_feature_frame(frame))
        with torch.no_grad():
            transformed = self.model(_tensor(features)).numpy().reshape(-1)
        return _inr_predictions(transformed)

    def save(self, directory: Path) -> dict:
        if self.model is None:
            raise RuntimeError("mlp adapter must be fitted before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        preprocessor_path = directory / "mlp_preprocessor.joblib"
        model_path = directory / "mlp.pt"
        joblib.dump(self.preprocessor, preprocessor_path)
        torch.save(
            {
                "input_dim": self.model.net[0].in_features,
                "hidden_dims": list(self.hidden_dims),
                "dropout": self.dropout,
                "model_state": self.model.state_dict(),
                "target_transform": "log1p",
            },
            model_path,
        )
        return {
            "model_type": "mlp",
            "candidate": self.config.to_dict(),
            "target_transform": "log1p",
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "artifacts": {
                "preprocessor": preprocessor_path.name,
                "model": model_path.name,
            },
        }


def _create_adapter(
    config: CandidateConfig,
    seed: int,
) -> CandidateAdapter:
    if not isinstance(config, CandidateConfig):
        raise TypeError("config must be a CandidateConfig")
    factories = {
        "catboost": CatBoostCandidateAdapter,
        "extra_trees": ExtraTreesCandidateAdapter,
        "mlp": MLPCandidateAdapter,
    }
    if config.model_type not in factories:
        raise ValueError(f"unsupported model_type: {config.model_type}")
    return factories[config.model_type](config, seed)


def evaluate_candidate(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    config: CandidateConfig,
    *,
    collection_year: int,
    seed: int = 42,
) -> dict[str, Any]:
    collection_year = _normalized_collection_year(collection_year)
    folds = _validated_evaluation_folds(frame, manifest, config, seed)
    fold_metrics = []
    observed_train_indices = []
    observed_validation_indices = []
    best_iterations = []
    for train_ids, validation_ids in folds:
        train_source = frame.loc[train_ids].copy()
        validation_source = frame.loc[validation_ids].copy()
        train_frame = _model_frame(train_source, collection_year)
        validation_frame = _model_frame(validation_source, collection_year)
        adapter = _create_adapter(config, int(seed))
        adapter.fit(train_frame, validation_frame)
        predictions = adapter.predict(validation_frame.loc[:, MODEL_FEATURES])
        actual = validation_source["price"].to_numpy(dtype=float)
        train_prices = train_source["price"].to_numpy(dtype=float)
        baseline = np.full(len(actual), float(np.mean(train_prices)))
        fold_metrics.append(calculate_metrics(actual, predictions, baseline))
        observed_train_indices.append(train_frame.index.tolist())
        observed_validation_indices.append(validation_frame.index.tolist())
        if config.model_type == "catboost":
            best_iterations.append(getattr(adapter, "best_iteration", None))

    cv = {}
    for metric_name in fold_metrics[0]:
        values = np.asarray(
            [metrics[metric_name] for metrics in fold_metrics],
            dtype=float,
        )
        cv[f"{metric_name}_mean"] = float(np.mean(values))
        cv[f"{metric_name}_std"] = float(np.std(values))

    result = {
        "name": config.name,
        "model_type": config.model_type,
        "config": _json_safe_value(config.params),
        "complexity": config.complexity,
        "collection_year": collection_year,
        "candidate": config.to_dict(),
        "fold_metrics": fold_metrics,
        "cv": cv,
        "observed_train_indices": observed_train_indices,
        "observed_validation_indices": observed_validation_indices,
    }
    if config.model_type == "catboost":
        result["best_iterations"] = best_iterations
    return result


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
