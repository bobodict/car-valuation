"""Version-aware loading and prediction for valuation model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_is_fitted

from services.feature_engineering import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    enrich_features,
)
from services.model_competition import MLPRegressor
from services.publication_validation import is_reparse_point


ARTIFACT_VERSION = "3.0.0"
FEATURE_VERSION = "3.0.0"
MODEL_CONTRACT_VERSION = "3.0.0"
_MAX_SAFE_PRICE = np.finfo(float).max / 4.0
_MAX_SAFE_LOG_TARGET = np.log1p(_MAX_SAFE_PRICE)
_MODEL_ARTIFACT_ROLES = {
    "catboost": {"model"},
    "extra_trees": {"bundle"},
    "mlp": {"preprocessor", "model"},
}


class ModelRuntimeError(RuntimeError):
    """Raised when a model runtime cannot be loaded or used safely."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _validated_models_root(root: Path) -> Path:
    if is_reparse_point(root):
        raise ValueError("models directory must not be a symlink or junction")
    if not root.is_dir():
        raise FileNotFoundError(f"models directory does not exist: {root}")
    return root.resolve(strict=True)


def _resolve_models_file(
    root: Path, relative_path: str | Path, label: str
) -> Path:
    resolved_root = _validated_models_root(root)
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"{label} must remain inside models directory")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} must remain inside models directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {relative_path}")
    return resolved


def _optional_models_file(root: Path, filename: str) -> Path | None:
    candidate = root / filename
    if candidate.is_symlink():
        raise ValueError(f"{filename} path must not contain symlinks")
    if not candidate.exists():
        return None
    return _resolve_models_file(root, filename, filename)


def _require_fields(
    value: Mapping[str, Any], required: set[str], label: str
) -> None:
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{label} is missing required fields: {sorted(missing)}")


def _validate_model_version(value: Any) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value.startswith("v3-")
        or len(value) <= len("v3-")
    ):
        raise ValueError("model_version must be a nonempty v3-* identity")
    return value


def _validate_collection_year(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 1980 <= int(value) <= 2100
    ):
        raise ValueError(
            "collection_year must be an integer between 1980 and 2100"
        )
    return int(value)


def _validate_manifest(manifest: Mapping[str, Any]) -> tuple[str, int]:
    _require_fields(
        manifest,
        {
            "artifact_version",
            "model_contract_version",
            "feature_version",
            "model_version",
            "model_type",
            "model_artifacts",
            "target_transform",
            "collection_year",
        },
        "model_manifest.json",
    )
    if manifest["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError("manifest artifact_version must be 3.0.0")
    if manifest["model_contract_version"] != MODEL_CONTRACT_VERSION:
        raise ValueError("manifest model_contract_version must be 3.0.0")
    if manifest["feature_version"] != FEATURE_VERSION:
        raise ValueError("manifest feature_version must be 3.0.0")
    _validate_model_version(manifest["model_version"])
    model_type = manifest["model_type"]
    if model_type not in _MODEL_ARTIFACT_ROLES:
        raise ValueError(f"unsupported manifest model_type: {model_type!r}")
    if manifest["target_transform"] != "log1p":
        raise ValueError("manifest target_transform must be log1p")
    return model_type, _validate_collection_year(manifest["collection_year"])


def _validate_feature_config(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    _require_fields(
        config,
        {
            "artifact_version",
            "feature_version",
            "model_version",
            "feature_cols",
            "numeric_features",
            "categorical_features",
            "target_transform",
            "collection_year",
        },
        "feature_config.json",
    )
    expected = {
        "artifact_version": ARTIFACT_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_cols": list(MODEL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_transform": "log1p",
    }
    for field, expected_value in expected.items():
        if config[field] != expected_value:
            raise ValueError(
                f"feature_config.json {field} does not match the v3 contract"
            )
    if config["model_version"] != manifest["model_version"]:
        raise ValueError("manifest and feature_config model_version must agree")
    if config["feature_version"] != manifest["feature_version"]:
        raise ValueError("manifest and feature_config feature_version must agree")
    config_year = _validate_collection_year(config["collection_year"])
    if config_year != manifest["collection_year"]:
        raise ValueError("manifest and feature_config collection_year must agree")
    if config["target_transform"] != manifest["target_transform"]:
        raise ValueError("manifest and feature_config target_transform must agree")


def _resolve_artifacts(
    root: Path, model_type: str, declared: Any
) -> dict[str, Path]:
    expected_roles = _MODEL_ARTIFACT_ROLES[model_type]
    if not isinstance(declared, Mapping) or set(declared) != expected_roles:
        raise ValueError(
            f"{model_type} model_artifacts must declare exactly "
            f"{sorted(expected_roles)}"
        )
    artifacts = {}
    for role, relative_path in declared.items():
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise TypeError(f"model artifact path for {role} must be a string")
        artifacts[role] = _resolve_models_file(
            root, relative_path, f"model artifact {role}"
        )
    return artifacts


def _require_fitted(
    component: Any, label: str, attributes: tuple[str, ...]
) -> None:
    try:
        check_is_fitted(component, attributes=attributes)
    except MemoryError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not fitted") from exc


def _validate_mlp_checkpoint(checkpoint: Any) -> tuple[int, tuple[int, ...], float]:
    required = {
        "input_dim",
        "hidden_dims",
        "dropout",
        "model_state",
        "target_transform",
    }
    if not isinstance(checkpoint, Mapping):
        raise TypeError("MLP checkpoint must be a mapping")
    _require_fields(checkpoint, required, "MLP checkpoint")
    input_dim = checkpoint["input_dim"]
    hidden_dims = checkpoint["hidden_dims"]
    dropout = checkpoint["dropout"]
    if (
        isinstance(input_dim, bool)
        or not isinstance(input_dim, Integral)
        or int(input_dim) <= 0
    ):
        raise ValueError("MLP checkpoint input_dim must be a positive integer")
    if (
        not isinstance(hidden_dims, (list, tuple))
        or not hidden_dims
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
            for value in hidden_dims
        )
    ):
        raise ValueError("MLP checkpoint hidden_dims must be positive integers")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, Real)
        or not math.isfinite(float(dropout))
        or not 0 <= float(dropout) < 1
    ):
        raise ValueError("MLP checkpoint dropout must be between 0 and 1")
    if not isinstance(checkpoint["model_state"], Mapping) or not checkpoint[
        "model_state"
    ]:
        raise ValueError("MLP checkpoint model_state must be a nonempty mapping")
    if checkpoint["target_transform"] != "log1p":
        raise ValueError("MLP checkpoint target_transform must be log1p")
    return (
        int(input_dim),
        tuple(int(value) for value in hidden_dims),
        float(dropout),
    )


def _prediction_vector(values: Any, expected_length: int, label: str) -> np.ndarray:
    try:
        predictions = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} predictions must be a numeric array") from exc
    if predictions.ndim != 1:
        raise ValueError(f"{label} predictions must be one-dimensional")
    if len(predictions) != expected_length:
        raise ValueError(
            f"{label} prediction length {len(predictions)} does not match "
            f"input length {expected_length}"
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"{label} predictions must be finite")
    return predictions


def _inverse_log_predictions(values: Any, expected_length: int) -> np.ndarray:
    transformed = _prediction_vector(values, expected_length, "log target")
    if np.any(transformed > _MAX_SAFE_LOG_TARGET):
        raise ValueError(
            "log target prediction exceeds the safe downstream price limit"
        )
    bounded = np.clip(transformed, 0.0, _MAX_SAFE_LOG_TARGET)
    with np.errstate(over="ignore", invalid="ignore"):
        predictions = np.expm1(bounded)
    predictions = _prediction_vector(predictions, expected_length, "price")
    if np.any(predictions < 0):
        raise ValueError("price predictions must be nonnegative")
    return predictions


def _preprocessor_output_width(
    preprocessor: Any, collection_year: int, label: str
) -> int:
    probe = {
        **{column: 1.0 for column in NUMERIC_FEATURES},
        **{column: "unknown" for column in CATEGORICAL_FEATURES},
        "year": collection_year - 1,
    }
    features = enrich_features(pd.DataFrame([probe]), collection_year).loc[
        :, MODEL_FEATURES
    ]
    transformed = preprocessor.transform(features)
    return _matrix_width(transformed, label)


def _matrix_width(transformed: Any, label: str) -> int:
    shape = getattr(transformed, "shape", None)
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or shape[0] != 1
        or not isinstance(shape[1], Integral)
        or int(shape[1]) <= 0
    ):
        raise ValueError(f"{label} output must be a nonempty 2-D feature matrix")
    return int(shape[1])


def _legacy_feature_names(config: Mapping[str, Any]) -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    validated = []
    for name in ("feature_cols", "numeric_features", "categorical_features"):
        values = config[name]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError(f"legacy {name} must be a list of feature names")
        if name == "feature_cols" and not values:
            raise ValueError("legacy feature_cols must be nonempty")
        if len(values) != len(set(values)):
            raise ValueError(f"legacy {name} must contain unique feature names")
        validated.append(tuple(values))
    feature_cols, numeric_features, categorical_features = validated
    numeric_set = set(numeric_features)
    categorical_set = set(categorical_features)
    if numeric_set.intersection(categorical_set):
        raise ValueError("legacy numeric and categorical features must not overlap")
    if numeric_set.union(categorical_set) != set(feature_cols):
        raise ValueError(
            "legacy numeric and categorical features must compose feature_cols"
        )
    return feature_cols, numeric_features, categorical_features


class _V3Runtime:
    def __init__(
        self,
        model_type: str,
        collection_year: int,
        model: Any,
        preprocessor: Any | None = None,
    ):
        self.model_type = model_type
        self.collection_year = collection_year
        self.model = model
        self.preprocessor = preprocessor

    @classmethod
    def from_directory(cls, root: Path, manifest: Mapping[str, Any]):
        model_type, collection_year = _validate_manifest(manifest)
        config_path = _resolve_models_file(
            root, "feature_config.json", "feature_config.json"
        )
        config = _load_json_object(config_path, "feature_config.json")
        _validate_feature_config(config, manifest)
        artifacts = _resolve_artifacts(
            root, model_type, manifest["model_artifacts"]
        )

        if model_type == "catboost":
            model = CatBoostRegressor()
            model.load_model(artifacts["model"])
            tree_count = getattr(model, "tree_count_", None)
            if (
                isinstance(tree_count, bool)
                or not isinstance(tree_count, Integral)
                or int(tree_count) <= 0
            ):
                raise ValueError("CatBoost artifact must contain fitted trees")
            is_fitted = getattr(model, "is_fitted", None)
            if callable(is_fitted) and not bool(is_fitted()):
                raise ValueError("CatBoost artifact is not fitted")
            if list(getattr(model, "feature_names_", ())) != list(MODEL_FEATURES):
                raise ValueError(
                    "CatBoost saved feature contract does not match MODEL_FEATURES"
                )
            expected_categorical_indices = [
                MODEL_FEATURES.index(column) for column in CATEGORICAL_FEATURES
            ]
            if list(model.get_cat_feature_indices()) != expected_categorical_indices:
                raise ValueError(
                    "CatBoost categorical feature indices do not match "
                    "CATEGORICAL_FEATURES"
                )
            return cls(model_type, collection_year, model)

        if model_type == "extra_trees":
            bundle = joblib.load(artifacts["bundle"])
            if not isinstance(bundle, Mapping):
                raise TypeError("ExtraTrees bundle must be a mapping")
            _require_fields(
                bundle,
                {"preprocessor", "model", "target_transform"},
                "ExtraTrees bundle",
            )
            if bundle["target_transform"] != "log1p":
                raise ValueError("ExtraTrees bundle target_transform must be log1p")
            preprocessor = bundle["preprocessor"]
            model = bundle["model"]
            if type(preprocessor) is not ColumnTransformer:
                raise TypeError(
                    "ExtraTrees preprocessor must be a ColumnTransformer"
                )
            if type(model) is not ExtraTreesRegressor:
                raise TypeError("ExtraTrees model must be an ExtraTreesRegressor")
            _require_fitted(
                preprocessor,
                "ExtraTrees preprocessor",
                ("transformers_", "n_features_in_"),
            )
            _require_fitted(
                model,
                "ExtraTrees model",
                ("estimators_", "n_features_in_"),
            )
            transformed_width = _preprocessor_output_width(
                preprocessor, collection_year, "ExtraTrees preprocessor"
            )
            model_width = getattr(model, "n_features_in_", None)
            if (
                isinstance(model_width, bool)
                or not isinstance(model_width, Integral)
                or int(model_width) <= 0
            ):
                raise ValueError(
                    "ExtraTrees model must declare a fitted feature width"
                )
            if transformed_width != int(model_width):
                raise ValueError(
                    "ExtraTrees feature/model contract width mismatch: "
                    f"preprocessor={transformed_width}, model={model_width}"
                )
            return cls(model_type, collection_year, model, preprocessor)

        preprocessor = joblib.load(artifacts["preprocessor"])
        if type(preprocessor) is not ColumnTransformer:
            raise TypeError("MLP preprocessor must be a ColumnTransformer")
        _require_fitted(
            preprocessor,
            "MLP preprocessor",
            ("transformers_", "n_features_in_"),
        )
        checkpoint = torch.load(
            artifacts["model"], map_location="cpu", weights_only=True
        )
        input_dim, hidden_dims, dropout = _validate_mlp_checkpoint(checkpoint)
        transformed_width = _preprocessor_output_width(
            preprocessor, collection_year, "MLP preprocessor"
        )
        if transformed_width != input_dim:
            raise ValueError(
                "MLP feature/model contract width mismatch: "
                f"preprocessor={transformed_width}, checkpoint={input_dim}"
            )
        model = MLPRegressor(input_dim, hidden_dims, dropout)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        return cls(model_type, collection_year, model, preprocessor)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.empty(0, dtype=float)
        features = enrich_features(frame, self.collection_year).loc[
            :, MODEL_FEATURES
        ]
        if self.model_type == "catboost":
            for column in CATEGORICAL_FEATURES:
                features[column] = features[column].astype(str)
            transformed = self.model.predict(features)
        else:
            prepared = self.preprocessor.transform(features)
            if self.model_type == "extra_trees":
                transformed = self.model.predict(prepared)
            else:
                if hasattr(prepared, "toarray"):
                    prepared = prepared.toarray()
                array = np.asarray(prepared, dtype=np.float32)
                if array.ndim != 2 or array.shape[0] != len(frame):
                    raise ValueError("MLP preprocessor output must be a matching 2-D array")
                tensor = torch.from_numpy(array)
                with torch.no_grad():
                    transformed = self.model(tensor).cpu().numpy().reshape(-1)
        return _inverse_log_predictions(transformed, len(frame))


class LegacyTorchRuntime:
    """Compatibility runtime for v2 preprocess.joblib and price_mlp.pt."""

    def __init__(
        self,
        feature_cols: tuple[str, ...],
        numeric_features: tuple[str, ...],
        preprocessor: Any,
        model: MLPRegressor,
        target_mean: float,
        target_std: float,
    ):
        self.feature_cols = feature_cols
        self.numeric_features = numeric_features
        self.preprocessor = preprocessor
        self.model = model
        self.target_mean = target_mean
        self.target_std = target_std

    @classmethod
    def from_directory(cls, root: Path):
        config_path = _optional_models_file(root, "legacy_feature_config.json")
        if config_path is None:
            config_path = _resolve_models_file(
                root, "feature_config.json", "feature_config.json"
            )
        preprocessor_path = _resolve_models_file(
            root, "preprocess.joblib", "legacy preprocessor artifact"
        )
        model_path = _resolve_models_file(
            root, "price_mlp.pt", "legacy model artifact"
        )
        config = _load_json_object(config_path, "feature_config.json")
        _require_fields(
            config,
            {"feature_cols", "numeric_features", "categorical_features"},
            "legacy feature_config.json",
        )
        feature_cols, numeric_features, categorical_features = (
            _legacy_feature_names(config)
        )

        preprocessor = joblib.load(preprocessor_path)
        if type(preprocessor) is not ColumnTransformer:
            raise TypeError("legacy preprocessor must be a ColumnTransformer")
        _require_fitted(
            preprocessor,
            "legacy preprocessor",
            ("transformers_", "n_features_in_"),
        )
        probe = {
            **{column: 1.0 for column in numeric_features},
            **{column: "unknown" for column in categorical_features},
        }
        transformed_width = _matrix_width(
            preprocessor.transform(
                pd.DataFrame([probe]).reindex(columns=feature_cols)
            ),
            "legacy preprocessor",
        )

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("legacy MLP checkpoint must be a mapping")
        _require_fields(
            checkpoint,
            {"input_dim", "hidden_dims", "model_state"},
            "legacy MLP checkpoint",
        )
        legacy_checkpoint = {
            **checkpoint,
            "dropout": checkpoint.get("dropout", 0.0),
            "target_transform": "log1p",
        }
        input_dim, hidden_dims, dropout = _validate_mlp_checkpoint(
            legacy_checkpoint
        )
        if transformed_width != input_dim:
            raise ValueError(
                "legacy feature/model contract width mismatch: "
                f"preprocessor={transformed_width}, checkpoint={input_dim}"
            )
        target_mean = checkpoint.get("target_mean", 0.0)
        target_std = checkpoint.get("target_std", 1.0)
        if (
            isinstance(target_mean, bool)
            or not isinstance(target_mean, Real)
            or not math.isfinite(float(target_mean))
        ):
            raise ValueError("legacy target_mean must be a finite number")
        if (
            isinstance(target_std, bool)
            or not isinstance(target_std, Real)
            or not math.isfinite(float(target_std))
            or float(target_std) <= 0
        ):
            raise ValueError("legacy target_std must be finite and positive")
        model = MLPRegressor(input_dim, hidden_dims, dropout)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        return cls(
            tuple(feature_cols),
            tuple(numeric_features),
            preprocessor,
            model,
            float(target_mean),
            float(target_std),
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.empty(0, dtype=float)
        features = frame.copy()
        if "car_age" in self.numeric_features and "car_age" not in features:
            if "collection_time" in features and "year" in features:
                collection_time = pd.to_datetime(
                    features["collection_time"], errors="coerce"
                )
                features["car_age"] = (
                    collection_time.dt.year.fillna(date.today().year)
                    - features["year"]
                ).clip(lower=0)
            elif "year" in features:
                features["car_age"] = date.today().year - features["year"]
        prepared = self.preprocessor.transform(
            features.reindex(columns=self.feature_cols)
        )
        if hasattr(prepared, "toarray"):
            prepared = prepared.toarray()
        array = np.asarray(prepared, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(frame):
            raise ValueError(
                "legacy preprocessor output must be a matching 2-D array"
            )
        with torch.no_grad():
            scaled = self.model(torch.from_numpy(array)).cpu().numpy().reshape(-1)
        scaled = _prediction_vector(scaled, len(frame), "legacy scaled target")
        predictions = scaled * self.target_std + self.target_mean
        return _prediction_vector(predictions, len(frame), "legacy price")


class ModelRuntime:
    """Facade selecting a v3 manifest runtime or the legacy v2 runtime."""

    def __init__(self, implementation: _V3Runtime | LegacyTorchRuntime):
        self._implementation = implementation

    @classmethod
    def from_directory(cls, models_dir: str | Path) -> ModelRuntime:
        root = Path(models_dir)
        try:
            _validated_models_root(root)
            manifest_path = _optional_models_file(root, "model_manifest.json")
            if manifest_path is not None:
                manifest = _load_json_object(
                    manifest_path, "model_manifest.json"
                )
                implementation = _V3Runtime.from_directory(root, manifest)
            else:
                implementation = LegacyTorchRuntime.from_directory(root)
            return cls(implementation)
        except (ModelRuntimeError, MemoryError):
            raise
        except Exception as exc:
            raise ModelRuntimeError(
                f"failed to load model runtime from {root}: {exc}"
            ) from exc

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not isinstance(frame, pd.DataFrame):
            raise ModelRuntimeError("predict frame must be a pandas DataFrame")
        try:
            predictions = self._implementation.predict(frame)
            return _prediction_vector(predictions, len(frame), "runtime")
        except (ModelRuntimeError, MemoryError):
            raise
        except Exception as exc:
            raise ModelRuntimeError(f"model prediction failed: {exc}") from exc

    def predict_one(self, vehicle: Mapping[str, Any]) -> float:
        if not isinstance(vehicle, Mapping):
            raise ModelRuntimeError("predict_one vehicle must be a mapping")
        predictions = self.predict(pd.DataFrame([dict(vehicle)]))
        if predictions.shape != (1,):
            raise ModelRuntimeError(
                "predict_one requires the runtime to return exactly one result"
            )
        return float(predictions[0])
