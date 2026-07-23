"""Train, audit, and atomically publish a v3 vehicle valuation model."""

import argparse
import hashlib
import json
import math
import os
import shutil
import threading
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from services.dataset_contract import load_dataset
from services.feature_engineering import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)
from services.model_competition import (
    CandidateConfig,
    CatBoostRegressor,
    MLPRegressor,
    _create_adapter,
    _model_frame,
    calculate_metrics,
    default_candidates,
    evaluate_candidate,
    fit_mlp_network,
    rank_candidates,
)
from services.split_service import build_split_manifest
from services.publication_validation import PUBLICATION_GENERATION_FILENAME


SEED = 42
DEFAULT_COLLECTION_YEAR = 2026
ARTIFACT_VERSION = "3.0.0"
MODEL_CONTRACT_VERSION = "3.0.0"
FEATURE_VERSION = "3.0.0"
TARGET_COL = "price"
_FIXED_QUALITY_THRESHOLDS = {"min_r2": 0.0, "min_acc_10": 0.5}
QUALITY_THRESHOLDS = dict(_FIXED_QUALITY_THRESHOLDS)
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
DEFAULT_RAW_DATASET = Path(__file__).resolve().parents[1] / "data" / "raw" / "car-details-v4.csv"
DEFAULT_PROCESSED_DATASET = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "normalized_training.csv"
)
SOURCE_URL = "https://raw.githubusercontent.com/chandanverma07/DataSets/master/car%20details%20v4.csv"
REQUIRED_REPORTS = (
    "model_manifest.json",
    "feature_config.json",
    "metrics.json",
    "leaderboard.json",
    "error_analysis.json",
    "model_card.json",
)
LEGACY_CATEGORY_FIELDS = ("vehicle_type", "accident_history")
DEFAULT_RARE_MODEL_FAMILY_THRESHOLD = 5
MODEL_OWNERSHIP_SENTINEL = ".model-publication-owner.json"
_MODEL_OWNER = "car-valuation-model-publication"
_MODEL_SENTINEL_VERSION = 1
RUN_STATUS_FILENAME = "run_status.json"
FAILURE_RECORD_FILENAME = "failure.json"
PUBLICATION_LOCK_FILENAME = "lock.json"
PUBLICATION_WARNING_FILENAME = "publication_warning.json"


@dataclass
class RefitResult:
    adapter: Any
    candidate: CandidateConfig
    strategy: str
    train_indices: list[int]
    validation_indices: list[int]
    collection_year: int
    development_mean: float


@dataclass
class PublicationMutex:
    kind: str
    resource: Any
    held: bool = True


@dataclass
class PublicationLock:
    path: Path
    formal_dir: Path
    token: str
    owner: dict[str, int]
    acquired_at: str
    mutex: PublicationMutex


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_run_id() -> str:
    timestamp = _utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def _validate_collection_year(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 1980 <= int(value) <= 2100
    ):
        raise ValueError("collection_year must be an integer between 1980 and 2100")
    return int(value)


def _validated_v3_model_version(value: Any) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value.startswith("v3-")
        or len(value) <= len("v3-")
    ):
        raise ValueError("model_version must match the nonempty v3- artifact identity")
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            normalized[key] = _json_value(value[key])
        return normalized
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if type(value) is float and not math.isfinite(value):
        raise ValueError("JSON artifacts cannot contain NaN or Infinity")
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"unsupported JSON artifact value: {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    normalized = _json_value(value)
    content = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    path.write_text(f"{content}\n", encoding="utf-8")
    return path


def _write_publication_generation(directory: Path) -> Path:
    return _write_json(
        directory / PUBLICATION_GENERATION_FILENAME,
        {
            "generation": uuid.uuid4().hex,
            "published_at": _utc_now().isoformat(),
        },
    )


def _reject_json_constant(value: str):
    raise ValueError(f"non-standard JSON constant: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"required artifact is missing: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return value


def _write_run_status(
    experiment_dir: Path,
    run_id: str,
    started_at: str,
    stage: str,
    status: str,
    **details: Any,
) -> Path:
    return _write_json(
        experiment_dir / RUN_STATUS_FILENAME,
        {
            "artifact_version": ARTIFACT_VERSION,
            "run_id": run_id,
            "status": status,
            "stage": stage,
            "started_at": started_at,
            "updated_at": _utc_now().isoformat(),
            **details,
        },
    )


def _record_run_failure(
    experiment_dir: Path,
    run_id: str,
    started_at: str,
    stage: str,
    exc: BaseException,
) -> Path:
    timestamp = _utc_now().isoformat()
    failure = _write_json(
        experiment_dir / FAILURE_RECORD_FILENAME,
        {
            "artifact_version": ARTIFACT_VERSION,
            "run_id": run_id,
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "timestamp": timestamp,
        },
    )
    _write_run_status(
        experiment_dir,
        run_id,
        started_at,
        stage,
        "failed",
        failure_record=FAILURE_RECORD_FILENAME,
    )
    return failure


def assess_quality_gate(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the fixed, non-bypassable outer-holdout quality gate."""
    if thresholds is not None and dict(thresholds) != _FIXED_QUALITY_THRESHOLDS:
        raise ValueError("quality thresholds are fixed for the v3 contract")

    warnings = []
    values = {}
    for field in ("r2", "acc_10", "rmse", "baseline_rmse"):
        raw_value = metrics.get(field)
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, Real)
            or not math.isfinite(float(raw_value))
        ):
            warnings.append(f"{field} is missing or nonfinite")
        else:
            values[field] = float(raw_value)

    if "r2" in values and values["r2"] <= _FIXED_QUALITY_THRESHOLDS["min_r2"]:
        warnings.append("test R2 does not exceed the configured threshold")
    if "acc_10" in values and (
        not 0.0 <= values["acc_10"] <= 1.0
        or values["acc_10"] < _FIXED_QUALITY_THRESHOLDS["min_acc_10"]
    ):
        warnings.append("10% error accuracy is below the configured threshold")
    if "rmse" in values and values["rmse"] < 0:
        warnings.append("model RMSE must be nonnegative")
    if "baseline_rmse" in values and values["baseline_rmse"] < 0:
        warnings.append("baseline RMSE must be nonnegative")
    if "rmse" in values and "baseline_rmse" in values:
        if values["rmse"] >= values["baseline_rmse"]:
            warnings.append("model RMSE does not strictly beat the development-mean baseline")

    return {
        "quality_gate": "fail" if warnings else "pass",
        "warnings": warnings,
        "thresholds": dict(_FIXED_QUALITY_THRESHOLDS),
    }


_CANDIDATE_METRICS = frozenset(
    {
        "mse",
        "rmse",
        "mae",
        "r2",
        "acc_10",
        "acc_20",
        "median_ape",
        "rmsle",
        "baseline_rmse",
        "baseline_r2",
    }
)
_CANDIDATE_CV_FIELDS = frozenset(
    f"{metric}_{statistic}"
    for metric in _CANDIDATE_METRICS
    for statistic in ("mean", "std")
)
_CANDIDATE_RESULT_FIELDS = frozenset(
    {
        "name",
        "model_type",
        "config",
        "complexity",
        "collection_year",
        "candidate",
        "fold_metrics",
        "cv",
        "observed_train_indices",
        "observed_validation_indices",
    }
)


class _SealedCompetitionManifest(Mapping[str, Any]):
    def __init__(self, manifest: Mapping[str, Any]):
        self._keys = tuple(manifest)
        required = {"development", "test", "folds"}
        if not required.issubset(self._keys):
            raise ValueError("manifest must include development, test, and folds")
        self._values = {
            key: _json_value(manifest[key])
            for key in self._keys
            if key != "test"
        }

    def __getitem__(self, key: str) -> Any:
        if key == "test":
            raise RuntimeError("outer-test manifest value is sealed during competition")
        return self._values[key]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


def _validated_candidate_number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"candidate result {path} must be a finite number")
    return float(value)


def _validated_candidate_metrics(
    value: Any,
    path: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"candidate result {path} must be an object")
    unexpected = set(value).difference(_CANDIDATE_METRICS)
    if unexpected:
        raise ValueError(
            f"candidate result {path} contains unexpected fields: {sorted(unexpected)}"
        )
    return {
        key: _validated_candidate_number(value[key], f"{path}.{key}")
        for key in sorted(value)
    }


def _validated_candidate_cv(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("candidate result cv must be an object")
    unexpected = set(value).difference(_CANDIDATE_CV_FIELDS)
    if unexpected:
        raise ValueError(
            f"candidate result cv contains unexpected fields: {sorted(unexpected)}"
        )
    required = {"acc_10_mean", "median_ape_mean", "r2_mean"}
    missing = required.difference(value)
    if missing:
        raise ValueError(
            f"candidate result cv is missing rank fields: {sorted(missing)}"
        )
    return {
        key: _validated_candidate_number(value[key], f"cv.{key}")
        for key in sorted(value)
    }


def _validated_candidate_index_groups(
    value: Any,
    path: str,
    development_ids: set[int],
) -> list[list[int]]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"candidate result {path} must be a list of row-ID lists")
    groups = []
    for group_index, group in enumerate(value):
        if not isinstance(group, (list, tuple)):
            raise TypeError(
                f"candidate result {path}[{group_index}] must be a row-ID list"
            )
        row_ids = []
        for row_id in group:
            if isinstance(row_id, bool) or not isinstance(row_id, Integral):
                raise TypeError(f"candidate result {path} must contain integer row IDs")
            row_ids.append(int(row_id))
        if not set(row_ids).issubset(development_ids):
            raise ValueError(
                f"candidate result {path} contains rows outside development"
            )
        groups.append(row_ids)
    return groups


def _validated_candidate_fold_evidence(
    result: Mapping[str, Any],
    development_ids: set[int],
    expected_folds: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, float]], dict[str, float], list[list[int]], list[list[int]]]:
    fold_metrics_value = result["fold_metrics"]
    if not isinstance(fold_metrics_value, (list, tuple)):
        raise TypeError("candidate result fold_metrics must be a list")
    fold_metrics = [
        _validated_candidate_metrics(metrics, f"fold_metrics[{index}]")
        for index, metrics in enumerate(fold_metrics_value)
    ]
    observed_train = _validated_candidate_index_groups(
        result["observed_train_indices"],
        "observed_train_indices",
        development_ids,
    )
    observed_validation = _validated_candidate_index_groups(
        result["observed_validation_indices"],
        "observed_validation_indices",
        development_ids,
    )
    reported_cv = _validated_candidate_cv(result["cv"])

    if not expected_folds:
        if fold_metrics or observed_train or observed_validation:
            raise ValueError("candidate fold evidence requires manifest folds")
        return fold_metrics, reported_cv, observed_train, observed_validation

    expected_train = [
        [int(row_id) for row_id in fold["train"]] for fold in expected_folds
    ]
    expected_validation = [
        [int(row_id) for row_id in fold["validation"]] for fold in expected_folds
    ]
    if (
        len(fold_metrics) != len(expected_folds)
        or observed_train != expected_train
        or observed_validation != expected_validation
    ):
        raise ValueError("candidate fold evidence must exactly match manifest folds")
    if any(set(metrics) != _CANDIDATE_METRICS for metrics in fold_metrics):
        raise ValueError("candidate fold_metrics must contain all shared metrics")

    canonical_cv = {}
    for metric_name in sorted(_CANDIDATE_METRICS):
        values = np.asarray(
            [metrics[metric_name] for metrics in fold_metrics],
            dtype=float,
        )
        canonical_cv[f"{metric_name}_mean"] = float(np.mean(values))
        canonical_cv[f"{metric_name}_std"] = float(np.std(values))
    if reported_cv != canonical_cv:
        raise ValueError("candidate cv must equal canonical fold_metrics aggregation")
    return fold_metrics, canonical_cv, observed_train, observed_validation


def _validated_candidate_result(
    result: Mapping[str, Any],
    candidate: CandidateConfig,
    collection_year: int,
    development_ids: set[int],
    expected_folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError("candidate result must be an object")
    allowed = set(_CANDIDATE_RESULT_FIELDS)
    if candidate.model_type == "catboost":
        allowed.add("best_iterations")
    unexpected = set(result).difference(allowed)
    if unexpected:
        raise ValueError(
            "candidate result contains unexpected or post-selection fields: "
            f"{sorted(unexpected)}"
        )
    missing = set(_CANDIDATE_RESULT_FIELDS).difference(result)
    if missing:
        raise ValueError(f"candidate result is missing fields: {sorted(missing)}")

    expected_candidate = candidate.to_dict()
    if result["name"] != candidate.name or result["model_type"] != candidate.model_type:
        raise ValueError("candidate result identity does not match evaluated candidate")
    if result["complexity"] != candidate.complexity:
        raise ValueError("candidate result complexity does not match evaluated candidate")
    if result["collection_year"] != collection_year:
        raise ValueError("candidate result collection_year does not match competition")
    if _json_value(result["config"]) != expected_candidate["params"]:
        raise ValueError("candidate result config does not match evaluated candidate")
    if _json_value(result["candidate"]) != expected_candidate:
        raise ValueError("candidate result candidate metadata does not match evaluated candidate")

    fold_metrics, cv, observed_train, observed_validation = (
        _validated_candidate_fold_evidence(
            result,
            development_ids,
            expected_folds,
        )
    )
    projected = {
        "name": candidate.name,
        "model_type": candidate.model_type,
        "config": expected_candidate["params"],
        "complexity": candidate.complexity,
        "collection_year": collection_year,
        "candidate": expected_candidate,
        "fold_metrics": fold_metrics,
        "cv": cv,
        "observed_train_indices": observed_train,
        "observed_validation_indices": observed_validation,
    }
    if "best_iterations" in result:
        values = result["best_iterations"]
        if not isinstance(values, (list, tuple)) or any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, Integral))
            for value in values
        ):
            raise TypeError("candidate result best_iterations must contain integers or null")
        projected["best_iterations"] = [
            None if value is None else int(value) for value in values
        ]
    return projected


def run_competition(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    candidates: Sequence[CandidateConfig],
    seed: int,
    collection_year: int,
    *,
    evaluator: Callable[..., Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all declared candidates using development folds only."""
    collection_year = _validate_collection_year(collection_year)
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("candidates must contain at least one configuration")
    evaluator = evaluator or evaluate_candidate
    development_ids = [int(value) for value in manifest["development"]]
    if len(development_ids) != len(set(development_ids)):
        raise ValueError("manifest development must contain unique row IDs")
    missing_development = set(development_ids).difference(
        int(value) for value in frame.index
    )
    if missing_development:
        raise ValueError("manifest development row IDs must exist in frame")
    development_frame = frame.loc[development_ids].copy()
    sealed_manifest = _SealedCompetitionManifest(manifest)
    development_set = set(development_ids)
    leaderboard = []
    for candidate in candidates:
        result = evaluator(
            development_frame.copy(),
            sealed_manifest,
            candidate,
            collection_year=collection_year,
            seed=seed,
        )
        leaderboard.append(
            _validated_candidate_result(
                result,
                candidate,
                collection_year,
                development_set,
                manifest["folds"],
            )
        )
    return leaderboard


def select_winner(leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a winner exclusively through the CV ranking contract."""
    return rank_candidates(leaderboard)


def _winner_config(winner: Mapping[str, Any]) -> CandidateConfig:
    candidate = winner.get("candidate")
    if isinstance(candidate, Mapping):
        return CandidateConfig(
            name=candidate["name"],
            model_type=candidate["model_type"],
            params=candidate["params"],
            complexity=candidate["complexity"],
        )
    return CandidateConfig(
        name=winner["name"],
        model_type=winner["model_type"],
        params=winner["config"],
        complexity=winner["complexity"],
    )


def _development_holdback_ids(
    development_frame: pd.DataFrame,
    seed: int,
) -> tuple[list[int], list[int]]:
    if len(development_frame) < 4:
        raise ValueError("development data must contain at least four rows for refit holdback")
    row_ids = np.asarray(sorted(int(value) for value in development_frame.index))
    validation_count = max(2, int(math.ceil(len(row_ids) * 0.15)))
    if validation_count >= len(row_ids):
        validation_count = len(row_ids) - 1
    train_ids, validation_ids = train_test_split(
        row_ids,
        test_size=validation_count,
        random_state=seed,
        shuffle=True,
    )
    return (
        sorted(int(value) for value in train_ids),
        sorted(int(value) for value in validation_ids),
    )


def refit_winner(
    development_frame: pd.DataFrame,
    winner: Mapping[str, Any],
    seed: int,
    collection_year: int,
    *,
    adapter_factory: Callable[[CandidateConfig, int], Any] | None = None,
) -> RefitResult:
    """Deterministically refit the CV winner without accessing outer-test rows."""
    collection_year = _validate_collection_year(collection_year)
    if development_frame.empty:
        raise ValueError("development_frame must not be empty")
    index_values = development_frame.index.tolist()
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in index_values
    ):
        raise TypeError("development_frame index must contain integer row IDs")
    normalized_index = [int(value) for value in index_values]
    if len(normalized_index) != len(set(normalized_index)):
        raise ValueError("development_frame index must contain unique row IDs")
    config = _winner_config(winner)
    adapter_factory = adapter_factory or _create_adapter
    adapter = adapter_factory(config, seed)
    development_mean = float(development_frame[TARGET_COL].astype(float).mean())

    if config.model_type == "extra_trees":
        train_ids = [int(value) for value in development_frame.index]
        validation_ids = []
        strategy = "full_development"
        train_frame = _model_frame(development_frame.loc[train_ids], collection_year)
        adapter.fit(train_frame, None)
    else:
        train_ids, validation_ids = _development_holdback_ids(development_frame, seed)
        strategy = "deterministic_development_holdback"
        train_frame = _model_frame(development_frame.loc[train_ids], collection_year)
        validation_frame = _model_frame(
            development_frame.loc[validation_ids], collection_year
        )
        adapter.fit(train_frame, validation_frame)

    return RefitResult(
        adapter=adapter,
        candidate=config,
        strategy=strategy,
        train_indices=train_ids,
        validation_indices=validation_ids,
        collection_year=collection_year,
        development_mean=development_mean,
    )


def evaluate_holdout(
    fitted: RefitResult,
    test_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate the selected model once on the sealed outer holdout."""
    if not isinstance(fitted, RefitResult):
        raise TypeError("fitted must be a RefitResult")
    model_frame = _model_frame(test_frame, fitted.collection_year)
    predictions = np.asarray(
        fitted.adapter.predict(model_frame.loc[:, MODEL_FEATURES]), dtype=float
    ).reshape(-1)
    actual = test_frame[TARGET_COL].astype(float).to_numpy()
    baseline = np.full(len(actual), fitted.development_mean, dtype=float)
    metrics = calculate_metrics(actual, predictions, baseline)
    return {
        "evaluation_scope": "recorded_test",
        "count": int(len(actual)),
        "indices": [int(value) for value in test_frame.index],
        "actual": actual.tolist(),
        "predictions": predictions.tolist(),
        "metrics": metrics,
        "development_mean_baseline": fitted.development_mean,
    }


def _single_or_multi_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    development_mean: float,
) -> dict[str, float | None]:
    metric_names = (
        "mse",
        "rmse",
        "mae",
        "r2",
        "acc_10",
        "acc_20",
        "median_ape",
        "rmsle",
        "baseline_rmse",
        "baseline_r2",
    )
    if len(actual) == 0:
        return {name: None for name in metric_names}
    baseline = np.full(len(actual), development_mean, dtype=float)
    if len(actual) >= 2:
        return calculate_metrics(actual, predicted, baseline)
    duplicated = calculate_metrics(
        np.repeat(actual, 2),
        np.repeat(predicted, 2),
        np.repeat(baseline, 2),
    )
    duplicated["r2"] = None
    duplicated["baseline_r2"] = None
    return duplicated


def _group_report(
    label: str,
    mask: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    development_mean: float,
) -> dict[str, Any]:
    group_actual = actual[mask]
    group_predicted = predicted[mask]
    minimum = float(np.min(group_actual)) if len(group_actual) else None
    maximum = float(np.max(group_actual)) if len(group_actual) else None
    range_label = (
        f"INR {minimum:,.0f} to {maximum:,.0f}"
        if minimum is not None
        else "no holdout rows"
    )
    return {
        "label": label,
        "count": int(len(group_actual)),
        "price_unit": "INR",
        "actual_price_range_inr": {"min": minimum, "max": maximum},
        "range_label": range_label,
        "metrics": _single_or_multi_metrics(
            group_actual,
            group_predicted,
            development_mean,
        ),
        "baseline_evidence": {
            "kind": "development_mean",
            "value_inr": float(development_mean),
        },
    }


def _normalized_model_names(frame: pd.DataFrame) -> pd.Series:
    names = frame["model"].astype("string").fillna("unknown").str.strip()
    return names.mask(names.eq(""), "unknown").astype(str)


def build_error_analysis(
    development_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    predictions: Sequence[float],
    development_mean: float,
    *,
    rare_threshold: int = DEFAULT_RARE_MODEL_FAMILY_THRESHOLD,
) -> dict[str, Any]:
    """Build post-selection diagnostics with development-only group definitions."""
    if isinstance(rare_threshold, bool) or not isinstance(rare_threshold, Integral):
        raise TypeError("rare_threshold must be a positive integer")
    if int(rare_threshold) <= 0:
        raise ValueError("rare_threshold must be a positive integer")
    rare_threshold = int(rare_threshold)
    actual = test_frame[TARGET_COL].astype(float).to_numpy()
    predicted = np.asarray(predictions, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError("test prices and predictions must have equal lengths")
    if not math.isfinite(float(development_mean)):
        raise ValueError("development_mean must be finite")

    development_prices = development_frame[TARGET_COL].astype(float).to_numpy()
    if (
        len(development_prices) == 0
        or not np.all(np.isfinite(development_prices))
        or np.any(development_prices < 0)
    ):
        raise ValueError("development prices must be nonempty, finite, and nonnegative")
    boundaries = np.quantile(
        development_prices,
        [0.25, 0.5, 0.75],
        method="linear",
    ).astype(float)
    quartile_ids = np.searchsorted(boundaries, actual, side="left")
    quartile_groups = []
    for quartile in range(4):
        group = _group_report(
            f"Q{quartile + 1}",
            quartile_ids == quartile,
            actual,
            predicted,
            development_mean,
        )
        group["definition_range_inr"] = {
            "lower_exclusive": (
                None if quartile == 0 else float(boundaries[quartile - 1])
            ),
            "upper_inclusive": (
                None if quartile == 3 else float(boundaries[quartile])
            ),
        }
        quartile_groups.append(group)

    development_models = _normalized_model_names(development_frame)
    test_models = _normalized_model_names(test_frame)
    development_families = development_models.str.split().str[0]
    test_families = test_models.str.split().str[0]
    frequencies = development_families.value_counts().sort_index()
    common_families = set(frequencies[frequencies >= rare_threshold].index)
    common_mask = test_families.isin(common_families).to_numpy(dtype=bool)
    family_groups = [
        _group_report("common", common_mask, actual, predicted, development_mean),
        _group_report("rare", ~common_mask, actual, predicted, development_mean),
    ]

    seen_models = set(development_models.tolist())
    seen_mask = test_models.isin(seen_models).to_numpy(dtype=bool)
    seen_groups = [
        _group_report("seen", seen_mask, actual, predicted, development_mean),
        _group_report("unseen", ~seen_mask, actual, predicted, development_mean),
    ]

    return {
        "development_mean_baseline": float(development_mean),
        "baseline_evidence": {
            "source": "development_only",
            "definition": "arithmetic mean of development prices",
            "value_inr": float(development_mean),
        },
        "price_quartiles": {
            "definition_source": "development_only",
            "boundary_source": "development_actual_price",
            "boundary_method": "linear_quantiles_25_50_75",
            "development_count": int(len(development_prices)),
            "boundaries_inr": boundaries.tolist(),
            "currency": "INR",
            "groups": quartile_groups,
        },
        "model_family_frequency": {
            "frequency_source": "development_only",
            "rare_threshold": rare_threshold,
            "rare_definition": "development frequency below rare_threshold, including unseen families",
            "development_frequencies": {
                str(key): int(value) for key, value in frequencies.items()
            },
            "groups": family_groups,
        },
        "full_model_seen_status": {
            "seen_set_source": "development_only",
            "development_seen_models": sorted(seen_models),
            "groups": seen_groups,
        },
    }


def _refit_metadata(fitted: Any) -> dict[str, Any]:
    if isinstance(fitted, RefitResult):
        return {
            "strategy": fitted.strategy,
            "train_indices": fitted.train_indices,
            "validation_indices": fitted.validation_indices,
            "development_mean": fitted.development_mean,
        }
    return {}


def _adapter_from_experiment(experiment: Mapping[str, Any]) -> Any:
    fitted = experiment["fitted_model"]
    return fitted.adapter if isinstance(fitted, RefitResult) else fitted


def _category_options(frame: pd.DataFrame, collection_year: int) -> dict[str, list[str]]:
    model_frame = _model_frame(frame, collection_year)
    options = {}
    for column in (*CATEGORICAL_FEATURES, *LEGACY_CATEGORY_FIELDS):
        source = model_frame[column] if column in model_frame else frame.get(column)
        if source is None:
            options[column] = ["unknown"]
            continue
        values = source.astype("string").fillna("unknown").str.strip()
        values = values.mask(values.eq(""), "unknown")
        options[column] = sorted(str(value) for value in values.unique())[:100]
    return options


def _split_indices(
    manifest: Mapping[str, Any],
    refit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "development": [int(value) for value in manifest["development"]],
        "test": [int(value) for value in manifest["test"]],
        "folds": _json_value(manifest.get("folds", [])),
        "train": [int(value) for value in refit.get("train_indices", [])],
        "validation": [
            int(value) for value in refit.get("validation_indices", [])
        ],
    }


def _validate_declared_artifacts(
    experiment_dir: Path,
    declared: Mapping[str, Any],
) -> list[Path]:
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError("model manifest must declare model artifact paths")
    root = experiment_dir.resolve()
    paths = []
    for role, relative_path in declared.items():
        if not isinstance(role, str) or not isinstance(relative_path, str):
            raise TypeError("declared model artifact roles and paths must be strings")
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError("declared model artifacts must remain inside experiment directory")
        resolved = (experiment_dir / candidate).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("declared model artifacts must remain inside experiment directory")
        if not resolved.is_file():
            raise FileNotFoundError(f"declared model artifact is missing: {relative_path}")
        paths.append(resolved)
    return paths


def _prepare_artifact_output_directory(output_dir: Path, run_id: str) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return
    if not output_dir.is_dir():
        raise FileExistsError("experiment artifact target already exists and is not a directory")
    entries = list(output_dir.iterdir())
    if len(entries) != 1 or entries[0].name != RUN_STATUS_FILENAME:
        raise FileExistsError(
            "experiment artifact target must be a newly initialized run directory"
        )
    status = _read_json(entries[0])
    if (
        status.get("artifact_version") != ARTIFACT_VERSION
        or status.get("run_id") != run_id
        or status.get("status") != "running"
    ):
        raise FileExistsError(
            "experiment artifact target does not belong to the active run"
        )


def build_artifacts(
    experiment: Mapping[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Write the complete, strict-JSON v3 experiment artifact set."""
    output_dir = Path(output_dir)
    run_id = str(experiment["run_id"])
    _prepare_artifact_output_directory(output_dir, run_id)
    frame = experiment["frame"]
    manifest = _json_value(experiment["split_manifest"])
    seed = int(experiment["seed"])
    collection_year = _validate_collection_year(experiment["collection_year"])
    development_ids = {int(value) for value in manifest["development"]}
    leaderboard = [
        _validated_candidate_result(
            result,
            _winner_config(result),
            collection_year,
            development_ids,
            manifest["folds"],
        )
        for result in experiment["leaderboard"]
    ]
    winner_source = experiment["winner"]
    winner = _validated_candidate_result(
        winner_source,
        _winner_config(winner_source),
        collection_year,
        development_ids,
        manifest["folds"],
    )
    fitted = experiment["fitted_model"]
    refit = _json_value(experiment.get("refit") or _refit_metadata(fitted))
    holdout = experiment["holdout"]
    metrics = _json_value(holdout["metrics"])
    gate = _json_value(experiment["gate"])
    model_version = str(experiment.get("model_version", f"v3-{run_id}"))
    created_at = str(experiment["created_at"])
    provenance = _json_value(experiment.get("provenance") or {})
    category_options = _category_options(frame, collection_year)
    split_indices = _split_indices(manifest, refit)
    development_mean = float(
        holdout.get(
            "development_mean_baseline",
            experiment.get("error_analysis", {}).get(
                "development_mean_baseline",
                frame.loc[manifest["development"], TARGET_COL].astype(float).mean(),
            ),
        )
    )
    error_analysis = experiment.get("error_analysis")
    if error_analysis is None:
        error_analysis = build_error_analysis(
            frame.loc[manifest["development"]],
            frame.loc[manifest["test"]],
            holdout["predictions"],
            development_mean,
            rare_threshold=int(
                experiment.get(
                    "rare_threshold", DEFAULT_RARE_MODEL_FAMILY_THRESHOLD
                )
            ),
        )
    error_analysis = _json_value(error_analysis)

    adapter = _adapter_from_experiment(experiment)
    model_metadata = _json_value(adapter.save(output_dir))
    model_artifacts = model_metadata.get("artifacts")
    _validate_declared_artifacts(output_dir, model_artifacts)
    model_type = str(model_metadata.get("model_type", winner["model_type"]))
    target_transform = str(model_metadata.get("target_transform", "log1p"))
    units = {"currency": "INR", "price": "INR", "mileage": "km"}
    counts = {
        "total": int(len(frame)),
        "development": int(len(manifest["development"])),
        "test": int(len(manifest["test"])),
        "candidates": int(len(leaderboard)),
    }
    limitations = [
        "The source represents the Indian used-car market.",
        "Prices are recorded in INR and are not converted to another currency.",
        "The independent holdout is used once after CV-only model selection.",
        "Legacy v2 category fields remain metadata-only when absent from v3 model features.",
    ]

    model_manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_version": model_version,
        "model_type": model_type,
        "model_family": model_type,
        "winner": winner,
        "collection_year": collection_year,
        "seed": seed,
        "target_transform": target_transform,
        "model_artifacts": model_artifacts,
        "model_artifact_metadata": model_metadata,
        "report_artifacts": list(REQUIRED_REPORTS),
        "refit_strategy": refit,
        "split_manifest": manifest,
        "split_indices": split_indices,
        "provenance": provenance,
        "units": units,
        "counts": counts,
        "created_at": created_at,
    }
    feature_config = {
        "artifact_version": ARTIFACT_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_version": model_version,
        "target_col": TARGET_COL,
        "feature_cols": list(MODEL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_transform": target_transform,
        "collection_year": collection_year,
        "category_options": category_options,
        "currency": "INR",
        "price_unit": "INR",
        "mileage_unit": "km",
        "units": units,
        "compatibility": {
            "legacy_artifact_version": "2.0.0",
            "legacy_category_fields": list(LEGACY_CATEGORY_FIELDS),
            "limitations": limitations,
        },
    }
    metrics_artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model_version": model_version,
        "model_type": model_type,
        "winner": winner["name"],
        "evaluation_scope": "recorded_test",
        "independent_holdout": True,
        "test_metrics": metrics,
        "development_mean_baseline": development_mean,
        "baseline_evidence": {
            "source": "development_only",
            "value_inr": development_mean,
        },
        "quality_gate": gate["quality_gate"],
        "warnings": gate["warnings"],
        "thresholds": gate["thresholds"],
        "split_manifest": manifest,
        "split_indices": split_indices,
        "seed": seed,
        "collection_year": collection_year,
        "sample_count": counts["total"],
        "split": {
            "train": len(split_indices["train"]),
            "validation": len(split_indices["validation"]),
            "development": counts["development"],
            "test": counts["test"],
        },
        "data_source": provenance,
        "provenance": provenance,
        "currency": "INR",
        "price_unit": "INR",
        "mileage_unit": "km",
        "units": units,
        "trained_at": created_at,
    }
    leaderboard_artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model_version": model_version,
        "selection_scope": "development_cv_only",
        "selection_method": "rank_candidates",
        "winner": winner["name"],
        "winner_model_type": model_type,
        "candidate_count": len(leaderboard),
        "candidates": leaderboard,
        "outer_test_metrics_in_candidates": False,
        "seed": seed,
        "collection_year": collection_year,
    }
    error_analysis_artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model_version": model_version,
        "evaluation_scope": "recorded_test_post_selection",
        "price_unit": "INR",
        **error_analysis,
    }
    model_card = {
        "artifact_version": ARTIFACT_VERSION,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_version": model_version,
        "model_type": model_type,
        "model_family": model_type,
        "winner": winner,
        "collection_year": collection_year,
        "seed": seed,
        "provenance": provenance,
        "data_source": provenance,
        "currency": "INR",
        "price_unit": "INR",
        "mileage_unit": "km",
        "sample_count": counts["total"],
        "split": {
            "train": len(split_indices["train"]),
            "validation": len(split_indices["validation"]),
            "development": counts["development"],
            "test": counts["test"],
        },
        "thresholds": gate["thresholds"],
        "quality_gate": gate["quality_gate"],
        "warnings": gate["warnings"],
        "test_metrics": metrics,
        "units": units,
        "counts": counts,
        "created_at": created_at,
        "target_transform": target_transform,
        "refit_strategy": refit,
        "development_mean_baseline": development_mean,
        "cv_selection": {
            "scope": "development_cv_only",
            "winner": winner["name"],
            "winner_cv": winner["cv"],
            "candidate_count": len(leaderboard),
        },
        "independent_holdout": {
            "scope": "recorded_test",
            "count": counts["test"],
            "metrics": metrics,
            "quality_gate": gate,
        },
        "error_analysis": error_analysis,
        "category_options": category_options,
        "feature_descriptions": {
            "car_age": "collection year minus registration year",
            "mileage": "driven distance in kilometers",
            "mileage_per_year": "mileage divided by car age, with a one-year floor",
            "power_per_liter": "brake horsepower divided by displacement",
            "target": "used-car price in INR transformed with log1p for fitting",
        },
        "limitations": limitations,
    }

    payloads = {
        "model_manifest.json": model_manifest,
        "feature_config.json": feature_config,
        "metrics.json": metrics_artifact,
        "leaderboard.json": leaderboard_artifact,
        "error_analysis.json": error_analysis_artifact,
        "model_card.json": model_card,
    }
    paths = [_write_json(output_dir / name, payload) for name, payload in payloads.items()]
    paths.extend(
        path for path in sorted(output_dir.iterdir()) if path not in paths
    )
    return paths


def write_experiment_artifacts(
    experiment: Mapping[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    return build_artifacts(experiment, output_dir)


def _load_saved_model_for_smoke(
    experiment_dir: Path,
    manifest: Mapping[str, Any],
) -> Any:
    model_type = manifest.get("model_type")
    artifacts = manifest["model_artifacts"]
    if model_type == "catboost":
        model = CatBoostRegressor()
        model.load_model(experiment_dir / artifacts["model"])
        tree_count = getattr(model, "tree_count_", None)
        if (
            isinstance(tree_count, bool)
            or not isinstance(tree_count, Integral)
            or int(tree_count) <= 0
        ):
            raise ValueError("CatBoost artifact must contain a fitted model with trees")
        is_fitted = getattr(model, "is_fitted", None)
        if callable(is_fitted) and not bool(is_fitted()):
            raise ValueError("CatBoost artifact model is not fitted")
        return model
    if model_type == "extra_trees":
        bundle = joblib.load(experiment_dir / artifacts["bundle"])
        if not isinstance(bundle, dict) or not {"preprocessor", "model"}.issubset(bundle):
            raise ValueError("invalid ExtraTrees artifact bundle")
        preprocessor = bundle["preprocessor"]
        model = bundle["model"]
        if not callable(getattr(preprocessor, "transform", None)) or not callable(
            getattr(model, "predict", None)
        ):
            raise ValueError(
                "ExtraTrees artifact bundle must contain transform/predict components"
            )
        try:
            check_is_fitted(preprocessor)
            check_is_fitted(model)
        except Exception as exc:
            raise ValueError(
                "ExtraTrees artifact bundle contains unfitted components"
            ) from exc
        return bundle
    if model_type == "mlp":
        preprocessor = joblib.load(experiment_dir / artifacts["preprocessor"])
        checkpoint = torch.load(
            experiment_dir / artifacts["model"],
            map_location="cpu",
            weights_only=True,
        )
        required_checkpoint = {
            "input_dim",
            "hidden_dims",
            "dropout",
            "model_state",
            "target_transform",
        }
        if not isinstance(checkpoint, Mapping) or not required_checkpoint.issubset(
            checkpoint
        ):
            raise ValueError("MLP checkpoint is missing required keys")
        input_dim = checkpoint["input_dim"]
        hidden_dims = checkpoint["hidden_dims"]
        dropout = checkpoint["dropout"]
        if (
            isinstance(input_dim, bool)
            or not isinstance(input_dim, Integral)
            or int(input_dim) <= 0
            or not isinstance(hidden_dims, (list, tuple))
            or not hidden_dims
            or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) <= 0
                for value in hidden_dims
            )
            or isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(float(dropout))
            or not 0.0 <= float(dropout) < 1.0
            or not isinstance(checkpoint["model_state"], Mapping)
            or not checkpoint["model_state"]
            or checkpoint["target_transform"] != "log1p"
        ):
            raise ValueError("MLP checkpoint contains invalid fitted-model metadata")
        if not callable(getattr(preprocessor, "transform", None)):
            raise ValueError("MLP preprocessor must support transform")
        try:
            check_is_fitted(preprocessor)
        except Exception as exc:
            raise ValueError("MLP preprocessor is not fitted") from exc
        model = MLPRegressor(
            int(input_dim),
            tuple(int(value) for value in hidden_dims),
            float(dropout),
        )
        try:
            model.load_state_dict(checkpoint["model_state"], strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("MLP checkpoint state is not loadable on CPU") from exc
        model.eval()
        return {"preprocessor": preprocessor, "model": model}
    raise ValueError(f"unsupported model_type in manifest: {model_type}")


def _validated_report_row_ids(
    name: str,
    values: Any,
    *,
    allow_empty: bool = False,
) -> list[int]:
    if not isinstance(values, list):
        raise TypeError(f"{name} must be a list of integer row IDs")
    row_ids = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must contain only integer non-bool row IDs")
        row_ids.append(int(value))
    if not allow_empty and not row_ids:
        raise ValueError(f"{name} must not be empty")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"{name} must contain unique row IDs")
    return row_ids


def _validated_report_count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _validate_split_report_semantics(
    split_manifest: Mapping[str, Any],
    split_indices: Mapping[str, Any],
    refit: Mapping[str, Any],
    counts: Mapping[str, Any],
    metrics: Mapping[str, Any],
    model_card: Mapping[str, Any],
    model_type: str,
) -> set[int]:
    development = _validated_report_row_ids(
        "split_indices.development", split_indices.get("development")
    )
    test = _validated_report_row_ids("split_indices.test", split_indices.get("test"))
    development_set = set(development)
    test_set = set(test)
    if not development_set.isdisjoint(test_set):
        raise ValueError("development and test split row IDs must be disjoint")

    total_count = _validated_report_count("counts.total", counts.get("total"))
    development_count = _validated_report_count(
        "counts.development", counts.get("development")
    )
    test_count = _validated_report_count("counts.test", counts.get("test"))
    if (
        development_set | test_set != set(range(total_count))
        or len(development) != development_count
        or len(test) != test_count
    ):
        raise ValueError("split membership and manifest counts must agree")

    expected_split_counts = {
        "train": len(split_indices.get("train", [])),
        "validation": len(split_indices.get("validation", [])),
        "development": development_count,
        "test": test_count,
    }
    if (
        metrics.get("sample_count") != total_count
        or metrics.get("split") != expected_split_counts
        or model_card.get("sample_count") != total_count
        or model_card.get("split") != expected_split_counts
    ):
        raise ValueError("metrics and model-card split counts must agree")

    folds = split_indices.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("split_indices.folds must be a nonempty list")
    n_splits = split_manifest.get("n_splits")
    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, Integral)
        or int(n_splits) != len(folds)
        or int(n_splits) < 2
    ):
        raise ValueError("split manifest n_splits must match at least two folds")
    validation_coverage: Counter[int] = Counter()
    fold_numbers = []
    for position, fold in enumerate(folds):
        if not isinstance(fold, Mapping):
            raise TypeError(f"split fold {position} must be an object")
        fold_number = fold.get("fold")
        if (
            isinstance(fold_number, bool)
            or not isinstance(fold_number, Integral)
        ):
            raise TypeError("split fold numbers must be integers")
        fold_numbers.append(int(fold_number))
        train_ids = _validated_report_row_ids(
            f"split fold {position}.train", fold.get("train")
        )
        validation_ids = _validated_report_row_ids(
            f"split fold {position}.validation", fold.get("validation")
        )
        train_set = set(train_ids)
        validation_set = set(validation_ids)
        if (
            not train_set.isdisjoint(validation_set)
            or train_set | validation_set != development_set
        ):
            raise ValueError(
                "each fold train/validation pair must disjointly partition development"
            )
        validation_coverage.update(validation_ids)
    if sorted(fold_numbers) != list(range(len(folds))) or any(
        validation_coverage[row_id] != 1 for row_id in development
    ):
        raise ValueError("validation folds must cover development exactly once")

    train_ids = _validated_report_row_ids(
        "refit train_indices", refit.get("train_indices")
    )
    validation_ids = _validated_report_row_ids(
        "refit validation_indices",
        refit.get("validation_indices"),
        allow_empty=True,
    )
    train_set = set(train_ids)
    validation_set = set(validation_ids)
    if (
        not train_set.isdisjoint(validation_set)
        or not train_set.issubset(development_set)
        or not validation_set.issubset(development_set)
    ):
        raise ValueError("refit rows must be disjoint subsets of development")
    strategy = refit.get("strategy")
    if strategy == "full_development" and model_type == "extra_trees":
        coherent = train_set == development_set and not validation_set
    elif (
        strategy == "deterministic_development_holdback"
        and model_type in {"catboost", "mlp"}
    ):
        coherent = (
            bool(train_set)
            and bool(validation_set)
            and train_set | validation_set == development_set
        )
    else:
        coherent = False
    if not coherent:
        raise ValueError("refit strategy and development membership are incoherent")
    return development_set


def _validate_feature_report_semantics(
    manifest: Mapping[str, Any],
    feature_config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    leaderboard: Mapping[str, Any],
    model_card: Mapping[str, Any],
) -> tuple[int, int]:
    if (
        feature_config.get("target_col") != TARGET_COL
        or feature_config.get("feature_cols") != list(MODEL_FEATURES)
        or feature_config.get("numeric_features") != list(NUMERIC_FEATURES)
        or feature_config.get("categorical_features") != list(CATEGORICAL_FEATURES)
    ):
        raise ValueError("feature_config.json must use the canonical ordered feature contract")
    model_metadata = manifest.get("model_artifact_metadata")
    transforms = {
        manifest.get("target_transform"),
        feature_config.get("target_transform"),
        model_card.get("target_transform"),
        model_metadata.get("target_transform")
        if isinstance(model_metadata, Mapping)
        else None,
    }
    if transforms != {"log1p"}:
        raise ValueError("all model reports must use target_transform log1p")

    collection_year = _validate_collection_year(manifest.get("collection_year"))
    if {
        feature_config.get("collection_year"),
        metrics.get("collection_year"),
        leaderboard.get("collection_year"),
        model_card.get("collection_year"),
    } != {collection_year}:
        raise ValueError("all model reports must agree on collection_year")
    seed = manifest.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or not 0 <= int(seed) <= 2**32 - 1
    ):
        raise ValueError("model seed must be an integer in uint32 range")
    seed = int(seed)
    split_manifest = manifest.get("split_manifest")
    if not isinstance(split_manifest, Mapping) or {
        split_manifest.get("seed"),
        metrics.get("seed"),
        leaderboard.get("seed"),
        model_card.get("seed"),
    } != {seed}:
        raise ValueError("all model reports must agree on seed")
    return collection_year, seed


def _validate_error_analysis_semantics(
    error_analysis: Mapping[str, Any],
    test_count: int,
    development_mean: float,
) -> None:
    price_quartiles = error_analysis.get("price_quartiles")
    family_frequency = error_analysis.get("model_family_frequency")
    seen_status = error_analysis.get("full_model_seen_status")
    if not all(
        isinstance(value, Mapping)
        for value in (price_quartiles, family_frequency, seen_status)
    ):
        raise TypeError("error-analysis subgroup reports must be objects")
    boundaries = price_quartiles.get("boundaries_inr")
    if (
        price_quartiles.get("definition_source") != "development_only"
        or price_quartiles.get("boundary_source") != "development_actual_price"
        or not isinstance(boundaries, list)
        or len(boundaries) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in boundaries
        )
        or [float(value) for value in boundaries]
        != sorted(float(value) for value in boundaries)
    ):
        raise ValueError("price quartiles must use finite ordered development boundaries")
    if family_frequency.get("frequency_source") != "development_only":
        raise ValueError("model-family frequency must be defined from development only")
    if seen_status.get("seen_set_source") != "development_only":
        raise ValueError("seen-model status must be defined from development only")

    for name, report, expected_groups in (
        ("price quartiles", price_quartiles, 4),
        ("model-family frequency", family_frequency, 2),
        ("seen-model status", seen_status, 2),
    ):
        groups = report.get("groups")
        if not isinstance(groups, list) or len(groups) != expected_groups:
            raise ValueError(f"{name} must contain the canonical groups")
        group_count = sum(
            _validated_report_count(f"{name} group count", group.get("count"))
            if isinstance(group, Mapping)
            else -1
            for group in groups
        )
        if group_count != test_count:
            raise ValueError(f"{name} counts must sum to the recorded test count")
        expected_baseline_evidence = {
            "kind": "development_mean",
            "value_inr": development_mean,
        }
        if any(
            group.get("baseline_evidence") != expected_baseline_evidence
            for group in groups
        ):
            raise ValueError(
                f"{name} subgroup baseline evidence must use the development mean"
            )


def _validate_experiment_directory(experiment_dir: Path) -> dict[str, Any]:
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {experiment_dir}")
    payloads = {name: _read_json(experiment_dir / name) for name in REQUIRED_REPORTS}
    versions = {payload.get("artifact_version") for payload in payloads.values()}
    if versions != {ARTIFACT_VERSION}:
        raise ValueError("all experiment reports must use artifact_version 3.0.0")
    manifest = payloads["model_manifest.json"]
    feature_config = payloads["feature_config.json"]
    metrics = payloads["metrics.json"]
    leaderboard = payloads["leaderboard.json"]
    error_analysis = payloads["error_analysis.json"]
    model_card = payloads["model_card.json"]

    required_fields = {
        "model_manifest.json": {
            "model_contract_version",
            "feature_version",
            "model_version",
            "model_type",
            "winner",
            "model_artifacts",
            "model_artifact_metadata",
            "report_artifacts",
            "refit_strategy",
            "split_manifest",
            "split_indices",
            "collection_year",
            "seed",
            "target_transform",
            "provenance",
            "units",
            "counts",
        },
        "feature_config.json": {
            "feature_version",
            "model_version",
            "target_col",
            "feature_cols",
            "numeric_features",
            "categorical_features",
            "target_transform",
            "collection_year",
            "category_options",
            "currency",
            "price_unit",
            "mileage_unit",
        },
        "metrics.json": {
            "model_version",
            "model_type",
            "winner",
            "evaluation_scope",
            "test_metrics",
            "development_mean_baseline",
            "baseline_evidence",
            "quality_gate",
            "warnings",
            "thresholds",
            "split_manifest",
            "split_indices",
            "seed",
            "collection_year",
            "sample_count",
            "split",
        },
        "leaderboard.json": {
            "model_version",
            "selection_scope",
            "selection_method",
            "winner",
            "winner_model_type",
            "candidate_count",
            "candidates",
            "outer_test_metrics_in_candidates",
            "seed",
            "collection_year",
        },
        "error_analysis.json": {
            "model_version",
            "evaluation_scope",
            "development_mean_baseline",
            "baseline_evidence",
            "price_quartiles",
            "model_family_frequency",
            "full_model_seen_status",
        },
        "model_card.json": {
            "model_contract_version",
            "feature_version",
            "model_version",
            "model_type",
            "winner",
            "collection_year",
            "seed",
            "data_source",
            "currency",
            "price_unit",
            "mileage_unit",
            "sample_count",
            "split",
            "thresholds",
            "quality_gate",
            "warnings",
            "test_metrics",
            "counts",
            "target_transform",
            "refit_strategy",
            "development_mean_baseline",
            "cv_selection",
            "independent_holdout",
            "error_analysis",
            "category_options",
            "limitations",
        },
    }
    for report_name, fields in required_fields.items():
        missing = fields.difference(payloads[report_name])
        if missing:
            raise ValueError(
                f"{report_name} is missing required fields: {sorted(missing)}"
            )

    model_version = _validated_v3_model_version(manifest.get("model_version"))
    if any(
        payload.get("model_version") != model_version
        for payload in payloads.values()
    ):
        raise ValueError("all experiment reports must agree on model_version")
    if {
        manifest["feature_version"],
        feature_config["feature_version"],
        model_card["feature_version"],
    } != {FEATURE_VERSION}:
        raise ValueError("experiment reports must agree on feature_version")
    if {
        manifest["model_contract_version"],
        model_card["model_contract_version"],
    } != {MODEL_CONTRACT_VERSION}:
        raise ValueError("experiment reports must agree on model contract version")

    model_type = manifest["model_type"]
    if model_type not in {"catboost", "extra_trees", "mlp"} or {
        metrics["model_type"],
        leaderboard["winner_model_type"],
        model_card["model_type"],
    } != {model_type}:
        raise ValueError("experiment reports must agree on model_type")

    collection_year, seed = _validate_feature_report_semantics(
        manifest,
        feature_config,
        metrics,
        leaderboard,
        model_card,
    )

    split_manifest = manifest["split_manifest"]
    split_indices = manifest["split_indices"]
    if not isinstance(split_manifest, Mapping) or not isinstance(
        split_indices, Mapping
    ):
        raise TypeError("model split manifest and indices must be objects")
    if metrics["split_manifest"] != split_manifest or metrics["split_indices"] != split_indices:
        raise ValueError("model_manifest.json and metrics.json split records must agree")
    for field in ("development", "test", "folds"):
        if split_indices.get(field) != split_manifest.get(field):
            raise ValueError(f"split_indices.{field} must agree with split_manifest")
    refit = manifest["refit_strategy"]
    if not isinstance(refit, Mapping) or model_card["refit_strategy"] != refit:
        raise ValueError("experiment reports must agree on refit strategy")
    if split_indices.get("train") != refit.get("train_indices") or split_indices.get(
        "validation"
    ) != refit.get("validation_indices"):
        raise ValueError("refit strategy indices must agree with split_indices")

    counts = manifest["counts"]
    if not isinstance(counts, Mapping) or model_card["counts"] != counts:
        raise ValueError("model manifest and model card counts must agree")
    development_ids = _validate_split_report_semantics(
        split_manifest,
        split_indices,
        refit,
        counts,
        metrics,
        model_card,
        model_type,
    )
    winner = _validated_candidate_result(
        manifest["winner"],
        _winner_config(manifest["winner"]),
        collection_year,
        development_ids,
        split_manifest["folds"],
    )
    if model_card["winner"] != winner:
        raise ValueError("model manifest and model card winner records must agree")
    winner_name = winner["name"]
    if metrics["winner"] != winner_name or leaderboard["winner"] != winner_name:
        raise ValueError("experiment reports must agree on winner")
    if winner["model_type"] != model_type:
        raise ValueError("winner model_type must agree with model manifest")

    candidates = leaderboard["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("leaderboard candidates must be a nonempty list")
    validated_candidates = [
        _validated_candidate_result(
            candidate,
            _winner_config(candidate),
            collection_year,
            development_ids,
            split_manifest["folds"],
        )
        for candidate in candidates
    ]
    ranked_winner = select_winner(validated_candidates)
    matching_winners = [
        candidate for candidate in validated_candidates
        if candidate["name"] == winner_name
    ]
    if (
        leaderboard["selection_scope"] != "development_cv_only"
        or leaderboard["selection_method"] != "rank_candidates"
        or leaderboard["outer_test_metrics_in_candidates"] is not False
        or leaderboard["candidate_count"] != len(validated_candidates)
        or ranked_winner["name"] != winner_name
        or len(matching_winners) != 1
        or matching_winners[0] != winner
    ):
        raise ValueError("leaderboard must describe CV-only winner selection")
    cv_selection = model_card["cv_selection"]
    if (
        not isinstance(cv_selection, Mapping)
        or cv_selection.get("scope") != "development_cv_only"
        or cv_selection.get("winner") != winner_name
        or cv_selection.get("winner_cv") != winner["cv"]
        or cv_selection.get("candidate_count") != len(validated_candidates)
        or counts.get("candidates") != len(validated_candidates)
    ):
        raise ValueError("model card CV selection must agree with ranked leaderboard")

    test_metrics = _validated_candidate_metrics(
        metrics["test_metrics"], "test_metrics"
    )
    if set(test_metrics) != _CANDIDATE_METRICS:
        raise ValueError("test_metrics must contain the complete shared metric fields")
    independent_holdout = model_card["independent_holdout"]
    if not isinstance(independent_holdout, Mapping):
        raise TypeError("model card independent_holdout must be an object")
    if (
        model_card["test_metrics"] != test_metrics
        or independent_holdout.get("metrics") != test_metrics
        or metrics["evaluation_scope"] != "recorded_test"
        or independent_holdout.get("scope") != "recorded_test"
    ):
        raise ValueError("experiment reports must agree on holdout metrics and scope")

    recorded_gate = {
        "quality_gate": metrics["quality_gate"],
        "warnings": metrics["warnings"],
        "thresholds": metrics["thresholds"],
    }
    recalculated_gate = assess_quality_gate(test_metrics, metrics["thresholds"])
    if recorded_gate != recalculated_gate:
        raise ValueError("recorded quality gate does not agree with recalculated gate")
    if (
        model_card["quality_gate"] != recorded_gate["quality_gate"]
        or model_card["warnings"] != recorded_gate["warnings"]
        or model_card["thresholds"] != recorded_gate["thresholds"]
        or independent_holdout.get("quality_gate") != recorded_gate
    ):
        raise ValueError("model card quality gate fields must agree with metrics.json")

    development_mean = metrics["development_mean_baseline"]
    metrics_baseline_evidence = metrics["baseline_evidence"]
    error_baseline_evidence = error_analysis["baseline_evidence"]
    nested_error_analysis = {
        key: value
        for key, value in error_analysis.items()
        if key not in {
            "artifact_version",
            "model_version",
            "evaluation_scope",
            "price_unit",
        }
    }
    if (
        isinstance(development_mean, bool)
        or not isinstance(development_mean, Real)
        or not math.isfinite(float(development_mean))
        or model_card["development_mean_baseline"] != development_mean
        or error_analysis["development_mean_baseline"] != development_mean
        or not isinstance(metrics_baseline_evidence, Mapping)
        or metrics_baseline_evidence.get("source") != "development_only"
        or metrics_baseline_evidence.get("value_inr") != development_mean
        or not isinstance(error_baseline_evidence, Mapping)
        or error_baseline_evidence.get("source") != "development_only"
        or error_baseline_evidence.get("value_inr") != development_mean
        or isinstance(refit.get("development_mean"), bool)
        or not isinstance(refit.get("development_mean"), Real)
        or not math.isfinite(float(refit["development_mean"]))
        or float(refit["development_mean"]) != float(development_mean)
        or model_card["error_analysis"] != nested_error_analysis
    ):
        raise ValueError("experiment reports must agree on development baseline")

    expected_split_counts = {
        "train": len(split_indices.get("train", [])),
        "validation": len(split_indices.get("validation", [])),
        "development": len(split_indices.get("development", [])),
        "test": len(split_indices.get("test", [])),
    }
    if (
        model_card["split"] != expected_split_counts
        or model_card["sample_count"] != counts.get("total")
        or independent_holdout.get("count") != expected_split_counts["test"]
    ):
        raise ValueError("model card compatibility counts must agree with split records")
    _validate_error_analysis_semantics(
        error_analysis,
        expected_split_counts["test"],
        float(development_mean),
    )
    if (
        model_card["data_source"] != manifest["provenance"]
        or feature_config["category_options"] != model_card["category_options"]
        or any(
            payload.get(field) != expected
            for payload in (feature_config, model_card)
            for field, expected in (
                ("currency", "INR"),
                ("price_unit", "INR"),
                ("mileage_unit", "km"),
            )
        )
    ):
        raise ValueError("model card compatibility fields must agree with v3 reports")

    expected_roles = {
        "catboost": {"model"},
        "extra_trees": {"bundle"},
        "mlp": {"preprocessor", "model"},
    }[model_type]
    model_artifacts = manifest["model_artifacts"]
    model_metadata = manifest["model_artifact_metadata"]
    if (
        not isinstance(model_artifacts, Mapping)
        or set(model_artifacts) != expected_roles
        or not isinstance(model_metadata, Mapping)
        or model_metadata.get("artifacts") != model_artifacts
        or model_metadata.get("model_type") != model_type
        or manifest["target_transform"] != feature_config["target_transform"]
        or manifest["target_transform"] != model_metadata.get("target_transform")
        or set(manifest["report_artifacts"]) != set(REQUIRED_REPORTS)
    ):
        raise ValueError("model artifact metadata and declared artifact roles must agree")
    _validate_declared_artifacts(
        experiment_dir,
        model_artifacts,
    )
    return payloads


def smoke_load_experiment(
    experiment_dir: str | Path,
    loader: Callable[[Path, Mapping[str, Any]], Any] | None = None,
) -> bool:
    """Strictly validate reports and structurally reload the saved winner."""
    experiment_dir = Path(experiment_dir)
    payloads = _validate_experiment_directory(experiment_dir)
    model_loader = loader or _load_saved_model_for_smoke
    model_loader(experiment_dir, payloads["model_manifest.json"])
    return True


def _rename_directory(source: str | Path, target: str | Path) -> Path:
    return Path(source).rename(target)


def _protected_publication_paths() -> set[Path]:
    backend_root = DEFAULT_MODELS_DIR.resolve().parent
    project_root = backend_root.parent
    protected_roots = {
        Path.home().resolve(),
        backend_root,
        project_root,
        Path.cwd().resolve(),
    }
    protected = set()
    for root in protected_roots:
        protected.add(root)
        protected.update(root.parents)
    return protected


def _has_ownership_sentinel(formal_dir: Path) -> bool:
    sentinel = formal_dir / MODEL_OWNERSHIP_SENTINEL
    if not sentinel.is_file():
        return False
    try:
        payload = _read_json(sentinel)
    except (OSError, TypeError, ValueError):
        return False
    return payload == {
        "owner": _MODEL_OWNER,
        "sentinel_version": _MODEL_SENTINEL_VERSION,
    }


def _has_legacy_model_signature(formal_dir: Path) -> bool:
    required = ("feature_config.json", "metrics.json", "model_card.json")
    try:
        payloads = [_read_json(formal_dir / name) for name in required]
    except (OSError, TypeError, ValueError):
        return False
    if {payload.get("artifact_version") for payload in payloads} != {"2.0.0"}:
        return False
    versions = {payload.get("model_version") for payload in payloads}
    if len(versions) != 1 or not next(iter(versions)):
        return False
    return (
        (formal_dir / "preprocess.joblib").is_file()
        and (formal_dir / "price_mlp.pt").is_file()
    )


def _has_v3_model_signature(formal_dir: Path) -> bool:
    try:
        _validate_experiment_directory(formal_dir)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _validate_publication_target(
    requested_formal_dir: str | Path,
    formal_dir: Path,
) -> None:
    requested = Path(requested_formal_dir)
    lexical = requested.absolute()
    if os.path.lexists(requested) and lexical != formal_dir:
        raise ValueError("formal model directory must not be a symlink or junction")
    if formal_dir in _protected_publication_paths():
        raise ValueError("formal model directory resolves to a protected broad path")
    if formal_dir.exists() and not formal_dir.is_dir():
        raise ValueError("formal model target exists as a file, not a directory")


def _resolve_publication_target(requested_formal_dir: str | Path) -> Path:
    requested = Path(requested_formal_dir)
    if os.path.lexists(requested):
        try:
            return requested.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise ValueError(
                "formal model target must not be a dangling link or junction"
            ) from exc
    return requested.resolve(strict=False)


def _validate_formal_ownership(formal_dir: Path) -> None:
    if not formal_dir.exists() or not any(formal_dir.iterdir()):
        return
    if formal_dir == DEFAULT_MODELS_DIR.resolve():
        if _has_legacy_model_signature(formal_dir) or _has_v3_model_signature(
            formal_dir
        ) or _has_ownership_sentinel(formal_dir):
            return
    elif _has_ownership_sentinel(formal_dir) or _has_legacy_model_signature(
        formal_dir
    ) or _has_v3_model_signature(formal_dir):
        return
    raise ValueError(
        "existing nonempty formal target is not an owned model directory"
    )


def _write_ownership_sentinel(directory: Path) -> Path:
    return _write_json(
        directory / MODEL_OWNERSHIP_SENTINEL,
        {
            "owner": _MODEL_OWNER,
            "sentinel_version": _MODEL_SENTINEL_VERSION,
        },
    )


def _publication_lock_prefix(formal_dir: Path) -> str:
    return f".{formal_dir.name}.publish.lock-"


def _publication_lock_path(formal_dir: Path, token: str) -> Path:
    if not _is_publication_lock_token(token):
        raise ValueError("publication lock token is invalid")
    return formal_dir.with_name(f"{_publication_lock_prefix(formal_dir)}{token}")


def _publication_lock_token_from_path(lock_dir: Path, formal_dir: Path) -> str:
    prefix = _publication_lock_prefix(formal_dir)
    token = lock_dir.name[len(prefix) :] if lock_dir.name.startswith(prefix) else ""
    if (
        lock_dir.parent != formal_dir.parent
        or not _is_publication_lock_token(token)
        or lock_dir != _publication_lock_path(formal_dir, token)
    ):
        raise ValueError("refusing publication lock operation outside exact lock path")
    return token


def _validate_publication_lock_path(lock_dir: Path, formal_dir: Path) -> None:
    _publication_lock_token_from_path(lock_dir, formal_dir)


def _publication_mutex_name(formal_dir: Path) -> str:
    identity = str(formal_dir.resolve(strict=False)).casefold().encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    return f"Local\\car-valuation-model-publish-{digest}"


def _acquire_publication_mutex(formal_dir: Path) -> PublicationMutex:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        )
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateMutexW(
            None,
            False,
            _publication_mutex_name(formal_dir),
        )
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        wait_result = kernel32.WaitForSingleObject(handle, 0)
        if wait_result not in {0x00000000, 0x00000080}:
            kernel32.CloseHandle(handle)
            if wait_result == 0x00000102:
                raise FileExistsError(
                    "another publication owns the model directory lock"
                )
            raise ctypes.WinError(ctypes.get_last_error())
        return PublicationMutex("windows", handle)

    import fcntl

    guard_path = formal_dir.with_name(f".{formal_dir.name}.publish.mutex")
    guard_file = guard_path.open("a+b")
    try:
        fcntl.flock(guard_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        guard_file.close()
        raise FileExistsError(
            "another publication owns the model directory lock"
        ) from None
    return PublicationMutex("posix", guard_file)


def _release_publication_mutex(mutex: PublicationMutex) -> None:
    if not mutex.held:
        return
    if mutex.kind == "windows":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel32.ReleaseMutex.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        if not kernel32.ReleaseMutex(mutex.resource):
            raise ctypes.WinError(ctypes.get_last_error())
        mutex.held = False
        kernel32.CloseHandle(mutex.resource)
        return

    import fcntl

    try:
        fcntl.flock(mutex.resource.fileno(), fcntl.LOCK_UN)
    finally:
        mutex.held = False
        mutex.resource.close()


def _publication_lock_paths(formal_dir: Path) -> list[Path]:
    prefix = _publication_lock_prefix(formal_dir)
    return sorted(
        path
        for path in formal_dir.parent.iterdir()
        if path.name.startswith(prefix)
    )


def _lock_payload(
    lock: PublicationLock,
    state: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    if state not in {"active", "completed", "failed"}:
        raise ValueError("publication lock state is invalid")
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "state": state,
        "token": lock.token,
        "owner": lock.owner,
        "formal_dir": str(lock.formal_dir),
        "acquired_at": lock.acquired_at,
        "updated_at": _utc_now().isoformat(),
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["message"] = str(error)
    return payload


def _is_publication_lock_token(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 32:
        return False
    try:
        return uuid.UUID(hex=value).hex == value
    except ValueError:
        return False


def _read_owned_publication_lock(lock: PublicationLock) -> dict[str, Any]:
    _validate_publication_lock_path(lock.path, lock.formal_dir)
    metadata_path = lock.path / PUBLICATION_LOCK_FILENAME
    if metadata_path.is_symlink():
        raise PermissionError("publication lock metadata must not be a symlink")
    try:
        metadata = _read_json(metadata_path)
    except (OSError, TypeError, ValueError) as exc:
        raise PermissionError("publication lock ownership cannot be verified") from exc
    if (
        not _is_publication_lock_token(metadata.get("token"))
        or metadata["token"] != lock.token
    ):
        raise PermissionError("publication lock token is owned by another publisher")
    return metadata


def _mark_publication_lock(
    lock: PublicationLock,
    state: str,
    error: BaseException | None = None,
) -> Path:
    _validate_publication_lock_path(lock.path, lock.formal_dir)
    metadata_path = lock.path / PUBLICATION_LOCK_FILENAME
    if metadata_path.exists():
        _read_owned_publication_lock(lock)
    elif state != "active":
        raise PermissionError("publication lock owner metadata is missing")
    marked = _write_json(
        metadata_path,
        _lock_payload(lock, state, error),
    )
    if state in {"completed", "failed"}:
        _release_publication_mutex(lock.mutex)
    return marked


def _recover_terminal_publication_lock(lock_dir: Path, formal_dir: Path) -> bool:
    try:
        directory_token = _publication_lock_token_from_path(lock_dir, formal_dir)
    except ValueError:
        return False
    lexical = lock_dir.absolute()
    try:
        resolved = lock_dir.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return False
    if lexical != resolved or not lock_dir.is_dir():
        return False
    metadata_path = lock_dir / PUBLICATION_LOCK_FILENAME
    if metadata_path.is_symlink():
        return False
    try:
        metadata = _read_json(metadata_path)
    except (OSError, TypeError, ValueError):
        return False
    if (
        metadata.get("state") not in {"completed", "failed"}
        or not _is_publication_lock_token(metadata.get("token"))
        or metadata["token"] != directory_token
    ):
        return False
    shutil.rmtree(lock_dir)
    return True


def _acquire_publication_lock(formal_dir: Path) -> PublicationLock:
    mutex = _acquire_publication_mutex(formal_dir)
    lock_dir = None
    try:
        for existing_lock in _publication_lock_paths(formal_dir):
            try:
                recovered = _recover_terminal_publication_lock(
                    existing_lock,
                    formal_dir,
                )
            except FileNotFoundError:
                recovered = True
            if not recovered:
                raise FileExistsError(
                    f"another publication owns the model directory lock: {existing_lock}"
                )

        token = uuid.uuid4().hex
        lock_dir = _publication_lock_path(formal_dir, token)
        lock_dir.mkdir()
        acquired_at = _utc_now().isoformat()
        lock = PublicationLock(
            path=lock_dir,
            formal_dir=formal_dir,
            token=token,
            owner={"pid": os.getpid(), "thread_id": threading.get_ident()},
            acquired_at=acquired_at,
            mutex=mutex,
        )
        try:
            _mark_publication_lock(lock, "active")
        except Exception:
            shutil.rmtree(lock_dir, ignore_errors=True)
            raise
        return lock
    except Exception:
        _release_publication_mutex(mutex)
        raise


def _release_publication_lock(lock: PublicationLock) -> None:
    try:
        _read_owned_publication_lock(lock)
        shutil.rmtree(lock.path)
    finally:
        _release_publication_mutex(lock.mutex)


@contextmanager
def _publication_lock(formal_dir: Path):
    lock = _acquire_publication_lock(formal_dir)
    try:
        yield lock
    except BaseException as exc:
        try:
            _mark_publication_lock(lock, "failed", exc)
        except OSError:
            pass
        try:
            _release_publication_lock(lock)
        except OSError:
            pass
        raise
    else:
        try:
            _mark_publication_lock(lock, "completed")
        except OSError:
            pass
        try:
            _release_publication_lock(lock)
        except OSError:
            pass


def _remove_generated_publication_tree(
    path: Path,
    formal_dir: Path,
    kind: str,
) -> None:
    expected_prefix = f".{formal_dir.name}.{kind}-"
    if (
        kind not in {"staging", "backup"}
        or path.parent != formal_dir.parent
        or not path.name.startswith(expected_prefix)
        or len(path.name) <= len(expected_prefix)
    ):
        raise ValueError("refusing cleanup outside generated publication paths")
    shutil.rmtree(path)


def _validate_publication_relationship(
    experiment_dir: Path,
    formal_dir: Path,
) -> None:
    if (
        experiment_dir == formal_dir
        or experiment_dir.is_relative_to(formal_dir)
        or formal_dir.is_relative_to(experiment_dir)
    ):
        raise ValueError(
            "experiment and formal model directories must not be equal, ancestor, or descendant"
        )


def _write_published_run_status(
    experiment_dir: Path,
    staging_dir: Path,
    manifest: Mapping[str, Any],
) -> Path:
    source_status = {}
    source_status_path = experiment_dir / RUN_STATUS_FILENAME
    if source_status_path.is_file():
        try:
            source_status = _read_json(source_status_path)
        except (OSError, TypeError, ValueError):
            source_status = {}
    model_version = str(manifest["model_version"])
    run_id = str(source_status.get("run_id") or model_version.removeprefix("v3-"))
    started_at = str(
        source_status.get("started_at")
        or manifest.get("created_at")
        or _utc_now().isoformat()
    )
    return _write_run_status(
        staging_dir,
        run_id,
        started_at,
        "publish_experiment",
        "completed",
        published=True,
        quality_gate="pass",
    )


def _write_publication_warning(
    experiment_dir: Path,
    backup_dir: Path,
    exc: OSError,
) -> Path:
    return _write_json(
        experiment_dir / PUBLICATION_WARNING_FILENAME,
        {
            "artifact_version": ARTIFACT_VERSION,
            "stage": "publish_experiment",
            "warning_type": "backup_cleanup_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "backup_name": backup_dir.name,
            "timestamp": _utc_now().isoformat(),
        },
    )


def publish_experiment(
    experiment_dir: str | Path,
    formal_dir: str | Path,
    *,
    smoke_loader: Callable[[Path, Mapping[str, Any]], Any] | None = None,
) -> bool:
    """Publish a passing experiment copy with backup and rollback."""
    requested_formal_dir = formal_dir
    experiment_dir = Path(experiment_dir).resolve(strict=True)
    formal_dir = _resolve_publication_target(formal_dir)
    _validate_publication_target(requested_formal_dir, formal_dir)
    _validate_publication_relationship(experiment_dir, formal_dir)
    metrics = _read_json(experiment_dir / "metrics.json")
    if metrics.get("quality_gate") != "pass":
        return False
    try:
        verified_gate = assess_quality_gate(
            metrics.get("test_metrics", {}),
            metrics.get("thresholds"),
        )
    except (TypeError, ValueError):
        return False
    if verified_gate["quality_gate"] != "pass":
        return False

    payloads = _validate_experiment_directory(experiment_dir)
    _validate_formal_ownership(formal_dir)
    formal_dir.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(formal_dir):
        current_formal_dir = _resolve_publication_target(requested_formal_dir)
        _validate_publication_target(requested_formal_dir, current_formal_dir)
        if current_formal_dir != formal_dir:
            raise ValueError("formal model target changed after publication lock")
        _validate_publication_relationship(experiment_dir, current_formal_dir)
        _validate_formal_ownership(formal_dir)
        token = uuid.uuid4().hex
        staging_dir = formal_dir.with_name(f".{formal_dir.name}.staging-{token}")
        backup_dir = formal_dir.with_name(f".{formal_dir.name}.backup-{token}")
        if staging_dir.exists() or backup_dir.exists():
            raise FileExistsError("publication staging or backup path already exists")

        backup_created = False
        try:
            shutil.copytree(experiment_dir, staging_dir)
            _write_ownership_sentinel(staging_dir)
            _write_publication_generation(staging_dir)
            _write_published_run_status(
                experiment_dir,
                staging_dir,
                payloads["model_manifest.json"],
            )
            staged_payloads = _validate_experiment_directory(staging_dir)
            staged_metrics = staged_payloads["metrics.json"]
            staged_gate = assess_quality_gate(
                staged_metrics["test_metrics"],
                staged_metrics["thresholds"],
            )
            if (
                staged_metrics["quality_gate"] != "pass"
                or staged_gate["quality_gate"] != "pass"
            ):
                raise ValueError("staged experiment must retain a passing quality gate")
            smoke_load_experiment(staging_dir, loader=smoke_loader)
            if formal_dir.exists():
                _rename_directory(formal_dir, backup_dir)
                backup_created = True
            _rename_directory(staging_dir, formal_dir)
        except Exception:
            if backup_created and backup_dir.exists() and not formal_dir.exists():
                _rename_directory(backup_dir, formal_dir)
            raise
        finally:
            if staging_dir.exists():
                _remove_generated_publication_tree(
                    staging_dir,
                    formal_dir,
                    "staging",
                )

        if backup_dir.exists():
            try:
                _remove_generated_publication_tree(
                    backup_dir,
                    formal_dir,
                    "backup",
                )
            except OSError as exc:
                # The formal swap is complete. Keep the backup recoverable rather
                # than report a false publication failure.
                try:
                    _write_publication_warning(experiment_dir, backup_dir, exc)
                except OSError:
                    pass
    return True


def train_and_publish(
    dataset_path: str | Path,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    seed: int = SEED,
    provenance: Mapping[str, Any] | None = None,
    *,
    collection_year: int,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    candidates: Sequence[CandidateConfig] | None = None,
    candidate_evaluator: Callable[..., Mapping[str, Any]] | None = None,
    adapter_factory: Callable[[CandidateConfig, int], Any] | None = None,
    smoke_loader: Callable[[Path, Mapping[str, Any]], Any] | None = None,
    rare_threshold: int = DEFAULT_RARE_MODEL_FAMILY_THRESHOLD,
) -> dict[str, Any]:
    """Run the fixed v3 train/evaluate/audit/publish orchestration."""
    run_id = _new_run_id()
    created_at = _utc_now().isoformat()
    experiment_dir = Path(experiments_dir) / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    stage = "initialize"
    _write_run_status(
        experiment_dir,
        run_id,
        created_at,
        stage,
        "running",
    )

    try:
        stage = "validate_collection_year"
        _write_run_status(
            experiment_dir, run_id, created_at, stage, "running"
        )
        collection_year = _validate_collection_year(collection_year)

        stage = "load_dataset"
        _write_run_status(
            experiment_dir, run_id, created_at, stage, "running"
        )
        frame = load_dataset(dataset_path).drop_duplicates().reset_index(drop=True)

        stage = "build_split_manifest"
        _write_run_status(
            experiment_dir, run_id, created_at, stage, "running"
        )
        manifest = build_split_manifest(frame, seed)

        stage = "run_competition"
        _write_run_status(
            experiment_dir, run_id, created_at, stage, "running"
        )
        competition_candidates = (
            default_candidates() if candidates is None else candidates
        )
        leaderboard = run_competition(
            frame,
            manifest,
            competition_candidates,
            seed,
            collection_year,
            evaluator=candidate_evaluator,
        )

        stage = "select_winner"
        _write_run_status(
            experiment_dir, run_id, created_at, stage, "running"
        )
        winner = select_winner(leaderboard)

        stage = "refit_winner"
        _write_run_status(
            experiment_dir, run_id, created_at, stage, "running"
        )
        fitted = refit_winner(
            frame.loc[manifest["development"]],
            winner,
            seed,
            collection_year,
            adapter_factory=adapter_factory,
        )

        stage = "evaluate_holdout"
        _write_run_status(
            experiment_dir, run_id, created_at, stage, "running"
        )
        holdout = evaluate_holdout(fitted, frame.loc[manifest["test"]])

        stage = "assess_quality_gate"
        _write_run_status(
            experiment_dir, run_id, created_at, stage, "running"
        )
        gate = assess_quality_gate(holdout["metrics"])

        source = {
            "source_id": "normalized-training-csv",
            "source_url": SOURCE_URL,
            "raw_path": str(Path(dataset_path)),
        }
        if provenance:
            source.update(provenance)
        experiment = {
            "run_id": run_id,
            "created_at": created_at,
            "frame": frame,
            "split_manifest": manifest,
            "leaderboard": leaderboard,
            "winner": winner,
            "fitted_model": fitted,
            "refit": _refit_metadata(fitted),
            "holdout": holdout,
            "gate": gate,
            "seed": seed,
            "collection_year": collection_year,
            "provenance": source,
            "rare_threshold": rare_threshold,
        }

        stage = "write_experiment_artifacts"
        _write_run_status(
            experiment_dir, run_id, created_at, stage, "running"
        )
        write_experiment_artifacts(experiment, experiment_dir)

        smoke_status = "pass"
        smoke_error = None
        stage = "smoke_load_experiment"
        _write_run_status(
            experiment_dir, run_id, created_at, stage, "running"
        )
        try:
            smoke_load_experiment(experiment_dir, loader=smoke_loader)
        except Exception as exc:
            smoke_status = "fail"
            smoke_error = f"{type(exc).__name__}: {exc}"
            _write_json(
                experiment_dir / "smoke_failure.json",
                {
                    "artifact_version": ARTIFACT_VERSION,
                    "stage": stage,
                    "error": smoke_error,
                    "created_at": _utc_now().isoformat(),
                },
            )
            _record_run_failure(
                experiment_dir,
                run_id,
                created_at,
                stage,
                exc,
            )
            return {
                "artifact_version": ARTIFACT_VERSION,
                "model_version": f"v3-{run_id}",
                "model_type": winner["model_type"],
                "winner": winner["name"],
                "collection_year": collection_year,
                "seed": seed,
                "quality_gate": gate["quality_gate"],
                "warnings": gate["warnings"],
                "thresholds": gate["thresholds"],
                "test_metrics": holdout["metrics"],
                "experiment_dir": str(experiment_dir),
                "smoke_status": smoke_status,
                "smoke_error": smoke_error,
                "published": False,
            }

        published = False
        if gate["quality_gate"] == "pass":
            stage = "publish_experiment"
            _write_run_status(
                experiment_dir, run_id, created_at, stage, "running"
            )
            published = publish_experiment(
                experiment_dir,
                models_dir,
                smoke_loader=smoke_loader,
            )
            if not published:
                rejection = RuntimeError(
                    "publication rejected a smoke-validated passing experiment"
                )
                _record_run_failure(
                    experiment_dir,
                    run_id,
                    created_at,
                    stage,
                    rejection,
                )
                smoke_error = None
                return {
                    "artifact_version": ARTIFACT_VERSION,
                    "model_version": f"v3-{run_id}",
                    "model_type": winner["model_type"],
                    "winner": winner["name"],
                    "collection_year": collection_year,
                    "seed": seed,
                    "quality_gate": gate["quality_gate"],
                    "warnings": gate["warnings"],
                    "thresholds": gate["thresholds"],
                    "test_metrics": holdout["metrics"],
                    "experiment_dir": str(experiment_dir),
                    "smoke_status": smoke_status,
                    "smoke_error": smoke_error,
                    "published": False,
                }

        stage = "complete"
        _write_run_status(
            experiment_dir,
            run_id,
            created_at,
            stage,
            "completed",
            published=published,
            quality_gate=gate["quality_gate"],
        )
    except Exception as exc:
        _record_run_failure(
            experiment_dir,
            run_id,
            created_at,
            stage,
            exc,
        )
        raise

    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_version": f"v3-{run_id}",
        "model_type": winner["model_type"],
        "winner": winner["name"],
        "collection_year": collection_year,
        "seed": seed,
        "quality_gate": gate["quality_gate"],
        "warnings": gate["warnings"],
        "thresholds": gate["thresholds"],
        "test_metrics": holdout["metrics"],
        "experiment_dir": str(experiment_dir),
        "smoke_status": smoke_status,
        "smoke_error": smoke_error,
        "published": published,
    }


def _prepare_downloaded_dataset() -> tuple[Path, dict[str, Any]]:
    from scripts.download_public_dataset import DEFAULT_METADATA, download_dataset
    from services.public_dataset_adapter import adapt_car_details_v4

    manifest = download_dataset(SOURCE_URL, DEFAULT_RAW_DATASET, DEFAULT_METADATA)
    normalized = adapt_car_details_v4(pd.read_csv(DEFAULT_RAW_DATASET))
    DEFAULT_PROCESSED_DATASET.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(DEFAULT_PROCESSED_DATASET, index=False)
    return DEFAULT_PROCESSED_DATASET, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the vehicle valuation model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PROCESSED_DATASET)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--collection-year", type=int, default=DEFAULT_COLLECTION_YEAR
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    provenance = None
    dataset = args.dataset
    if args.download:
        dataset, provenance = _prepare_downloaded_dataset()
    result = train_and_publish(
        dataset,
        models_dir=args.models_dir,
        experiments_dir=args.experiments_dir,
        seed=args.seed,
        collection_year=args.collection_year,
        provenance=provenance,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
