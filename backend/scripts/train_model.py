"""Train, audit, and atomically publish a v3 vehicle valuation model."""

import argparse
import json
import math
import shutil
import uuid
from collections.abc import Mapping, Sequence
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


SEED = 42
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


@dataclass
class RefitResult:
    adapter: Any
    candidate: CandidateConfig
    strategy: str
    train_indices: list[int]
    validation_indices: list[int]
    collection_year: int
    development_mean: float


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


def _without_post_selection_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    prohibited = {
        "error_analysis",
        "holdout_metrics",
        "outer_test_metrics",
        "subgroup_diagnostics",
        "test_metrics",
    }
    return {
        key: _json_value(value)
        for key, value in result.items()
        if key not in prohibited
    }


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
    leaderboard = []
    for candidate in candidates:
        result = evaluator(
            frame,
            manifest,
            candidate,
            collection_year=collection_year,
            seed=seed,
        )
        leaderboard.append(_without_post_selection_fields(result))
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

    error = predicted - actual
    relative_error = np.abs(error) / np.maximum(np.abs(actual), 1e-8)
    log_error = np.log1p(np.maximum(predicted, 0.0)) - np.log1p(actual)
    return {
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": None,
        "acc_10": float(np.mean(relative_error <= 0.10)),
        "acc_20": float(np.mean(relative_error <= 0.20)),
        "median_ape": float(np.median(relative_error)),
        "rmsle": float(np.sqrt(np.mean(log_error**2))),
        "baseline_rmse": float(np.sqrt(np.mean((actual - baseline) ** 2))),
        "baseline_r2": None,
    }


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

    quartile_count = min(4, len(actual))
    if quartile_count:
        quartile_ids = pd.qcut(
            pd.Series(actual).rank(method="first"),
            q=quartile_count,
            labels=False,
        ).to_numpy(dtype=int)
    else:
        quartile_ids = np.asarray([], dtype=int)
    quartile_groups = [
        _group_report(
            f"Q{quartile + 1}",
            quartile_ids == quartile,
            actual,
            predicted,
            development_mean,
        )
        for quartile in range(quartile_count)
    ]

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
            "definition_source": "recorded_test_actual_price",
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


def build_artifacts(
    experiment: Mapping[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Write the complete, strict-JSON v3 experiment artifact set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = experiment["frame"]
    manifest = _json_value(experiment["split_manifest"])
    leaderboard = [
        _without_post_selection_fields(result)
        for result in experiment["leaderboard"]
    ]
    winner = _without_post_selection_fields(experiment["winner"])
    fitted = experiment["fitted_model"]
    refit = _json_value(experiment.get("refit") or _refit_metadata(fitted))
    holdout = experiment["holdout"]
    metrics = _json_value(holdout["metrics"])
    gate = _json_value(experiment["gate"])
    seed = int(experiment["seed"])
    collection_year = _validate_collection_year(experiment["collection_year"])
    run_id = str(experiment["run_id"])
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
        return model
    if model_type == "extra_trees":
        bundle = joblib.load(experiment_dir / artifacts["bundle"])
        if not isinstance(bundle, dict) or not {"preprocessor", "model"}.issubset(bundle):
            raise ValueError("invalid ExtraTrees artifact bundle")
        return bundle
    if model_type == "mlp":
        preprocessor = joblib.load(experiment_dir / artifacts["preprocessor"])
        checkpoint = torch.load(
            experiment_dir / artifacts["model"],
            map_location="cpu",
            weights_only=True,
        )
        model = MLPRegressor(
            int(checkpoint["input_dim"]),
            tuple(checkpoint["hidden_dims"]),
            float(checkpoint["dropout"]),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return {"preprocessor": preprocessor, "model": model}
    raise ValueError(f"unsupported model_type in manifest: {model_type}")


def _validate_experiment_directory(experiment_dir: Path) -> dict[str, Any]:
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {experiment_dir}")
    payloads = {name: _read_json(experiment_dir / name) for name in REQUIRED_REPORTS}
    versions = {payload.get("artifact_version") for payload in payloads.values()}
    if versions != {ARTIFACT_VERSION}:
        raise ValueError("all experiment reports must use artifact_version 3.0.0")
    manifest = payloads["model_manifest.json"]
    _validate_declared_artifacts(
        experiment_dir,
        manifest.get("model_artifacts"),
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


def publish_experiment(
    experiment_dir: str | Path,
    formal_dir: str | Path,
) -> bool:
    """Publish a passing experiment copy with backup and rollback."""
    experiment_dir = Path(experiment_dir)
    formal_dir = Path(formal_dir)
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

    _validate_experiment_directory(experiment_dir)
    if experiment_dir.resolve() == formal_dir.resolve():
        raise ValueError("experiment and formal model directories must be different")
    formal_dir.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging_dir = formal_dir.with_name(f".{formal_dir.name}.staging-{token}")
    backup_dir = formal_dir.with_name(f".{formal_dir.name}.backup-{token}")
    if staging_dir.exists() or backup_dir.exists():
        raise FileExistsError("publication staging or backup path already exists")

    backup_created = False
    try:
        shutil.copytree(experiment_dir, staging_dir)
        _validate_experiment_directory(staging_dir)
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
            shutil.rmtree(staging_dir)

    if backup_dir.exists():
        shutil.rmtree(backup_dir)
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
    collection_year = _validate_collection_year(collection_year)
    frame = load_dataset(dataset_path).drop_duplicates().reset_index(drop=True)
    manifest = build_split_manifest(frame, seed)
    competition_candidates = default_candidates() if candidates is None else candidates
    leaderboard = run_competition(
        frame,
        manifest,
        competition_candidates,
        seed,
        collection_year,
        evaluator=candidate_evaluator,
    )
    winner = select_winner(leaderboard)
    fitted = refit_winner(
        frame.loc[manifest["development"]],
        winner,
        seed,
        collection_year,
        adapter_factory=adapter_factory,
    )
    holdout = evaluate_holdout(fitted, frame.loc[manifest["test"]])
    gate = assess_quality_gate(holdout["metrics"])

    run_id = _new_run_id()
    created_at = _utc_now().isoformat()
    experiment_dir = Path(experiments_dir) / run_id
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
    write_experiment_artifacts(experiment, experiment_dir)

    smoke_status = "pass"
    smoke_error = None
    try:
        smoke_load_experiment(experiment_dir, loader=smoke_loader)
    except Exception as exc:
        smoke_status = "fail"
        smoke_error = f"{type(exc).__name__}: {exc}"
        _write_json(
            experiment_dir / "smoke_failure.json",
            {
                "artifact_version": ARTIFACT_VERSION,
                "stage": "smoke_load_experiment",
                "error": smoke_error,
                "created_at": _utc_now().isoformat(),
            },
        )

    published = False
    if gate["quality_gate"] == "pass" and smoke_status == "pass":
        published = publish_experiment(experiment_dir, models_dir)

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
    parser.add_argument("--collection-year", type=int, required=True)
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
