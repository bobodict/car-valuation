"""Evaluate a formal model artifact on its recorded outer-test rows.

Usage from backend:
    python -m scripts.evaluate_model path/to/used_cars.csv
"""

import argparse
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import settings
from services.dataset_contract import load_dataset
from services.model_competition import calculate_metrics


def _reject_json_constant(value: str):
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"required model artifact is missing: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"model artifact must contain a JSON object: {path}")
    return value


def _is_v3(artifact: Mapping[str, Any]) -> bool:
    return str(artifact.get("artifact_version", "")).split(".", 1)[0] == "3"


def _recorded_test_indices(frame, artifact: Mapping[str, Any]) -> list[int] | None:
    split_indices = artifact.get("split_indices")
    if not isinstance(split_indices, Mapping):
        if _is_v3(artifact):
            raise ValueError("v3 model_manifest.json requires split_indices.test")
        return None
    values = split_indices.get("test")
    if values is None:
        if _is_v3(artifact):
            raise ValueError("v3 model_manifest.json requires split_indices.test")
        return None
    if not isinstance(values, list) or not values:
        if _is_v3(artifact):
            raise ValueError("v3 split_indices.test must be a nonempty list")
        return None

    indices = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("split_indices.test must contain only integer row IDs")
        indices.append(int(value))
    if len(indices) != len(set(indices)):
        raise ValueError("split_indices.test must not contain duplicate row IDs")
    missing = sorted(set(indices).difference(int(value) for value in frame.index))
    if missing:
        raise ValueError("split_indices.test contains row IDs absent from the dataset")
    return indices


def select_test_frame(frame, artifact: Mapping[str, Any]):
    test_indices = _recorded_test_indices(frame, artifact)
    if test_indices is None:
        return frame
    return frame.loc[test_indices]


def _recorded_development_mean(
    metrics_artifact: Mapping[str, Any],
) -> float:
    value = metrics_artifact.get("development_mean_baseline")
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            "v3 metrics.json requires a finite development_mean_baseline"
        )
    return float(value)


def baseline_for_test(
    frame,
    artifact: Mapping[str, Any],
    metrics_artifact: Mapping[str, Any] | None = None,
) -> np.ndarray:
    evaluation_count = len(select_test_frame(frame, artifact))
    if _is_v3(artifact):
        source = metrics_artifact if metrics_artifact is not None else artifact
        mean = _recorded_development_mean(source)
        return np.full(evaluation_count, mean, dtype=float)

    split_indices = artifact.get("split_indices", {})
    train_indices = split_indices.get("train") if isinstance(split_indices, Mapping) else None
    if not train_indices:
        train_mean = frame["price"].astype(float).mean()
    else:
        train_mean = frame.loc[train_indices, "price"].astype(float).mean()
    return np.full(evaluation_count, train_mean, dtype=float)


def load_model_manifest(models_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(models_dir) if models_dir is not None else settings.models_dir
    return _load_json_object(root / "model_manifest.json")


def load_metrics_artifact(models_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(models_dir) if models_dir is not None else settings.models_dir
    return _load_json_object(root / "metrics.json")


def load_runtime(models_dir: str | Path):
    """Import Task 7's runtime only after dataset and artifact validation."""
    from services.model_runtime import ModelRuntime

    return ModelRuntime.from_directory(Path(models_dir))


def _validate_v3_artifact_agreement(
    manifest: Mapping[str, Any],
    metrics_artifact: Mapping[str, Any],
) -> None:
    if not (_is_v3(manifest) or _is_v3(metrics_artifact)):
        return
    for field in ("artifact_version", "model_version", "model_type"):
        manifest_value = manifest.get(field)
        metrics_value = metrics_artifact.get(field)
        if (
            not isinstance(manifest_value, str)
            or not manifest_value
            or manifest_value != metrics_value
        ):
            raise ValueError(
                f"v3 model_manifest.json and metrics.json must agree on {field}"
            )
    if not _is_v3(manifest) or not _is_v3(metrics_artifact):
        raise ValueError(
            "v3 model_manifest.json and metrics.json artifact versions must match"
        )
    manifest_split = manifest.get("split_indices")
    metrics_split = metrics_artifact.get("split_indices")
    if (
        not isinstance(manifest_split, Mapping)
        or not isinstance(metrics_split, Mapping)
        or dict(manifest_split) != dict(metrics_split)
    ):
        raise ValueError(
            "v3 model_manifest.json and metrics.json split_indices must agree"
        )


def _predict(runtime: Any, evaluation_frame) -> np.ndarray:
    feature_frame = evaluation_frame.drop(columns=["price"])
    if hasattr(runtime, "predict"):
        predictions = runtime.predict(feature_frame)
    elif hasattr(runtime, "predict_one"):
        predictions = [
            runtime.predict_one(row.to_dict())
            for _, row in feature_frame.iterrows()
        ]
    else:
        raise TypeError("ModelRuntime must expose predict(frame) or predict_one(record)")
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 1:
        raise ValueError("runtime predictions must be exactly one-dimensional")
    if len(values) != len(evaluation_frame):
        raise ValueError("runtime prediction count does not match recorded test count")
    return values


def evaluate_model(
    dataset_path: str | Path,
    *,
    models_dir: str | Path | None = None,
    runtime_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    # Dataset validation intentionally precedes the optional Task 7 import/load.
    frame = load_dataset(dataset_path).drop_duplicates().reset_index(drop=True)
    root = Path(models_dir) if models_dir is not None else settings.models_dir
    manifest = load_model_manifest(root)
    metrics_artifact = load_metrics_artifact(root)
    _validate_v3_artifact_agreement(manifest, metrics_artifact)
    evaluation_frame = select_test_frame(frame, manifest)
    baseline = baseline_for_test(frame, manifest, metrics_artifact)

    loader = runtime_loader or load_runtime
    runtime = loader(root)
    predictions = _predict(runtime, evaluation_frame)
    actual = evaluation_frame["price"].astype(float).to_numpy()
    metrics = calculate_metrics(actual, predictions, baseline)
    result = {
        "evaluation_scope": "recorded_test" if _is_v3(manifest) else (
            "recorded_test" if _recorded_test_indices(frame, manifest) else "full_dataset"
        ),
        "model_version": manifest.get(
            "model_version", metrics_artifact.get("model_version")
        ),
        "model_type": manifest.get(
            "model_type", metrics_artifact.get("model_type")
        ),
        "count": int(len(actual)),
        "metrics": metrics,
    }
    result.update(metrics)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the supplied car price model.")
    parser.add_argument("dataset", help="CSV dataset containing the required feature columns")
    parser.add_argument("--models-dir", type=Path, default=settings.models_dir)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_model(args.dataset, models_dir=args.models_dir),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
