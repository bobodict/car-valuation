"""Train and publish the deterministic vehicle valuation artifact."""

import argparse
import copy
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from services.dataset_contract import REQUIRED_DATASET_COLUMNS, load_dataset


SEED = 42
TARGET_COL = "price"
NUMERIC_FEATURES = ["mileage", "displacement", "seats", "owner_count", "car_age"]
CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "city",
    "transmission",
    "fuel_type",
    "vehicle_type",
    "color",
    "accident_history",
]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
QUALITY_THRESHOLDS = {"min_r2": 0.0, "min_acc_10": 0.5}
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_RAW_DATASET = Path(__file__).resolve().parents[1] / "data" / "raw" / "car-details-v4.csv"
DEFAULT_PROCESSED_DATASET = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "normalized_training.csv"
)
SOURCE_URL = "https://raw.githubusercontent.com/chandanverma07/DataSets/master/car%20details%20v4.csv"


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.0):
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


def split_dataset(frame: pd.DataFrame, seed: int = SEED):
    if len(frame) < 30:
        raise ValueError("training dataset must contain at least 30 rows")
    train_validation, test = train_test_split(frame, test_size=0.15, random_state=seed)
    validation_fraction = 0.15 / 0.85
    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction,
        random_state=seed,
    )
    return train, validation, test


def _error_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    relative_error = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-8)
    return float(np.mean(relative_error <= 0.10))


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline_prediction: np.ndarray,
) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    baseline_prediction = np.asarray(baseline_prediction, dtype=float)
    return {
        "mse": float(mean_squared_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "acc_10": _error_accuracy(actual, predicted),
        "baseline_rmse": float(mean_squared_error(actual, baseline_prediction) ** 0.5),
        "baseline_r2": float(r2_score(actual, baseline_prediction)),
    }


def assess_quality_gate(metrics: dict, thresholds: dict | None = None) -> dict:
    thresholds = thresholds or QUALITY_THRESHOLDS
    warnings = []
    if float(metrics.get("r2", float("nan"))) <= float(thresholds["min_r2"]):
        warnings.append("test R2 does not exceed the configured threshold")
    if float(metrics.get("acc_10", float("nan"))) < float(thresholds["min_acc_10"]):
        warnings.append("10% error accuracy is below the configured threshold")
    if float(metrics.get("rmse", float("inf"))) > float(metrics.get("baseline_rmse", float("inf"))):
        warnings.append("model RMSE does not beat the mean-price baseline")
    return {
        "quality_gate": "fail" if warnings else "pass",
        "warnings": warnings,
        "thresholds": thresholds,
    }


def publish_artifacts(
    source_dir: str | Path,
    target_dir: str | Path,
    quality_gate: str,
    allow_failed_publish: bool = False,
) -> bool:
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if quality_gate == "fail" and not allow_failed_publish:
        return False

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = target_dir.with_name(f".{target_dir.name}.backup-{uuid.uuid4().hex}")
    had_target = target_dir.exists()
    if had_target:
        target_dir.rename(backup_dir)
    try:
        source_dir.rename(target_dir)
    except Exception:
        if had_target and backup_dir.exists():
            backup_dir.rename(target_dir)
        raise
    finally:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    return True


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def _with_derived_features(frame: pd.DataFrame, collection_year: int) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["car_age"] = (collection_year - pd.to_numeric(enriched["year"])).clip(lower=0)
    return enriched


def _tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def _fit_mlp(X_train, y_train, X_validation, y_validation, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPRegressor(X_train.shape[1], hidden_dims=(128, 64), dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    train_features = _tensor(X_train)
    train_targets = _tensor(y_train).reshape(-1, 1)
    validation_features = _tensor(X_validation)
    validation_targets = _tensor(y_validation).reshape(-1, 1)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    patience = 35
    stale_epochs = 0

    for _ in range(300):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(train_features), train_targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(model(validation_features), validation_targets).item()
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_validation_loss ** 0.5


def _predict_scaled(model, values: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(_tensor(values)).numpy().reshape(-1)


def _category_options(frame: pd.DataFrame) -> dict:
    return {
        column: sorted(frame[column].dropna().astype(str).unique().tolist())[:100]
        for column in CATEGORICAL_FEATURES
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def train_and_publish(
    dataset_path: str | Path,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    seed: int = SEED,
    allow_failed_publish: bool = False,
    provenance: dict | None = None,
) -> dict:
    dataset_path = Path(dataset_path)
    models_dir = Path(models_dir)
    frame = load_dataset(dataset_path).drop_duplicates().reset_index(drop=True)
    collection_year = datetime.now(timezone.utc).year
    frame = _with_derived_features(frame, collection_year)
    train, validation, test = split_dataset(frame, seed=seed)

    preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    X_train = preprocessor.fit_transform(train[FEATURE_COLS])
    X_validation = preprocessor.transform(validation[FEATURE_COLS])
    X_test = preprocessor.transform(test[FEATURE_COLS])
    y_train = train[TARGET_COL].astype(float).to_numpy()
    y_validation = validation[TARGET_COL].astype(float).to_numpy()
    y_test = test[TARGET_COL].astype(float).to_numpy()
    target_mean = float(y_train.mean())
    target_std = float(y_train.std() or 1.0)
    scaled_train = (y_train - target_mean) / target_std
    scaled_validation = (y_validation - target_mean) / target_std
    model, best_val_scaled_rmse = _fit_mlp(
        X_train,
        scaled_train,
        X_validation,
        scaled_validation,
        seed,
    )
    prediction_scaled = _predict_scaled(model, X_test)
    prediction = prediction_scaled * target_std + target_mean
    baseline_prediction = np.full_like(y_test, target_mean)
    metrics = calculate_metrics(y_test, prediction, baseline_prediction)
    gate = assess_quality_gate(metrics)

    model_version = f"mlp-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    source = {
        "source_id": "normalized-training-csv",
        "source_url": SOURCE_URL,
        "raw_path": str(dataset_path),
    }
    if provenance:
        source.update(provenance)
    metrics_artifact = {
        "artifact_version": "2.0.0",
        "model_version": model_version,
        "best_val_rmse": float(best_val_scaled_rmse * target_std),
        "test_metrics": metrics,
        "quality_gate": gate["quality_gate"],
        "warnings": gate["warnings"],
        "thresholds": gate["thresholds"],
        "data_source": source,
        "currency": "INR",
        "price_unit": "INR",
        "mileage_unit": "km",
        "seed": seed,
        "split": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "split_indices": {
            "train": [int(index) for index in train.index],
            "validation": [int(index) for index in validation.index],
            "test": [int(index) for index in test.index],
        },
        "sample_count": len(frame),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    feature_config = {
        "artifact_version": "2.0.0",
        "target_col": TARGET_COL,
        "feature_cols": FEATURE_COLS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "currency": "INR",
        "price_unit": "INR",
        "mileage_unit": "km",
        "model_version": model_version,
    }
    model_card = {
        **metrics_artifact,
        "category_options": _category_options(frame),
        "feature_descriptions": {
            "mileage": "driven distance in kilometers",
            "displacement": "engine displacement in liters",
            "car_age": "collection year minus registration year",
            "accident_history": "unknown because the public source does not provide it",
        },
        "limitations": [
            "The source represents the Indian used-car market.",
            "Prices are INR, not converted to CNY.",
            "Accident history is unknown for every source row.",
        ],
    }

    temporary_dir = Path(tempfile.mkdtemp(prefix="car-valuation-model-", dir=models_dir.parent))
    try:
        joblib.dump(preprocessor, temporary_dir / "preprocess.joblib")
        torch.save(
            {
                "input_dim": X_train.shape[1],
                "hidden_dims": [128, 64],
                "dropout": 0.0,
                "model_state": model.state_dict(),
                "artifact_version": "2.0.0",
                "target_mean": target_mean,
                "target_std": target_std,
                "model_version": model_version,
            },
            temporary_dir / "price_mlp.pt",
        )
        _write_json(temporary_dir / "feature_config.json", feature_config)
        _write_json(temporary_dir / "metrics.json", metrics_artifact)
        _write_json(temporary_dir / "model_card.json", model_card)
        published = publish_artifacts(
            temporary_dir,
            models_dir,
            gate["quality_gate"],
            allow_failed_publish=allow_failed_publish,
        )
        if not published:
            return {**metrics_artifact, "published": False}
        temporary_dir = None
        return {**metrics_artifact, "published": True}
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def _prepare_downloaded_dataset() -> tuple[Path, dict]:
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--allow-failed-publish", action="store_true")
    args = parser.parse_args()

    provenance = None
    dataset = args.dataset
    if args.download:
        dataset, provenance = _prepare_downloaded_dataset()
    result = train_and_publish(
        dataset,
        models_dir=args.models_dir,
        seed=args.seed,
        allow_failed_publish=args.allow_failed_publish,
        provenance=provenance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
