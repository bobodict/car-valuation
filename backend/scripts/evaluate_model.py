"""Evaluate the supplied artifact against a reproducible CSV dataset.

Usage from backend:
    python -m scripts.evaluate_model path/to/used_cars.csv
"""

import argparse
import json

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import settings
from services.dataset_contract import load_dataset


def select_test_frame(frame, artifact):
    split_indices = artifact.get("split_indices", {})
    test_indices = split_indices.get("test")
    if not test_indices:
        return frame
    return frame.loc[test_indices]


def baseline_for_test(frame, artifact):
    split_indices = artifact.get("split_indices", {})
    train_indices = split_indices.get("train")
    if not train_indices:
        train_mean = frame["price"].astype(float).mean()
    else:
        train_mean = frame.loc[train_indices, "price"].astype(float).mean()
    return np.full(len(select_test_frame(frame, artifact)), train_mean, dtype=float)


def load_metrics_artifact() -> dict:
    with settings.metrics_path.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def evaluate_model(dataset_path: str) -> dict:
    frame = load_dataset(dataset_path)
    artifact = load_metrics_artifact()
    evaluation_frame = select_test_frame(frame, artifact)
    from predict_service import predict_price_one

    actual = evaluation_frame["price"].astype(float).to_numpy()
    predictions = np.array(
        [
            predict_price_one(row.drop(labels=["price"]).to_dict())
            for _, row in evaluation_frame.iterrows()
        ],
        dtype=float,
    )
    mean_baseline = baseline_for_test(frame, artifact)
    metrics = {
        "count": int(len(actual)),
        "mse": float(mean_squared_error(actual, predictions)),
        "rmse": float(mean_squared_error(actual, predictions) ** 0.5),
        "mae": float(mean_absolute_error(actual, predictions)),
        "r2": float(r2_score(actual, predictions)),
        "acc_10": float(np.mean(np.abs(predictions - actual) / np.maximum(np.abs(actual), 1e-8) <= 0.1)),
        "baseline_rmse": float(mean_squared_error(actual, mean_baseline) ** 0.5),
        "baseline_r2": float(r2_score(actual, mean_baseline)),
        "evaluation_scope": "recorded_test" if artifact.get("split_indices") else "full_dataset",
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the supplied car price model.")
    parser.add_argument("dataset", help="CSV dataset containing the required feature columns")
    args = parser.parse_args()
    print(json.dumps(evaluate_model(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
