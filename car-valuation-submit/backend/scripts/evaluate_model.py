"""Evaluate the supplied artifact against a reproducible CSV dataset.

Usage from backend:
    python -m scripts.evaluate_model path/to/used_cars.csv
"""

import argparse
import json

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from services.dataset_contract import load_dataset


def evaluate_model(dataset_path: str) -> dict:
    frame = load_dataset(dataset_path)
    from predict_service import predict_price_one

    actual = frame["price"].astype(float).to_numpy()
    predictions = np.array(
        [predict_price_one(row.drop(labels=["price"]).to_dict()) for _, row in frame.iterrows()],
        dtype=float,
    )
    mean_baseline = np.full_like(actual, actual.mean())
    metrics = {
        "count": int(len(actual)),
        "mse": float(mean_squared_error(actual, predictions)),
        "rmse": float(mean_squared_error(actual, predictions) ** 0.5),
        "mae": float(mean_absolute_error(actual, predictions)),
        "r2": float(r2_score(actual, predictions)),
        "acc_10": float(np.mean(np.abs(predictions - actual) / np.maximum(np.abs(actual), 1e-8) <= 0.1)),
        "baseline_r2": float(r2_score(actual, mean_baseline)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the supplied car price model.")
    parser.add_argument("dataset", help="CSV dataset containing the required feature columns")
    args = parser.parse_args()
    print(json.dumps(evaluate_model(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
