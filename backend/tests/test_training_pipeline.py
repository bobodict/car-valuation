import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_model import (
    assess_quality_gate,
    publish_artifacts,
    split_dataset,
)
from services.model_competition import calculate_metrics


def make_fixture_frame(size=30):
    return pd.DataFrame(
        {
            "price": np.arange(size, dtype=float) * 1000 + 100000,
            "mileage": np.arange(size, dtype=float) * 1000,
            "displacement": np.full(size, 1.2),
            "seats": np.full(size, 5),
            "owner_count": np.ones(size),
            "year": np.arange(size) % 8 + 2015,
            "brand": ["Honda" if index % 2 else "Toyota" for index in range(size)],
            "model": [f"model-{index % 3}" for index in range(size)],
            "city": ["Pune" if index % 2 else "Delhi" for index in range(size)],
            "transmission": ["Manual"] * size,
            "fuel_type": ["Petrol"] * size,
            "vehicle_type": ["car"] * size,
            "color": ["Grey"] * size,
            "accident_history": ["unknown"] * size,
        }
    )


class TrainingPipelineTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        frame = make_fixture_frame()

        first = split_dataset(frame, seed=42)
        second = split_dataset(frame, seed=42)

        self.assertEqual(
            [part.index.tolist() for part in first],
            [part.index.tolist() for part in second],
        )
        self.assertTrue(set(first[0].index).isdisjoint(first[1].index))
        self.assertTrue(set(first[1].index).isdisjoint(first[2].index))
        self.assertTrue(set(first[0].index).isdisjoint(first[2].index))

    def test_metrics_include_training_mean_baseline(self):
        metrics = calculate_metrics(
            np.array([100.0, 200.0, 300.0]),
            np.array([110.0, 190.0, 310.0]),
            np.array([200.0, 200.0, 200.0]),
        )

        self.assertTrue(
            {
                "rmse",
                "mae",
                "r2",
                "acc_10",
                "baseline_rmse",
                "baseline_r2",
            }.issubset(metrics)
        )
        self.assertGreater(metrics["baseline_rmse"], 0)

    def test_quality_gate_requires_model_to_beat_baseline(self):
        failing = assess_quality_gate(
            {"r2": -0.01, "acc_10": 0.8, "rmse": 20.0, "baseline_rmse": 10.0},
            {"min_r2": 0.0, "min_acc_10": 0.5},
        )
        passing = assess_quality_gate(
            {"r2": 0.8, "acc_10": 0.8, "rmse": 5.0, "baseline_rmse": 10.0},
            {"min_r2": 0.0, "min_acc_10": 0.5},
        )

        self.assertEqual(failing["quality_gate"], "fail")
        self.assertEqual(passing["quality_gate"], "pass")

    def test_failed_gate_does_not_replace_existing_artifacts_without_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "new"
            target = root / "models"
            source.mkdir()
            target.mkdir()
            (source / "metrics.json").write_text("new", encoding="utf-8")
            marker = target / "marker.txt"
            marker.write_text("old", encoding="utf-8")

            published = publish_artifacts(source, target, "fail", allow_failed_publish=False)

            self.assertFalse(published)
            self.assertEqual(marker.read_text(encoding="utf-8"), "old")


if __name__ == "__main__":
    unittest.main()
