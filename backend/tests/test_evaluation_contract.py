import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scripts import evaluate_model as evaluator


def write_v3_artifacts(models_dir, test_indices=(2, 3), baseline=150.0):
    manifest = {
        "artifact_version": "3.0.0",
        "model_version": "v3-fixture",
        "model_type": "extra_trees",
        "split_indices": {"development": [0, 1], "test": list(test_indices)},
    }
    metrics = {
        "artifact_version": "3.0.0",
        "model_version": "v3-fixture",
        "development_mean_baseline": baseline,
    }
    (models_dir / "model_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (models_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return manifest, metrics


class EvaluationContractTests(unittest.TestCase):
    def test_evaluator_uses_recorded_v3_test_and_development_mean_baseline(self):
        frame = pd.DataFrame(
            {
                "price": [100.0, 200.0, 300.0, 400.0],
                "model": ["A", "B", "C", "D"],
            }
        )

        class Runtime:
            def __init__(self):
                self.indices = None

            def predict(self, prediction_frame):
                self.indices = prediction_frame.index.tolist()
                return np.array([310.0, 390.0])

        runtime = Runtime()
        loader = Mock(return_value=runtime)
        with tempfile.TemporaryDirectory() as directory:
            models_dir = Path(directory)
            write_v3_artifacts(models_dir, baseline=123.0)
            with patch.object(evaluator, "load_dataset", return_value=frame):
                result = evaluator.evaluate_model(
                    "fixture.csv",
                    models_dir=models_dir,
                    runtime_loader=loader,
                )

        self.assertEqual(runtime.indices, [2, 3])
        loader.assert_called_once_with(models_dir)
        self.assertEqual(result["evaluation_scope"], "recorded_test")
        self.assertEqual(result["model_version"], "v3-fixture")
        self.assertEqual(result["model_type"], "extra_trees")
        self.assertEqual(result["count"], 2)
        expected_baseline_rmse = float(np.sqrt(np.mean((np.array([300.0, 400.0]) - 123.0) ** 2)))
        self.assertAlmostEqual(result["metrics"]["baseline_rmse"], expected_baseline_rmse)

    def test_v3_manifest_requires_nonempty_recorded_test_indices(self):
        frame = pd.DataFrame({"price": [100.0, 200.0], "model": ["A", "B"]})
        with tempfile.TemporaryDirectory() as directory:
            models_dir = Path(directory)
            write_v3_artifacts(models_dir, test_indices=())
            loader = Mock()
            with patch.object(evaluator, "load_dataset", return_value=frame):
                with self.assertRaisesRegex(ValueError, "split_indices.test"):
                    evaluator.evaluate_model(
                        "fixture.csv", models_dir=models_dir, runtime_loader=loader
                    )
            loader.assert_not_called()

    def test_dataset_validation_happens_before_runtime_import_or_load(self):
        loader = Mock(side_effect=AssertionError("runtime must not load"))
        with patch.object(evaluator, "load_dataset", side_effect=ValueError("bad dataset")):
            with self.assertRaisesRegex(ValueError, "bad dataset"):
                evaluator.evaluate_model(
                    "bad.csv", models_dir=Path("unused"), runtime_loader=loader
                )
        loader.assert_not_called()

    def test_runtime_loader_seam_imports_task7_lazily(self):
        source = Path(evaluator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from services.model_runtime import ModelRuntime\n", source.split("def load_runtime", 1)[0])

    def test_legacy_helpers_remain_compatible_without_weakening_v3(self):
        frame = pd.DataFrame({"price": [100.0, 200.0, 300.0, 400.0]})
        legacy = {"artifact_version": "2.0.0", "split_indices": {"train": [0, 1], "test": [2, 3]}}

        selected = evaluator.select_test_frame(frame, legacy)
        baseline = evaluator.baseline_for_test(frame, legacy)

        self.assertEqual(selected.index.tolist(), [2, 3])
        self.assertEqual(baseline.tolist(), [150.0, 150.0])


if __name__ == "__main__":
    unittest.main()
