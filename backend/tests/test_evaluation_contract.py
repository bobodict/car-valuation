import unittest

import pandas as pd

from scripts.evaluate_model import baseline_for_test, select_test_frame


class EvaluationContractTests(unittest.TestCase):
    def test_evaluator_selects_recorded_test_indexes(self):
        frame = pd.DataFrame({"price": [100.0, 200.0, 300.0, 400.0]})
        artifact = {"split_indices": {"train": [0, 1], "test": [2, 3]}}

        selected = select_test_frame(frame, artifact)

        self.assertEqual(selected.index.tolist(), [2, 3])

    def test_baseline_uses_training_mean_not_full_dataset_mean(self):
        frame = pd.DataFrame({"price": [100.0, 200.0, 300.0, 400.0]})
        artifact = {"split_indices": {"train": [0, 1], "test": [2, 3]}}

        baseline = baseline_for_test(frame, artifact)

        self.assertEqual(baseline.tolist(), [150.0, 150.0])


if __name__ == "__main__":
    unittest.main()
