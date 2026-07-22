import unittest

import pandas as pd

from services.dataset_contract import REQUIRED_DATASET_COLUMNS, validate_dataset_columns
from services.model_quality_service import assess_metrics


class ModelQualityTests(unittest.TestCase):
    def test_negative_r2_and_low_accuracy_fail_quality_gate(self):
        assessment = assess_metrics(
            {
                "mse": 177.17,
                "rmse": 13.31,
                "mae": 11.5,
                "r2": -0.015,
                "acc_10": 0.117,
            }
        )

        self.assertEqual(assessment["quality_gate"], "fail")
        self.assertTrue(any("R²" in warning for warning in assessment["warnings"]))
        self.assertTrue(any("10%" in warning for warning in assessment["warnings"]))


class DatasetContractTests(unittest.TestCase):
    def test_dataset_contract_reports_missing_columns(self):
        frame = pd.DataFrame({"price": [10000], "brand": ["大众"]})

        missing = validate_dataset_columns(frame.columns)

        self.assertIn("mileage", missing)
        self.assertIn("city", missing)
        self.assertEqual(len(REQUIRED_DATASET_COLUMNS), 14)


if __name__ == "__main__":
    unittest.main()
