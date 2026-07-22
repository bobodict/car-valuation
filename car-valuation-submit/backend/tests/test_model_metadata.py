import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import model_card
from services.model_metadata import load_model_card
from services.model_quality_service import get_model_health


class ModelMetadataTests(unittest.TestCase):
    def test_model_card_requires_currency_and_positive_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_card.json"
            valid = {
                "source_id": "car-details-v4",
                "source_url": "https://example.test/cars.csv",
                "currency": "INR",
                "price_unit": "INR",
                "mileage_unit": "km",
                "sample_count": 10,
                "feature_version": "2.0.0",
                "model_version": "mlp-test",
                "split": {"train": 7, "validation": 1, "test": 2},
                "thresholds": {"min_r2": 0.0, "min_acc_10": 0.5},
                "category_options": {},
            }
            path.write_text(json.dumps(valid), encoding="utf-8")

            self.assertEqual(load_model_card(path)["currency"], "INR")
            invalid = {**valid, "currency": ""}
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model_card(path)

            invalid = {**valid, "sample_count": 0}
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model_card(path)

    def test_api_model_card_returns_source_and_model_facts(self):
        result = model_card()

        self.assertEqual(result["currency"], "INR")
        self.assertEqual(result["mileage_unit"], "km")
        self.assertEqual(result["data_source"]["source_id"], "car-details-v4")
        self.assertIn("category_options", result)

    @patch("services.model_quality_service.load_metrics")
    def test_health_uses_stored_quality_gate(self, load_metrics_mock):
        load_metrics_mock.return_value = {
            "quality_gate": "pass",
            "warnings": [],
            "test_metrics": {
                "mse": 1.0,
                "rmse": 1.0,
                "mae": 1.0,
                "r2": -0.2,
                "acc_10": 0.1,
            },
            "currency": "INR",
            "price_unit": "INR",
            "mileage_unit": "km",
            "data_source": {"source_id": "test"},
            "model_version": "test",
            "sample_count": 10,
        }

        result = get_model_health()

        self.assertEqual(result["quality_gate"], "pass")


if __name__ == "__main__":
    unittest.main()
