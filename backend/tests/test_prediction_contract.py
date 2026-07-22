import sys
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from schemas import PredictRequest
from services.model_service import ModelServiceError, build_model_input, call_model_api


def make_request():
    return PredictRequest(
        brand="Honda",
        model="Amaze",
        city="Pune",
        mileage=87150,
        year=2017,
        month=6,
        gearbox="Manual",
        emission="unknown",
        fuel_type="Petrol",
        displacement=1.198,
        seats=5,
        owner_count=1,
        vehicle_type="car",
        color="Grey",
        accident_history="unknown",
    )


class PredictionContractTests(unittest.TestCase):
    def test_adapter_passes_full_source_features_and_km(self):
        model_input = build_model_input(make_request())

        self.assertEqual(model_input["mileage"], 87150.0)
        self.assertEqual(model_input["transmission"], "Manual")
        self.assertEqual(model_input["fuel_type"], "Petrol")
        self.assertEqual(model_input["displacement"], 1.198)
        self.assertEqual(model_input["seats"], 5)
        self.assertEqual(model_input["owner_count"], 1)
        self.assertEqual(model_input["vehicle_type"], "car")
        self.assertEqual(model_input["color"], "Grey")
        self.assertEqual(model_input["accident_history"], "unknown")

    def test_request_rejects_negative_seats(self):
        with self.assertRaises(ValidationError):
            PredictRequest(**{**make_request().model_dump(), "seats": -1})

    @patch("services.model_service.load_metrics")
    @patch("services.model_service.predict_price_one", return_value=505000.0)
    def test_response_keeps_raw_inr_price_and_gate_metadata(self, _predict, load_metrics_mock):
        load_metrics_mock.return_value = {
            "quality_gate": "fail",
            "model_status": "experimental",
            "currency": "INR",
            "price_unit": "INR",
            "mileage_unit": "km",
            "model_version": "mlp-test",
            "test_metrics": {
                "mse": 1.0,
                "rmse": 1.0,
                "mae": 1.0,
                "r2": 0.1,
                "acc_10": 0.2,
            },
        }

        result = call_model_api(make_request())

        self.assertEqual(result["price"], 505000.0)
        self.assertEqual(result["currency"], "INR")
        self.assertEqual(result["price_unit"], "INR")
        self.assertEqual(result["quality_gate"], "fail")
        self.assertEqual(result["model_status"], "experimental")
        self.assertEqual(result["range"], {"low": 464600.0, "high": 545400.0})

    @patch("services.model_service.predict_price_one", side_effect=RuntimeError("broken"))
    def test_model_failure_is_not_hidden_by_a_rule_fallback(self, _predict):
        with self.assertRaises(ModelServiceError):
            call_model_api(make_request())

    @patch("services.model_service.load_metrics")
    @patch(
        "services.model_service.predict_price_one",
        return_value=sys.float_info.max,
    )
    def test_response_rejects_price_when_reference_range_would_overflow(
        self, _predict, load_metrics_mock
    ):
        load_metrics_mock.return_value = {
            "quality_gate": "fail",
            "test_metrics": {"r2": 0.0},
        }

        with self.assertRaisesRegex(ModelServiceError, "range|finite|invalid"):
            call_model_api(make_request())


if __name__ == "__main__":
    unittest.main()
