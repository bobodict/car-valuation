import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from config import settings
from schemas import HistoryQuery, PredictRequest
from services.metrics_service import load_metrics
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


class RequestContractTests(unittest.TestCase):
    def test_valid_request_keeps_public_units_and_values(self):
        request = make_request()

        self.assertEqual(request.mileage, 87150)
        self.assertEqual(request.model, "Amaze")
        self.assertEqual(request.fuel_type, "Petrol")

    def test_request_rejects_impossible_vehicle_values(self):
        with self.assertRaises(ValidationError):
            PredictRequest(**{**make_request().model_dump(), "mileage": -1})

        with self.assertRaises(ValidationError):
            PredictRequest(**{**make_request().model_dump(), "month": 13})

    def test_history_query_rejects_unbounded_limits(self):
        self.assertEqual(HistoryQuery().limit, 20)
        with self.assertRaises(ValidationError):
            HistoryQuery(limit=201)


class ModelAdapterTests(unittest.TestCase):
    def test_adapter_uses_source_categories_and_kilometers(self):
        model_input = build_model_input(make_request())

        self.assertEqual(model_input["mileage"], 87150)
        self.assertEqual(model_input["transmission"], "Manual")
        self.assertEqual(model_input["fuel_type"], "Petrol")
        self.assertEqual(model_input["displacement"], 1.198)
        self.assertEqual(model_input["vehicle_type"], "car")
        self.assertEqual(model_input["accident_history"], "unknown")

    @patch("services.model_service.load_metrics")
    @patch("services.model_service.predict_price_one", return_value=55742.34)
    def test_response_reports_experimental_status_without_fake_confidence(
        self, _predict, load_metrics_mock
    ):
        load_metrics_mock.return_value = {
            "quality_gate": "fail",
            "test_metrics": {
                "mse": 680398226160.8066,
                "rmse": 824862.5498595549,
                "mae": 403990.84203074436,
                "r2": 0.8635845112741601,
                "acc_10": 0.2912621359223301,
                "baseline_rmse": 2234030.4642809303,
                "baseline_r2": -0.000641919560930404,
            },
            "currency": "INR",
            "price_unit": "INR",
            "mileage_unit": "km",
            "model_version": "mlp-test",
        }

        result = call_model_api(make_request())

        self.assertEqual(result["price"], 55742.34)
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["model_status"], "experimental")
        self.assertEqual(result["quality_gate"], "fail")
        self.assertEqual(result["currency"], "INR")
        self.assertIn("R2=0.864", result["comment"])

    @patch("services.model_service.predict_price_one", side_effect=RuntimeError("broken"))
    def test_model_failure_is_not_hidden_by_a_rule_fallback(self, _predict):
        with self.assertRaises(ModelServiceError):
            call_model_api(make_request())


class MetricsTests(unittest.TestCase):
    def test_artifact_metrics_are_loaded_from_backend_model_directory(self):
        artifact = json.loads(settings.metrics_path.read_text(encoding="utf-8"))
        feature_config = json.loads(
            settings.feature_config_path.read_text(encoding="utf-8")
        )
        metrics = load_metrics()
        test_metrics = metrics["test_metrics"]

        self.assertEqual(artifact["artifact_version"], "3.0.0")
        self.assertEqual(artifact["model_type"], "catboost")
        self.assertRegex(artifact["model_version"], r"^v3-.+")
        self.assertEqual(feature_config["feature_version"], "3.0.0")
        self.assertEqual(metrics["feature_version"], feature_config["feature_version"])
        self.assertEqual(metrics["model_type"], artifact["model_type"])
        self.assertEqual(metrics["model_version"], artifact["model_version"])
        self.assertEqual(metrics["quality_gate"], "pass")
        self.assertGreater(test_metrics["r2"], 0)
        self.assertGreaterEqual(test_metrics["acc_10"], 0.50)
        self.assertLess(test_metrics["rmse"], test_metrics["baseline_rmse"])
        self.assertEqual(metrics["currency"], "INR")
        self.assertEqual(metrics["mileage_unit"], "km")


if __name__ == "__main__":
    unittest.main()
