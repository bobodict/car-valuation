import unittest
from unittest.mock import patch

from pydantic import ValidationError

from schemas import HistoryQuery, PredictRequest
from services.metrics_service import load_metrics
from services.model_service import ModelServiceError, build_model_input, call_model_api


class RequestContractTests(unittest.TestCase):
    def test_valid_request_keeps_public_units_and_values(self):
        request = PredictRequest(
            brand="大众",
            model="帕萨特",
            city="广州",
            mileage=6.5,
            year=2018,
            month=6,
            gearbox="自动",
            emission="国六",
        )

        self.assertEqual(request.mileage, 6.5)
        self.assertEqual(request.model, "帕萨特")

    def test_request_rejects_impossible_vehicle_values(self):
        with self.assertRaises(ValidationError):
            PredictRequest(
                brand="大众",
                model="帕萨特",
                city="广州",
                mileage=-1,
                year=2018,
                month=6,
                gearbox="自动",
                emission="国六",
            )

        with self.assertRaises(ValidationError):
            PredictRequest(
                brand="大众",
                model="帕萨特",
                city="广州",
                mileage=6.5,
                year=2018,
                month=13,
                gearbox="自动",
                emission="国六",
            )

    def test_history_query_rejects_unbounded_limits(self):
        self.assertEqual(HistoryQuery().limit, 20)
        with self.assertRaises(ValidationError):
            HistoryQuery(limit=201)


class ModelAdapterTests(unittest.TestCase):
    def setUp(self):
        self.request = PredictRequest(
            brand="大众",
            model="帕萨特",
            city="广州",
            mileage=6.5,
            year=2018,
            month=6,
            gearbox="自动",
            emission="国六",
        )

    def test_adapter_uses_fitted_chinese_categories_and_kilometers(self):
        model_input = build_model_input(self.request)

        self.assertEqual(model_input["mileage"], 65000)
        self.assertEqual(model_input["transmission"], "自动")
        self.assertEqual(model_input["fuel_type"], "汽油")
        self.assertEqual(model_input["vehicle_type"], "轿车")
        self.assertEqual(model_input["color"], "白色")
        self.assertEqual(model_input["accident_history"], "无事故")

    @patch("services.model_service.load_metrics")
    @patch("services.model_service.predict_price_one", return_value=55742.34)
    def test_response_reports_experimental_status_without_fake_confidence(
        self, _predict, load_metrics_mock
    ):
        load_metrics_mock.return_value = {
            "best_val_rmse": 13.177239209284734,
            "test_metrics": {
                "mse": 177.17701721191406,
                "rmse": 13.310785747352185,
                "mae": 11.497577667236328,
                "r2": -0.015273571014404297,
                "acc_10": 0.11777777777777777,
            },
        }

        result = call_model_api(self.request)

        self.assertEqual(result["price"], 5.57)
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["model_status"], "experimental")
        self.assertEqual(result["metrics"]["r2"], -0.015273571014404297)
        self.assertIn("参考", result["comment"])

    @patch("services.model_service.predict_price_one", side_effect=RuntimeError("broken"))
    def test_model_failure_is_not_hidden_by_a_rule_fallback(self, _predict):
        with self.assertRaises(ModelServiceError):
            call_model_api(self.request)


class MetricsTests(unittest.TestCase):
    def test_artifact_metrics_are_loaded_from_backend_model_directory(self):
        metrics = load_metrics()

        self.assertEqual(metrics["test_metrics"]["r2"], -0.015273571014404297)
        self.assertEqual(metrics["test_metrics"]["acc_10"], 0.11777777777777777)


if __name__ == "__main__":
    unittest.main()
