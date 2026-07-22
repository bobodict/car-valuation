import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from main import app
from schemas import PredictRequest, PredictResponse
from services.model_runtime import ModelRuntimeError
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


NEW_TECHNICAL_VALUES = {
    "seller_type": "Dealer",
    "drivetrain": "FWD",
    "max_power_bhp": 87.0,
    "power_rpm": 6000.0,
    "max_torque_nm": 109.0,
    "torque_rpm": 4500.0,
    "length_mm": 3995.0,
    "width_mm": 1695.0,
    "height_mm": 1501.0,
    "fuel_tank_liter": 35.0,
}


NUMERIC_BOUNDARIES = {
    "max_power_bhp": ((1e-9, 2000.0), (0.0, 2000.1)),
    "power_rpm": ((1e-9, 25000.0), (0.0, 25000.1)),
    "max_torque_nm": ((1e-9, 10000.0), (0.0, 10000.1)),
    "torque_rpm": ((1e-9, 25000.0), (0.0, 25000.1)),
    "length_mm": ((1000.0, 10000.0), (999.9, 10000.1)),
    "width_mm": ((1000.0, 5000.0), (999.9, 5000.1)),
    "height_mm": ((500.0, 5000.0), (499.9, 5000.1)),
    "fuel_tank_liter": ((1e-9, 1000.0), (0.0, 1000.1)),
}


client = TestClient(app)


class PredictionContractTests(unittest.TestCase):
    def test_adapter_passes_complete_v3_features_without_unit_conversion(self):
        request = PredictRequest(
            **{
                **make_request().model_dump(),
                **NEW_TECHNICAL_VALUES,
                "seller_type": "  Dealer  ",
                "drivetrain": "  FWD  ",
            }
        )

        self.assertEqual(
            build_model_input(request),
            {
                "brand": "Honda",
                "model": "Amaze",
                "year": 2017,
                "mileage": 87150.0,
                "city": "Pune",
                "transmission": "Manual",
                "fuel_type": "Petrol",
                "displacement": 1.198,
                "vehicle_type": "car",
                "color": "Grey",
                "seats": 5,
                "accident_history": "unknown",
                "owner_count": 1,
                **NEW_TECHNICAL_VALUES,
            },
        )

    def test_optional_numeric_features_accept_boundaries_and_none(self):
        base = make_request().model_dump()
        for field, (accepted, _rejected) in NUMERIC_BOUNDARIES.items():
            for value in (*accepted, None):
                with self.subTest(field=field, value=value):
                    result = PredictRequest(**{**base, field: value}).model_dump()
                    self.assertIn(field, result)
                    self.assertEqual(result[field], value)

    def test_optional_numeric_features_reject_values_outside_boundaries(self):
        base = make_request().model_dump()
        for field, (_accepted, rejected) in NUMERIC_BOUNDARIES.items():
            for value in rejected:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValidationError):
                        PredictRequest(**{**base, field: value})

    def test_optional_strings_are_trimmed_and_reject_blank_values(self):
        base = make_request().model_dump()
        request = PredictRequest(
            **{**base, "seller_type": "  Individual  ", "drivetrain": "  AWD  "}
        )
        result = request.model_dump()
        self.assertEqual(result.get("seller_type"), "Individual")
        self.assertEqual(result.get("drivetrain"), "AWD")

        for field in ("seller_type", "drivetrain"):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    PredictRequest(**{**base, field: "   "})

    def test_legacy_request_keeps_new_features_optional(self):
        result = make_request().model_dump()

        for field in NEW_TECHNICAL_VALUES:
            with self.subTest(field=field):
                self.assertIn(field, result)
                self.assertIsNone(result[field])

    def test_api_returns_422_for_blank_optional_string(self):
        payload = {**make_request().model_dump(), "seller_type": "   "}
        with patch(
            "main.call_model_api", side_effect=ModelServiceError("unavailable")
        ) as call_model_mock:
            response = client.post("/api/predict", json=payload)

        self.assertEqual(response.status_code, 422)
        call_model_mock.assert_not_called()

    def test_api_returns_503_for_model_service_error(self):
        with patch(
            "main.call_model_api", side_effect=ModelServiceError("model unavailable")
        ):
            response = client.post("/api/predict", json=make_request().model_dump())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "model unavailable"})
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
        self.assertEqual(result["model_type"], "mlp")
        self.assertEqual(result["feature_version"], "2.0.0")
        response = PredictResponse.model_validate(result).model_dump()
        self.assertEqual(response["model_type"], "mlp")
        self.assertEqual(response["feature_version"], "2.0.0")

    @patch(
        "services.model_service.predict_price_one",
        side_effect=ModelRuntimeError("failed at D:\\private\\models\\secret.pt"),
    )
    def test_model_runtime_error_is_sanitized_as_model_service_error(self, _predict):
        with self.assertRaises(ModelServiceError) as raised:
            call_model_api(make_request())

        self.assertNotIn("private", str(raised.exception))
        self.assertNotIn("secret.pt", str(raised.exception))

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
