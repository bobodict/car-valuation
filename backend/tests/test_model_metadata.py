import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app, model_card
from schemas import MetricsResponse, ModelCardResponse, ModelHealthResponse
from services import metrics_service
from services.model_metadata import load_model_card
from services.model_quality_service import get_model_health


client = TestClient(app)


def make_card(split=None, **overrides):
    card = {
        "artifact_version": "2.0.0",
        "currency": "INR",
        "price_unit": "INR",
        "mileage_unit": "km",
        "sample_count": 10,
        "model_version": "mlp-test",
        "data_source": {"source_id": "car-details-v4"},
        "split": split or {"train": 7, "validation": 1, "test": 2},
        "thresholds": {"min_r2": 0.0, "min_acc_10": 0.5},
        "quality_gate": "fail",
        "test_metrics": {
            "mse": 1.0,
            "rmse": 1.0,
            "mae": 1.0,
            "r2": 0.1,
            "acc_10": 0.2,
        },
        "category_options": {},
    }
    card.update(overrides)
    return card


def make_metrics(**overrides):
    artifact = {
        "quality_gate": "fail",
        "test_metrics": {
            "mse": 1.0,
            "rmse": 1.0,
            "mae": 1.0,
            "r2": 0.1,
            "acc_10": 0.2,
        },
        "currency": "INR",
        "price_unit": "INR",
        "mileage_unit": "km",
        "data_source": {"source_id": "test"},
        "model_version": "test",
        "sample_count": 10,
    }
    artifact.update(overrides)
    return artifact


class ModelMetadataTests(unittest.TestCase):
    def test_api_response_models_return_v3_metadata_and_evidence(self):
        metrics = make_metrics(model_type="catboost", feature_version="3.0.0")
        health = {
            "model_status": "experimental",
            "quality_gate": "fail",
            "warnings": [],
            "metrics": metrics["test_metrics"],
            "data_status": "recorded",
            **{
                key: metrics[key]
                for key in (
                    "currency",
                    "price_unit",
                    "mileage_unit",
                    "data_source",
                    "model_version",
                    "model_type",
                    "feature_version",
                    "sample_count",
                )
            },
        }
        card = make_card(
            feature_version="3.0.0",
            model_type="catboost",
            leaderboard={"winner": "catboost"},
            error_analysis={"segment": "rare"},
        )

        with patch("main.load_metrics", return_value=metrics):
            metrics_response = client.get("/api/metrics")
        with patch("main.get_model_health", return_value=health):
            health_response = client.get("/api/model-health")
        with patch("main.load_model_card", return_value=card):
            card_response = client.get("/api/model-card")

        self.assertEqual(metrics_response.json()["model_type"], "catboost")
        self.assertEqual(health_response.json()["feature_version"], "3.0.0")
        self.assertEqual(card_response.json()["leaderboard"], {"winner": "catboost"})
        self.assertEqual(
            card_response.json()["error_analysis"], {"segment": "rare"}
        )

    def test_v2_model_card_gets_safe_evidence_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_card.json"
            path.write_text(json.dumps(make_card()), encoding="utf-8")

            result = load_model_card(path)

        self.assertEqual(result["feature_version"], "2.0.0")
        self.assertEqual(result["model_type"], "mlp")
        self.assertEqual(result["leaderboard"], {})
        self.assertEqual(result["error_analysis"], {})
        response = ModelCardResponse.model_validate(result).model_dump()
        self.assertEqual(response["model_type"], "mlp")

    def test_v3_model_card_accepts_split_and_loads_companion_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model_card.json"
            path.write_text(
                json.dumps(
                    make_card(
                        split={"development": 8, "test": 2, "folds": 5},
                        artifact_version="3.0.0",
                        feature_version="3.0.0",
                        model_type="catboost",
                    )
                ),
                encoding="utf-8",
            )
            (root / "leaderboard.json").write_text(
                json.dumps({"winner": "catboost"}), encoding="utf-8"
            )
            (root / "error_analysis.json").write_text(
                json.dumps({"worst_segment": "rare"}), encoding="utf-8"
            )

            result = load_model_card(path)

        self.assertEqual(result["model_type"], "catboost")
        self.assertEqual(result["leaderboard"], {"winner": "catboost"})
        self.assertEqual(result["error_analysis"], {"worst_segment": "rare"})

    def test_model_card_embedded_evidence_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model_card.json"
            path.write_text(
                json.dumps(
                    make_card(
                        leaderboard={"source": "embedded"},
                        error_analysis={"source": "embedded"},
                    )
                ),
                encoding="utf-8",
            )
            (root / "leaderboard.json").write_text(
                json.dumps({"source": "companion"}), encoding="utf-8"
            )
            (root / "error_analysis.json").write_text(
                json.dumps({"source": "companion"}), encoding="utf-8"
            )

            result = load_model_card(path)

        self.assertEqual(result["leaderboard"], {"source": "embedded"})
        self.assertEqual(result["error_analysis"], {"source": "embedded"})

    def test_malformed_companion_evidence_fails_explicitly(self):
        for filename in ("leaderboard.json", "error_analysis.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "model_card.json"
                path.write_text(json.dumps(make_card()), encoding="utf-8")
                (root / filename).write_text("{not-json", encoding="utf-8")

                with self.assertRaisesRegex(ValueError, filename):
                    load_model_card(path)

    def test_metrics_loads_v3_metadata_and_defaults_for_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            for artifact, expected in (
                (make_metrics(), ("mlp", "2.0.0")),
                (
                    make_metrics(model_type="extra_trees", feature_version="3.0.0"),
                    ("extra_trees", "3.0.0"),
                ),
            ):
                with self.subTest(expected=expected):
                    path.write_text(json.dumps(artifact), encoding="utf-8")
                    metrics_service.load_metrics.cache_clear()
                    with patch.object(
                        metrics_service,
                        "settings",
                        SimpleNamespace(metrics_path=path),
                    ):
                        result = metrics_service.load_metrics()
                    self.assertEqual(
                        (result["model_type"], result["feature_version"]), expected
                    )
                    response = MetricsResponse.model_validate(result).model_dump()
                    self.assertEqual(response["model_type"], expected[0])
            metrics_service.load_metrics.cache_clear()

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
            "model_type": "catboost",
            "feature_version": "3.0.0",
            "sample_count": 10,
        }

        result = get_model_health()

        self.assertEqual(result["quality_gate"], "pass")
        self.assertEqual(result["model_type"], "catboost")
        self.assertEqual(result["feature_version"], "3.0.0")
        response = ModelHealthResponse.model_validate(result).model_dump()
        self.assertEqual(response["model_type"], "catboost")


if __name__ == "__main__":
    unittest.main()
