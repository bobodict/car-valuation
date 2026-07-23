import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from main import get_metrics, model_card, model_health
from schemas import (
    MetricsResponse,
    ModelCardResponse,
    ModelHealthResponse,
    PredictRequest,
    PredictResponse,
)
from services import metrics_service
from services.model_metadata import load_model_card
from services.model_quality_service import get_model_health
from services.model_runtime import ModelRuntimeError
from services.model_service import call_model_api


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
            metrics_response = MetricsResponse.model_validate(
                get_metrics()
            ).model_dump()
        with patch("main.get_model_health", return_value=health):
            health_response = ModelHealthResponse.model_validate(
                model_health()
            ).model_dump()
        with patch("main.load_model_card", return_value=card):
            card_response = ModelCardResponse.model_validate(model_card()).model_dump()

        self.assertEqual(metrics_response["model_type"], "catboost")
        self.assertEqual(health_response["feature_version"], "3.0.0")
        self.assertEqual(card_response["leaderboard"], {"winner": "catboost"})
        self.assertEqual(
            card_response["error_analysis"], {"segment": "rare"}
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

    def test_v3_model_card_accepts_current_train_validation_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_card.json"
            path.write_text(
                json.dumps(
                    make_card(
                        split={
                            "train": 7,
                            "validation": 1,
                            "development": 8,
                            "test": 2,
                        },
                        artifact_version="3.0.0",
                        feature_version="3.0.0",
                    )
                ),
                encoding="utf-8",
            )

            result = load_model_card(path)

        self.assertEqual(result["split"]["development"], 8)

    def test_model_card_rejects_v3_split_without_folds_or_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_card.json"
            path.write_text(
                json.dumps(
                    make_card(
                        split={"development": 8, "test": 2},
                        artifact_version="3.0.0",
                        feature_version="3.0.0",
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "split"):
                load_model_card(path)

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
                    make_metrics(
                        artifact_version="3.0.0",
                        feature_version="3.1.0",
                        model_type="extra_trees",
                    ),
                    ("extra_trees", "3.1.0"),
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

    def test_metrics_refresh_after_atomic_replacement_without_cache_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "metrics.json"
            replacement = root / "metrics.next.json"
            old_identity = ("publication", "old")
            new_identity = ("publication", "new")
            old_artifact = make_metrics(
                artifact_version="3.0.0",
                model_version="old-v3",
                model_type="old_type",
                quality_gate="fail",
            )
            new_artifact = make_metrics(
                artifact_version="3.1.0",
                model_version="new-v3",
                model_type="new_type",
                quality_gate="pass",
            )
            path.write_text(
                json.dumps(old_artifact, sort_keys=True), encoding="utf-8"
            )
            old_stat = path.stat()
            replacement.write_text(
                json.dumps(new_artifact, sort_keys=True), encoding="utf-8"
            )
            self.assertEqual(path.stat().st_size, replacement.stat().st_size)
            os.utime(
                replacement,
                ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns),
            )

            metrics_service.load_metrics.cache_clear()
            try:
                with (
                    patch.object(
                        metrics_service,
                        "settings",
                        SimpleNamespace(metrics_path=path),
                    ),
                    patch.object(
                        metrics_service,
                        "get_model_publication_state",
                        create=True,
                        side_effect=[
                            (old_identity, False),
                            (old_identity, False),
                            (new_identity, False),
                            (new_identity, False),
                        ],
                    ) as publication_state,
                ):
                    old_metrics = metrics_service.load_metrics()
                    replacement.replace(path)
                    self.assertEqual(path.stat().st_mtime_ns, old_stat.st_mtime_ns)
                    new_metrics = metrics_service.load_metrics()
            finally:
                metrics_service.load_metrics.cache_clear()

        self.assertEqual(old_metrics["model_version"], "old-v3")
        self.assertEqual(new_metrics["model_version"], "new-v3")
        self.assertEqual(new_metrics["model_type"], "new_type")
        self.assertEqual(new_metrics["feature_version"], "3.1.0")
        self.assertEqual(new_metrics["quality_gate"], "pass")
        self.assertEqual(publication_state.call_count, 4)

    def test_metrics_route_retries_a_publication_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(
                json.dumps(make_metrics(model_version="new-v3")), encoding="utf-8"
            )
            old_identity = ("publication", "old")
            new_identity = ("publication", "new")

            metrics_service.load_metrics.cache_clear()
            try:
                with (
                    patch.object(
                        metrics_service,
                        "settings",
                        SimpleNamespace(metrics_path=path),
                    ),
                    patch.object(
                        metrics_service,
                        "get_model_publication_state",
                        side_effect=[
                            (old_identity, False),
                            (new_identity, False),
                            (new_identity, False),
                            (new_identity, False),
                        ],
                    ) as publication_state,
                ):
                    response = MetricsResponse.model_validate(get_metrics())
            finally:
                metrics_service.load_metrics.cache_clear()

        self.assertEqual(response.model_version, "new-v3")
        self.assertEqual(publication_state.call_count, 4)

    def test_model_health_route_exhausts_publication_change_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(make_metrics()), encoding="utf-8")
            states = [
                (("publication", attempt, phase), False)
                for attempt in range(3)
                for phase in ("before", "after")
            ]

            metrics_service.load_metrics.cache_clear()
            try:
                with (
                    patch.object(
                        metrics_service,
                        "settings",
                        SimpleNamespace(metrics_path=path),
                    ),
                    patch.object(
                        metrics_service,
                        "get_model_publication_state",
                        side_effect=states,
                    ) as publication_state,
                ):
                    with self.assertRaisesRegex(
                        metrics_service.MetricsPublicationChanged,
                        "repeatedly",
                    ):
                        model_health()
            finally:
                metrics_service.load_metrics.cache_clear()

        self.assertEqual(publication_state.call_count, 6)

    def test_cached_metrics_returns_are_mutation_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(
                json.dumps(
                    make_metrics(
                        warnings=["original warning"],
                        data_source={"source_id": "original-source"},
                    )
                ),
                encoding="utf-8",
            )
            identity = ("publication", "stable")

            metrics_service.load_metrics.cache_clear()
            try:
                with (
                    patch.object(
                        metrics_service,
                        "settings",
                        SimpleNamespace(metrics_path=path),
                    ),
                    patch.object(
                        metrics_service,
                        "get_model_publication_state",
                        return_value=(identity, False),
                    ),
                ):
                    first = metrics_service.load_metrics()
                    first["test_metrics"]["r2"] = 999.0
                    first["data_source"]["source_id"] = "mutated-source"
                    first["warnings"].append("mutated warning")
                    second = metrics_service.load_metrics()
            finally:
                metrics_service.load_metrics.cache_clear()

        self.assertEqual(second["test_metrics"]["r2"], 0.1)
        self.assertEqual(second["data_source"]["source_id"], "original-source")
        self.assertEqual(second["warnings"], ["original warning"])

    def test_metrics_gap_uses_cached_snapshot_once_then_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "models"
            root.mkdir()
            path = root / "metrics.json"
            backup = parent / "models.backup"
            identity = ("publication", "old")
            new_identity = ("publication", "new")
            path.write_text(
                json.dumps(make_metrics(model_version="old-v3")), encoding="utf-8"
            )

            metrics_service.load_metrics.cache_clear()
            try:
                with (
                    patch.object(
                        metrics_service,
                        "settings",
                        SimpleNamespace(metrics_path=path),
                    ),
                    patch.object(
                        metrics_service,
                        "get_model_publication_state",
                        create=True,
                        side_effect=[
                            (identity, False),
                            (identity, False),
                            (identity, True),
                            ModelRuntimeError("publication gap persisted"),
                            (new_identity, False),
                            (new_identity, False),
                        ],
                    ),
                ):
                    cached = metrics_service.load_metrics()
                    root.rename(backup)
                    outcomes = []
                    for _ in range(2):
                        try:
                            outcomes.append(metrics_service.load_metrics())
                        except Exception as exc:
                            outcomes.append(exc)
                    backup.rename(root)
                    path.write_text(
                        json.dumps(make_metrics(model_version="new-v3")),
                        encoding="utf-8",
                    )
                    refreshed = metrics_service.load_metrics()
            finally:
                metrics_service.load_metrics.cache_clear()

        self.assertEqual(cached["model_version"], "old-v3")
        self.assertIsInstance(outcomes[0], dict)
        self.assertEqual(outcomes[0]["model_version"], "old-v3")
        self.assertIsInstance(outcomes[1], ModelRuntimeError)
        self.assertEqual(refreshed["model_version"], "new-v3")

    def test_metrics_gap_without_validated_snapshot_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing" / "metrics.json"
            identity = ("publication", "missing")
            metrics_service.load_metrics.cache_clear()
            try:
                with (
                    patch.object(
                        metrics_service,
                        "settings",
                        SimpleNamespace(metrics_path=path),
                    ),
                    patch.object(
                        metrics_service,
                        "get_model_publication_state",
                        create=True,
                        return_value=(identity, True),
                    ),
                ):
                    try:
                        outcome = metrics_service.load_metrics()
                    except Exception as exc:
                        outcome = exc
            finally:
                metrics_service.load_metrics.cache_clear()

        self.assertIsInstance(outcome, ModelRuntimeError)

    def test_artifact_version_only_v3_metrics_flow_to_health_and_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(
                json.dumps(
                    make_metrics(
                        artifact_version="3.0.0",
                        model_type="catboost",
                    )
                ),
                encoding="utf-8",
            )
            metrics_service.load_metrics.cache_clear()
            try:
                with patch.object(
                    metrics_service,
                    "settings",
                    SimpleNamespace(metrics_path=path),
                ):
                    metrics = metrics_service.load_metrics()
            finally:
                metrics_service.load_metrics.cache_clear()

        with patch(
            "services.model_quality_service.load_metrics", return_value=metrics
        ):
            health = get_model_health()
        with (
            patch("services.model_service.load_metrics", return_value=metrics),
            patch("services.model_service.predict_price_one", return_value=505000.0),
        ):
            prediction = call_model_api(
                PredictRequest(
                    brand="Honda",
                    city="Pune",
                    mileage=10000,
                    year=2020,
                    month=1,
                )
            )

        self.assertEqual(
            MetricsResponse.model_validate(metrics).feature_version, "3.0.0"
        )
        self.assertEqual(
            ModelHealthResponse.model_validate(health).feature_version, "3.0.0"
        )
        self.assertEqual(
            PredictResponse.model_validate(prediction).feature_version, "3.0.0"
        )

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
