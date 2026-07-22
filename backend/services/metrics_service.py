import hashlib
import json
import threading

from config import settings
from schemas import MetricsResponse


_cache_lock = threading.RLock()
_cached_fingerprint: bytes | None = None
_cached_metrics: dict | None = None


def load_metrics() -> dict:
    global _cached_fingerprint, _cached_metrics

    with _cache_lock:
        contents = settings.metrics_path.read_bytes()
        fingerprint = hashlib.sha256(contents).digest()
        if fingerprint == _cached_fingerprint and _cached_metrics is not None:
            return _cached_metrics

        artifact = json.loads(contents)
        response = MetricsResponse(
            model_status=(
                "usable" if artifact.get("quality_gate") == "pass" else "experimental"
            ),
            quality_gate=artifact.get("quality_gate", "fail"),
            best_val_rmse=artifact.get("best_val_rmse"),
            test_metrics=artifact["test_metrics"],
            currency=artifact.get("currency", "INR"),
            price_unit=artifact.get("price_unit", "INR"),
            mileage_unit=artifact.get("mileage_unit", "km"),
            data_source=artifact.get("data_source", {}),
            model_version=artifact.get("model_version", "unknown"),
            model_type=artifact.get("model_type", "mlp"),
            feature_version=(
                artifact["feature_version"]
                if "feature_version" in artifact
                else artifact.get("artifact_version", "2.0.0")
            ),
            sample_count=artifact.get("sample_count", 0),
            warnings=artifact.get("warnings", []),
        )
        _cached_fingerprint = fingerprint
        _cached_metrics = response.model_dump()
        return _cached_metrics


def _clear_metrics_cache() -> None:
    global _cached_fingerprint, _cached_metrics

    with _cache_lock:
        _cached_fingerprint = None
        _cached_metrics = None


load_metrics.cache_clear = _clear_metrics_cache
