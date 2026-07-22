from collections import OrderedDict
import hashlib
import json
import threading
from typing import Any

from config import settings
from predict_service import get_model_publication_state
from schemas import MetricsResponse
from services.model_runtime import ModelRuntimeError


class MetricsPublicationChanged(ModelRuntimeError):
    """Raised when metrics no longer match the requested model publication."""


_cache_lock = threading.RLock()
_cached_snapshots: OrderedDict[
    tuple[Any, ...], tuple[bytes, dict]
] = OrderedDict()
_CACHED_PUBLICATION_LIMIT = 2


def _cached_metrics(identity: tuple[Any, ...]) -> dict | None:
    snapshot = _cached_snapshots.get(identity)
    if snapshot is None:
        return None
    _cached_snapshots.move_to_end(identity)
    return snapshot[1]


def _require_cached_metrics(identity: tuple[Any, ...]) -> dict:
    metrics = _cached_metrics(identity)
    if metrics is None:
        raise ModelRuntimeError(
            "no validated metrics snapshot is cached for the model publication"
        )
    return metrics


def _store_metrics(
    identity: tuple[Any, ...], fingerprint: bytes, metrics: dict
) -> dict:
    _cached_snapshots[identity] = (fingerprint, metrics)
    _cached_snapshots.move_to_end(identity)
    while len(_cached_snapshots) > _CACHED_PUBLICATION_LIMIT:
        _cached_snapshots.popitem(last=False)
    return metrics


def _validated_metrics(contents: bytes) -> dict:
    try:
        artifact = json.loads(contents)
        if not isinstance(artifact, dict):
            raise TypeError("metrics artifact must contain a JSON object")
        response = MetricsResponse(
            model_status=(
                "usable"
                if artifact.get("quality_gate") == "pass"
                else "experimental"
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
    except MemoryError:
        raise
    except Exception as exc:
        raise ModelRuntimeError("published metrics artifact is invalid") from exc
    return response.model_dump()


def _load_metrics_for_state(
    expected_identity: tuple[Any, ...],
    initial_state: tuple[tuple[Any, ...], bool],
) -> dict:
    initial_identity, used_cached_runtime = initial_state
    if initial_identity != expected_identity:
        raise MetricsPublicationChanged("model publication changed before metrics load")
    if used_cached_runtime:
        return _require_cached_metrics(expected_identity)

    try:
        contents = settings.metrics_path.read_bytes()
    except MemoryError:
        raise
    except OSError as exc:
        current_identity, current_is_stale = get_model_publication_state()
        if current_identity != expected_identity:
            raise MetricsPublicationChanged(
                "model publication changed while metrics were read"
            ) from exc
        if current_is_stale:
            return _require_cached_metrics(expected_identity)
        raise ModelRuntimeError("published metrics artifact is unavailable") from exc

    current_identity, current_is_stale = get_model_publication_state()
    if current_identity != expected_identity:
        raise MetricsPublicationChanged(
            "model publication changed while metrics were read"
        )
    if current_is_stale:
        return _require_cached_metrics(expected_identity)

    fingerprint = hashlib.sha256(contents).digest()
    cached = _cached_snapshots.get(expected_identity)
    if cached is not None and cached[0] == fingerprint:
        _cached_snapshots.move_to_end(expected_identity)
        return cached[1]
    return _store_metrics(
        expected_identity, fingerprint, _validated_metrics(contents)
    )


def load_metrics_for_publication(
    expected_identity: tuple[Any, ...], *, used_cached_runtime: bool = False
) -> dict:
    """Load metrics that match one runtime publication identity."""
    with _cache_lock:
        if used_cached_runtime:
            return _require_cached_metrics(expected_identity)
        return _load_metrics_for_state(
            expected_identity, get_model_publication_state()
        )


def load_metrics() -> dict:
    """Load metrics for the current complete model publication."""
    with _cache_lock:
        publication = get_model_publication_state()
        identity, _ = publication
        return _load_metrics_for_state(identity, publication)


def _clear_metrics_cache() -> None:
    with _cache_lock:
        _cached_snapshots.clear()


load_metrics.cache_clear = _clear_metrics_cache
