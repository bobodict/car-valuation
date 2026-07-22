"""Lazy, publication-aware model-runtime adapter used by the API service."""

from collections.abc import Mapping
import hashlib
from pathlib import Path
import threading
from typing import Any

from config import settings
from services.model_runtime import ModelRuntime, ModelRuntimeError


_runtime_lock = threading.RLock()
_cached_runtime: ModelRuntime | None = None
_cached_identity: tuple[Any, ...] | None = None


def _stable_file_stat(path: Path) -> tuple[Any, ...]:
    stat = path.stat()
    if not path.is_file():
        raise FileNotFoundError(f"published model artifact is missing: {path.name}")
    return (
        path.name,
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _manifest_content_identity(path: Path) -> tuple[Any, ...]:
    before = _stable_file_stat(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = _stable_file_stat(path)
    if before != after:
        raise OSError("model manifest changed while its identity was read")
    return ("v3", digest)


def _published_artifact_identity(models_dir: Path) -> tuple[Any, ...]:
    root = Path(models_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"published models directory is unavailable: {root}")
    manifest_path = root / "model_manifest.json"
    if manifest_path.exists():
        return _manifest_content_identity(manifest_path)
    legacy_paths = (
        root / "feature_config.json",
        root / "preprocess.joblib",
        root / "price_mlp.pt",
    )
    return ("legacy", *(_stable_file_stat(path) for path in legacy_paths))


def _load_current_runtime(models_dir: Path, identity: tuple[Any, ...]) -> ModelRuntime:
    global _cached_identity, _cached_runtime

    try:
        runtime = ModelRuntime.from_directory(models_dir)
    except MemoryError:
        raise
    except ModelRuntimeError:
        if _cached_runtime is not None:
            try:
                current_identity = _published_artifact_identity(models_dir)
            except (OSError, ValueError):
                return _cached_runtime
            if current_identity != identity:
                return _cached_runtime
        raise
    _cached_runtime = runtime
    _cached_identity = identity
    return runtime


def get_model_runtime() -> ModelRuntime:
    """Return the runtime for the current complete published artifact identity."""
    global _cached_identity, _cached_runtime

    models_dir = settings.published_models_dir
    with _runtime_lock:
        try:
            identity = _published_artifact_identity(models_dir)
        except MemoryError:
            raise
        except (OSError, ValueError) as exc:
            if _cached_runtime is not None:
                return _cached_runtime
            raise ModelRuntimeError(
                f"published model artifacts are unavailable in {models_dir}: {exc}"
            ) from exc

        if _cached_runtime is not None:
            if identity == _cached_identity:
                return _cached_runtime
            if _cached_identity and _cached_identity[0] == "v3" and identity[0] == "legacy":
                return _cached_runtime
        return _load_current_runtime(models_dir, identity)


def clear_model_runtime_cache() -> None:
    """Atomically clear the process-local published runtime."""
    global _cached_identity, _cached_runtime

    with _runtime_lock:
        _cached_runtime = None
        _cached_identity = None


def reload_model_runtime() -> ModelRuntime:
    """Atomically clear and load the current complete publication."""
    global _cached_identity, _cached_runtime

    with _runtime_lock:
        _cached_runtime = None
        _cached_identity = None
        return get_model_runtime()


def predict_price_one(car_dict: Mapping[str, Any]) -> float:
    """Predict one vehicle price using the current published runtime."""
    return get_model_runtime().predict_one(car_dict)


get_model_runtime.cache_clear = clear_model_runtime_cache
