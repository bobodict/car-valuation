"""Lazy, publication-aware model-runtime adapter used by the API service."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import stat
import threading
from typing import Any

from config import settings
from services.model_runtime import ModelRuntime, ModelRuntimeError


_runtime_lock = threading.RLock()
_cached_runtime: ModelRuntime | None = None
_cached_identity: tuple[Any, ...] | None = None
_identity_failure_count = 0

_STALE_IDENTITY_FAILURE_LIMIT = 1
_PUBLICATION_LOAD_ATTEMPTS = 3


def _file_stat(path: Path, identity_name: str) -> tuple[Any, ...]:
    details = path.stat()
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"published model artifact is not a file: {identity_name}")
    return (
        identity_name,
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _directory_stat(path: Path) -> tuple[Any, ...]:
    details = path.stat()
    if not stat.S_ISDIR(details.st_mode):
        raise FileNotFoundError(
            f"published models directory is unavailable: {path}"
        )
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _json_object(contents: bytes, label: str) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} contains invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, hashlib.sha256(contents).hexdigest()


def _artifact_paths(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[tuple[str, str, Path], ...]:
    declared = manifest.get("model_artifacts")
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError("model_manifest.json model_artifacts must be an object")
    resolved_root = root.resolve(strict=True)
    artifacts = []
    for role, relative_name in declared.items():
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(relative_name, str)
            or not relative_name.strip()
        ):
            raise ValueError(
                "model_manifest.json model_artifacts must map names to paths"
            )
        relative_path = Path(relative_name)
        if relative_path.is_absolute():
            raise ValueError(f"model artifact path is outside models directory: {role}")
        artifact_path = root / relative_path
        resolved_artifact = artifact_path.resolve(strict=False)
        if resolved_artifact == resolved_root or resolved_root not in resolved_artifact.parents:
            raise ValueError(f"model artifact path is outside models directory: {role}")
        artifacts.append((role, relative_name, artifact_path))
    return tuple(sorted(artifacts))


def _v3_identity(
    root: Path,
    directory_before: tuple[Any, ...],
    manifest_path: Path,
    manifest_before: tuple[Any, ...],
) -> tuple[Any, ...]:
    manifest_contents = manifest_path.read_bytes()
    manifest, manifest_digest = _json_object(
        manifest_contents, "model_manifest.json"
    )
    artifact_paths = _artifact_paths(root, manifest)
    feature_path = root / "feature_config.json"
    feature_before = _file_stat(feature_path, "feature_config.json")
    artifacts_before = tuple(
        (role, _file_stat(path, relative_name))
        for role, relative_name, path in artifact_paths
    )

    feature_contents = feature_path.read_bytes()
    _, feature_digest = _json_object(feature_contents, "feature_config.json")

    directory_after = _directory_stat(root)
    manifest_after = _file_stat(manifest_path, "model_manifest.json")
    feature_after = _file_stat(feature_path, "feature_config.json")
    artifacts_after = tuple(
        (role, _file_stat(path, relative_name))
        for role, relative_name, path in artifact_paths
    )
    if (
        directory_before != directory_after
        or manifest_before != manifest_after
        or feature_before != feature_after
        or artifacts_before != artifacts_after
    ):
        raise OSError("v3 publication changed while its identity was read")
    return (
        "v3",
        directory_after,
        (manifest_after, manifest_digest),
        (feature_after, feature_digest),
        artifacts_after,
    )


def _legacy_identity(
    root: Path, directory_before: tuple[Any, ...], manifest_path: Path
) -> tuple[Any, ...]:
    feature_path = root / "feature_config.json"
    preprocess_path = root / "preprocess.joblib"
    model_path = root / "price_mlp.pt"
    feature_before = _file_stat(feature_path, "feature_config.json")
    preprocess_before = _file_stat(preprocess_path, "preprocess.joblib")
    model_before = _file_stat(model_path, "price_mlp.pt")

    feature_contents = feature_path.read_bytes()
    _, feature_digest = _json_object(feature_contents, "feature_config.json")

    directory_after = _directory_stat(root)
    feature_after = _file_stat(feature_path, "feature_config.json")
    preprocess_after = _file_stat(preprocess_path, "preprocess.joblib")
    model_after = _file_stat(model_path, "price_mlp.pt")
    try:
        manifest_path.stat()
    except FileNotFoundError:
        pass
    else:
        raise OSError("model manifest appeared while legacy identity was read")
    if (
        directory_before != directory_after
        or feature_before != feature_after
        or preprocess_before != preprocess_after
        or model_before != model_after
    ):
        raise OSError("legacy publication changed while its identity was read")
    return (
        "legacy",
        directory_after,
        (feature_after, feature_digest),
        preprocess_after,
        model_after,
    )


def _published_artifact_identity(models_dir: Path) -> tuple[Any, ...]:
    root = Path(models_dir)
    directory_before = _directory_stat(root)
    manifest_path = root / "model_manifest.json"
    try:
        manifest_before = _file_stat(manifest_path, "model_manifest.json")
    except FileNotFoundError:
        return _legacy_identity(root, directory_before, manifest_path)
    return _v3_identity(root, directory_before, manifest_path, manifest_before)


def _read_published_identity(models_dir: Path) -> tuple[Any, ...] | None:
    global _identity_failure_count
    try:
        identity = _published_artifact_identity(models_dir)
    except MemoryError:
        raise
    except ValueError as exc:
        raise ModelRuntimeError(
            f"published model identity is invalid in {models_dir}: {exc}"
        ) from exc
    except OSError as exc:
        _identity_failure_count += 1
        if (
            _cached_runtime is not None
            and _identity_failure_count <= _STALE_IDENTITY_FAILURE_LIMIT
        ):
            return None
        raise ModelRuntimeError(
            f"published model artifacts are unavailable in {models_dir}: {exc}"
        ) from exc
    _identity_failure_count = 0
    return identity


def get_model_runtime() -> ModelRuntime:
    """Return the runtime for the current complete published artifact identity."""
    global _cached_identity, _cached_runtime

    models_dir = Path(settings.published_models_dir)
    with _runtime_lock:
        identity = _read_published_identity(models_dir)
        if identity is None:
            return _cached_runtime

        for _ in range(_PUBLICATION_LOAD_ATTEMPTS):
            if _cached_runtime is not None and identity == _cached_identity:
                return _cached_runtime
            try:
                candidate = ModelRuntime.from_directory(models_dir)
            except MemoryError:
                raise
            except ModelRuntimeError:
                current_identity = _read_published_identity(models_dir)
                if current_identity is None:
                    return _cached_runtime
                if current_identity != identity:
                    identity = current_identity
                    continue
                raise

            current_identity = _read_published_identity(models_dir)
            if current_identity is None:
                return _cached_runtime
            if current_identity == identity:
                _cached_runtime = candidate
                _cached_identity = identity
                return candidate
            identity = current_identity

        raise ModelRuntimeError(
            f"published model changed repeatedly while loading from {models_dir}"
        )


def clear_model_runtime_cache() -> None:
    """Atomically clear the process-local published runtime."""
    global _cached_identity, _cached_runtime, _identity_failure_count

    with _runtime_lock:
        _cached_runtime = None
        _cached_identity = None
        _identity_failure_count = 0


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
