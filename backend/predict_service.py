"""Lazy model-runtime adapter used by the API service."""

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from config import settings
from services.model_runtime import ModelRuntime


@lru_cache(maxsize=1)
def get_model_runtime() -> ModelRuntime:
    """Load and cache the currently published model runtime on first use."""
    return ModelRuntime.from_directory(settings.experiment_path)


def clear_model_runtime_cache() -> None:
    """Clear the process-local runtime after publishing new artifacts."""
    get_model_runtime.cache_clear()


def reload_model_runtime() -> ModelRuntime:
    """Clear and immediately reload the currently published runtime."""
    clear_model_runtime_cache()
    return get_model_runtime()


def predict_price_one(car_dict: Mapping[str, Any]) -> float:
    """Predict one vehicle price using the cached runtime."""
    return get_model_runtime().predict_one(car_dict)
