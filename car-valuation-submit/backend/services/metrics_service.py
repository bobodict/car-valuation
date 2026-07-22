import json
from functools import lru_cache

from config import settings
from schemas import MetricsResponse


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    with settings.metrics_path.open("r", encoding="utf-8") as metrics_file:
        artifact = json.load(metrics_file)

    response = MetricsResponse(
        model_status="experimental",
        best_val_rmse=artifact.get("best_val_rmse"),
        test_metrics=artifact["test_metrics"],
    )
    return response.model_dump()
