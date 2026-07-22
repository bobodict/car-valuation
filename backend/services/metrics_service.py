import json
from functools import lru_cache

from config import settings
from schemas import MetricsResponse


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    with settings.metrics_path.open("r", encoding="utf-8") as metrics_file:
        artifact = json.load(metrics_file)

    response = MetricsResponse(
        model_status="usable" if artifact.get("quality_gate") == "pass" else "experimental",
        quality_gate=artifact.get("quality_gate", "fail"),
        best_val_rmse=artifact.get("best_val_rmse"),
        test_metrics=artifact["test_metrics"],
        currency=artifact.get("currency", "INR"),
        price_unit=artifact.get("price_unit", "INR"),
        mileage_unit=artifact.get("mileage_unit", "km"),
        data_source=artifact.get("data_source", {}),
        model_version=artifact.get("model_version", "unknown"),
        sample_count=artifact.get("sample_count", 0),
        warnings=artifact.get("warnings", []),
    )
    return response.model_dump()
