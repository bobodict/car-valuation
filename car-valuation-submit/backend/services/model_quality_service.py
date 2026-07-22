import math

from services.metrics_service import load_metrics


def assess_metrics(metrics: dict) -> dict:
    warnings = []
    r2 = metrics.get("r2")
    acc_10 = metrics.get("acc_10")

    if r2 is None or not math.isfinite(float(r2)):
        warnings.append("R² 缺失或无效，无法判断模型是否优于均值基线。")
    elif float(r2) < 0:
        warnings.append("R² 小于 0，当前模型在测试集上不如简单均值基线。")

    if acc_10 is None or not math.isfinite(float(acc_10)):
        warnings.append("10% 误差命中率缺失或无效。")
    elif float(acc_10) < 0.5:
        warnings.append("10% 误差命中率低于 50%，不应宣传高准确率。")

    return {
        "quality_gate": "fail" if warnings else "pass",
        "warnings": warnings,
    }


def get_model_health() -> dict:
    artifact = load_metrics()
    metrics = artifact["test_metrics"]
    quality_gate = artifact.get("quality_gate")
    warnings = list(artifact.get("warnings", []))
    if quality_gate not in {"pass", "fail"}:
        assessment = assess_metrics(metrics)
        quality_gate = assessment["quality_gate"]
        warnings = assessment["warnings"]
    return {
        "model_status": "usable" if quality_gate == "pass" else "experimental",
        "quality_gate": quality_gate,
        "warnings": warnings,
        "metrics": metrics,
        "data_status": "public source and reproducible training pipeline are recorded",
        "currency": artifact.get("currency", "INR"),
        "price_unit": artifact.get("price_unit", "INR"),
        "mileage_unit": artifact.get("mileage_unit", "km"),
        "data_source": artifact.get("data_source", {}),
        "model_version": artifact.get("model_version", "unknown"),
        "sample_count": artifact.get("sample_count", 0),
    }
