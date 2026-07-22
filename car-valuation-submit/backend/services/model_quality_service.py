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
    assessment = assess_metrics(metrics)
    return {
        "model_status": "experimental",
        "quality_gate": assessment["quality_gate"],
        "warnings": assessment["warnings"],
        "metrics": metrics,
        "data_status": "training dataset and original training script are not included",
    }
