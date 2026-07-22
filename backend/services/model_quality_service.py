import math

from services.metrics_service import load_metrics


def assess_metrics(metrics: dict) -> dict:
    warnings = []
    r2 = metrics.get("r2")
    acc_10 = metrics.get("acc_10")

    if r2 is None or not math.isfinite(float(r2)):
        warnings.append(
            "R2 is missing or invalid; model quality cannot be compared with the mean baseline."
        )
    elif float(r2) < 0:
        warnings.append(
            "R2 is below 0; the model underperforms a simple mean baseline on the test set."
        )

    if acc_10 is None or not math.isfinite(float(acc_10)):
        warnings.append("The 10% error hit rate is missing or invalid.")
    elif float(acc_10) < 0.5:
        warnings.append(
            "The 10% error hit rate is below 50%; do not claim high accuracy."
        )

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
        "model_type": artifact.get("model_type", "mlp"),
        "feature_version": artifact.get("feature_version", "2.0.0"),
        "sample_count": artifact.get("sample_count", 0),
    }
