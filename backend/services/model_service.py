import math

from predict_service import (
    predict_price_one,
    take_last_prediction_publication_state,
)
from schemas import PredictRequest
from services.metrics_service import (
    MetricsPublicationChanged,
    load_metrics,
    load_metrics_for_publication,
)
from services.model_runtime import ModelRuntimeError


class ModelServiceError(RuntimeError):
    """Raised when the supplied model cannot produce a usable prediction."""


_PREDICTION_BUNDLE_ATTEMPTS = 3


def _normalize_gearbox(value: str) -> str:
    normalized = value.strip().lower()
    mapping = {
        "automatic": "Automatic",
        "auto": "Automatic",
        "manual": "Manual",
        "amt": "Automatic",
        "cvt": "Automatic",
        "dct": "Automatic",
        "other": "unknown",
        "unknown": "unknown",
        "自动": "Automatic",
        "手动": "Manual",
        "其他": "unknown",
    }
    return mapping.get(normalized, value.strip())


def build_model_input(req: PredictRequest) -> dict:
    """Translate the public request into the fitted model feature schema."""
    return {
        "brand": req.brand,
        "model": req.model or "unknown",
        "year": req.year,
        "mileage": float(req.mileage),
        "city": req.city,
        "transmission": _normalize_gearbox(req.gearbox),
        "fuel_type": req.fuel_type,
        "displacement": float(req.displacement),
        "vehicle_type": req.vehicle_type,
        "color": req.color,
        "seats": req.seats,
        "accident_history": req.accident_history,
        "owner_count": req.owner_count,
        "seller_type": req.seller_type,
        "drivetrain": req.drivetrain,
        "max_power_bhp": req.max_power_bhp,
        "power_rpm": req.power_rpm,
        "max_torque_nm": req.max_torque_nm,
        "torque_rpm": req.torque_rpm,
        "length_mm": req.length_mm,
        "width_mm": req.width_mm,
        "height_mm": req.height_mm,
        "fuel_tank_liter": req.fuel_tank_liter,
    }


def predict_price_one_with_identity(model_input: dict):
    """Preserve legacy predictor injection while capturing runtime identity."""
    take_last_prediction_publication_state()
    raw_price = predict_price_one(model_input)
    return raw_price, take_last_prediction_publication_state()


def _prediction_bundle(model_input: dict) -> tuple[float, dict]:
    for _ in range(_PREDICTION_BUNDLE_ATTEMPTS):
        raw_price, publication = predict_price_one_with_identity(model_input)
        if publication is None:
            return float(raw_price), load_metrics()
        identity, used_cached_runtime = publication
        try:
            metrics = load_metrics_for_publication(
                identity, used_cached_runtime=used_cached_runtime
            )
        except MetricsPublicationChanged:
            continue
        return float(raw_price), metrics
    raise ModelRuntimeError(
        "model publication changed repeatedly while prediction metadata was loaded"
    )


def call_model_api(req: PredictRequest) -> dict:
    model_input = build_model_input(req)
    try:
        raw_price, metrics = _prediction_bundle(model_input)
    except ModelRuntimeError as exc:
        raise ModelServiceError(
            "model service is unavailable; check model artifacts and runtime dependencies"
        ) from exc
    except Exception as exc:
        raise ModelServiceError(
            "model service is unavailable; check model artifacts and runtime dependencies"
        ) from exc

    if not math.isfinite(raw_price) or raw_price <= 0:
        raise ModelServiceError("model returned an invalid positive price")

    price = round(raw_price, 2)
    reference_low = round(price * 0.92, 2)
    reference_high = round(price * 1.08, 2)
    if not all(
        math.isfinite(value)
        for value in (price, reference_low, reference_high)
    ):
        raise ModelServiceError("model returned an invalid non-finite price range")
    quality_gate = metrics.get("quality_gate", "fail")
    model_status = "usable" if quality_gate == "pass" else "experimental"
    source_id = metrics.get("data_source", {}).get("source_id", "unknown source")
    r2 = metrics["test_metrics"]["r2"]

    return {
        "price": price,
        "range": {"low": reference_low, "high": reference_high},
        "confidence": None,
        "model_status": model_status,
        "quality_gate": quality_gate,
        "currency": metrics.get("currency", "INR"),
        "price_unit": metrics.get("price_unit", "INR"),
        "mileage_unit": metrics.get("mileage_unit", "km"),
        "model_version": metrics.get("model_version", "unknown"),
        "model_type": metrics.get("model_type", "mlp"),
        "feature_version": metrics.get("feature_version", "2.0.0"),
        "metrics": metrics["test_metrics"],
        "comment": (
            f"Experimental estimate from {source_id}; test R2={r2:.3f}. "
            "The reference range is not a statistical confidence interval."
        ),
    }
