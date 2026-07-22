import math

from schemas import PredictRequest
from predict_service import predict_price_one
from services.metrics_service import load_metrics


class ModelServiceError(RuntimeError):
    """Raised when the supplied model cannot produce a usable prediction."""


def _normalize_gearbox(value: str) -> str:
    return {"自动": "自动", "手动": "手动", "其他": "自动"}.get(value, "自动")


def build_model_input(req: PredictRequest) -> dict:
    """Translate public Chinese UI fields into the fitted model's feature schema."""
    return {
        "brand": req.brand,
        "model": req.model or f"{req.brand}车型",
        "year": req.year,
        "mileage": float(req.mileage) * 10000,
        "city": req.city,
        "transmission": _normalize_gearbox(req.gearbox),
        "fuel_type": "汽油",
        "displacement": 2.0,
        "vehicle_type": "轿车",
        "color": "白色",
        "seats": 5,
        "accident_history": "无事故",
        "owner_count": 1,
    }


def call_model_api(req: PredictRequest) -> dict:
    model_input = build_model_input(req)
    try:
        raw_price = float(predict_price_one(model_input))
        metrics = load_metrics()
    except Exception as exc:
        raise ModelServiceError("模型服务暂时不可用，请检查模型文件和运行环境。") from exc

    if not math.isfinite(raw_price) or raw_price <= 0:
        raise ModelServiceError("模型返回了无效价格，无法生成估值结果。")

    price_wan = round(raw_price / 10000, 2)
    reference_low = round(price_wan * 0.92, 2)
    reference_high = round(price_wan * 1.08, 2)
    r2 = metrics["test_metrics"]["r2"]

    return {
        "price": price_wan,
        "range": {"low": reference_low, "high": reference_high},
        "confidence": None,
        "model_status": "experimental",
        "metrics": metrics["test_metrics"],
        "comment": (
            f"实验模型预测结果，仅供研究演示。当前测试集 R²={r2:.3f}；"
            "区间为临时参考范围，不是统计置信区间。"
        ),
    }
