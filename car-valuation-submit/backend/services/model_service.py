# services/model_service.py
from datetime import datetime
from schemas import PredictRequest
from predict_service import predict_price_one


from schemas import PredictRequest
from predict_service import predict_price_one

def call_model_api(req: PredictRequest) -> dict:
    """
    调用队友训练好的神经网络模型，返回价格和区间等信息。
    """

    # 1. 前端的“变速箱”字段 -> 模型里的 transmission
    if "自" in (req.gearbox or ""):
        transmission = "AT"      # 自动档
    elif "手" in (req.gearbox or ""):
        transmission = "MT"      # 手动档
    else:
        transmission = "AT"      # 默认自动

    # 2. 里程单位转换
    # 前端是“万公里”，模型训练用的是“公里”
    mileage_km = float(req.mileage) * 10000

    # 3. 组装成模型需要的特征字典
    #    key 要和你刚才发的 predict_service.py 里说明的一致
    car_dict = {
        "brand": req.brand,
        "model": req.model or f"{req.brand}车型",
        "year": req.year,
        "mileage": mileage_km,
        "city": req.city,
        "transmission": transmission,
        "fuel_type": "Petrol",          # 默认汽油
        "displacement": 2.0,            # 默认 2.0 排量
        "vehicle_type": "Sedan",        # 默认轿车
        "color": "White",
        "seats": 5,
        "accident_history": "No",       # 默认无重大事故
        "owner_count": 1,
        # "collection_time" 可以不传，predict_service 里会用 2025 - year 算 car_age
    }

    try:
        # 4. 调用你队友的模型
        raw_price = float(predict_price_one(car_dict))

        # 你刚才跑脚本得到的是 55742.34 这种数量级，
        # 十有八九是“元”，前端界面展示的是“万元”，这里 /10000
        price_wan = round(raw_price / 10000, 2)

        return {
            "price": price_wan,
            "lower": round(price_wan * 0.92, 2),
            "upper": round(price_wan * 1.08, 2),
            "confidence": 0.9,  # 0~1 之间的小数，前端显示“约 90%”
            "comment": "已接入队友训练的神经网络模型预测结果，仅供参考。"
        }

    except Exception as e:
        # 任何错误都打印出来，并回退到一个简单规则，防止接口直接 500 掉
        print("调用神经网络模型失败：", e)

        base = max(3.0, 20 - 0.8 * (2025 - req.year) - 0.3 * float(req.mileage))
        base = round(base, 2)
        return {
            "price": base,
            "lower": round(base * 0.92, 2),
            "upper": round(base * 1.08, 2),
            "confidence": 0.6,
            "comment": "模型调用异常，已回退到规则估值结果。"
        }
