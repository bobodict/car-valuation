"""Load the supplied regression artifact and run one deterministic prediction."""

from datetime import date
import json

import joblib
import pandas as pd
import torch
from torch import nn

from config import settings

MODELS_DIR = settings.models_dir
PREPROCESS_PATH = settings.preprocess_path
FEATURE_CONFIG_PATH = settings.feature_config_path
MODEL_PATH = settings.model_path

# 读特征配置
with open(FEATURE_CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

FEATURE_COLS = cfg["feature_cols"]
NUMERIC_FEATURES = cfg["numeric_features"]
CATEGORICAL_FEATURES = cfg["categorical_features"]

preprocess = joblib.load(PREPROCESS_PATH)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(MODEL_PATH, map_location=device)
input_dim = ckpt["input_dim"]
hidden_dims = ckpt["hidden_dims"]
dropout = ckpt.get("dropout", 0.0)
TARGET_MEAN = float(ckpt.get("target_mean", 0.0))
TARGET_STD = float(ckpt.get("target_std", 1.0))

model = MLPRegressor(input_dim, hidden_dims, dropout).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

def predict_price_one(car_dict: dict) -> float:
    """
    car_dict: 车辆属性字典，包含训练时使用的字段：
      brand, model, year, mileage, city, transmission, fuel_type,
      displacement, vehicle_type, color, seats, accident_history,
      owner_count, collection_time (car_age 会在训练时生成，如果前端不给就由后端算)
    这里简化为：你传入 FEATURE_COLS 中的字段即可。
    """
    df = pd.DataFrame([car_dict])
    # 如果前端没算 car_age，这里可以算一下（可选）
    if "car_age" in NUMERIC_FEATURES and "car_age" not in df.columns:
        if "collection_time" in df.columns and "year" in df.columns:
            df["collection_time"] = pd.to_datetime(df["collection_time"], errors="coerce")
            df["car_age"] = (
                df["collection_time"].dt.year.fillna(date.today().year) - df["year"]
            ).clip(lower=0)
        elif "year" in df.columns:
            df["car_age"] = date.today().year - df["year"]

    X = df.reindex(columns=FEATURE_COLS)
    X_proc = preprocess.transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    X_tensor = torch.from_numpy(X_proc.astype("float32")).to(device)

    with torch.no_grad():
        scaled_prediction = model(X_tensor).cpu().numpy().flatten()[0]

    return float(scaled_prediction * TARGET_STD + TARGET_MEAN)

if __name__ == "__main__":
    example = {
        "brand": "丰田",
        "model": "凯美瑞",
        "year": 2018,
        "mileage": 60000,
        "city": "广州",
        "transmission": "自动",
        "fuel_type": "汽油",
        "displacement": 2.0,
        "vehicle_type": "轿车",
        "color": "白色",
        "seats": 5,
        "accident_history": "无事故",
        "owner_count": 1,
        "collection_time": "2026-01-01",
    }
    print("预测价格：", predict_price_one(example))
