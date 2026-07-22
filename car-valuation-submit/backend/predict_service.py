# predict_service.py
import os
import json
import pandas as pd
import joblib
import torch
from torch import nn

MODELS_DIR = "models"
PREPROCESS_PATH = os.path.join(MODELS_DIR, "preprocess.joblib")
FEATURE_CONFIG_PATH = os.path.join(MODELS_DIR, "feature_config.json")
MODEL_PATH = os.path.join(MODELS_DIR, "price_mlp.pt")

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
                df["collection_time"].dt.year.fillna(2025) - df["year"]
            ).clip(lower=0)
        elif "year" in df.columns:
            df["car_age"] = 2025 - df["year"]

    X = df.reindex(columns=FEATURE_COLS)
    X_proc = preprocess.transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    X_tensor = torch.from_numpy(X_proc.astype("float32")).to(device)

    with torch.no_grad():
        pred = model(X_tensor).cpu().numpy().flatten()[0]

    return float(pred)

if __name__ == "__main__":
    example = {
        "brand": "Toyota",
        "model": "Camry",
        "year": 2018,
        "mileage": 60000,
        "city": "HK",
        "transmission": "AT",
        "fuel_type": "Petrol",
        "displacement": 2.0,
        "vehicle_type": "Sedan",
        "color": "White",
        "seats": 5,
        "accident_history": "No",
        "owner_count": 1,
        "collection_time": "2024-01-01",
    }
    print("预测价格：", predict_price_one(example))
