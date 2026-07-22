from pathlib import Path

import pandas as pd


REQUIRED_DATASET_COLUMNS = (
    "price",
    "mileage",
    "displacement",
    "seats",
    "owner_count",
    "year",
    "brand",
    "model",
    "city",
    "transmission",
    "fuel_type",
    "vehicle_type",
    "color",
    "accident_history",
)


def validate_dataset_columns(columns) -> list[str]:
    available = set(columns)
    return [column for column in REQUIRED_DATASET_COLUMNS if column not in available]


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集不存在：{dataset_path}")
    frame = pd.read_csv(dataset_path)
    missing = validate_dataset_columns(frame.columns)
    if missing:
        raise ValueError(f"数据集缺少必要字段：{', '.join(missing)}")
    return frame
