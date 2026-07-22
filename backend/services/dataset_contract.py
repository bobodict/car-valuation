from datetime import date
from pathlib import Path

import numpy as np
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
    return validate_normalized_frame(frame)


def validate_normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the normalized training contract without imputing values."""
    missing = validate_dataset_columns(frame.columns)
    if missing:
        raise ValueError(f"dataset missing normalized columns: {', '.join(missing)}")

    price = pd.to_numeric(frame["price"], errors="coerce")
    mileage = pd.to_numeric(frame["mileage"], errors="coerce")
    year = pd.to_numeric(frame["year"], errors="coerce")
    current_year = date.today().year

    if price.isna().any() or (price <= 0).any():
        raise ValueError("price must contain only positive numeric values")
    if mileage.isna().any() or (mileage < 0).any():
        raise ValueError("mileage must contain only non-negative numeric values")
    if year.isna().any() or (year < 1980).any() or (year > current_year).any():
        raise ValueError("year must be between 1980 and the current year")

    return frame
