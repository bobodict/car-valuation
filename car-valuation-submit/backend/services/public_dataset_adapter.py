"""Normalize the public car details v4 dataset into the training contract."""

import re

import numpy as np
import pandas as pd

from services.dataset_contract import REQUIRED_DATASET_COLUMNS, validate_normalized_frame


RAW_REQUIRED_COLUMNS = (
    "Make",
    "Model",
    "Price",
    "Year",
    "Kilometer",
    "Fuel Type",
    "Transmission",
    "Location",
    "Color",
    "Owner",
    "Engine",
    "Seating Capacity",
)


def parse_engine_liters(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    number = float(match.group(1))
    return number / 1000 if "cc" in text else number


def map_owner_count(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
    }.get(str(value).strip().lower(), np.nan)


def adapt_car_details_v4(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"raw dataset missing columns: {', '.join(missing)}")

    normalized = pd.DataFrame(
        {
            "price": pd.to_numeric(raw["Price"], errors="coerce"),
            "mileage": pd.to_numeric(raw["Kilometer"], errors="coerce"),
            "displacement": raw["Engine"].map(parse_engine_liters),
            "seats": pd.to_numeric(raw["Seating Capacity"], errors="coerce"),
            "owner_count": raw["Owner"].map(map_owner_count),
            "year": pd.to_numeric(raw["Year"], errors="coerce"),
            "brand": raw["Make"].fillna("unknown").astype(str).str.strip(),
            "model": raw["Model"].fillna("unknown").astype(str).str.strip(),
            "city": raw["Location"].fillna("unknown").astype(str).str.strip(),
            "transmission": raw["Transmission"].fillna("unknown").astype(str).str.strip(),
            "fuel_type": raw["Fuel Type"].fillna("unknown").astype(str).str.strip(),
            "vehicle_type": pd.Series("car", index=raw.index),
            "color": raw["Color"].fillna("unknown").astype(str).str.strip(),
            "accident_history": pd.Series("unknown", index=raw.index),
        },
        index=raw.index,
    )
    normalized = normalized.reindex(columns=REQUIRED_DATASET_COLUMNS)
    return validate_normalized_frame(normalized)
