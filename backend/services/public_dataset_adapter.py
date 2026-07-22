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
    "Seller Type",
    "Max Power",
    "Max Torque",
    "Drivetrain",
    "Length",
    "Width",
    "Height",
    "Fuel Tank Capacity",
)

_POWER_CONVERSIONS = {"bhp": 1.0, "ps": 0.98632, "kw": 1.34102}
_TORQUE_CONVERSIONS = {"nm": 1.0, "kgm": 9.80665, "kg-m": 9.80665}


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


def _parse_measurement(value, conversions) -> tuple[float, float]:
    if value is None or pd.isna(value):
        return np.nan, np.nan
    text = str(value).strip().lower()
    value_match = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z-]+)", text)
    rpm_match = re.search(r"@\s*(\d+(?:\.\d+)?)\s*rpm", text)
    if not value_match or value_match.group(2) not in conversions:
        return np.nan, np.nan
    amount = float(value_match.group(1)) * conversions[value_match.group(2)]
    rpm = float(rpm_match.group(1)) if rpm_match else np.nan
    return amount, rpm


def parse_power(value) -> tuple[float, float]:
    return _parse_measurement(value, _POWER_CONVERSIONS)


def parse_torque(value) -> tuple[float, float]:
    return _parse_measurement(value, _TORQUE_CONVERSIONS)


def adapt_car_details_v4(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"raw dataset missing columns: {', '.join(missing)}")

    power = pd.DataFrame(
        raw["Max Power"].map(parse_power).tolist(),
        columns=("max_power_bhp", "power_rpm"),
        index=raw.index,
    )
    torque = pd.DataFrame(
        raw["Max Torque"].map(parse_torque).tolist(),
        columns=("max_torque_nm", "torque_rpm"),
        index=raw.index,
    )
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
            "seller_type": raw["Seller Type"].fillna("unknown").astype(str).str.strip(),
            "drivetrain": raw["Drivetrain"].fillna("unknown").astype(str).str.strip(),
            "max_power_bhp": power["max_power_bhp"],
            "power_rpm": power["power_rpm"],
            "max_torque_nm": torque["max_torque_nm"],
            "torque_rpm": torque["torque_rpm"],
            "length_mm": pd.to_numeric(raw["Length"], errors="coerce"),
            "width_mm": pd.to_numeric(raw["Width"], errors="coerce"),
            "height_mm": pd.to_numeric(raw["Height"], errors="coerce"),
            "fuel_tank_liter": pd.to_numeric(
                raw["Fuel Tank Capacity"], errors="coerce"
            ),
        },
        index=raw.index,
    )
    normalized = normalized.reindex(columns=REQUIRED_DATASET_COLUMNS)
    return validate_normalized_frame(normalized)
