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
    "seller_type",
    "drivetrain",
    "max_power_bhp",
    "power_rpm",
    "max_torque_nm",
    "torque_rpm",
    "length_mm",
    "width_mm",
    "height_mm",
    "fuel_tank_liter",
)

_PHYSICAL_VALUE_BOUNDS = {
    "max_power_bhp": (0.0, 2000.0, False),
    "power_rpm": (0.0, 25000.0, False),
    "max_torque_nm": (0.0, 10000.0, False),
    "torque_rpm": (0.0, 25000.0, False),
    "length_mm": (1000.0, 10000.0, True),
    "width_mm": (1000.0, 5000.0, True),
    "height_mm": (500.0, 5000.0, True),
    "fuel_tank_liter": (0.0, 1000.0, False),
}


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


def _validate_physical_values(frame: pd.DataFrame) -> None:
    for column, (lower, upper, lower_inclusive) in _PHYSICAL_VALUE_BOUNDS.items():
        source = frame[column]
        invalid_type = source.map(
            lambda value: isinstance(
                value, (bool, np.bool_, complex, np.complexfloating)
            )
        )
        if invalid_type.any():
            raise ValueError(
                f"{column} must be numeric and within its allowed physical range when provided"
            )

        values = pd.to_numeric(source, errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        missing = source.isna().to_numpy(dtype=bool)
        invalid_numeric = ~missing & np.isnan(values)
        non_finite = ~missing & ~np.isfinite(values)
        below_lower = values < lower if lower_inclusive else values <= lower
        out_of_bounds = below_lower | (values > upper)
        if invalid_numeric.any() or non_finite.any() or out_of_bounds.any():
            raise ValueError(
                f"{column} must be numeric and within its allowed physical range when provided"
            )


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

    _validate_physical_values(frame)

    return frame
