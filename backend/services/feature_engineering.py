"""Shared vehicle feature engineering contract."""

import numpy as np
import pandas as pd


NUMERIC_FEATURES = (
    "mileage",
    "displacement",
    "seats",
    "owner_count",
    "car_age",
    "max_power_bhp",
    "power_rpm",
    "max_torque_nm",
    "torque_rpm",
    "length_mm",
    "width_mm",
    "height_mm",
    "fuel_tank_liter",
    "mileage_per_year",
    "power_per_liter",
    "footprint_m2",
)
CATEGORICAL_FEATURES = (
    "brand",
    "model",
    "model_family",
    "city",
    "transmission",
    "fuel_type",
    "color",
    "seller_type",
    "drivetrain",
)
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _finite_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_values = numerator.to_numpy(dtype=float, na_value=np.nan)
    denominator_values = denominator.to_numpy(dtype=float, na_value=np.nan)
    valid = (
        np.isfinite(numerator_values)
        & np.isfinite(denominator_values)
        & (denominator_values > 0)
    )
    result = np.full(numerator_values.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(numerator_values, denominator_values, out=result, where=valid)
    result[~np.isfinite(result)] = np.nan
    return pd.Series(result, index=numerator.index)


def enrich_features(frame: pd.DataFrame, collection_year: int) -> pd.DataFrame:
    """Return a normalized copy with all model features and derivations."""
    enriched = frame.copy()

    for column in NUMERIC_FEATURES:
        if column not in enriched.columns:
            enriched[column] = np.nan
        enriched[column] = _finite_numeric(enriched[column])

    for column in CATEGORICAL_FEATURES:
        if column not in enriched.columns:
            enriched[column] = "unknown"
        normalized = (
            enriched[column].astype("string").fillna("unknown").str.strip()
        )
        enriched[column] = normalized.mask(normalized.eq(""), "unknown")

    enriched["model_family"] = enriched["model"].str.split().str[0]

    if "year" in enriched.columns:
        year = _finite_numeric(enriched["year"])
    else:
        year = pd.Series(np.nan, index=enriched.index, dtype=float)
    with np.errstate(invalid="ignore", over="ignore"):
        car_age = (collection_year - year).clip(lower=0)
    enriched["car_age"] = _finite_numeric(car_age)

    age_denominator = enriched["car_age"].clip(lower=1)
    enriched["mileage_per_year"] = _safe_ratio(
        enriched["mileage"], age_denominator
    )
    enriched["power_per_liter"] = _safe_ratio(
        enriched["max_power_bhp"], enriched["displacement"]
    )
    with np.errstate(invalid="ignore", over="ignore"):
        footprint = enriched["length_mm"] * enriched["width_mm"] / 1_000_000
    enriched["footprint_m2"] = _finite_numeric(footprint)

    return enriched


def transform_target(values) -> np.ndarray:
    return np.log1p(np.asarray(values, dtype=float))


def inverse_target(values) -> np.ndarray:
    return np.expm1(np.asarray(values, dtype=float))
