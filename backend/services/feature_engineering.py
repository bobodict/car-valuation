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


def enrich_features(frame: pd.DataFrame, collection_year: int) -> pd.DataFrame:
    """Return a normalized copy with all model features and derivations."""
    enriched = frame.copy()

    for column in NUMERIC_FEATURES:
        if column not in enriched.columns:
            enriched[column] = np.nan
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        if column not in enriched.columns:
            enriched[column] = "unknown"
        normalized = enriched[column].fillna("unknown").astype(str).str.strip()
        enriched[column] = normalized.mask(normalized.eq(""), "unknown")

    enriched["model_family"] = enriched["model"].str.split().str[0]

    if "year" in enriched.columns:
        year = pd.to_numeric(enriched["year"], errors="coerce")
    else:
        year = pd.Series(np.nan, index=enriched.index, dtype=float)
    enriched["car_age"] = (collection_year - year).clip(lower=0)

    age_denominator = enriched["car_age"].clip(lower=1)
    enriched["mileage_per_year"] = (
        enriched["mileage"] / age_denominator
    ).replace([np.inf, -np.inf], np.nan)

    displacement = enriched["displacement"].where(
        enriched["displacement"].notna() & enriched["displacement"].ne(0)
    )
    enriched["power_per_liter"] = enriched["max_power_bhp"] / displacement
    enriched["footprint_m2"] = (
        enriched["length_mm"] * enriched["width_mm"] / 1_000_000
    )

    return enriched


def transform_target(values) -> np.ndarray:
    return np.log1p(np.asarray(values, dtype=float))


def inverse_target(values) -> np.ndarray:
    return np.expm1(np.asarray(values, dtype=float))
