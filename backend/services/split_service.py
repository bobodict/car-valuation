"""Deterministic, leakage-safe dataset split manifests."""

import math
from decimal import Decimal
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


SPLIT_VERSION = "3.0.0"
STRATIFICATION = "log_price_quantiles"
MAX_PRICE_BINS = 10
MAX_IJSON_INTEGER = 2**53 - 1
MAX_SKLEARN_SEED = 2**32 - 1
PRICE_ERROR = "price must contain only finite positive real numeric values"


def _validate_integer_index(frame: pd.DataFrame) -> None:
    index_is_integer = not isinstance(frame.index, pd.MultiIndex) and all(
        isinstance(value, (Integral, np.integer))
        and not isinstance(value, (bool, np.bool_))
        for value in frame.index
    )
    if not frame.index.is_unique or not index_is_integer:
        raise ValueError("frame must have a unique integer index")
    if any(abs(int(value)) > MAX_IJSON_INTEGER for value in frame.index):
        raise ValueError("frame index IDs must be within the I-JSON safe integer range")


def _validated_prices(frame: pd.DataFrame) -> np.ndarray:
    if "price" not in frame.columns:
        raise ValueError("frame must contain a price column")

    source = frame["price"]
    invalid_dtype = (
        is_bool_dtype(source.dtype)
        or is_complex_dtype(source.dtype)
        or is_datetime64_any_dtype(source.dtype)
        or is_timedelta64_dtype(source.dtype)
    )
    if invalid_dtype:
        raise ValueError(PRICE_ERROR)

    prices = []
    for value in source.array:
        is_real_number = isinstance(
            value,
            (Real, Decimal, np.integer, np.floating),
        ) and not isinstance(value, (bool, np.bool_))
        if not is_real_number:
            raise ValueError(PRICE_ERROR)
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(PRICE_ERROR) from None
        if not math.isfinite(converted) or converted <= 0:
            raise ValueError(PRICE_ERROR)
        prices.append(converted)

    return np.asarray(prices, dtype=float)


def _validated_seed(seed: int) -> int:
    if not isinstance(seed, (Integral, np.integer)) or isinstance(
        seed, (bool, np.bool_)
    ):
        raise ValueError("seed must be an integer in the uint32 range")
    normalized = int(seed)
    if not 0 <= normalized <= MAX_SKLEARN_SEED:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    return normalized


def _normalized_test_fraction(test_size) -> float:
    is_real_number = isinstance(
        test_size,
        (Real, Decimal, np.integer, np.floating),
    ) and not isinstance(test_size, (bool, np.bool_))
    if not is_real_number:
        raise ValueError("test_size must be a finite fraction between 0 and 1")
    try:
        normalized = float(test_size)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "test_size must be a finite fraction between 0 and 1"
        ) from None
    if not math.isfinite(normalized) or not 0 < normalized < 1:
        raise ValueError("test_size must be a finite fraction between 0 and 1")
    return normalized


def _validated_n_splits(n_splits: int) -> int:
    if (
        not isinstance(n_splits, (Integral, np.integer))
        or isinstance(n_splits, (bool, np.bool_))
        or int(n_splits) < 2
    ):
        raise ValueError("n_splits must be an integer of at least 2")
    return int(n_splits)


def _price_bins(prices: np.ndarray, n_bins: int) -> np.ndarray:
    ranked_log_prices = pd.Series(np.log1p(prices)).rank(method="first")
    bins = pd.qcut(
        ranked_log_prices,
        q=n_bins,
        labels=False,
        duplicates="drop",
    )
    return bins.to_numpy(dtype=int)


def _outer_split(
    positions: np.ndarray,
    prices: np.ndarray,
    *,
    seed: int,
    test_size: float,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    row_count = len(positions)
    test_count = math.ceil(row_count * test_size)
    development_count = row_count - test_count
    effective_bins = min(
        MAX_PRICE_BINS,
        test_count,
        development_count,
        row_count // 2,
    )
    if effective_bins < 2 or development_count < 2 * n_splits:
        raise ValueError(
            f"frame must contain enough rows for a stratified outer split and {n_splits} folds"
        )

    bins = _price_bins(prices, effective_bins)
    development, test = train_test_split(
        positions,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=bins,
    )
    return development, test, int(effective_bins)


def build_split_manifest(
    frame: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.15,
    n_splits: int = 5,
) -> dict[str, object]:
    """Return stable outer-test and development-fold row IDs for ``frame``."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    _validate_integer_index(frame)
    prices = _validated_prices(frame)
    seed = _validated_seed(seed)
    test_size = _normalized_test_fraction(test_size)
    n_splits = _validated_n_splits(n_splits)
    ordered_positions = np.asarray(
        sorted(range(len(frame)), key=lambda position: int(frame.index[position])),
        dtype=int,
    )
    ordered_prices = prices[ordered_positions]
    development, test, effective_outer_bins = _outer_split(
        ordered_positions,
        ordered_prices,
        seed=seed,
        test_size=test_size,
        n_splits=n_splits,
    )

    development = np.asarray(
        sorted(development, key=lambda position: int(frame.index[position])),
        dtype=int,
    )
    test_ids = sorted(int(frame.index[position]) for position in test)
    development_ids = [int(frame.index[position]) for position in development]
    effective_development_bins = min(
        MAX_PRICE_BINS,
        len(development) // n_splits,
    )
    if effective_development_bins < 2:
        raise ValueError(
            f"frame must contain enough rows for a stratified outer split and {n_splits} folds"
        )
    development_bins = _price_bins(
        prices[development],
        effective_development_bins,
    )

    folds = []
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    for fold_number, (train_offset, validation_offset) in enumerate(
        splitter.split(development, development_bins)
    ):
        train_ids = sorted(
            int(frame.index[position]) for position in development[train_offset]
        )
        validation_ids = sorted(
            int(frame.index[position])
            for position in development[validation_offset]
        )
        folds.append(
            {
                "fold": int(fold_number),
                "train": train_ids,
                "validation": validation_ids,
            }
        )

    return {
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "test_fraction": test_size,
        "actual_test_fraction": len(test_ids) / len(frame),
        "n_splits": n_splits,
        "effective_outer_bins": int(effective_outer_bins),
        "effective_development_bins": int(effective_development_bins),
        "stratification": STRATIFICATION,
        "development": development_ids,
        "test": test_ids,
        "folds": folds,
    }
