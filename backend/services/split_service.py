"""Deterministic, leakage-safe dataset split manifests."""

from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


SPLIT_VERSION = "3.0.0"
STRATIFICATION = "log_price_quantiles"
MAX_PRICE_BINS = 10


def _validate_integer_index(frame: pd.DataFrame) -> None:
    index_is_integer = not isinstance(frame.index, pd.MultiIndex) and all(
        isinstance(value, (Integral, np.integer))
        and not isinstance(value, (bool, np.bool_))
        for value in frame.index
    )
    if not frame.index.is_unique or not index_is_integer:
        raise ValueError("frame must have a unique integer index")


def _validated_prices(frame: pd.DataFrame) -> np.ndarray:
    if "price" not in frame.columns:
        raise ValueError("frame must contain a price column")

    source = frame["price"]
    invalid_type = source.map(
        lambda value: isinstance(
            value, (bool, np.bool_, complex, np.complexfloating)
        )
    )
    numeric = pd.to_numeric(source, errors="coerce")
    try:
        prices = numeric.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        raise ValueError(
            "price must contain only finite positive numeric values"
        ) from None

    if invalid_type.any() or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("price must contain only finite positive numeric values")
    return prices


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for n_bins in range(min(MAX_PRICE_BINS, len(positions)), 1, -1):
        bins = _price_bins(prices, n_bins)
        try:
            development, test = train_test_split(
                positions,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
                stratify=bins,
            )
        except ValueError:
            continue

        bin_by_position = np.empty(len(positions), dtype=int)
        bin_by_position[positions] = bins
        development_counts = np.bincount(
            bin_by_position[development], minlength=n_bins
        )
        test_counts = np.bincount(bin_by_position[test], minlength=n_bins)
        if (development_counts >= n_splits).all() and (test_counts >= 1).all():
            return development, test, bin_by_position

    raise ValueError(
        f"frame must contain enough rows for a stratified outer split and {n_splits} folds"
    )


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

    if not isinstance(seed, (Integral, np.integer)) or isinstance(
        seed, (bool, np.bool_)
    ):
        raise ValueError("seed must be an integer")
    if (
        not isinstance(test_size, Real)
        or isinstance(test_size, (bool, np.bool_))
        or not np.isfinite(test_size)
        or not 0 < float(test_size) < 1
    ):
        raise ValueError("test_size must be a fraction between 0 and 1")
    if (
        not isinstance(n_splits, (Integral, np.integer))
        or isinstance(n_splits, (bool, np.bool_))
        or int(n_splits) < 2
    ):
        raise ValueError("n_splits must be an integer of at least 2")

    seed = int(seed)
    test_size = float(test_size)
    n_splits = int(n_splits)
    ordered_positions = np.asarray(
        sorted(range(len(frame)), key=lambda position: int(frame.index[position])),
        dtype=int,
    )
    ordered_prices = prices[ordered_positions]
    development, test, bin_by_position = _outer_split(
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
    development_bins = bin_by_position[development]

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
        "stratification": STRATIFICATION,
        "development": development_ids,
        "test": test_ids,
        "folds": folds,
    }
