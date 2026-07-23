import json
import unittest
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.model_selection import StratifiedKFold

from services.split_service import build_split_manifest


def make_price_fixture(size=200, index=None):
    prices = np.geomspace(50_000, 5_000_000, size)
    return pd.DataFrame(
        {
            "price": prices,
            "brand": ["Honda" if row % 2 else "Toyota" for row in range(size)],
        },
        index=index,
    )


class SplitServiceTests(unittest.TestCase):
    def test_manifest_is_deterministic_and_has_versioned_metadata(self):
        frame = make_price_fixture()

        first = build_split_manifest(frame, seed=42)
        second = build_split_manifest(frame, seed=42)

        self.assertEqual(first, second)
        self.assertTrue(
            {
                "actual_test_fraction",
                "n_splits",
                "effective_outer_bins",
                "effective_development_bins",
            }.issubset(first),
            "manifest must include split audit metadata",
        )
        self.assertEqual(first["split_version"], "3.0.0")
        self.assertEqual(first["seed"], 42)
        self.assertEqual(first["test_fraction"], 0.15)
        self.assertEqual(
            first["actual_test_fraction"], len(first["test"]) / len(frame)
        )
        self.assertEqual(first["n_splits"], 5)
        self.assertEqual(first["effective_outer_bins"], 10)
        self.assertEqual(first["effective_development_bins"], 10)
        self.assertEqual(first["stratification"], "log_price_quantiles")

    def test_outer_split_is_complete_and_disjoint(self):
        frame = make_price_fixture()

        manifest = build_split_manifest(frame)
        development = set(manifest["development"])
        test = set(manifest["test"])

        self.assertTrue(development.isdisjoint(test))
        self.assertEqual(development | test, set(frame.index))

    def test_folds_partition_development_without_exposing_test(self):
        manifest = build_split_manifest(make_price_fixture())
        development = set(manifest["development"])
        test = set(manifest["test"])
        validation_counts = Counter()

        self.assertEqual(len(manifest["folds"]), 5)
        for expected_fold, fold in enumerate(manifest["folds"]):
            train = set(fold["train"])
            validation = set(fold["validation"])
            self.assertEqual(fold["fold"], expected_fold)
            self.assertTrue(train.isdisjoint(validation))
            self.assertEqual(train | validation, development)
            self.assertTrue(test.isdisjoint(train | validation))
            validation_counts.update(fold["validation"])

        self.assertEqual(validation_counts, Counter({row_id: 1 for row_id in development}))

    def test_manifest_ids_are_builtin_ints_and_json_serializable(self):
        manifest = build_split_manifest(make_price_fixture())
        stored_ids = [*manifest["development"], *manifest["test"]]
        for fold in manifest["folds"]:
            self.assertIs(type(fold["fold"]), int)
            stored_ids.extend(fold["train"])
            stored_ids.extend(fold["validation"])

        self.assertTrue(stored_ids)
        self.assertTrue(all(type(row_id) is int for row_id in stored_ids))
        json.dumps(manifest)

    def test_changed_seed_changes_assignment(self):
        frame = make_price_fixture()

        first = build_split_manifest(frame, seed=42)
        second = build_split_manifest(frame, seed=43)

        self.assertNotEqual(first["test"], second["test"])

    def test_seed_accepts_builtin_and_numpy_integers_within_uint32_range(self):
        frame = make_price_fixture()

        for seed in (0, np.int64(42), 2**32 - 1):
            with self.subTest(seed=seed):
                manifest = build_split_manifest(frame, seed=seed)

                self.assertEqual(manifest["seed"], int(seed))
                self.assertIs(type(manifest["seed"]), int)

    def test_invalid_seed_reports_seed_error_before_sklearn(self):
        frame = make_price_fixture()

        for seed in (-1, 2**32, True, 42.0, "42"):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "seed"):
                    build_split_manifest(frame, seed=seed)

    def test_fraction_and_decimal_test_sizes_are_normalized(self):
        frame = make_price_fixture()

        for test_size in (Fraction(3, 20), Decimal("0.15")):
            with self.subTest(test_size=test_size):
                try:
                    manifest = build_split_manifest(frame, test_size=test_size)
                except Exception as exc:
                    self.fail(f"valid test_size raised {exc!r}")

                self.assertEqual(manifest["test_fraction"], 0.15)
                self.assertIs(type(manifest["test_fraction"]), float)
                self.assertEqual(
                    manifest["actual_test_fraction"],
                    len(manifest["test"]) / len(frame),
                )

    def test_invalid_test_sizes_are_rejected_clearly(self):
        frame = make_price_fixture()
        invalid_sizes = (
            True,
            0,
            1,
            float("nan"),
            float("inf"),
            float("-inf"),
            Decimal("NaN"),
            Decimal("Infinity"),
            0.15 + 0j,
            "0.15",
            None,
        )

        for test_size in invalid_sizes:
            with self.subTest(test_size=test_size):
                with self.assertRaisesRegex(ValueError, "test_size"):
                    build_split_manifest(frame, test_size=test_size)

    def test_n_splits_must_be_an_integer_of_at_least_two(self):
        frame = make_price_fixture()

        for n_splits in (True, 1, 5.0, "5"):
            with self.subTest(n_splits=n_splits):
                with self.assertRaisesRegex(ValueError, "n_splits"):
                    build_split_manifest(frame, n_splits=n_splits)

    def test_custom_n_splits_is_recorded_in_manifest(self):
        manifest = build_split_manifest(make_price_fixture(), n_splits=4)

        self.assertIn("n_splits", manifest)
        self.assertIn("effective_development_bins", manifest)
        self.assertEqual(manifest["n_splits"], 4)
        self.assertEqual(len(manifest["folds"]), 4)
        self.assertEqual(manifest["effective_development_bins"], 10)

    def test_custom_unique_integer_index_is_preserved(self):
        custom_index = pd.Index([10_001 + row * 7 for row in range(200)])
        frame = make_price_fixture(index=custom_index)

        manifest = build_split_manifest(frame)

        self.assertEqual(
            set(manifest["development"]) | set(manifest["test"]),
            set(custom_index),
        )

    def test_row_ids_outside_ijson_safe_integer_range_are_rejected(self):
        for unsafe_id in (2**53, -(2**53)):
            with self.subTest(unsafe_id=unsafe_id):
                frame = make_price_fixture(
                    index=pd.Index([unsafe_id, *range(1, 200)])
                )

                with self.assertRaisesRegex(ValueError, "I-JSON"):
                    build_split_manifest(frame)

    def test_duplicate_index_is_rejected(self):
        frame = make_price_fixture()
        frame.index = [0, 0, *range(2, len(frame))]

        with self.assertRaisesRegex(ValueError, "unique integer index"):
            build_split_manifest(frame)

    def test_non_integer_index_is_rejected(self):
        frame = make_price_fixture(index=[f"vehicle-{row}" for row in range(200)])

        with self.assertRaisesRegex(ValueError, "unique integer index"):
            build_split_manifest(frame)

    def test_invalid_price_is_rejected(self):
        cases = {
            "missing": None,
            "not numeric": "invalid",
            "missing value": np.nan,
            "positive infinity": np.inf,
            "negative infinity": -np.inf,
            "zero": 0,
            "negative": -1,
            "boolean": True,
            "complex": 1 + 2j,
        }

        for label, invalid_price in cases.items():
            with self.subTest(label=label):
                frame = make_price_fixture()
                if invalid_price is None:
                    frame = frame.drop(columns="price")
                else:
                    frame["price"] = frame["price"].astype(object)
                    frame.loc[frame.index[0], "price"] = invalid_price

                with self.assertRaisesRegex(ValueError, "price"):
                    build_split_manifest(frame)

    def test_datetime_price_dtype_is_rejected_before_numeric_coercion(self):
        frame = pd.DataFrame(
            {
                "price": pd.Series(
                    pd.date_range("2020-01-01", periods=200, freq="D"),
                    dtype="datetime64[ns]",
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "price"):
            build_split_manifest(frame)

    def test_timedelta_price_dtype_is_rejected_before_numeric_coercion(self):
        frame = pd.DataFrame(
            {
                "price": pd.Series(
                    pd.to_timedelta(np.arange(1, 201), unit="D"),
                    dtype="timedelta64[ns]",
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "price"):
            build_split_manifest(frame)

    def test_real_numpy_and_pandas_numeric_price_dtypes_are_accepted(self):
        cases = {
            "numpy int64": np.arange(1, 201, dtype=np.int64) * 1_000,
            "numpy float64": np.arange(1, 201, dtype=np.float64) * 1_000,
            "pandas Int64": pd.Series(range(1, 201), dtype="Int64") * 1_000,
            "pandas Float64": pd.Series(range(1, 201), dtype="Float64") * 1_000,
        }

        for label, prices in cases.items():
            with self.subTest(label=label):
                frame = pd.DataFrame({"price": prices})

                try:
                    manifest = build_split_manifest(frame)
                except Exception as exc:
                    self.fail(f"valid {label} prices raised {exc!r}")

                self.assertEqual(
                    set(manifest["development"]) | set(manifest["test"]),
                    set(frame.index),
                )

    def test_fraction_and_decimal_price_values_are_accepted(self):
        cases = {
            "Fraction": [
                Fraction(100_000 + row * 1_000, 3) for row in range(200)
            ],
            "Decimal": [
                Decimal(100_000 + row * 1_000) / Decimal(3)
                for row in range(200)
            ],
        }

        for label, prices in cases.items():
            with self.subTest(label=label):
                frame = pd.DataFrame({"price": prices})

                try:
                    manifest = build_split_manifest(frame)
                except Exception as exc:
                    self.fail(f"valid {label} prices raised {exc!r}")

                self.assertEqual(
                    set(manifest["development"]) | set(manifest["test"]),
                    set(frame.index),
                )

    def test_duplicate_heavy_and_equal_prices_split_deterministically(self):
        cases = {
            "duplicate heavy": np.repeat(
                [100_000.0, 200_000.0, 400_000.0, 800_000.0], 50
            ),
            "all equal": np.full(200, 250_000.0),
        }

        for label, prices in cases.items():
            with self.subTest(label=label):
                frame = pd.DataFrame({"price": prices})

                first = build_split_manifest(frame)
                second = build_split_manifest(frame)

                self.assertEqual(first, second)
                self.assertIn("effective_outer_bins", first)
                self.assertIn("effective_development_bins", first)
                self.assertEqual(first["effective_outer_bins"], 10)
                self.assertEqual(first["effective_development_bins"], 10)

    def test_shuffled_row_order_does_not_change_manifest(self):
        frame = make_price_fixture()
        shuffled = frame.sample(frac=1, random_state=17)

        self.assertEqual(
            build_split_manifest(frame),
            build_split_manifest(shuffled),
        )

    def test_smallest_supported_frame_records_fallback_bin_counts(self):
        frame = make_price_fixture(size=12)

        manifest = build_split_manifest(frame)

        self.assertIn("effective_outer_bins", manifest)
        self.assertIn("effective_development_bins", manifest)
        self.assertIn("actual_test_fraction", manifest)
        self.assertEqual(manifest["effective_outer_bins"], 2)
        self.assertEqual(manifest["effective_development_bins"], 2)
        self.assertEqual(manifest["actual_test_fraction"], 2 / 12)

    def test_development_folds_ignore_test_prices_after_membership_is_fixed(self):
        frame = make_price_fixture(size=61)
        membership = build_split_manifest(frame, seed=42)
        fixed_development = np.asarray(membership["development"], dtype=int)
        fixed_test = np.asarray(membership["test"], dtype=int)
        altered = frame.copy()
        rng = np.random.default_rng(22_200)
        altered.loc[fixed_test, "price"] = np.exp(
            rng.uniform(np.log(1_000), np.log(50_000_000), len(fixed_test))
        )

        def fixed_outer_membership(_positions, **_kwargs):
            return fixed_development.copy(), fixed_test.copy()

        with patch(
            "services.split_service.train_test_split",
            side_effect=fixed_outer_membership,
        ):
            baseline = build_split_manifest(frame, seed=42)
            changed = build_split_manifest(altered, seed=42)

        def expected_folds(bin_labels):
            folds = []
            splitter = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42,
            )
            for fold_number, (train_offset, validation_offset) in enumerate(
                splitter.split(fixed_development, bin_labels)
            ):
                folds.append(
                    {
                        "fold": fold_number,
                        "train": sorted(
                            int(row_id)
                            for row_id in fixed_development[train_offset]
                        ),
                        "validation": sorted(
                            int(row_id)
                            for row_id in fixed_development[validation_offset]
                        ),
                    }
                )
            return folds

        development_prices = frame.loc[fixed_development, "price"].to_numpy(
            dtype=float
        )
        development_ranks = pd.Series(np.log1p(development_prices)).rank(
            method="first"
        )
        development_bins = pd.qcut(
            development_ranks,
            q=min(10, len(fixed_development) // 5),
            labels=False,
        ).to_numpy(dtype=int)
        full_price_ranks = pd.Series(
            np.log1p(altered["price"].to_numpy(dtype=float)),
            index=altered.index,
        ).rank(method="first")
        full_data_bins = pd.qcut(
            full_price_ranks,
            q=10,
            labels=False,
        ).loc[fixed_development].to_numpy(dtype=int)

        development_only_folds = expected_folds(development_bins)

        self.assertEqual(baseline["test"], changed["test"])
        self.assertEqual(baseline["folds"], development_only_folds)
        self.assertEqual(changed["folds"], development_only_folds)
        self.assertFalse(np.array_equal(development_bins, full_data_bins))

    def test_too_small_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "enough rows"):
            build_split_manifest(make_price_fixture(size=11))

    def test_input_frame_is_unchanged(self):
        frame = make_price_fixture(
            index=pd.Index([500 + row * 3 for row in range(200)], name="vehicle_id")
        )
        original = frame.copy(deep=True)

        build_split_manifest(frame)

        assert_frame_equal(frame, original)

    def test_outer_test_is_fifteen_percent_and_represents_price_deciles(self):
        frame = make_price_fixture()

        manifest = build_split_manifest(frame)
        ranked_log_prices = pd.Series(
            np.log1p(frame["price"].to_numpy(dtype=float)),
            index=frame.index,
        ).rank(method="first")
        bins = pd.qcut(ranked_log_prices, q=10, labels=False)
        expected_bins = set(bins.unique())

        self.assertAlmostEqual(len(manifest["test"]) / len(frame), 0.15, places=2)
        self.assertEqual(set(bins.loc[manifest["test"]]), expected_bins)
        self.assertEqual(set(bins.loc[manifest["development"]]), expected_bins)


if __name__ == "__main__":
    unittest.main()
