import json
import unittest
from collections import Counter

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

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
        self.assertEqual(first["split_version"], "3.0.0")
        self.assertEqual(first["seed"], 42)
        self.assertEqual(first["test_fraction"], 0.15)
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

    def test_custom_unique_integer_index_is_preserved(self):
        custom_index = pd.Index([10_001 + row * 7 for row in range(200)])
        frame = make_price_fixture(index=custom_index)

        manifest = build_split_manifest(frame)

        self.assertEqual(
            set(manifest["development"]) | set(manifest["test"]),
            set(custom_index),
        )

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

    def test_too_small_frame_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "enough rows"):
            build_split_manifest(make_price_fixture(size=10))

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
