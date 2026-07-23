import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from services import feature_engineering


EXPECTED_NUMERIC_FEATURES = (
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
EXPECTED_CATEGORICAL_FEATURES = (
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


class FeatureEngineeringTests(unittest.TestCase):
    def _require_callable(self, name):
        function = getattr(feature_engineering, name, None)
        self.assertTrue(callable(function), f"{name} must be callable")
        return function

    def test_feature_contract_uses_exact_immutable_tuples_without_constants(self):
        numeric_features = getattr(feature_engineering, "NUMERIC_FEATURES", None)
        categorical_features = getattr(
            feature_engineering, "CATEGORICAL_FEATURES", None
        )
        model_features = getattr(feature_engineering, "MODEL_FEATURES", None)

        self.assertEqual(numeric_features, EXPECTED_NUMERIC_FEATURES)
        self.assertEqual(categorical_features, EXPECTED_CATEGORICAL_FEATURES)
        self.assertEqual(
            model_features,
            EXPECTED_NUMERIC_FEATURES + EXPECTED_CATEGORICAL_FEATURES,
        )
        self.assertIsInstance(numeric_features, tuple)
        self.assertIsInstance(categorical_features, tuple)
        self.assertIsInstance(model_features, tuple)
        self.assertNotIn("vehicle_type", model_features)
        self.assertNotIn("accident_history", model_features)

    def test_enrich_features_derives_amaze_fixture_and_adds_all_model_columns(self):
        enrich_features = self._require_callable("enrich_features")
        frame = pd.DataFrame(
            {
                "model": ["  Amaze VX CVT  "],
                "year": ["2020"],
                "mileage": ["60000"],
                "displacement": ["1.2"],
                "max_power_bhp": ["90"],
                "length_mm": ["4000"],
                "width_mm": ["1700"],
            }
        )

        result = enrich_features(frame, collection_year=2026)

        self.assertEqual(result.loc[0, "model"], "Amaze VX CVT")
        self.assertEqual(result.loc[0, "model_family"], "Amaze")
        self.assertEqual(result.loc[0, "car_age"], 6)
        self.assertEqual(result.loc[0, "mileage_per_year"], 10000)
        self.assertEqual(result.loc[0, "power_per_liter"], 75)
        self.assertEqual(result.loc[0, "footprint_m2"], 6.8)
        self.assertTrue(
            set(EXPECTED_NUMERIC_FEATURES + EXPECTED_CATEGORICAL_FEATURES).issubset(
                result.columns
            )
        )

    def test_enrich_features_returns_copy_without_mutating_caller(self):
        enrich_features = self._require_callable("enrich_features")
        frame = pd.DataFrame(
            {
                "model": [" Amaze S "],
                "year": ["2020"],
                "mileage": ["60000"],
                "brand": [None],
            }
        )
        original = frame.copy(deep=True)

        result = enrich_features(frame, collection_year=2026)

        self.assertIsNot(result, frame)
        assert_frame_equal(frame, original)

    def test_zero_age_uses_one_year_mileage_denominator(self):
        enrich_features = self._require_callable("enrich_features")
        frame = pd.DataFrame(
            {
                "year": [2027],
                "mileage": [12000],
            }
        )

        result = enrich_features(frame, collection_year=2026)

        self.assertEqual(result.loc[0, "car_age"], 0)
        self.assertEqual(result.loc[0, "mileage_per_year"], 12000)
        self.assertTrue(np.isfinite(result.loc[0, "mileage_per_year"]))

    def test_missing_and_zero_denominators_produce_nan_without_infinity(self):
        enrich_features = self._require_callable("enrich_features")
        frame = pd.DataFrame(
            {
                "year": [None, 2020, 2020],
                "mileage": [60000, None, 60000],
                "displacement": [1.2, 0, None],
                "max_power_bhp": [90, 90, 90],
            }
        )

        result = enrich_features(frame, collection_year=2026)

        self.assertTrue(pd.isna(result.loc[0, "car_age"]))
        self.assertTrue(pd.isna(result.loc[0, "mileage_per_year"]))
        self.assertTrue(pd.isna(result.loc[1, "mileage_per_year"]))
        self.assertTrue(pd.isna(result.loc[1, "power_per_liter"]))
        self.assertTrue(pd.isna(result.loc[2, "power_per_liter"]))
        self.assertFalse(np.isinf(result["mileage_per_year"]).any())

    def test_numeric_source_features_convert_infinities_to_nan(self):
        enrich_features = self._require_callable("enrich_features")
        source_columns = (
            "mileage",
            "displacement",
            "seats",
            "owner_count",
            "max_power_bhp",
            "power_rpm",
            "max_torque_nm",
            "torque_rpm",
            "length_mm",
            "width_mm",
            "height_mm",
            "fuel_tank_liter",
        )
        frame = pd.DataFrame(
            {
                column: [np.inf, -np.inf]
                for column in source_columns
            }
        )

        result = enrich_features(frame, collection_year=2026)

        for column in source_columns:
            with self.subTest(column=column):
                self.assertTrue(result[column].isna().all())

    def test_derived_features_reject_invalid_ratios_and_arithmetic_overflow(self):
        enrich_features = self._require_callable("enrich_features")
        frame = pd.DataFrame(
            {
                "year": [2020, 2020, 2020, 2020, 2020, np.inf, 2020],
                "mileage": [60000, 60000, 60000, 60000, 60000, 1000, np.inf],
                "displacement": [1.2, -1, 0, np.nan, np.inf, 1e-308, 1.2],
                "max_power_bhp": [90, 90, 90, 90, 90, 1e308, 90],
                "length_mm": [4000, 4000, 4000, 4000, 4000, 4000, 1e308],
                "width_mm": [1700, 1700, 1700, 1700, 1700, 1700, 1e308],
            },
            index=[19, 2, 13, 5, 11, 3, 17],
        )

        result = enrich_features(frame, collection_year=2026)

        self.assertEqual(result.loc[19, "mileage_per_year"], 10000)
        self.assertEqual(result.loc[19, "power_per_liter"], 75)
        self.assertEqual(result.loc[19, "footprint_m2"], 6.8)
        for index in (2, 13, 5, 11, 3):
            with self.subTest(index=index, feature="power_per_liter"):
                self.assertTrue(pd.isna(result.loc[index, "power_per_liter"]))
        self.assertTrue(pd.isna(result.loc[3, "car_age"]))
        self.assertTrue(pd.isna(result.loc[3, "mileage_per_year"]))
        self.assertTrue(pd.isna(result.loc[17, "mileage_per_year"]))
        self.assertTrue(pd.isna(result.loc[17, "footprint_m2"]))
        derived = result[["mileage_per_year", "power_per_liter", "footprint_m2"]]
        self.assertTrue(np.isfinite(derived.stack().to_numpy(dtype=float)).all())

    def test_blank_missing_and_absent_categories_become_unknown(self):
        enrich_features = self._require_callable("enrich_features")
        frame = pd.DataFrame(
            {
                "brand": [None, "   "],
                "model": [np.nan, "\t"],
                "city": ["", None],
                "transmission": [None, "\n"],
                "fuel_type": ["  ", np.nan],
                "color": [None, ""],
            }
        )

        result = enrich_features(frame, collection_year=2026)

        for column in EXPECTED_CATEGORICAL_FEATURES:
            with self.subTest(column=column):
                self.assertEqual(result[column].tolist(), ["unknown", "unknown"])

    def test_categorical_missing_values_preserve_custom_index_deterministically(self):
        enrich_features = self._require_callable("enrich_features")
        frame = pd.DataFrame(
            {"model": ["Amaze S", "City ZX", "Civic V"]},
            index=[17, 3, 11],
        )
        frame["brand"] = pd.Categorical(
            ["Honda", None, "Honda"],
            categories=["Honda"],
        )

        try:
            first = enrich_features(frame, collection_year=2026)
            second = enrich_features(frame, collection_year=2026)
        except Exception as exc:
            self.fail(f"categorical normalization raised {exc!r}")

        self.assertEqual(first.index.tolist(), [17, 3, 11])
        self.assertEqual(first["brand"].tolist(), ["Honda", "unknown", "Honda"])
        self.assertTrue(all(isinstance(value, str) for value in first["brand"]))
        assert_frame_equal(first, second)

    def test_target_transform_round_trips_vehicle_prices(self):
        transform_target = self._require_callable("transform_target")
        inverse_target = self._require_callable("inverse_target")
        prices = np.array([49_000, 825_000, 35_000_000])

        transformed = transform_target(prices)
        restored = inverse_target(transformed)

        self.assertTrue(np.issubdtype(transformed.dtype, np.floating))
        self.assertTrue(np.issubdtype(restored.dtype, np.floating))
        np.testing.assert_allclose(restored, prices, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
