import unittest
from datetime import date

import numpy as np
import pandas as pd

from services.dataset_contract import (
    REQUIRED_DATASET_COLUMNS,
    validate_dataset_columns,
    validate_normalized_frame,
)
from services.model_quality_service import assess_metrics


def make_valid_normalized_frame():
    return pd.DataFrame(
        [
            {
                "price": 505000,
                "mileage": 87150,
                "displacement": 1.198,
                "seats": 5,
                "owner_count": 1,
                "year": date.today().year,
                "brand": "Honda",
                "model": "Amaze",
                "city": "Pune",
                "transmission": "Manual",
                "fuel_type": "Petrol",
                "vehicle_type": "car",
                "color": "Grey",
                "accident_history": "unknown",
                "seller_type": "Corporate",
                "drivetrain": "FWD",
                "max_power_bhp": 87.0,
                "power_rpm": 6000.0,
                "max_torque_nm": 109.0,
                "torque_rpm": 4500.0,
                "length_mm": 3995.0,
                "width_mm": 1695.0,
                "height_mm": 1501.0,
                "fuel_tank_liter": 35.0,
            }
        ]
    )


class ModelQualityTests(unittest.TestCase):
    def test_negative_r2_and_low_accuracy_fail_quality_gate(self):
        assessment = assess_metrics(
            {
                "mse": 177.17,
                "rmse": 13.31,
                "mae": 11.5,
                "r2": -0.015,
                "acc_10": 0.117,
            }
        )

        self.assertEqual(assessment["quality_gate"], "fail")
        self.assertTrue(any("R²" in warning for warning in assessment["warnings"]))
        self.assertTrue(any("10%" in warning for warning in assessment["warnings"]))


class DatasetContractTests(unittest.TestCase):
    def test_dataset_contract_reports_missing_columns(self):
        frame = pd.DataFrame({"price": [10000], "brand": ["大众"]})

        missing = validate_dataset_columns(frame.columns)

        self.assertIn("mileage", missing)
        self.assertIn("city", missing)

    def test_dataset_contract_has_exact_normalized_column_order(self):
        self.assertEqual(
            REQUIRED_DATASET_COLUMNS,
            (
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
            ),
        )

    def test_physical_values_accept_exact_valid_boundaries(self):
        frame = make_valid_normalized_frame()
        boundaries = {
            "max_power_bhp": (np.nextafter(0.0, 1.0), 2000.0),
            "power_rpm": (np.nextafter(0.0, 1.0), 25000.0),
            "max_torque_nm": (np.nextafter(0.0, 1.0), 10000.0),
            "torque_rpm": (np.nextafter(0.0, 1.0), 25000.0),
            "length_mm": (1000.0, 10000.0),
            "width_mm": (1000.0, 5000.0),
            "height_mm": (500.0, 5000.0),
            "fuel_tank_liter": (np.nextafter(0.0, 1.0), 1000.0),
        }

        for column, values in boundaries.items():
            for value in values:
                with self.subTest(column=column, value=value):
                    candidate = frame.copy()
                    candidate.loc[0, column] = value
                    validate_normalized_frame(candidate)

    def test_physical_values_reject_values_outside_exact_bounds(self):
        frame = make_valid_normalized_frame()
        invalid_values = {
            "max_power_bhp": (0.0, 2000.1),
            "power_rpm": (0.0, 25000.1),
            "max_torque_nm": (0.0, 10000.1),
            "torque_rpm": (0.0, 25000.1),
            "length_mm": (999.9, 10000.1),
            "width_mm": (999.9, 5000.1),
            "height_mm": (499.9, 5000.1),
            "fuel_tank_liter": (0.0, 1000.1),
        }

        for column, values in invalid_values.items():
            for value in values:
                with self.subTest(column=column, value=value):
                    candidate = frame.copy()
                    candidate.loc[0, column] = value
                    with self.assertRaisesRegex(ValueError, column):
                        validate_normalized_frame(candidate)

    def test_missing_physical_values_remain_nan_and_are_valid(self):
        frame = make_valid_normalized_frame()
        physical_columns = (
            "max_power_bhp",
            "power_rpm",
            "max_torque_nm",
            "torque_rpm",
            "length_mm",
            "width_mm",
            "height_mm",
            "fuel_tank_liter",
        )
        frame.loc[0, list(physical_columns)] = np.nan

        result = validate_normalized_frame(frame)

        for column in physical_columns:
            with self.subTest(column=column):
                self.assertTrue(pd.isna(result.loc[0, column]))

    def test_non_numeric_physical_values_are_invalid_not_missing(self):
        frame = make_valid_normalized_frame()
        frame["max_power_bhp"] = frame["max_power_bhp"].astype(object)
        frame.loc[0, "max_power_bhp"] = "not supplied"

        with self.assertRaisesRegex(ValueError, "max_power_bhp"):
            validate_normalized_frame(frame)

    def test_bool_and_complex_physical_values_are_rejected_without_type_errors(self):
        invalid_values = (
            True,
            np.bool_(True),
            1 + 2j,
            np.complex64(1 + 2j),
            np.complex128(1 + 2j),
        )

        for value in invalid_values:
            with self.subTest(value_type=type(value).__name__):
                frame = make_valid_normalized_frame()
                frame["max_power_bhp"] = pd.Series([value], dtype=object)
                with self.assertRaisesRegex(ValueError, "max_power_bhp"):
                    validate_normalized_frame(frame)

    def test_infinite_physical_values_are_rejected(self):
        for value in (np.inf, -np.inf):
            with self.subTest(value=value):
                frame = make_valid_normalized_frame()
                frame.loc[0, "max_power_bhp"] = value
                with self.assertRaisesRegex(ValueError, "max_power_bhp"):
                    validate_normalized_frame(frame)


if __name__ == "__main__":
    unittest.main()
