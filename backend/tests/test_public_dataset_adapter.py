import unittest

import numpy as np
import pandas as pd

from services import public_dataset_adapter
from services.dataset_contract import REQUIRED_DATASET_COLUMNS
from services.public_dataset_adapter import (
    adapt_car_details_v4,
    map_owner_count,
    parse_engine_liters,
)


class PublicDatasetAdapterTests(unittest.TestCase):
    def test_raw_contract_has_exact_source_column_order(self):
        self.assertEqual(
            public_dataset_adapter.RAW_REQUIRED_COLUMNS,
            (
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
            ),
        )

    def test_adapt_car_details_v4_maps_units_and_missing_fields(self):
        raw = pd.DataFrame(
            [
                {
                    "Make": "Honda",
                    "Model": "Amaze",
                    "Price": 505000,
                    "Year": 2017,
                    "Kilometer": 87150,
                    "Fuel Type": "Petrol",
                    "Transmission": "Manual",
                    "Location": "Pune",
                    "Color": "Grey",
                    "Owner": "First",
                    "Engine": "1198 cc",
                    "Seating Capacity": 5,
                    "Seller Type": "Corporate",
                    "Max Power": "87 bhp @ 6000 rpm",
                    "Max Torque": "109 Nm @ 4500 rpm",
                    "Drivetrain": "FWD",
                    "Length": 3995,
                    "Width": 1695,
                    "Height": 1501,
                    "Fuel Tank Capacity": 35,
                },
                {
                    "Make": "Unknown",
                    "Model": "X",
                    "Price": 100000,
                    "Year": 2017,
                    "Kilometer": 1000,
                    "Fuel Type": "Petrol",
                    "Transmission": "Manual",
                    "Location": "Pune",
                    "Color": "Grey",
                    "Owner": "Fifth",
                    "Engine": "not supplied",
                    "Seating Capacity": 5,
                    "Seller Type": None,
                    "Max Power": None,
                    "Max Torque": "not supplied",
                    "Drivetrain": None,
                    "Length": None,
                    "Width": None,
                    "Height": None,
                    "Fuel Tank Capacity": None,
                },
            ]
        )

        result = adapt_car_details_v4(raw)

        self.assertEqual(list(result.columns), list(REQUIRED_DATASET_COLUMNS))
        self.assertEqual(result.loc[0, "mileage"], 87150)
        self.assertAlmostEqual(result.loc[0, "displacement"], 1.198)
        self.assertEqual(result.loc[0, "owner_count"], 1)
        self.assertEqual(result.loc[0, "vehicle_type"], "car")
        self.assertEqual(result.loc[0, "accident_history"], "unknown")
        v3_columns = {
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
        }
        self.assertTrue(v3_columns.issubset(result.columns))
        self.assertEqual(result.loc[0, "seller_type"], "Corporate")
        self.assertEqual(result.loc[0, "drivetrain"], "FWD")
        self.assertEqual(result.loc[0, "max_power_bhp"], 87.0)
        self.assertEqual(result.loc[0, "power_rpm"], 6000.0)
        self.assertEqual(result.loc[0, "max_torque_nm"], 109.0)
        self.assertEqual(result.loc[0, "torque_rpm"], 4500.0)
        self.assertEqual(result.loc[0, "length_mm"], 3995)
        self.assertEqual(result.loc[0, "width_mm"], 1695)
        self.assertEqual(result.loc[0, "height_mm"], 1501)
        self.assertEqual(result.loc[0, "fuel_tank_liter"], 35)
        self.assertTrue(pd.isna(result.loc[1, "displacement"]))
        self.assertTrue(pd.isna(result.loc[1, "owner_count"]))
        for column in (
            "max_power_bhp",
            "power_rpm",
            "max_torque_nm",
            "torque_rpm",
            "length_mm",
            "width_mm",
            "height_mm",
            "fuel_tank_liter",
        ):
            with self.subTest(column=column):
                self.assertTrue(pd.isna(result.loc[1, column]))

    def test_parse_power_extracts_bhp_and_rpm(self):
        parse_power = getattr(public_dataset_adapter, "parse_power", None)
        self.assertTrue(callable(parse_power))
        self.assertEqual(
            parse_power("87 bhp @ 6000 rpm"),
            (87.0, 6000.0),
        )

    def test_parse_power_supports_compact_source_format(self):
        for value, expected in (
            ("165@5500", (165.0, 5500.0)),
            ("207@4200", (207.0, 4200.0)),
        ):
            with self.subTest(value=value):
                self.assertEqual(public_dataset_adapter.parse_power(value), expected)

    def test_parse_power_accepts_leading_dot_decimal(self):
        amount, rpm = public_dataset_adapter.parse_power(".5 bhp")

        self.assertEqual(amount, 0.5)
        self.assertTrue(pd.isna(rpm))

    def test_parse_power_preserves_amount_for_source_rows_with_terminal_at(self):
        cases = (("112 bhp @", 112.0), ("189 bhp @", 189.0))
        for value, expected_amount in cases:
            with self.subTest(value=value):
                amount, rpm = public_dataset_adapter.parse_power(value)
                self.assertEqual(amount, expected_amount)
                self.assertTrue(pd.isna(rpm))

    def test_parse_power_converts_ps_and_kw_to_bhp(self):
        parse_power = getattr(public_dataset_adapter, "parse_power", None)
        self.assertTrue(callable(parse_power))
        for value, expected in (("100 ps", 98.632), ("100 kw", 134.102)):
            with self.subTest(value=value):
                amount, rpm = parse_power(value)
                self.assertAlmostEqual(amount, expected)
                self.assertTrue(pd.isna(rpm))

    def test_parse_torque_extracts_nm_and_rpm(self):
        parse_torque = getattr(public_dataset_adapter, "parse_torque", None)
        self.assertTrue(callable(parse_torque))
        self.assertEqual(
            parse_torque("109 Nm @ 4500 rpm"),
            (109.0, 4500.0),
        )

    def test_parse_torque_supports_compact_source_format(self):
        for value, expected in (
            ("240@3000", (240.0, 3000.0)),
            ("500@1600", (500.0, 1600.0)),
        ):
            with self.subTest(value=value):
                self.assertEqual(public_dataset_adapter.parse_torque(value), expected)

    def test_parse_torque_converts_kgm_and_kg_m_to_nm(self):
        parse_torque = getattr(public_dataset_adapter, "parse_torque", None)
        self.assertTrue(callable(parse_torque))
        for value in ("10 kgm", "10 kg-m"):
            with self.subTest(value=value):
                amount, rpm = parse_torque(value)
                self.assertAlmostEqual(amount, 98.0665)
                self.assertTrue(pd.isna(rpm))

    def test_measurement_parsers_return_nan_pairs_for_invalid_or_missing_values(self):
        parse_power = getattr(public_dataset_adapter, "parse_power", None)
        parse_torque = getattr(public_dataset_adapter, "parse_torque", None)
        self.assertTrue(callable(parse_power))
        self.assertTrue(callable(parse_torque))
        cases = (
            (parse_power, None),
            (parse_power, np.nan),
            (parse_power, "not supplied"),
            (parse_power, "87 hp @ 6000 rpm"),
            (parse_power, "1,000 bhp"),
            (parse_power, "87 bhp @ 6000 rpm junk"),
            (parse_power, "junk 165@5500"),
            (parse_torque, None),
            (parse_torque, np.nan),
            (parse_torque, "not supplied"),
            (parse_torque, "109 lb-ft @ 4500 rpm"),
            (parse_torque, "junk 109 Nm junk"),
            (parse_torque, "109 Nm @ 4500 rpm junk"),
            (parse_torque, "junk 240@3000"),
        )

        for parser, value in cases:
            with self.subTest(parser=parser.__name__, value=value):
                amount, rpm = parser(value)
                self.assertTrue(pd.isna(amount))
                self.assertTrue(pd.isna(rpm))

    def test_parse_engine_liters_supports_liters_and_cc(self):
        self.assertAlmostEqual(parse_engine_liters("2.0L"), 2.0)
        self.assertAlmostEqual(parse_engine_liters("1498 cc"), 1.498)

    def test_map_owner_count_supports_ordinal_labels(self):
        self.assertEqual(map_owner_count("Second"), 2)
        self.assertTrue(pd.isna(map_owner_count("Fifth")))

    def test_adapt_rejects_missing_source_column(self):
        raw = pd.DataFrame({"Make": ["Honda"]})

        with self.assertRaisesRegex(ValueError, "Price"):
            adapt_car_details_v4(raw)


if __name__ == "__main__":
    unittest.main()
