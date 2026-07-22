import unittest

import pandas as pd

from services.dataset_contract import REQUIRED_DATASET_COLUMNS
from services.public_dataset_adapter import (
    adapt_car_details_v4,
    map_owner_count,
    parse_engine_liters,
)


class PublicDatasetAdapterTests(unittest.TestCase):
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
        self.assertTrue(pd.isna(result.loc[1, "displacement"]))
        self.assertTrue(pd.isna(result.loc[1, "owner_count"]))

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
