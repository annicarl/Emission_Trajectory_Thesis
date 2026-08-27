import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(MODEL_DIRECTORY))

from ghg_model import (  # noqa: E402
    add_lca_mapping,
    numeric_series,
    parse_number,
)


class NumberParsingTests(unittest.TestCase):
    def test_german_and_standard_numbers_are_parsed(self):
        self.assertEqual(parse_number("1.234,5"), 1234.5)
        self.assertEqual(parse_number("12.5"), 12.5)
        self.assertTrue(np.isnan(parse_number("na")))

    def test_numeric_series_preserves_numeric_values(self):
        result = numeric_series(pd.Series([1, 2.5]))
        pd.testing.assert_series_equal(result, pd.Series([1.0, 2.5]))


class MappingTests(unittest.TestCase):
    def test_fleet_categories_are_mapped_to_lca_groups(self):
        frame = pd.DataFrame(
            {
                "Segment": ["Kleinwagen", "SUVs"],
                "Antriebsart": ["Elektro (BEV)", "Diesel"],
            }
        )
        result = add_lca_mapping(frame, "test")
        self.assertEqual(result["size_class"].tolist(), ["Small", "Large"])
        self.assertEqual(
            result["drivetrain_group_std"].tolist(), ["BEV", "ICE_Diesel"]
        )

    def test_unknown_categories_are_rejected(self):
        frame = pd.DataFrame(
            {"Segment": ["Unknown"], "Antriebsart": ["Elektro (BEV)"]}
        )
        with self.assertRaises(ValueError):
            add_lca_mapping(frame, "test")


if __name__ == "__main__":
    unittest.main()
