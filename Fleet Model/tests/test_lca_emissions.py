import sys
import unittest
from pathlib import Path

import pandas as pd


FLEET_MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLEET_MODEL_DIR))

from lca_emissions import calculate_emissions, summarize  # noqa: E402


def lca_frame() -> pd.DataFrame:
    columns = [
        "Parameter", "BEV_small", "BEV_medium", "BEV_large",
        "ICEVs-Petrol_small", "ICEVs-Petrol_medium", "ICEVs-Petrol_large",
        "ICEVs-Diesel_small", "ICEVs-Diesel_medium", "ICEVs-Diesel_large",
        "Explanation_parameter",
    ]
    values = {
        "Vehicle_production": [5, 6, 7, 5.5, 6.7, 7.8, 5.8, 7, 8.2],
        "Battery_production": [108, 108, 108, None, None, None, None, None, None],
        "Battery_capacity": [45, 60, 75, None, None, None, None, None, None],
        "Consumption": [16, 17.5, 19, 6.5, 7.5, 8.8, 5.4, 6.2, 6.7],
        "WTT": [None, None, None, 680, 680, 680, 980, 980, 980],
        "TTW": [None, None, None, 2240, 2240, 2240, 2440, 2440, 2440],
        "Emissions_electricity": [229, 229, 229, None, None, None, None, None, None],
        "Maintenance": [4.17, 4.17, 4.17, 8.34, 8.34, 8.34, 8.34, 8.34, 8.34],
        "End_of_life": [1.6, 2, 2.4, 1.1, 1.3, 1.6, 1.1, 1.3, 1.6],
    }
    rows = [[name, *row, "test"] for name, row in values.items()]
    return pd.DataFrame(rows, columns=columns)


def activity(new_rows: list[dict]) -> dict[str, pd.DataFrame]:
    base_columns = [
        "Jahr", "Segment", "Antriebsart", "size_class", "drivetrain_group"
    ]
    new = pd.DataFrame(new_rows)
    empty_stock = pd.DataFrame(columns=[*base_columns, "finaler_bestand"])
    empty_exits = pd.DataFrame(columns=[*base_columns, "exits"])
    return {"new": new, "stock": empty_stock, "exits": empty_exits}


class BatteryProductionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"Jahr": 2026, "Segment": "Minis", "Antriebsart": "Elektro (BEV)",
             "size_class": "Small", "drivetrain_group": "BEV", "neue_fahrzeuge": 1},
            {"Jahr": 2026, "Segment": "Kompaktklasse", "Antriebsart": "Elektro (BEV)",
             "size_class": "Medium", "drivetrain_group": "BEV", "neue_fahrzeuge": 2},
            {"Jahr": 2026, "Segment": "SUVs", "Antriebsart": "Elektro (BEV)",
             "size_class": "Large", "drivetrain_group": "BEV", "neue_fahrzeuge": 1},
            {"Jahr": 2026, "Segment": "Minis", "Antriebsart": "Benzin",
             "size_class": "Small", "drivetrain_group": "ICE_Petrol", "neue_fahrzeuge": 3},
            {"Jahr": 2026, "Segment": "Minis", "Antriebsart": "Diesel",
             "size_class": "Small", "drivetrain_group": "ICE_Diesel", "neue_fahrzeuge": 4},
        ]

    def calculate(self, lca=None):
        return calculate_emissions(activity(self.rows), lca if lca is not None else lca_frame())

    def test_battery_emissions_vary_by_bev_size_and_vehicle_count(self):
        result = self.calculate()
        battery = result[result["phase"].eq("battery_production")]
        by_size = battery.groupby("size_class")["emissions_tco2e"].sum()
        self.assertAlmostEqual(by_size["Small"], 4.86)
        self.assertAlmostEqual(by_size["Medium"], 12.96)
        self.assertAlmostEqual(by_size["Large"], 8.10)

    def test_combustion_vehicles_have_no_battery_emissions(self):
        battery = self.calculate().query("phase == 'battery_production'")
        combustion = battery[~battery["drivetrain_group"].eq("BEV")]
        self.assertTrue((combustion["emissions_tco2e"] == 0).all())

    def test_summary_reports_combined_production_without_double_counting(self):
        result = self.calculate()
        metrics = summarize(result, 2025, 2050, 2050)
        production = metrics["by_phase"]["vehicle_production"]
        battery = metrics["by_phase"]["battery_production"]
        self.assertAlmostEqual(metrics["production_total_tco2e"], production + battery)
        self.assertAlmostEqual(
            metrics["cumulative_total_tco2e"], sum(metrics["by_phase"].values())
        )

    def test_capacity_and_intensity_both_affect_battery_sensitivity(self):
        baseline = self.calculate().query("phase == 'battery_production'")["emissions_tco2e"].sum()
        for parameter in ["Battery_capacity", "Battery_production"]:
            varied = lca_frame()
            mask = varied["Parameter"].eq(parameter)
            varied.loc[mask, "BEV_small"] = varied.loc[mask, "BEV_small"] * 1.1
            changed = self.calculate(varied).query("phase == 'battery_production'")["emissions_tco2e"].sum()
            self.assertGreater(changed, baseline, parameter)

    def test_missing_bev_battery_parameter_is_rejected(self):
        lca = lca_frame()
        lca.loc[lca["Parameter"].eq("Battery_capacity"), "BEV_small"] = None
        with self.assertRaisesRegex(ValueError, "Missing battery-production parameters"):
            self.calculate(lca)


if __name__ == "__main__":
    unittest.main()
