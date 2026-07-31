import sys
import unittest
from pathlib import Path

import pandas as pd


FLEET_MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLEET_MODEL_DIR))

from fleet_model_with_policies import (  # noqa: E402
    adjusted_nzl_distribution,
    apply_scrappage_policy,
    bev_target_share,
    exclude_drive_types,
    policy_scenario_names,
)


class ScrappagePolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "calibration": {"max_hazard_rate": 1.0},
            "policies": {
                "scrappage_active_years": [2026, 2028],
                "scrappage_hazard_multiplier": 3.0,
                "scrappage_target_patterns": ["diesel", "benzin"],
            },
        }
        self.rates = pd.Series([0.1, 0.4, 0.2])
        self.drives = pd.Series(["Benzin", "Diesel", "Elektro (BEV)"])

    def test_multiplier_only_applies_in_active_destination_year(self):
        result = apply_scrappage_policy(
            self.rates, self.drives, 2026, "abwrackpraemie", self.config
        )
        for actual, expected in zip(result.tolist(), [0.3, 1.0, 0.2]):
            self.assertAlmostEqual(actual, expected)

    def test_rates_are_unchanged_outside_active_years(self):
        result = apply_scrappage_policy(
            self.rates, self.drives, 2027, "abwrackpraemie", self.config
        )
        self.assertEqual(result.tolist(), self.rates.tolist())

    def test_rates_are_unchanged_in_other_scenarios(self):
        result = apply_scrappage_policy(
            self.rates, self.drives, 2026, "baseline_ev_growth", self.config
        )
        self.assertEqual(result.tolist(), self.rates.tolist())


class BevTargetPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "years": {"start_year": 2025},
            "policies": {
                "scenarios": ["diesel_fahrverbot", "abwrackpraemie"],
                "bev_growth_reference_scenario": "test_path",
                "diesel_ban_year": 2032,
                "diesel_patterns": ["diesel"],
                "bev_patterns": ["elektro (bev)"],
                "bev_target_scenarios": {
                    "test_path": {
                        "2025": 20.0,
                        "2030": 50.0,
                        "2035": 80.0,
                        "2040": 100.0,
                    },
                    "light_path": {"2025": 20.0, "2035": 90.0},
                },
            }
        }
        self.base = pd.DataFrame(
            {
                "Segment": ["Small", "Large", "Small", "Large", "Small"],
                "Antriebsart": [
                    "Elektro (BEV)", "Elektro (BEV)", "Benzin", "Benzin", "Diesel"
                ],
                "ist_verbrenner": [False, False, True, True, True],
                "ist_diesel": [False, False, False, False, True],
                "anteil_original": [0.15, 0.05, 0.32, 0.08, 0.40],
            }
        )

    def distribution(self, year):
        return adjusted_nzl_distribution(self.base, year, "test_path", self.config)

    def test_targets_and_linear_interpolation(self):
        for year, expected in [(2025, 0.2), (2030, 0.5), (2032.5, 0.65), (2035, 0.8), (2040, 1.0)]:
            self.assertAlmostEqual(bev_target_share(year, "test_path", self.config), expected)

    def test_last_target_is_held_after_2040(self):
        self.assertAlmostEqual(bev_target_share(2050, "test_path", self.config), 1.0)

    def test_first_target_is_held_before_2025(self):
        self.assertAlmostEqual(bev_target_share(2020, "test_path", self.config), 0.2)

    def test_named_paths_are_added_to_fixed_scenarios(self):
        self.assertEqual(
            policy_scenario_names(self.config),
            [
                "diesel_fahrverbot", "abwrackpraemie", "test_path", "light_path",
            ],
        )

    def test_distribution_preserves_group_proportions_and_sums_to_one(self):
        result = self.distribution(2030)
        bev = result["Antriebsart"].eq("Elektro (BEV)")
        self.assertAlmostEqual(result.loc[bev, "anteil_szenario"].sum(), 0.5)
        self.assertAlmostEqual(result["anteil_szenario"].sum(), 1.0)
        self.assertAlmostEqual(
            result.loc[bev & result["Segment"].eq("Small"), "anteil_szenario"].iloc[0]
            / result.loc[bev & result["Segment"].eq("Large"), "anteil_szenario"].iloc[0],
            3.0,
        )
        petrol = result["Antriebsart"].eq("Benzin")
        diesel = result["Antriebsart"].eq("Diesel")
        self.assertAlmostEqual(
            result.loc[petrol, "anteil_szenario"].sum()
            / result.loc[diesel, "anteil_szenario"].sum(),
            1.0,
        )

    def test_hundred_percent_target_removes_non_bev_registrations(self):
        result = self.distribution(2045)
        non_bev = ~result["Antriebsart"].eq("Elektro (BEV)")
        self.assertTrue((result.loc[non_bev, "anteil_szenario"] == 0).all())

    def test_scrappage_uses_reference_bev_growth_path(self):
        direct = self.distribution(2030)
        scrappage = adjusted_nzl_distribution(
            self.base, 2030, "abwrackpraemie", self.config
        )
        self.assertEqual(
            scrappage["anteil_szenario"].tolist(), direct["anteil_szenario"].tolist()
        )

    def test_diesel_ban_keeps_bev_target_and_redistributes_non_bev_remainder(self):
        result = adjusted_nzl_distribution(
            self.base, 2035, "diesel_fahrverbot", self.config
        )
        bev = result["Antriebsart"].eq("Elektro (BEV)")
        diesel = result["Antriebsart"].eq("Diesel")
        self.assertAlmostEqual(result.loc[bev, "anteil_szenario"].sum(), 0.8)
        self.assertAlmostEqual(result.loc[diesel, "anteil_szenario"].sum(), 0.0)
        self.assertAlmostEqual(result["anteil_szenario"].sum(), 1.0)

    def test_empty_and_out_of_range_targets_are_rejected(self):
        self.config["policies"]["bev_target_scenarios"]["test_path"] = {}
        with self.assertRaisesRegex(ValueError, "at least one"):
            bev_target_share(2030, "test_path", self.config)
        for invalid in (-1, 101):
            self.config["policies"]["bev_target_scenarios"]["test_path"] = {"2030": invalid}
            with self.assertRaisesRegex(ValueError, "between 0 and 100"):
                bev_target_share(2030, "test_path", self.config)

    def test_name_collisions_and_empty_names_are_rejected(self):
        self.config["policies"]["bev_target_scenarios"]["abwrackpraemie"] = {"2025": 20}
        with self.assertRaisesRegex(ValueError, "collide"):
            policy_scenario_names(self.config)
        del self.config["policies"]["bev_target_scenarios"]["abwrackpraemie"]
        self.config["policies"]["bev_target_scenarios"][""] = {"2025": 20}
        with self.assertRaisesRegex(ValueError, "non-empty"):
            policy_scenario_names(self.config)


class DriveTypeExclusionTests(unittest.TestCase):
    def test_configured_drive_types_are_removed_from_input(self):
        frame = pd.DataFrame(
            {
                "Antriebsart": [
                    "Benzin",
                    "Diesel",
                    "Elektro (BEV)",
                    "Hybrid",
                    "Erdgas (CNG) (einschl. bivalent)",
                    "Flüssiggas (LPG) (einschl. bivalent)",
                    "Sonstige",
                ],
                "Anzahl": [1, 2, 3, 4, 5, 6, 7],
            }
        )
        config = {
            "excluded_drive_patterns": [
                "erdgas", "cng", "flüssiggas", "lpg", "sonstige"
            ]
        }

        result = exclude_drive_types(frame, config)

        self.assertEqual(
            result["Antriebsart"].tolist(),
            ["Benzin", "Diesel", "Elektro (BEV)", "Hybrid"],
        )
        self.assertEqual(result["Anzahl"].sum(), 10)

    def test_missing_or_empty_exclusion_list_keeps_all_rows(self):
        frame = pd.DataFrame({"Antriebsart": ["Benzin", "Sonstige"]})
        self.assertEqual(len(exclude_drive_types(frame, {})), 2)
        self.assertEqual(len(exclude_drive_types(frame, {"excluded_drive_patterns": []})), 2)

    def test_non_list_exclusion_config_is_rejected(self):
        frame = pd.DataFrame({"Antriebsart": ["Benzin"]})
        with self.assertRaisesRegex(ValueError, "must be a list"):
            exclude_drive_types(frame, {"excluded_drive_patterns": "sonstige"})


if __name__ == "__main__":
    unittest.main()
