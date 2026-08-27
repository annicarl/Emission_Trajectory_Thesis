import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo


FLEET_MODEL_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(FLEET_MODEL_DIR))

import fleet_model_with_policies as fleet_model  # noqa: E402


class RunNameTests(unittest.TestCase):
    def test_valid_run_name_is_returned(self):
        self.assertEqual(
            fleet_model.validate_run_name({"name": "baseline_2035-v1.2"}),
            "baseline_2035-v1.2",
        )

    def test_missing_or_unsafe_run_names_are_rejected(self):
        for value in (None, "", "with space", "../outside", "name/path"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fleet_model.validate_run_name({"name": value})


class RunDirectoryTests(unittest.TestCase):
    def test_directory_uses_run_name_and_berlin_timestamp(self):
        timestamp = datetime(2026, 8, 27, 14, 30, 52, tzinfo=ZoneInfo("Europe/Berlin"))
        output_root = Path("test-output")
        with patch.object(fleet_model, "OUTPUT_ROOT", output_root):
            with patch.object(Path, "mkdir") as mkdir:
                result = fleet_model.create_run_directory(
                    "baseline", timestamp
                )
        self.assertEqual(result.name, "baseline__2026-08-27_143052")
        mkdir.assert_called_once_with(parents=True, exist_ok=False)

    def test_existing_run_directory_is_not_overwritten(self):
        timestamp = datetime(2026, 8, 27, 14, 30, 52, tzinfo=ZoneInfo("Europe/Berlin"))
        with patch.object(fleet_model, "OUTPUT_ROOT", Path("test-output")):
            with patch.object(Path, "mkdir", side_effect=FileExistsError):
                with self.assertRaises(FileExistsError):
                    fleet_model.create_run_directory("baseline", timestamp)


if __name__ == "__main__":
    unittest.main()
