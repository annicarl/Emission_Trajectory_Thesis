import copy
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo


MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "baseline.json"
sys.path.insert(0, str(MODEL_DIRECTORY))

import run_pipeline  # noqa: E402


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.timestamp = datetime(
            2026, 8, 28, 10, 15, tzinfo=ZoneInfo("Europe/Berlin")
        )
        self.fleet_directory = Path("fleet") / "baseline__2026-08-28_101500"
        self.ghg_directory = Path("ghg") / "baseline__2026-08-28_101500"
        self.manifests = []

    def capture_manifest(self, _path, manifest):
        self.manifests.append(copy.deepcopy(manifest))

    def common_patches(self):
        return (
            patch.object(Path, "mkdir"),
            patch.object(run_pipeline, "write_manifest", side_effect=self.capture_manifest),
            patch.object(run_pipeline, "file_sha256", return_value="hash"),
            patch.object(
                run_pipeline,
                "model_reference",
                side_effect=[
                    {"status": "success", "directory": str(self.fleet_directory)},
                    {"status": "success", "directory": str(self.ghg_directory)},
                ],
            ),
        )

    def test_models_receive_same_timestamp_and_fleet_override(self):
        mkdir, write, sha, reference = self.common_patches()
        with mkdir, write, sha, reference:
            with patch.object(
                run_pipeline.fleet_model, "execute", return_value=self.fleet_directory
            ) as fleet_execute:
                with patch.object(
                    run_pipeline.ghg_model, "execute", return_value=self.ghg_directory
                ) as ghg_execute:
                    result = run_pipeline.execute(CONFIG_PATH, self.timestamp)

        fleet_execute.assert_called_once_with(
            CONFIG_PATH.resolve(), run_timestamp=self.timestamp
        )
        ghg_execute.assert_called_once_with(
            CONFIG_PATH.resolve(),
            fleet_run_override=self.fleet_directory.name,
            run_timestamp=self.timestamp,
        )
        self.assertEqual(result.name, "baseline__2026-08-28_101500")
        self.assertEqual(self.manifests[-1]["status"], "success")

    def test_ghg_is_not_started_after_fleet_failure(self):
        mkdir, write, sha, reference = self.common_patches()
        with mkdir, write, sha, reference:
            with patch.object(
                run_pipeline.fleet_model,
                "execute",
                side_effect=RuntimeError("fleet failed"),
            ):
                with patch.object(run_pipeline.ghg_model, "execute") as ghg_execute:
                    with self.assertRaisesRegex(RuntimeError, "fleet failed"):
                        run_pipeline.execute(CONFIG_PATH, self.timestamp)

        ghg_execute.assert_not_called()
        self.assertEqual(self.manifests[-1]["status"], "failed")
        self.assertEqual(self.manifests[-1]["ghg_model"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
