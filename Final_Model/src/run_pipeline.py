from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import fleet_model_with_policies as fleet_model
import ghg_model


BERLIN_TIMEZONE = ZoneInfo("Europe/Berlin")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    return (path if path.is_absolute() else base_dir / path).resolve()


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def model_reference(run_directory: Path) -> dict:
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "status": model_manifest.get("status"),
        "run_directory_name": run_directory.name,
        "directory": str(run_directory.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "started_at": model_manifest.get("started_at"),
        "finished_at": model_manifest.get("finished_at"),
    }


def execute(config_path: str | Path, run_timestamp: datetime | None = None) -> Path:
    config_path = Path(config_path).resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_name = fleet_model.validate_run_name(config)
    pipeline_config = config.get("pipeline")
    if not isinstance(pipeline_config, dict) or not pipeline_config.get("output_dir"):
        raise ValueError("Config must contain pipeline.output_dir.")

    started_at = datetime.now(BERLIN_TIMEZONE)
    directory_timestamp = run_timestamp or started_at
    output_root = resolve_path(pipeline_config["output_dir"], config_path.parent)
    pipeline_directory = output_root / f"{run_name}__{directory_timestamp:%Y-%m-%d_%H%M%S}"
    pipeline_directory.mkdir(parents=True, exist_ok=False)
    manifest_path = pipeline_directory / "run_manifest.json"

    manifest = {
        "model": "fleet_ghg_pipeline",
        "run_name": run_name,
        "status": "running",
        "timezone": "Europe/Berlin",
        "run_id_timestamp": iso_timestamp(directory_timestamp),
        "started_at": iso_timestamp(started_at),
        "finished_at": None,
        "config": {
            "source_path": str(config_path),
            "source_filename": config_path.name,
            "source_modified_at": iso_timestamp(
                datetime.fromtimestamp(config_path.stat().st_mtime, BERLIN_TIMEZONE)
            ),
            "sha256": file_sha256(config_path),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "pipeline_script": str(Path(__file__).resolve()),
            "pipeline_script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "fleet_model": {"status": "pending"},
        "ghg_model": {"status": "pending"},
        "error": None,
    }
    write_manifest(manifest_path, manifest)
    run_directory_name = pipeline_directory.name
    expected_fleet_directory = fleet_model.OUTPUT_ROOT / run_directory_name
    expected_ghg_directory = (
        resolve_path(config["ghg"]["output_dir"], config_path.parent)
        / run_directory_name
    )

    try:
        manifest["fleet_model"] = {"status": "running"}
        write_manifest(manifest_path, manifest)
        fleet_directory = fleet_model.execute(config_path, run_timestamp=directory_timestamp)
        manifest["fleet_model"] = model_reference(fleet_directory)
        write_manifest(manifest_path, manifest)

        manifest["ghg_model"] = {"status": "running"}
        write_manifest(manifest_path, manifest)
        ghg_directory = ghg_model.execute(
            config_path,
            fleet_run_override=fleet_directory.name,
            run_timestamp=directory_timestamp,
        )
        manifest["ghg_model"] = model_reference(ghg_directory)
        manifest["status"] = "success"
    except Exception as exc:
        for key, expected_directory in [
            ("fleet_model", expected_fleet_directory),
            ("ghg_model", expected_ghg_directory),
        ]:
            if (expected_directory / "run_manifest.json").is_file():
                manifest[key] = model_reference(expected_directory)
        manifest["status"] = "failed"
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        manifest["finished_at"] = iso_timestamp(datetime.now(BERLIN_TIMEZONE))
        write_manifest(manifest_path, manifest)

    print(f"Pipeline run directory: {pipeline_directory.resolve()}")
    print(f"Fleet run directory:    {fleet_directory}")
    print(f"GHG run directory:      {ghg_directory}")
    return pipeline_directory.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fleet and GHG models sequentially with one config."
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    execute(args.config)


if __name__ == "__main__":
    main()
