from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
BERLIN_TIMEZONE = ZoneInfo("Europe/Berlin")

SEGMENT_TO_SIZE_CLASS = {
    "Minis": "Small",
    "Kleinwagen": "Small",
    "Kompaktklasse": "Medium",
    "Mittelklasse": "Medium",
    "Mini-Vans": "Medium",
    "Sonstige": "Medium",
    "Obere Mittelklasse": "Large",
    "Oberklasse": "Large",
    "SUVs": "Large",
    "Geländewagen": "Large",
    "Sportwagen": "Large",
    "Großraum-Vans": "Large",
    "Utilities": "Large",
    "Wohnmobile": "Large",
}

DRIVE_TO_GROUP = {
    "Elektro (BEV)": "BEV",
    "Benzin": "ICE_Petrol",
    "Diesel": "ICE_Diesel",
    "Hybrid": "ICE_Petrol",
    "Erdgas (CNG) (einschl. bivalent)": "ICE_Petrol",
    "Flüssiggas (LPG) (einschl. bivalent)": "ICE_Petrol",
    "Sonstige": "ICE_Petrol",
}

PARAM_DRIVE_STANDARDIZATION = {
    "BEV": "BEV",
    "ICEVs-Petrol": "ICE_Petrol",
    "ICEVs-Diesel": "ICE_Diesel",
}

EMISSION_COLUMNS = [
    "Szenario",
    "emission_year",
    "phase",
    "Segment",
    "Antriebsart",
    "drivetrain_group_std",
    "size_class",
    "vehicles",
    "emissions_tco2e",
]


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    return config


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    return (path if path.is_absolute() else base_dir / path).resolve()


def validate_run_name(config: dict) -> str:
    run_name = config.get("name")
    if not isinstance(run_name, str) or not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "Config field 'name' must start with a letter or number and contain "
            "only letters, numbers, dots, underscores, and hyphens."
        )
    return run_name


def require_columns(frame: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {frame_name}: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def read_semicolon_csv(path: Path, encoding: str = "utf-8-sig") -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, sep=";", encoding=encoding)


def parse_number(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none"}:
        return np.nan
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    return float(text)


def numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.map(parse_number).astype(float)


def add_lca_mapping(frame: pd.DataFrame, frame_name: str) -> pd.DataFrame:
    require_columns(frame, ["Segment", "Antriebsart"], frame_name)
    result = frame.copy()
    result["size_class"] = result["Segment"].map(SEGMENT_TO_SIZE_CLASS)
    result["drivetrain_group_std"] = result["Antriebsart"].map(DRIVE_TO_GROUP)
    missing = result[result[["size_class", "drivetrain_group_std"]].isna().any(axis=1)]
    if not missing.empty:
        examples = missing[["Segment", "Antriebsart"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Missing LCA mappings in {frame_name}: {examples}")
    return result


def load_emission_parameters(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = read_semicolon_csv(path)
    require_columns(wide, ["Parameter", "Explanation_parameter"], path.name)
    value_columns = [
        column
        for column in wide.columns
        if column not in {"Parameter", "Explanation_parameter"}
    ]
    long = wide.melt(
        id_vars=["Parameter", "Explanation_parameter"],
        value_vars=value_columns,
        var_name="vehicle_type_raw",
        value_name="value_raw",
    )
    long["value"] = long["value_raw"].map(parse_number)
    split = long["vehicle_type_raw"].str.rsplit("_", n=1, expand=True)
    long["drivetrain_group_raw"] = split[0]
    long["size_class_raw"] = split[1]
    long["drivetrain_group_std"] = long["drivetrain_group_raw"].map(
        PARAM_DRIVE_STANDARDIZATION
    )
    long["size_class"] = long["size_class_raw"].str.lower().map(
        {"small": "Small", "medium": "Medium", "large": "Large"}
    )
    lookup = (
        long.pivot_table(
            index=["drivetrain_group_std", "size_class"],
            columns="Parameter",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return long, lookup


def merge_parameters(
    frame: pd.DataFrame,
    lookup: pd.DataFrame,
    required_parameters: list[str],
    frame_name: str,
) -> pd.DataFrame:
    result = frame.merge(
        lookup, on=["drivetrain_group_std", "size_class"], how="left"
    )
    for parameter in required_parameters:
        if parameter not in result.columns:
            print(
                f"Warning: LCA parameter {parameter!r} is absent; "
                f"it is set to 0.0 for {frame_name}."
            )
            result[parameter] = 0.0
        else:
            missing_count = int(result[parameter].isna().sum())
            if missing_count:
                print(
                    f"Warning: {missing_count} values for {parameter!r} are "
                    f"missing in {frame_name}; they are set to 0.0."
                )
                result[parameter] = result[parameter].fillna(0.0)
    return result


def load_fleet_inputs(config: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    config_path = Path(config["_config_path"])
    base_dir = config_path.parent
    ghg = config["ghg"]
    fleet_root = resolve_path(ghg["fleet_outputs_dir"], base_dir)
    fleet_run_name = ghg["fleet_run"]
    fleet_run_dir = fleet_root / fleet_run_name
    manifest_path = fleet_run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Fleet run manifest not found: {manifest_path}")
    fleet_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if fleet_manifest.get("status") != "success":
        raise ValueError(
            f"Fleet run {fleet_run_name!r} has status {fleet_manifest.get('status')!r}."
        )
    fleet_prefix = fleet_manifest.get("run_name")
    if not fleet_prefix:
        raise ValueError(f"Fleet manifest has no run_name: {manifest_path}")

    filenames = {
        "new_vehicles": "policy_new_vehicles_all.csv",
        "final_stock": "policy_final_stock_all.csv",
        "exits": "policy_exits_all.csv",
    }
    frames = {
        key: read_semicolon_csv(fleet_run_dir / f"{fleet_prefix}__{filename}")
        for key, filename in filenames.items()
    }
    return frames, {
        "directory": fleet_run_dir,
        "manifest_path": manifest_path,
        "manifest": fleet_manifest,
    }


def calculate_ghg(config: dict) -> dict[str, pd.DataFrame | dict | Path]:
    config_path = Path(config["_config_path"])
    base_dir = config_path.parent
    ghg = config["ghg"]
    start_year = int(ghg["analysis_start_year"])
    end_year = int(ghg["analysis_end_year"])
    annual_km = float(ghg["annual_km_per_vehicle"])
    offsets = ghg["year_offsets"]

    fleet, fleet_source = load_fleet_inputs(config)
    data_dir = resolve_path(config["data_dir"], base_dir)
    historical_exits_path = data_dir / config["input_files"]["abs"]
    parameter_path = resolve_path(ghg["emission_parameters_file"], base_dir)
    historical_exits = read_semicolon_csv(
        historical_exits_path, encoding=config["csv"]["encoding"]
    )
    parameter_long, parameter_lookup = load_emission_parameters(parameter_path)

    prepared_new = add_lca_mapping(fleet["new_vehicles"], "new vehicles")
    prepared_stock = add_lca_mapping(fleet["final_stock"], "final stock")
    prepared_exits = add_lca_mapping(fleet["exits"], "exits")

    require_columns(
        prepared_new,
        ["Szenario", "Jahr", "Segment", "Antriebsart", "neue_fahrzeuge"],
        "new vehicles",
    )
    production = prepared_new.copy()
    production["neue_fahrzeuge"] = numeric_series(
        production["neue_fahrzeuge"]
    ).fillna(0.0)
    production = merge_parameters(
        production, parameter_lookup, ["Vehicle_production"], "production"
    )
    registration_year_column = next(
        (
            column
            for column in [
                "Jahr der Erstzulassung",
                "Jahr_der_Erstzulassung",
                "Erstzulassungsjahr",
            ]
            if column in production.columns
        ),
        "Jahr",
    )
    production["emission_year"] = (
        numeric_series(production[registration_year_column])
        + int(offsets["production"])
    )
    production["phase"] = "production"
    production["vehicles"] = production["neue_fahrzeuge"]
    production["emissions_tco2e"] = (
        production["neue_fahrzeuge"] * production["Vehicle_production"]
    )
    production_emissions = production[EMISSION_COLUMNS].copy()

    battery_parameters = ["Battery_production", "Battery_capacity"]
    missing_battery_columns = [
        parameter
        for parameter in battery_parameters
        if parameter not in parameter_lookup.columns
    ]
    battery_vehicle = production["drivetrain_group_std"].eq("BEV") | production[
        "Antriebsart"
    ].eq("Hybrid")
    if missing_battery_columns and battery_vehicle.any():
        raise ValueError(f"Missing battery parameters: {missing_battery_columns}")
    battery_lookup = parameter_lookup.loc[
        parameter_lookup["drivetrain_group_std"].eq("BEV"),
        ["size_class", *battery_parameters],
    ].rename(columns={name: f"bev_{name}" for name in battery_parameters})
    production = production.merge(
        battery_lookup, on="size_class", how="left", validate="many_to_one"
    )
    battery_vehicle = production["drivetrain_group_std"].eq("BEV") | production[
        "Antriebsart"
    ].eq("Hybrid")
    battery_value_columns = [f"bev_{name}" for name in battery_parameters]
    if production.loc[battery_vehicle, battery_value_columns].isna().any(axis=1).any():
        raise ValueError("Missing BEV battery parameters for BEVs or hybrids.")
    battery = production.copy()
    battery["phase"] = "battery_production"
    battery["emissions_tco2e"] = np.where(
        battery_vehicle,
        battery["neue_fahrzeuge"]
        * battery["bev_Battery_production"]
        * battery["bev_Battery_capacity"]
        / 1_000.0,
        0.0,
    )
    battery_emissions = battery[EMISSION_COLUMNS].copy()

    require_columns(
        prepared_stock,
        ["Szenario", "Jahr", "Segment", "Antriebsart", "finaler_bestand"],
        "final stock",
    )
    use = prepared_stock.copy()
    use["finaler_bestand"] = numeric_series(use["finaler_bestand"]).fillna(0.0)
    maintenance_parameter = ghg["maintenance_parameter"]
    use = merge_parameters(
        use,
        parameter_lookup,
        ["Consumption", "WTT", "TTW", maintenance_parameter],
        "use",
    )
    use["emission_year"] = numeric_series(use["Jahr"]) + int(offsets["use"])
    electricity_mix = {
        int(year): float(value)
        for year, value in ghg["electricity_mix_gco2e_per_kwh"].items()
    }
    missing_electricity_years = sorted(
        set(range(start_year, end_year + 1)).difference(electricity_mix)
    )
    if missing_electricity_years:
        raise ValueError(
            f"Electricity-mix path lacks years: {missing_electricity_years}"
        )
    use["electricity_mix_gco2e_per_kwh"] = use["emission_year"].map(
        electricity_mix
    )
    is_bev = use["drivetrain_group_std"].eq("BEV")
    use["use_emissions_gco2e_per_km"] = np.where(
        is_bev,
        use["Consumption"] / 100.0 * use["electricity_mix_gco2e_per_kwh"]
        + use[maintenance_parameter],
        use["Consumption"] / 100.0 * (use["WTT"] + use["TTW"])
        + use[maintenance_parameter],
    )
    use["phase"] = "use"
    use["vehicles"] = use["finaler_bestand"]
    use["emissions_tco2e"] = (
        use["finaler_bestand"]
        * annual_km
        * use["use_emissions_gco2e_per_km"]
        / 1_000_000.0
    )
    use_emissions = use[EMISSION_COLUMNS].copy()

    require_columns(
        prepared_exits,
        ["Szenario", "Jahr", "Segment", "Antriebsart", "exits"],
        "exits",
    )
    end_of_life = prepared_exits.copy()
    end_of_life["exits"] = numeric_series(end_of_life["exits"]).fillna(0.0)
    end_of_life = merge_parameters(
        end_of_life, parameter_lookup, ["End_of_life"], "end of life"
    )
    end_of_life["emission_year"] = (
        numeric_series(end_of_life["Jahr"]) + int(offsets["end_of_life"])
    )
    end_of_life["phase"] = "end_of_life"
    end_of_life["vehicles"] = end_of_life["exits"]
    end_of_life["emissions_tco2e"] = (
        end_of_life["exits"] * end_of_life["End_of_life"]
    )
    end_of_life_emissions = end_of_life[EMISSION_COLUMNS].copy()

    historical = historical_exits.loc[
        historical_exits["Berichtsjahr"].isin([2024, 2025]),
        ["Berichtsjahr", "Segment", "Antriebsart", "Anzahl"],
    ].copy()
    historical = add_lca_mapping(historical, "historical exits")
    historical["exits"] = numeric_series(historical["Anzahl"]).fillna(0.0)
    historical = merge_parameters(
        historical, parameter_lookup, ["End_of_life"], "historical end of life"
    )
    historical["emission_year"] = (
        numeric_series(historical["Berichtsjahr"])
        + int(offsets["end_of_life"])
    )
    historical["phase"] = "end_of_life"
    historical["vehicles"] = historical["exits"]
    historical["emissions_tco2e"] = historical["exits"] * historical["End_of_life"]
    scenario_names = pd.DataFrame(
        {"Szenario": sorted(prepared_stock["Szenario"].unique())}
    )
    historical = historical.merge(scenario_names, how="cross")
    end_of_life_emissions = pd.concat(
        [historical[EMISSION_COLUMNS], end_of_life_emissions], ignore_index=True
    )

    emissions = pd.concat(
        [
            production_emissions,
            battery_emissions,
            use_emissions,
            end_of_life_emissions,
        ],
        ignore_index=True,
    )
    emissions["emission_year"] = emissions["emission_year"].astype(int)
    emissions["vehicles"] = emissions["vehicles"].fillna(0.0)
    emissions["emissions_tco2e"] = emissions["emissions_tco2e"].fillna(0.0)
    emissions = emissions.loc[
        emissions["emission_year"].between(start_year, end_year)
    ].copy()
    emissions = emissions.sort_values(
        ["Szenario", "emission_year", "phase", "Segment", "Antriebsart"]
    ).reset_index(drop=True)

    by_year_phase = emissions.groupby(
        ["Szenario", "emission_year", "phase"], as_index=False
    )["emissions_tco2e"].sum()
    by_year_total = (
        emissions.groupby(["Szenario", "emission_year"], as_index=False)[
            "emissions_tco2e"
        ]
        .sum()
        .rename(columns={"emissions_tco2e": "total_emissions_tco2e"})
    )
    total_by_scenario = (
        by_year_total.groupby("Szenario", as_index=False)["total_emissions_tco2e"]
        .sum()
        .sort_values("total_emissions_tco2e")
    )
    return {
        "prepared_new_vehicles_lca": prepared_new,
        "prepared_final_stock_lca": prepared_stock,
        "prepared_exits_lca": prepared_exits,
        "emissions_by_phase": emissions,
        "emissions_summary_by_year_phase": by_year_phase,
        "emissions_summary_by_year_total": by_year_total,
        "emissions_summary_total_by_scenario": total_by_scenario,
        "parameter_long": parameter_long,
        "parameter_path": parameter_path,
        "historical_exits_path": historical_exits_path,
        "fleet_source": fleet_source,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def execute(
    config_path: str | Path,
    fleet_run_override: str | None = None,
    run_timestamp: datetime | None = None,
) -> Path:
    config_path = Path(config_path).resolve(strict=True)
    config = load_config(config_path)
    run_name = validate_run_name(config)
    ghg = config.get("ghg")
    if not isinstance(ghg, dict):
        raise ValueError("Config must contain a 'ghg' object.")
    configured_fleet_run = ghg.get("fleet_run")
    effective_fleet_run = fleet_run_override or configured_fleet_run
    if not isinstance(effective_fleet_run, str) or not effective_fleet_run:
        raise ValueError(
            "A fleet run must be supplied through --fleet-run or ghg.fleet_run."
        )
    effective_config = copy.deepcopy(config)
    effective_config["ghg"]["fleet_run"] = effective_fleet_run

    started_at = datetime.now(BERLIN_TIMEZONE)
    directory_timestamp = run_timestamp or started_at
    output_root = resolve_path(ghg["output_dir"], config_path.parent)
    output_dir = output_root / f"{run_name}__{directory_timestamp:%Y-%m-%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot_path = output_dir / "config.snapshot.json"
    manifest_path = output_dir / "run_manifest.json"
    shutil.copy2(config_path, snapshot_path)

    manifest = {
        "model": "ghg_model",
        "run_name": run_name,
        "status": "running",
        "timezone": "Europe/Berlin",
        "run_id_timestamp": iso_timestamp(directory_timestamp),
        "started_at": iso_timestamp(started_at),
        "finished_at": None,
        "config": {
            "source_path": str(config_path),
            "source_modified_at": iso_timestamp(
                datetime.fromtimestamp(config_path.stat().st_mtime, BERLIN_TIMEZONE)
            ),
            "sha256": file_sha256(config_path),
            "snapshot_filename": snapshot_path.name,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "model_script": str(Path(__file__).resolve()),
            "model_script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "fleet_input": {
            "configured_run_directory_name": configured_fleet_run,
            "run_directory_name": effective_fleet_run,
            "override_used": fleet_run_override is not None,
        },
        "emission_parameters": {},
        "output_directory": str(output_dir.resolve()),
        "result_files": [],
        "error": None,
    }
    write_manifest(manifest_path, manifest)

    try:
        results = calculate_ghg(effective_config)
        saved = []
        for table_name in [
            "prepared_new_vehicles_lca",
            "prepared_final_stock_lca",
            "prepared_exits_lca",
            "emissions_by_phase",
            "emissions_summary_by_year_phase",
            "emissions_summary_by_year_total",
            "emissions_summary_total_by_scenario",
        ]:
            output_path = output_dir / f"{run_name}__{table_name}.csv"
            results[table_name].to_csv(
                output_path, sep=";", encoding="utf-8-sig", index=False
            )
            saved.append(output_path)
        fleet_source = results["fleet_source"]
        manifest["fleet_input"].update(
            {
                "directory": str(fleet_source["directory"].resolve()),
                "manifest_sha256": file_sha256(fleet_source["manifest_path"]),
                "run_name": fleet_source["manifest"]["run_name"],
            }
        )
        parameter_path = results["parameter_path"]
        manifest["emission_parameters"] = {
            "path": str(parameter_path.resolve()),
            "sha256": file_sha256(parameter_path),
        }
        manifest["status"] = "success"
        manifest["result_files"] = [path.name for path in saved]
    except Exception as exc:
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

    print(f"GHG run directory: {output_dir.resolve()}")
    print(f"Fleet input run: {effective_fleet_run}")
    print("Saved result files:")
    for path in saved:
        print(f"- {path.name}")
    return output_dir.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fleet life-cycle GHG model.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--fleet-run",
        help="Fleet run directory name; overrides ghg.fleet_run from the config.",
    )
    args = parser.parse_args()
    execute(args.config, fleet_run_override=args.fleet_run)


if __name__ == "__main__":
    main()
