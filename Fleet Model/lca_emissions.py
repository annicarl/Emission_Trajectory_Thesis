"""Reusable life-cycle emission calculation for the fleet model.

This module is the programmatic equivalent of ``GHG_Model_Annika.ipynb``.
It deliberately accepts in-memory fleet results and LCA data so sensitivity
runs never have to overwrite the original input files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


SEGMENT_TO_SIZE_CLASS = {
    "Minis": "Small", "Kleinwagen": "Small",
    "Kompaktklasse": "Medium", "Mittelklasse": "Medium",
    "Mini-Vans": "Medium", "Sonstige": "Medium",
    "Obere Mittelklasse": "Large", "Oberklasse": "Large",
    "SUVs": "Large", "Geländewagen": "Large", "GelÃ¤ndewagen": "Large",
    "Sportwagen": "Large", "Großraum-Vans": "Large", "GroÃŸraum-Vans": "Large",
    "Utilities": "Large", "Wohnmobile": "Large",
}
DRIVE_TO_GROUP = {
    "Elektro (BEV)": "BEV", "Benzin": "ICE_Petrol", "Diesel": "ICE_Diesel",
    "Hybrid": "ICE_Petrol", "Erdgas (CNG) (einschl. bivalent)": "ICE_Petrol",
    "Flüssiggas (LPG) (einschl. bivalent)": "ICE_Petrol",
    "FlÃ¼ssiggas (LPG) (einschl. bivalent)": "ICE_Petrol", "Sonstige": "ICE_Petrol",
}
PARAM_DRIVE = {"BEV": "BEV", "ICEVs-Petrol": "ICE_Petrol", "ICEVs-Diesel": "ICE_Diesel"}
PHASES = ["vehicle_production", "battery_production", "use", "energy_supply", "end_of_life"]


def parse_number(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none"}:
        return np.nan
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    return float(text)


def load_lca_wide(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    required = {"Parameter", "Explanation_parameter"}
    if not required.issubset(frame.columns):
        raise ValueError(f"LCA file lacks columns: {sorted(required - set(frame.columns))}")
    return frame


def lca_long(frame: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in frame.columns if c not in {"Parameter", "Explanation_parameter"}]
    out = frame.melt(["Parameter", "Explanation_parameter"], value_cols,
                     var_name="vehicle_type", value_name="raw_value")
    out["value"] = out["raw_value"].map(parse_number)
    split = out["vehicle_type"].str.rsplit("_", n=1, expand=True)
    out["drivetrain_group"] = split[0].map(PARAM_DRIVE)
    out["size_class"] = split[1].str.lower().map({"small": "Small", "medium": "Medium", "large": "Large"})
    return out


def parameter_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    return (lca_long(frame).pivot_table(index=["drivetrain_group", "size_class"],
            columns="Parameter", values="value", aggfunc="first").reset_index().rename_axis(None, axis=1))


def _map_vehicle(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["size_class"] = out["Segment"].map(SEGMENT_TO_SIZE_CLASS)
    out["drivetrain_group"] = out["Antriebsart"].map(DRIVE_TO_GROUP)
    missing = out[out[["size_class", "drivetrain_group"]].isna().any(axis=1)]
    if not missing.empty:
        examples = missing[["Segment", "Antriebsart"]].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"Missing LCA mapping for vehicle categories: {examples}")
    return out


def build_activity(fleet_results: dict, scenario: str) -> dict[str, pd.DataFrame]:
    """Aggregate fleet output once; LCA OAT runs then only apply new factors."""
    try:
        result = fleet_results["policy_outputs"][scenario]
    except KeyError as exc:
        raise ValueError(f"Scenario {scenario!r} is absent from fleet results") from exc

    def collect(table: str, value: str) -> pd.DataFrame:
        frames = []
        for year, tables in result["results_by_year"].items():
            df = tables.get(table)
            if df is not None and not df.empty:
                frames.append(df.assign(Jahr=year))
        if not frames:
            return pd.DataFrame(columns=["Jahr", "Segment", "Antriebsart", "size_class", "drivetrain_group", value])
        df = _map_vehicle(pd.concat(frames, ignore_index=True))
        df[value] = pd.to_numeric(df[value], errors="coerce").fillna(0.0)
        keys = ["Jahr", "Segment", "Antriebsart", "size_class", "drivetrain_group"]
        return df.groupby(keys, as_index=False)[value].sum()

    return {
        "new": collect("neue_fahrzeuge", "neue_fahrzeuge"),
        "stock": collect("finaler_bestand", "finaler_bestand"),
        "exits": collect("abgaenge", "exits"),
    }


def calculate_emissions(activity: dict[str, pd.DataFrame], lca_wide: pd.DataFrame,
                        annual_km: float = 12_500, offsets: dict | None = None) -> pd.DataFrame:
    offsets = {"production": -1, "use": 0, "end_of_life": 1, **(offsets or {})}
    lookup = parameter_lookup(lca_wide)

    def joined(name: str) -> pd.DataFrame:
        return activity[name].merge(lookup, on=["drivetrain_group", "size_class"], how="left")

    base_cols = ["emission_year", "phase", "Segment", "Antriebsart", "drivetrain_group", "size_class", "emissions_tco2e"]
    parts = []
    prod = joined("new")
    prod["emission_year"] = prod["Jahr"] + offsets["production"]
    prod["phase"] = "vehicle_production"
    prod["emissions_tco2e"] = prod["neue_fahrzeuge"] * prod.get("Vehicle_production", 0).fillna(0)
    parts.append(prod[base_cols])

    # No battery-production intensity exists in the supplied LCA file. Keep an
    # explicit zero phase; Battery_capacity consequently has zero sensitivity.
    battery = prod[base_cols].copy()
    battery["phase"] = "battery_production"
    battery["emissions_tco2e"] = 0.0
    parts.append(battery)

    use = joined("stock")
    for name in ["Consumption", "WTT", "TTW", "Emissions_electricity", "Maintenance"]:
        if name not in use:
            use[name] = 0.0
        use[name] = use[name].fillna(0.0)
    use["emission_year"] = use["Jahr"] + offsets["use"]
    bev = use["drivetrain_group"].eq("BEV")
    use_phase = use.copy()
    use_phase["phase"] = "use"
    use_phase["emissions_tco2e"] = use_phase["finaler_bestand"] * annual_km * (
        np.where(bev, 0.0, use_phase["Consumption"] / 100 * use_phase["TTW"]) + use_phase["Maintenance"]
    ) / 1_000_000
    parts.append(use_phase[base_cols])

    supply = use.copy()
    supply["phase"] = "energy_supply"
    supply["emissions_tco2e"] = supply["finaler_bestand"] * annual_km * np.where(
        bev, supply["Consumption"] / 100 * supply["Emissions_electricity"],
        supply["Consumption"] / 100 * supply["WTT"]
    ) / 1_000_000
    parts.append(supply[base_cols])

    eol = joined("exits")
    eol["emission_year"] = eol["Jahr"] + offsets["end_of_life"]
    eol["phase"] = "end_of_life"
    eol["emissions_tco2e"] = eol["exits"] * eol.get("End_of_life", 0).fillna(0)
    parts.append(eol[base_cols])
    out = pd.concat(parts, ignore_index=True)
    out["emissions_tco2e"] = out["emissions_tco2e"].fillna(0.0)
    return out


def summarize(emissions: pd.DataFrame, start_year: int, end_year: int, target_year: int) -> dict:
    data = emissions[emissions["emission_year"].between(start_year, end_year)].copy()
    annual = data.groupby("emission_year")["emissions_tco2e"].sum().sort_index()
    by_phase = data.groupby("phase")["emissions_tco2e"].sum().reindex(PHASES, fill_value=0.0)
    by_drive = data.groupby("Antriebsart")["emissions_tco2e"].sum()
    by_segment = data.groupby("Segment")["emissions_tco2e"].sum()
    total = float(annual.sum())
    return {
        "cumulative_total_tco2e": total,
        "target_year_emissions_tco2e": float(annual.get(target_year, 0.0)),
        "peak_annual_emissions_tco2e": float(annual.max()) if len(annual) else 0.0,
        "peak_year": int(annual.idxmax()) if len(annual) else None,
        "by_phase": by_phase.to_dict(), "by_drive": by_drive.to_dict(),
        "by_segment": by_segment.to_dict(), "annual": annual.to_dict(),
    }


def infer_unit(parameter: str, explanation: str) -> str:
    match = re.search(r"\[([^]]+)]", str(explanation))
    if match:
        return match.group(1)
    defaults = {"Battery_capacity": "kWh", "End_of_life": "tCO2e/vehicle"}
    return defaults.get(parameter, "unknown")

