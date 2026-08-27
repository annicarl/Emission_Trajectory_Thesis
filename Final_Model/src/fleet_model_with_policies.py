from __future__ import annotations

import argparse
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


GROUP_COLS = ["Berichtsjahr", "Segment", "Antriebsart", "Jahr der Erstzulassung"]
STOCK_COLS = GROUP_COLS + ["Anzahl"]
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
BERLIN_TIMEZONE = ZoneInfo("Europe/Berlin")
FINAL_MODEL_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = FINAL_MODEL_DIR / "outputs" / "fleet_model"


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    config["_config_path"] = str(config_path.resolve())
    return config


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def report_year(series: pd.Series, stock_logic: bool = False) -> pd.Series:
    year_num = pd.to_numeric(series, errors="coerce")
    year_date = pd.to_datetime(series, errors="coerce", dayfirst=True).dt.year
    year = year_num.fillna(year_date)
    if stock_logic:
        year = year - 1
    return year.astype("Int64")


def safe_sum(series: pd.Series) -> float:
    return pd.to_numeric(series, errors="coerce").fillna(0).sum()


def contains_any_pattern(text: object, patterns: list[str]) -> bool:
    text_lower = str(text).lower()
    return any(str(pattern).lower() in text_lower for pattern in patterns)


def exclude_drive_types(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Remove out-of-scope drive types before any model calculations."""
    patterns = config.get("excluded_drive_patterns", [])
    if not isinstance(patterns, list):
        raise ValueError("excluded_drive_patterns must be a list.")
    if not patterns:
        return df.copy()
    excluded = df["Antriebsart"].apply(lambda value: contains_any_pattern(value, patterns))
    return df.loc[~excluded].copy()


def is_ice(drive_type: object, config: dict) -> bool:
    policies = config["policies"]
    ice_match = contains_any_pattern(drive_type, policies["ice_patterns"])
    non_ice_match = contains_any_pattern(drive_type, policies["non_ice_patterns"])
    return ice_match and not non_ice_match


def is_diesel(drive_type: object, config: dict) -> bool:
    return contains_any_pattern(drive_type, config["policies"]["diesel_patterns"])


def bev_target_share(year: float, scenario: str, config: dict) -> float:
    """Return the linearly interpolated BEV target as a share between zero and one."""
    target_scenarios = config["policies"].get("bev_target_scenarios", {})
    if not isinstance(target_scenarios, dict):
        raise ValueError("bev_target_scenarios must be an object of named target paths.")
    if scenario not in target_scenarios:
        raise ValueError(f"No BEV target path configured for scenario {scenario!r}.")
    raw_targets = target_scenarios[scenario]
    if not isinstance(raw_targets, dict) or not raw_targets:
        raise ValueError(f"BEV target scenario {scenario!r} must contain at least one year and target.")

    parsed_targets: dict[float, float] = {}
    for raw_year, raw_percent in raw_targets.items():
        try:
            target_year = float(raw_year)
            target_percent = float(raw_percent)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "BEV target years and percentages must be numeric."
            ) from exc
        if not np.isfinite(target_year) or not np.isfinite(target_percent):
            raise ValueError("BEV target years and percentages must be finite.")
        if target_percent < 0 or target_percent > 100:
            raise ValueError(
                f"BEV target for {raw_year} must be between 0 and 100 percent."
            )
        if target_year in parsed_targets:
            raise ValueError(f"Duplicate BEV target year after parsing: {raw_year}.")
        parsed_targets[target_year] = target_percent

    years = np.array(sorted(parsed_targets), dtype=float)
    percentages = np.array([parsed_targets[target_year] for target_year in years], dtype=float)
    return float(np.interp(float(year), years, percentages) / 100)


def policy_scenario_names(config: dict) -> list[str]:
    """Validate and combine fixed policies with named BEV target scenarios."""
    policies = config["policies"]
    fixed_scenarios = policies.get("scenarios", [])
    target_scenarios = policies.get("bev_target_scenarios", {})
    if not isinstance(fixed_scenarios, list):
        raise ValueError("policies.scenarios must be a list.")
    if not isinstance(target_scenarios, dict):
        raise ValueError("bev_target_scenarios must be an object of named target paths.")

    for name in [*fixed_scenarios, *target_scenarios]:
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("Scenario names must be non-empty strings without surrounding whitespace.")
    if len(fixed_scenarios) != len(set(fixed_scenarios)):
        raise ValueError("policies.scenarios contains duplicate scenario names.")
    collisions = set(fixed_scenarios).intersection(target_scenarios)
    if collisions:
        raise ValueError(
            "BEV target scenario names collide with fixed scenarios: "
            + ", ".join(sorted(collisions))
        )

    growth_overlays = {"abwrackpraemie", "diesel_fahrverbot"}.intersection(fixed_scenarios)
    reference_scenario = policies.get("bev_growth_reference_scenario")
    if growth_overlays and reference_scenario not in target_scenarios:
        raise ValueError(
            "bev_growth_reference_scenario must name a configured BEV target scenario "
            "when scrappage or diesel-ban policies are enabled."
        )

    for scenario in target_scenarios:
        bev_target_share(config["years"]["start_year"], scenario, config)
    return [*fixed_scenarios, *target_scenarios]


def apply_bev_target_distribution(
    base_distribution: pd.DataFrame,
    year: float,
    scenario: str,
    config: dict,
    excluded_non_bev_patterns: list[str] | None = None,
) -> pd.Series:
    """Allocate a BEV target while preserving 2025 proportions within both groups."""
    bev_patterns = config["policies"].get("bev_patterns")
    if not isinstance(bev_patterns, list) or not bev_patterns:
        raise ValueError("bev_patterns must contain at least one BEV identification pattern.")

    original = pd.to_numeric(base_distribution["anteil_original"], errors="coerce").fillna(0)
    bev_mask = base_distribution["Antriebsart"].apply(
        lambda drive_type: contains_any_pattern(drive_type, bev_patterns)
    )
    excluded_non_bev_patterns = excluded_non_bev_patterns or []
    excluded_mask = base_distribution["Antriebsart"].apply(
        lambda drive_type: contains_any_pattern(drive_type, excluded_non_bev_patterns)
    )
    eligible_non_bev_mask = ~bev_mask & ~excluded_mask
    bev_base_total = safe_sum(original[bev_mask])
    non_bev_base_total = safe_sum(original[eligible_non_bev_mask])
    target_share = bev_target_share(year, scenario, config)

    if bev_base_total <= 0 and target_share > 0:
        raise ValueError("The 2025 new-vehicle distribution contains no matching BEVs.")
    if non_bev_base_total <= 0 and target_share < 1:
        raise ValueError("The 2025 new-vehicle distribution contains no non-BEV vehicles.")

    adjusted = pd.Series(0.0, index=base_distribution.index, dtype=float)
    if target_share > 0:
        adjusted.loc[bev_mask] = original.loc[bev_mask] / bev_base_total * target_share
    if target_share < 1:
        adjusted.loc[eligible_non_bev_mask] = (
            original.loc[eligible_non_bev_mask]
            / non_bev_base_total
            * (1 - target_share)
        )
    return adjusted


def add_age_bin(df: pd.DataFrame, current_year: int, bin_width: int) -> pd.DataFrame:
    df = df.copy()
    df["Alter"] = current_year - df["Jahr der Erstzulassung"]
    df = df[df["Alter"] >= 0].copy()
    df["Altersklasse_Start"] = (df["Alter"] // bin_width) * bin_width
    df["Altersklasse_Ende"] = df["Altersklasse_Start"] + bin_width - 1
    df["Altersklasse"] = (
        df["Altersklasse_Start"].astype(int).astype(str)
        + "-"
        + df["Altersklasse_Ende"].astype(int).astype(str)
    )
    return df


def load_input_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_dir = Path(config["_config_path"]).parent
    data_dir = resolve_path(config["data_dir"], base_dir)
    sep = config["csv"]["separator"]
    encoding = config["csv"]["encoding"]

    abs_df = pd.read_csv(data_dir / config["input_files"]["abs"], sep=sep, encoding=encoding)
    bestand_df = pd.read_csv(data_dir / config["input_files"]["bestand"], sep=sep, encoding=encoding)
    nzl_df = pd.read_csv(data_dir / config["input_files"]["nzl"], sep=sep, encoding=encoding)

    abs_df["Berichtsjahr"] = report_year(abs_df["Berichtsjahr"], stock_logic=False)
    bestand_df["Berichtsjahr_Datum"] = bestand_df["Berichtsjahr"]
    bestand_df["Berichtsjahr"] = report_year(bestand_df["Berichtsjahr"], stock_logic=True)
    nzl_df["Berichtsjahr"] = report_year(nzl_df["Berichtsjahr"], stock_logic=False)

    for df in (abs_df, bestand_df):
        df["Jahr der Erstzulassung"] = pd.to_numeric(
            df["Jahr der Erstzulassung"], errors="coerce"
        ).astype("Int64")
        df["Anzahl"] = pd.to_numeric(df["Anzahl"], errors="coerce")
    nzl_df["Anzahl"] = pd.to_numeric(nzl_df["Anzahl"], errors="coerce")

    abs_df = exclude_drive_types(abs_df, config)
    bestand_df = exclude_drive_types(bestand_df, config)
    nzl_df = exclude_drive_types(nzl_df, config)

    return abs_df, bestand_df, nzl_df


def start_stock_total(bestand_df: pd.DataFrame, start_year: int) -> float:
    stock = bestand_df[["Berichtsjahr", "Anzahl"]].copy()
    stock["Anzahl"] = pd.to_numeric(stock["Anzahl"], errors="coerce")
    total = stock.loc[stock["Berichtsjahr"] == start_year, "Anzahl"].sum()
    if pd.isna(total) or total <= 0:
        raise ValueError(f"No valid stock found for start year {start_year}.")
    return float(total)


def create_fleet_target_path(
    bestand_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    percent_change: float,
    method: str,
) -> pd.DataFrame:
    start_stock = start_stock_total(bestand_df, start_year)
    end_stock = start_stock * (1 + percent_change / 100)
    years = np.arange(start_year, end_year + 1)
    periods = end_year - start_year

    if method == "linear":
        target = np.linspace(start_stock, end_stock, len(years))
    elif method == "growth":
        growth_rate = (end_stock / start_stock) ** (1 / periods) - 1
        target = start_stock * ((1 + growth_rate) ** np.arange(len(years)))
    else:
        raise ValueError("fleet target method must be 'linear' or 'growth'.")

    return pd.DataFrame({"Jahr": years, "gesamtbestand_target": target})


def calculate_hazard_rates(
    abs_df: pd.DataFrame, bestand_df: pd.DataFrame, calibration_years: int, bin_width: int
) -> pd.DataFrame:
    abs_work = abs_df[STOCK_COLS].copy()
    bestand_work = bestand_df[STOCK_COLS].copy()

    for df in (abs_work, bestand_work):
        df["Berichtsjahr"] = pd.to_numeric(df["Berichtsjahr"], errors="coerce")
        df["Jahr der Erstzulassung"] = pd.to_numeric(
            df["Jahr der Erstzulassung"], errors="coerce"
        )
        df["Anzahl"] = pd.to_numeric(df["Anzahl"], errors="coerce")
        df.dropna(subset=STOCK_COLS, inplace=True)
        df["Berichtsjahr"] = df["Berichtsjahr"].astype(int)
        df["Jahr der Erstzulassung"] = df["Jahr der Erstzulassung"].astype(int)

    latest_year = int(abs_work["Berichtsjahr"].max())
    base_years = range(latest_year - calibration_years + 1, latest_year + 1)
    abs_work = abs_work[abs_work["Berichtsjahr"].isin(base_years)].copy()
    bestand_work = bestand_work[bestand_work["Berichtsjahr"].isin(base_years)].copy()

    exits = (
        abs_work.groupby(GROUP_COLS, as_index=False)["Anzahl"]
        .sum()
        .rename(columns={"Anzahl": "abgaenge"})
    )
    stock = (
        bestand_work.groupby(GROUP_COLS, as_index=False)["Anzahl"]
        .sum()
        .rename(columns={"Anzahl": "bestand"})
    )

    hazard_base = exits.merge(stock, on=GROUP_COLS, how="inner")
    hazard_base["Alter"] = hazard_base["Berichtsjahr"] - hazard_base["Jahr der Erstzulassung"]
    hazard_base = hazard_base[
        (hazard_base["bestand"].notna())
        & (hazard_base["bestand"] > 0)
        & (hazard_base["Alter"] >= 0)
    ].copy()
    hazard_base["Altersklasse_Start"] = (hazard_base["Alter"] // bin_width) * bin_width
    hazard_base["Altersklasse_Ende"] = hazard_base["Altersklasse_Start"] + bin_width - 1
    hazard_base["Altersklasse"] = (
        hazard_base["Altersklasse_Start"].astype(str)
        + "-"
        + hazard_base["Altersklasse_Ende"].astype(str)
    )

    pooled = (
        hazard_base.groupby(
            ["Segment", "Antriebsart", "Altersklasse_Start", "Altersklasse_Ende", "Altersklasse"],
            as_index=False,
        )
        .agg(
            summe_abgaenge=("abgaenge", "sum"),
            summe_bestand=("bestand", "sum"),
            anzahl_betrachtete_jahre=("Berichtsjahr", "nunique"),
        )
        .sort_values(["Segment", "Antriebsart", "Altersklasse_Start"])
        .reset_index(drop=True)
    )
    pooled["pooled_hazard_rate"] = pooled["summe_abgaenge"] / pooled["summe_bestand"]
    return pooled


def calculate_reentry_rates(
    abs_df: pd.DataFrame,
    nzl_df: pd.DataFrame,
    bestand_df: pd.DataFrame,
    calibration_years: int,
    bin_width: int,
    max_reentry_rate: float,
    max_age: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    abs_work = abs_df[STOCK_COLS].copy()
    abs_work["Anzahl"] = pd.to_numeric(abs_work["Anzahl"], errors="coerce")
    abs_work = abs_work.dropna(subset=STOCK_COLS)
    exits = (
        abs_work.groupby(GROUP_COLS, as_index=False)["Anzahl"]
        .sum()
        .rename(columns={"Anzahl": "exits"})
    )

    nzl_work = nzl_df[["Berichtsjahr", "Segment", "Antriebsart", "Anzahl"]].copy()
    nzl_work["Anzahl"] = pd.to_numeric(nzl_work["Anzahl"], errors="coerce")
    nzl_work["Jahr der Erstzulassung"] = nzl_work["Berichtsjahr"]
    nzl_work = nzl_work.dropna(subset=STOCK_COLS)
    entries = (
        nzl_work.groupby(GROUP_COLS, as_index=False)["Anzahl"]
        .sum()
        .rename(columns={"Anzahl": "entries"})
    )

    bestand_work = bestand_df[STOCK_COLS].copy()
    bestand_work["Anzahl"] = pd.to_numeric(bestand_work["Anzahl"], errors="coerce")
    bestand_work = bestand_work.dropna(subset=STOCK_COLS)
    stock = (
        bestand_work.groupby(GROUP_COLS, as_index=False)["Anzahl"]
        .sum()
        .rename(columns={"Anzahl": "bestand"})
    )

    stock_t = stock.rename(columns={"bestand": "bestand_t"}).copy()
    stock_t_plus_1 = stock.copy()
    stock_t_plus_1["Berichtsjahr"] = stock_t_plus_1["Berichtsjahr"] - 1
    stock_t_plus_1 = stock_t_plus_1.rename(columns={"bestand": "bestand_t_plus_1"})

    balance = stock_t.merge(stock_t_plus_1, on=GROUP_COLS, how="inner")
    balance = balance.merge(exits, on=GROUP_COLS, how="left")
    balance = balance.merge(entries, on=GROUP_COLS, how="left")
    balance["exits"] = balance["exits"].fillna(0)
    balance["entries"] = balance["entries"].fillna(0)
    balance["reentries_raw"] = (
        balance["bestand_t_plus_1"] - balance["bestand_t"] + balance["exits"] - balance["entries"]
    )
    balance["reentry_raw_negativ"] = balance["reentries_raw"] < 0
    balance["reentries"] = balance["reentries_raw"].clip(lower=0)
    balance["Alter"] = balance["Berichtsjahr"] - balance["Jahr der Erstzulassung"]
    balance = balance[(balance["Alter"] >= 0) & (balance["Alter"] <= max_age)].copy()
    balance["Altersklasse_Start"] = (balance["Alter"] // bin_width) * bin_width
    balance["Altersklasse_Ende"] = balance["Altersklasse_Start"] + bin_width - 1
    balance["Altersklasse"] = (
        balance["Altersklasse_Start"].astype(str)
        + "-"
        + balance["Altersklasse_Ende"].astype(str)
    )

    yearly = (
        balance.groupby(
            ["Berichtsjahr", "Segment", "Antriebsart", "Altersklasse_Start", "Altersklasse_Ende", "Altersklasse"],
            as_index=False,
        )
        .agg(
            summe_exits=("exits", "sum"),
            summe_reentries=("reentries", "sum"),
            summe_reentries_raw=("reentries_raw", "sum"),
            anzahl_negative_reentry_raw=("reentry_raw_negativ", "sum"),
        )
        .sort_values(["Berichtsjahr", "Segment", "Antriebsart", "Altersklasse_Start"])
        .reset_index(drop=True)
    )
    yearly["reentry_quote_exits"] = np.where(
        yearly["summe_exits"] > 0, yearly["summe_reentries"] / yearly["summe_exits"], 0
    )
    yearly["reentry_quote_exits"] = yearly["reentry_quote_exits"].clip(
        lower=0, upper=max_reentry_rate
    )

    latest_year = int(yearly["Berichtsjahr"].max())
    calibration_window = range(latest_year - calibration_years + 1, latest_year + 1)
    calibration_base = yearly[yearly["Berichtsjahr"].isin(calibration_window)].copy()

    pooled = (
        calibration_base.groupby(
            ["Segment", "Antriebsart", "Altersklasse_Start", "Altersklasse_Ende", "Altersklasse"],
            as_index=False,
        )
        .agg(
            summe_reentries=("summe_reentries", "sum"),
            summe_exits=("summe_exits", "sum"),
            anzahl_betrachtete_jahre=("Berichtsjahr", "nunique"),
        )
        .sort_values(["Segment", "Antriebsart", "Altersklasse_Start"])
        .reset_index(drop=True)
    )
    pooled["pooled_reentry_quote_exits"] = np.where(
        pooled["summe_exits"] > 0, pooled["summe_reentries"] / pooled["summe_exits"], 0
    )
    pooled["pooled_reentry_quote_exits"] = pooled["pooled_reentry_quote_exits"].clip(
        lower=0, upper=max_reentry_rate
    )
    return yearly, pooled


def adjusted_nzl_distribution(base_distribution: pd.DataFrame, year: int, scenario: str, config: dict) -> pd.DataFrame:
    policies = config["policies"]
    dist = base_distribution.copy()
    dist["anteil_original"] = pd.to_numeric(dist["anteil_original"], errors="coerce").fillna(0)
    dist["anteil_szenario"] = dist["anteil_original"]

    if scenario in {"diesel_fahrverbot", "abwrackpraemie"}:
        reference_scenario = policies["bev_growth_reference_scenario"]
        excluded_patterns = (
            policies["diesel_patterns"]
            if scenario == "diesel_fahrverbot" and year >= policies["diesel_ban_year"]
            else None
        )
        dist["anteil_szenario"] = apply_bev_target_distribution(
            dist,
            year,
            reference_scenario,
            config,
            excluded_non_bev_patterns=excluded_patterns,
        )
    elif scenario in policies.get("bev_target_scenarios", {}):
        dist["anteil_szenario"] = apply_bev_target_distribution(
            dist, year, scenario, config
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    share_sum = safe_sum(dist["anteil_szenario"])
    if share_sum <= 0:
        raise ValueError(f"No valid new-vehicle distribution for {scenario} in {year}.")
    dist["anteil_szenario"] = dist["anteil_szenario"] / share_sum
    return dist[
        ["Segment", "Antriebsart", "ist_verbrenner", "ist_diesel", "anteil_original", "anteil_szenario"]
    ].copy()


def prepare_rates(hazard_pooled_df: pd.DataFrame, reentry_pooled_df: pd.DataFrame, config: dict, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = config["calibration"]
    hazard = hazard_pooled_df[
        ["Segment", "Antriebsart", "Altersklasse_Start", "Altersklasse_Ende", "pooled_hazard_rate"]
    ].copy()
    hazard["pooled_hazard_rate"] = pd.to_numeric(
        hazard["pooled_hazard_rate"], errors="coerce"
    ).fillna(0).clip(lower=0, upper=calibration["max_hazard_rate"])
    hazard["Altersklasse_Start"] = pd.to_numeric(hazard["Altersklasse_Start"], errors="coerce").astype("Int64")
    hazard["Altersklasse_Ende"] = pd.to_numeric(hazard["Altersklasse_Ende"], errors="coerce").astype("Int64")
    hazard = hazard.dropna()

    reentry = reentry_pooled_df[
        ["Segment", "Antriebsart", "Altersklasse_Start", "Altersklasse_Ende", "pooled_reentry_quote_exits"]
    ].copy()
    reentry["pooled_reentry_quote_exits"] = pd.to_numeric(
        reentry["pooled_reentry_quote_exits"], errors="coerce"
    ).fillna(0).clip(lower=0, upper=calibration["max_reentry_rate"])
    reentry["Altersklasse_Start"] = pd.to_numeric(reentry["Altersklasse_Start"], errors="coerce").astype("Int64")
    reentry["Altersklasse_Ende"] = pd.to_numeric(reentry["Altersklasse_Ende"], errors="coerce").astype("Int64")
    reentry = reentry.dropna()
    return hazard, reentry


def apply_scrappage_policy(hazard_rate: pd.Series, drive_type: pd.Series, year: int, scenario: str, config: dict) -> pd.Series:
    """Increase hazards for configured drive types in configured destination years."""
    policies = config["policies"]
    active_years = {int(active_year) for active_year in policies["scrappage_active_years"]}
    adjusted = pd.to_numeric(hazard_rate, errors="coerce").fillna(0)

    if scenario != "abwrackpraemie" or year not in active_years:
        return adjusted

    targets = policies["scrappage_target_patterns"]
    multiplier = policies["scrappage_hazard_multiplier"]
    target_mask = drive_type.apply(lambda value: contains_any_pattern(value, targets))
    return adjusted.where(~target_mask, adjusted * multiplier).clip(
        lower=0, upper=config["calibration"]["max_hazard_rate"]
    )


def run_policy_scenario(
    scenario: str,
    config: dict,
    bestand_df: pd.DataFrame,
    nzl_df: pd.DataFrame,
    hazard_pooled_df: pd.DataFrame,
    reentry_pooled_df: pd.DataFrame,
    fleet_total_df: pd.DataFrame,
) -> dict[str, object]:
    start_year = config["years"]["start_year"]
    end_year = config["years"]["end_year"]
    bin_width = config["calibration"]["age_bin_width"]
    hazard_rates, reentry_rates = prepare_rates(hazard_pooled_df, reentry_pooled_df, config, scenario)

    nzl_start = nzl_df[nzl_df["Berichtsjahr"] == start_year].copy()
    nzl_base = (
        nzl_start.groupby(["Segment", "Antriebsart"], as_index=False)["Anzahl"]
        .sum()
        .rename(columns={"Anzahl": "nzl_startjahr"})
    )
    nzl_total = safe_sum(nzl_base["nzl_startjahr"])
    if nzl_total <= 0:
        raise ValueError(f"No valid new registrations found for start year {start_year}.")
    nzl_base["anteil_original"] = nzl_base["nzl_startjahr"] / nzl_total
    nzl_base["ist_verbrenner"] = nzl_base["Antriebsart"].apply(lambda x: is_ice(x, config))
    nzl_base["ist_diesel"] = nzl_base["Antriebsart"].apply(lambda x: is_diesel(x, config))
    nzl_base = nzl_base[["Segment", "Antriebsart", "ist_verbrenner", "ist_diesel", "anteil_original"]].copy()

    target_work = fleet_total_df.copy()
    target_work["Jahr"] = pd.to_numeric(target_work["Jahr"], errors="coerce").astype("Int64")
    target_work["gesamtbestand_target"] = pd.to_numeric(
        target_work["gesamtbestand_target"], errors="coerce"
    )
    target_work = target_work.dropna(subset=["Jahr", "gesamtbestand_target"])
    target_by_year = dict(zip(target_work["Jahr"].astype(int), target_work["gesamtbestand_target"]))
    missing_years = [year for year in range(start_year, end_year + 1) if year not in target_by_year]
    if missing_years:
        raise ValueError("Missing target years: " + ", ".join(map(str, missing_years)))

    stock_work = bestand_df[STOCK_COLS].copy().dropna(subset=STOCK_COLS)
    current_stock = (
        stock_work[stock_work["Berichtsjahr"] == start_year]
        .groupby(["Segment", "Antriebsart", "Jahr der Erstzulassung"], as_index=False)["Anzahl"]
        .sum()
        .rename(columns={"Anzahl": "bestand"})
    )
    if current_stock.empty:
        raise ValueError(f"No valid stock found for start year {start_year}.")
    current_stock["bestand"] = pd.to_numeric(current_stock["bestand"], errors="coerce").fillna(0)
    current_stock = current_stock[current_stock["bestand"] > 0].copy()

    by_year: dict[int, dict[str, pd.DataFrame]] = {}
    summary_rows: list[dict[str, float | int | str]] = []

    by_year[start_year] = {
        "abgaenge": pd.DataFrame(),
        "rueckkehrer": pd.DataFrame(),
        "bestand_vor_neuen_fahrzeugen": pd.DataFrame(),
        "fehlende_fahrzeuge": pd.DataFrame(
            {"Berichtsjahr": [start_year], "fehlende_fahrzeuge": [max(target_by_year[start_year] - safe_sum(current_stock["bestand"]), 0)]}
        ),
        "neue_fahrzeuge": pd.DataFrame(),
        "finaler_bestand": current_stock.assign(Berichtsjahr=start_year)
        .rename(columns={"bestand": "finaler_bestand"})[
            ["Berichtsjahr", "Segment", "Antriebsart", "Jahr der Erstzulassung", "finaler_bestand"]
        ]
        .copy(),
    }
    summary_rows.append(
        {
            "Szenario": scenario,
            "Jahr": start_year,
            "zielbestand": target_by_year[start_year],
            "bestand_vor_neuen_fahrzeugen": safe_sum(current_stock["bestand"]),
            "fehlende_fahrzeuge": max(target_by_year[start_year] - safe_sum(current_stock["bestand"]), 0),
            "finaler_bestand": safe_sum(current_stock["bestand"]),
        }
    )

    for year in range(start_year, end_year):
        next_year = year + 1
        projection = add_age_bin(current_stock, current_year=year, bin_width=bin_width)
        projection = projection.merge(
            hazard_rates,
            on=["Segment", "Antriebsart", "Altersklasse_Start", "Altersklasse_Ende"],
            how="left",
        )
        projection = projection.merge(
            reentry_rates,
            on=["Segment", "Antriebsart", "Altersklasse_Start", "Altersklasse_Ende"],
            how="left",
        )
        projection["pooled_hazard_rate"] = projection["pooled_hazard_rate"].fillna(0).clip(
            lower=0, upper=config["calibration"]["max_hazard_rate"]
        )
        projection["pooled_hazard_rate"] = apply_scrappage_policy(
            projection["pooled_hazard_rate"],
            projection["Antriebsart"],
            next_year,
            scenario,
            config,
        )
        projection["pooled_reentry_quote_exits"] = projection["pooled_reentry_quote_exits"].fillna(0).clip(
            lower=0, upper=config["calibration"]["max_reentry_rate"]
        )
        projection["exits"] = projection["bestand"] * projection["pooled_hazard_rate"]
        projection["reentries"] = projection["exits"] * projection["pooled_reentry_quote_exits"]
        projection["bestand_vor_neuen_fahrzeugen"] = (
            projection["bestand"] - projection["exits"] + projection["reentries"]
        ).clip(lower=0)
        projection["policy_forced_exits"] = 0.0

        if scenario == "diesel_fahrverbot" and next_year >= config["policies"]["diesel_ban_year"]:
            projection["ist_diesel"] = projection["Antriebsart"].apply(lambda x: is_diesel(x, config))
            # The diesel ban removes the diesel stock remaining after regular
            # hazard-based exits and reentries. Record these forced removals as
            # exits as well, so downstream end-of-life accounting includes them.
            projection["policy_forced_exits"] = projection[
                "bestand_vor_neuen_fahrzeugen"
            ].where(projection["ist_diesel"], 0.0)
            projection["exits"] = (
                projection["exits"] + projection["policy_forced_exits"]
            )
            projection["bestand_vor_neuen_fahrzeugen"] = np.where(
                projection["ist_diesel"], 0, projection["bestand_vor_neuen_fahrzeugen"]
            )

        existing_stock = safe_sum(projection["bestand_vor_neuen_fahrzeugen"])
        missing_vehicles = max(target_by_year[next_year] - existing_stock, 0)

        nzl_distribution = adjusted_nzl_distribution(nzl_base, next_year, scenario, config)
        new_vehicles = nzl_distribution.copy()
        new_vehicles["Jahr der Erstzulassung"] = next_year
        new_vehicles["neue_fahrzeuge"] = missing_vehicles * new_vehicles["anteil_szenario"]

        current_stock = pd.concat(
            [
                projection[["Segment", "Antriebsart", "Jahr der Erstzulassung", "bestand_vor_neuen_fahrzeugen"]]
                .rename(columns={"bestand_vor_neuen_fahrzeugen": "bestand"}),
                new_vehicles[["Segment", "Antriebsart", "Jahr der Erstzulassung", "neue_fahrzeuge"]]
                .rename(columns={"neue_fahrzeuge": "bestand"}),
            ],
            ignore_index=True,
        )
        current_stock["bestand"] = pd.to_numeric(current_stock["bestand"], errors="coerce").fillna(0)
        current_stock = (
            current_stock.groupby(["Segment", "Antriebsart", "Jahr der Erstzulassung"], as_index=False)["bestand"]
            .sum()
        )
        current_stock = current_stock[current_stock["bestand"] > 0].copy()
        final_stock = safe_sum(current_stock["bestand"])

        by_year[next_year] = {
            "abgaenge": projection.assign(Berichtsjahr=next_year)[
                [
                    "Berichtsjahr",
                    "Segment",
                    "Antriebsart",
                    "Jahr der Erstzulassung",
                    "exits",
                    "policy_forced_exits",
                ]
            ].copy(),
            "rueckkehrer": projection.assign(Berichtsjahr=next_year)[
                ["Berichtsjahr", "Segment", "Antriebsart", "Jahr der Erstzulassung", "reentries"]
            ].copy(),
            "bestand_vor_neuen_fahrzeugen": projection.assign(Berichtsjahr=next_year)[
                ["Berichtsjahr", "Segment", "Antriebsart", "Jahr der Erstzulassung", "bestand_vor_neuen_fahrzeugen"]
            ].copy(),
            "fehlende_fahrzeuge": pd.DataFrame(
                {"Berichtsjahr": [next_year], "fehlende_fahrzeuge": [missing_vehicles]}
            ),
            "neue_fahrzeuge": new_vehicles.assign(Berichtsjahr=next_year)[
                [
                    "Berichtsjahr",
                    "Segment",
                    "Antriebsart",
                    "Jahr der Erstzulassung",
                    "ist_verbrenner",
                    "ist_diesel",
                    "anteil_original",
                    "anteil_szenario",
                    "neue_fahrzeuge",
                ]
            ].copy(),
            "finaler_bestand": current_stock.assign(Berichtsjahr=next_year)
            .rename(columns={"bestand": "finaler_bestand"})[
                ["Berichtsjahr", "Segment", "Antriebsart", "Jahr der Erstzulassung", "finaler_bestand"]
            ]
            .copy(),
        }
        summary_rows.append(
            {
                "Szenario": scenario,
                "Jahr": next_year,
                "zielbestand": target_by_year[next_year],
                "bestand_vor_neuen_fahrzeugen": existing_stock,
                "fehlende_fahrzeuge": missing_vehicles,
                "finaler_bestand": final_stock,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary["abweichung_final_vs_ziel"] = summary["finaler_bestand"] - summary["zielbestand"]
    return {"results_by_year": by_year, "summary": summary}


def concat_yearly_outputs(outputs: dict[str, dict], table_name: str) -> pd.DataFrame:
    frames = []
    for scenario, output in outputs.items():
        for year, tables in output["results_by_year"].items():
            table = tables.get(table_name)
            if table is not None and not table.empty:
                frames.append(table.assign(Szenario=scenario, Jahr=year))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_model(config: dict) -> dict[str, object]:
    abs_df, bestand_df, nzl_df = load_input_data(config)
    start_year = config["years"]["start_year"]
    end_year = config["years"]["end_year"]
    calibration = config["calibration"]

    fleet_total_df = create_fleet_target_path(
        bestand_df,
        start_year=start_year,
        end_year=end_year,
        percent_change=config["fleet_target"]["percent_change"],
        method=config["fleet_target"]["method"],
    )
    hazard_pooled_df = calculate_hazard_rates(
        abs_df, bestand_df, calibration["years"], calibration["age_bin_width"]
    )
    reentry_yearly_df, reentry_pooled_df = calculate_reentry_rates(
        abs_df,
        nzl_df,
        bestand_df,
        calibration["years"],
        calibration["age_bin_width"],
        calibration["max_reentry_rate"],
        calibration["max_reentry_age"],
    )

    policy_outputs = {
        scenario: run_policy_scenario(
            scenario, config, bestand_df, nzl_df, hazard_pooled_df, reentry_pooled_df, fleet_total_df
        )
        for scenario in policy_scenario_names(config)
    }

    fleet_size_outputs = {}
    fleet_size_targets = {}
    fleet_size_config = config.get("fleet_size_scenarios", {})
    if fleet_size_config.get("enabled", False):
        policy_mode = fleet_size_config["policy_mode"]
        for scenario_name, percent_change in fleet_size_config["changes_percent"].items():
            target_path = create_fleet_target_path(
                bestand_df,
                start_year=start_year,
                end_year=end_year,
                percent_change=percent_change,
                method=fleet_size_config["target_method"],
            )
            output = run_policy_scenario(
                policy_mode, config, bestand_df, nzl_df, hazard_pooled_df, reentry_pooled_df, target_path
            )
            output["percent_change_2050"] = percent_change
            output["policy_mode"] = policy_mode
            fleet_size_outputs[scenario_name] = output
            fleet_size_targets[scenario_name] = target_path.assign(Szenario=scenario_name)

    return {
        "fleet_total_df": fleet_total_df,
        "hazard_pooled_df": hazard_pooled_df,
        "reentry_yearly_df": reentry_yearly_df,
        "reentry_pooled_df": reentry_pooled_df,
        "policy_outputs": policy_outputs,
        "fleet_size_outputs": fleet_size_outputs,
        "fleet_size_targets": fleet_size_targets,
    }


def validate_run_name(config: dict) -> str:
    run_name = config.get("name")
    if not isinstance(run_name, str) or not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "Config field 'name' must start with a letter or number and contain "
            "only letters, numbers, dots, underscores, and hyphens."
        )
    return run_name


def save_results(
    results: dict[str, object],
    config: dict,
    results_dir: Path | None = None,
    filename_prefix: str | None = None,
) -> dict[str, Path]:
    if results_dir is None:
        base_dir = Path(config["_config_path"]).parent
        results_dir = resolve_path(config["results_dir"], base_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    prefix = f"{filename_prefix}__" if filename_prefix else ""

    tables = {
        "fleet_total": results["fleet_total_df"],
        "hazard_pooled": results["hazard_pooled_df"],
        "reentry_yearly": results["reentry_yearly_df"],
        "reentry_pooled": results["reentry_pooled_df"],
        "policy_summary_all": pd.concat(
            [output["summary"] for output in results["policy_outputs"].values()], ignore_index=True
        ),
        "policy_final_stock_all": concat_yearly_outputs(results["policy_outputs"], "finaler_bestand"),
        "policy_exits_all": concat_yearly_outputs(results["policy_outputs"], "abgaenge"),
        "policy_reentries_all": concat_yearly_outputs(results["policy_outputs"], "rueckkehrer"),
        "policy_stock_before_new_vehicles_all": concat_yearly_outputs(
            results["policy_outputs"], "bestand_vor_neuen_fahrzeugen"
        ),
        "policy_missing_vehicles_all": concat_yearly_outputs(results["policy_outputs"], "fehlende_fahrzeuge"),
        "policy_new_vehicles_all": concat_yearly_outputs(results["policy_outputs"], "neue_fahrzeuge"),
    }

    if results["fleet_size_outputs"]:
        tables.update(
            {
                "fleet_size_summary_all": pd.concat(
                    [
                        output["summary"].assign(
                            Fleet_Size_Szenario=name,
                            percent_change_2050=output["percent_change_2050"],
                            policy_mode=output["policy_mode"],
                        )
                        for name, output in results["fleet_size_outputs"].items()
                    ],
                    ignore_index=True,
                ),
                "fleet_size_target_paths_all": pd.concat(
                    results["fleet_size_targets"].values(), ignore_index=True
                ),
                "fleet_size_final_stock_all": concat_yearly_outputs(
                    results["fleet_size_outputs"], "finaler_bestand"
                ),
                "fleet_size_new_vehicles_all": concat_yearly_outputs(
                    results["fleet_size_outputs"], "neue_fahrzeuge"
                ),
            }
        )

    for name, df in tables.items():
        path = results_dir / f"{prefix}{name}.csv"
        df.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
        saved[name] = path
    return saved


def iso_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_run_directory(run_name: str, started_at: datetime) -> Path:
    run_directory = OUTPUT_ROOT / f"{run_name}__{started_at:%Y-%m-%d_%H%M%S}"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def execute(config_path: str | Path, run_timestamp: datetime | None = None) -> Path:
    config_path = Path(config_path).resolve(strict=True)
    config = load_config(config_path)
    run_name = validate_run_name(config)
    started_at = datetime.now(BERLIN_TIMEZONE)
    directory_timestamp = run_timestamp or started_at
    run_directory = create_run_directory(run_name, directory_timestamp)
    snapshot_path = run_directory / "config.snapshot.json"
    manifest_path = run_directory / "run_manifest.json"
    shutil.copy2(config_path, snapshot_path)

    manifest = {
        "model": "fleet_model",
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
            "snapshot_filename": snapshot_path.name,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "model_script": str(Path(__file__).resolve()),
            "model_script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "output_directory": str(run_directory.resolve()),
        "result_files": [],
        "error": None,
    }
    write_manifest(manifest_path, manifest)

    try:
        results = run_model(config)
        saved = save_results(
            results,
            config,
            results_dir=run_directory,
            filename_prefix=run_name,
        )
        manifest["status"] = "success"
        manifest["result_files"] = [path.name for path in saved.values()]
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

    print(f"Run directory: {run_directory.resolve()}")
    print("Saved result files:")
    for name, path in saved.items():
        print(f"- {name}: {path}")
    return run_directory.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fleet model policy scenarios.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to JSON config file.",
    )
    args = parser.parse_args()
    execute(args.config)


if __name__ == "__main__":
    main()
