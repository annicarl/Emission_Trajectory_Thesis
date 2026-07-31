"""One-at-a-time sensitivity analysis for fleet life-cycle emissions.

The existing fleet model is loaded and called at ``run_fleet`` below. Inputs
are deep-copied and edited in memory; source JSON/CSV files are never written.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
from pathlib import Path
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lca_emissions import (PHASES, build_activity, calculate_emissions, infer_unit,
                           lca_long, load_lca_wide, summarize)

LOG = logging.getLogger("sensitivity")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def import_fleet_model(path: Path):
    spec = importlib.util.spec_from_file_location("fleet_model_existing", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import fleet model: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def json_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            yield from json_leaves(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from json_leaves(child, f"{prefix}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def get_json_path(data: dict, path: str):
    current = data
    for part in path.split("."):
        current = current[part]
    return current


def set_json_path(data: dict, path: str, value: float):
    current = data
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]
    old = current[parts[-1]]
    current[parts[-1]] = int(round(value)) if isinstance(old, int) else float(value)


def rule_for(parameter_id: str, rules: dict) -> dict:
    # precedence: exact, LCA parameter wildcard, all-source wildcard
    if parameter_id in rules:
        return rules[parameter_id]
    bits = parameter_id.split(":")
    if len(bits) >= 3 and f"{bits[0]}:{bits[1]}:*" in rules:
        return rules[f"{bits[0]}:{bits[1]}:*"]
    return rules.get(f"{bits[0]}:*", {})


def is_protected(path: str, protected: list[str]) -> bool:
    return any(path == item or path.startswith(item + ".") or path.startswith(item + "[") for item in protected)


def bounds(baseline: float, rule: dict, default: float) -> tuple[float, float]:
    kind = rule.get("variation_type", "relative")
    lower_delta, upper_delta = rule.get("lower", -default), rule.get("upper", default)
    if kind == "relative":
        low, high = baseline * (1 + lower_delta), baseline * (1 + upper_delta)
    elif kind == "absolute":
        low, high = baseline + lower_delta, baseline + upper_delta
    elif kind == "values":
        low, high = lower_delta, upper_delta
    else:
        raise ValueError(f"Unknown variation_type {kind!r}")
    return (float(min(low, high)), float(max(low, high)))


def discover_parameters(model_config: dict, lca: pd.DataFrame, sens: dict) -> pd.DataFrame:
    rows = []
    default = float(sens["default_relative_variation"])
    protected = sens.get("protected_json_paths", [])
    rules = sens.get("parameter_rules", {})
    for path, value in json_leaves({k: v for k, v in model_config.items() if not k.startswith("_")}):
        pid = f"config:{path}"
        rule = rule_for(pid, rules)
        excluded = is_protected(path, protected) or not rule.get("enabled", False)
        low, high = bounds(value, rule, default)
        rows.append(dict(parameter_id=pid, parameter_name=path.split(".")[-1], source="fleet_model_config.json",
            location=path, baseline_value=value, lower_value=low, upper_value=high, unit="unknown",
            lifecycle_phase="fleet", drivetrain="all", segment="all", enabled=not excluded,
            exclusion_reason="protected/technical or not enabled" if excluded else "", variation_type=rule.get("variation_type", "relative")))
    long = lca_long(lca)
    for idx, item in long.dropna(subset=["value"]).iterrows():
        pid = f"lca:{item.Parameter}:{item.vehicle_type}"
        rule = rule_for(pid, rules)
        low, high = bounds(float(item.value), rule, default)
        phase_map = {"Vehicle_production": "vehicle_production",
                     "Battery_production": "battery_production", "Battery_capacity": "battery_production",
                     "Consumption": "use/energy_supply", "TTW": "use", "Maintenance": "use",
                     "WTT": "energy_supply", "Emissions_electricity": "energy_supply", "End_of_life": "end_of_life"}
        rows.append(dict(parameter_id=pid, parameter_name=item.Parameter, source="LCA_Emissions_Python.csv",
            location=f"row {idx + 2}, column {item.vehicle_type}", baseline_value=float(item.value),
            lower_value=low, upper_value=high, unit=infer_unit(item.Parameter, item.Explanation_parameter),
            lifecycle_phase=phase_map.get(item.Parameter, "unknown"), drivetrain=item.drivetrain_group,
            segment=item.size_class, enabled=rule.get("enabled", False),
            exclusion_reason="not enabled" if not rule.get("enabled", False) else "",
            variation_type=rule.get("variation_type", "relative")))
    return pd.DataFrame(rows)


def validate_value(row: pd.Series, value: float, sens: dict):
    if not np.isfinite(value):
        raise ValueError("parameter value is not finite")
    name = str(row.parameter_name).lower()
    patterns = sens.get("plausibility", {}).get("nonnegative_name_patterns", [])
    if any(p.lower() in name for p in patterns) and value < 0:
        raise ValueError("nonnegative parameter would become negative")
    if row.lower_value > row.upper_value:
        raise ValueError("lower bound exceeds upper bound")


def run_fleet(module, config: dict, scenario: str):
    """CALL SITE OF THE EXISTING MODEL (no reimplementation of fleet logic)."""
    cfg = copy.deepcopy(config)
    if scenario in cfg["policies"].get("bev_target_scenarios", {}):
        cfg["policies"]["scenarios"] = []
    else:
        cfg["policies"]["scenarios"] = [scenario]
    cfg.setdefault("fleet_size_scenarios", {})["enabled"] = False
    return module.run_model(cfg)


def modify_lca(lca: pd.DataFrame, parameter_id: str, value: float) -> pd.DataFrame:
    _, parameter, vehicle_type = parameter_id.split(":", 2)
    out = lca.copy(deep=True)
    mask = out["Parameter"].eq(parameter)
    if mask.sum() != 1 or vehicle_type not in out:
        raise KeyError(f"Cannot uniquely locate {parameter_id}")
    out.loc[mask, vehicle_type] = str(value)
    return out


def flatten_metrics(metrics: dict) -> dict:
    row = {k: v for k, v in metrics.items() if k not in {"by_phase", "by_drive", "by_segment", "annual"}}
    row.update({f"phase__{k}": v for k, v in metrics["by_phase"].items()})
    row.update({f"drive__{k}": v for k, v in metrics["by_drive"].items()})
    row.update({f"segment__{k}": v for k, v in metrics["by_segment"].items()})
    row["annual_emissions_json"] = json.dumps(metrics["annual"], ensure_ascii=False)
    return row


def result_row(meta: pd.Series, variation: str, value: float, metrics: dict, baseline: dict, scenario: str) -> dict:
    rel_input = (value / meta.baseline_value - 1) if meta.baseline_value else np.nan
    delta = metrics["cumulative_total_tco2e"] - baseline["cumulative_total_tco2e"]
    rel_output = delta / baseline["cumulative_total_tco2e"] if baseline["cumulative_total_tco2e"] else np.nan
    out = dict(scenario=scenario, parameter_id=meta.parameter_id, parameter_name=meta.parameter_name,
        variation=variation, baseline_value=meta.baseline_value, parameter_value=value,
        absolute_parameter_change=value-meta.baseline_value, relative_parameter_change=rel_input,
        absolute_change_vs_baseline_tco2e=delta, percent_change_vs_baseline=100*rel_output,
        emission_savings_vs_baseline_tco2e=-delta,
        elasticity=rel_output/rel_input if rel_input not in (0, np.nan) and np.isfinite(rel_input) else np.nan)
    out.update(flatten_metrics(metrics))
    for phase, base_value in baseline["by_phase"].items():
        out[f"phase_delta__{phase}"] = metrics["by_phase"].get(phase, 0) - base_value
    for drive, base_value in baseline["by_drive"].items():
        out[f"drive_delta__{drive}"] = metrics["by_drive"].get(drive, 0) - base_value
    return out


def make_ranking(results: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (scenario, pid, name), group in results.groupby(["scenario", "parameter_id", "parameter_name"]):
        low = group[group.variation == "low"].iloc[0]
        high = group[group.variation == "high"].iloc[0]
        records.append(dict(scope="overall", category="total", scenario=scenario, parameter_id=pid,
            parameter_name=name, sensitivity_span_tco2e=abs(high.cumulative_total_tco2e-low.cumulative_total_tco2e),
            max_absolute_change_tco2e=max(abs(low.absolute_change_vs_baseline_tco2e), abs(high.absolute_change_vs_baseline_tco2e)),
            max_percent_change=max(abs(low.percent_change_vs_baseline), abs(high.percent_change_vs_baseline)),
            max_absolute_elasticity=np.nanmax(np.abs(group.elasticity.to_numpy(dtype=float)))))
        for prefix, scope in [("phase_delta__", "phase"), ("drive_delta__", "drivetrain")]:
            for col in [c for c in group.columns if c.startswith(prefix)]:
                records.append(dict(scope=scope, category=col[len(prefix):], scenario=scenario, parameter_id=pid,
                    parameter_name=name, sensitivity_span_tco2e=abs(high[col]-low[col]),
                    max_absolute_change_tco2e=max(abs(low[col]), abs(high[col])), max_percent_change=np.nan,
                    max_absolute_elasticity=np.nan))
    rank = pd.DataFrame(records)
    rank = rank.sort_values(["scope", "category", "max_absolute_change_tco2e", "max_percent_change", "max_absolute_elasticity"],
                            ascending=[True, True, False, False, False], na_position="last")
    rank["rank"] = rank.groupby(["scope", "category", "scenario"]).cumcount() + 1
    return rank


def save_figure(fig, path: Path, save_pdf: bool):
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight")
    if save_pdf: fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plots(results: pd.DataFrame, ranking: pd.DataFrame, baseline: dict, outdir: Path, sens: dict):
    overall = ranking[(ranking.scope == "overall") & (ranking.category == "total")].head(int(sens.get("top_n_plots", 15)))
    selected = results[results.parameter_id.isin(overall.parameter_id)]
    pivot = selected.pivot(index="parameter_id", columns="variation", values="percent_change_vs_baseline").reindex(overall.parameter_id)
    fig, ax = plt.subplots(figsize=(10, max(5, .42*len(pivot))))
    y = np.arange(len(pivot)); ax.barh(y, pivot.get("low", 0), label="Lower value"); ax.barh(y, pivot.get("high", 0), label="Upper value")
    ax.axvline(0, color="black", lw=.8); ax.set_yticks(y, pivot.index); ax.invert_yaxis(); ax.set_xlabel("Change in cumulative emissions [%]"); ax.legend()
    save_figure(fig, outdir/"tornado_cumulative_emissions.png", sens.get("save_pdf", False))

    elast = selected.groupby("parameter_id").elasticity.apply(lambda x: np.nanmax(np.abs(x))).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(5, .4*len(elast)))); elast.plot.barh(ax=ax); ax.set_xlabel("Absolute elasticity")
    save_figure(fig, outdir/"normalized_sensitivities.png", sens.get("save_pdf", False))

    cols = ["cumulative_total_tco2e", "target_year_emissions_tco2e"] + [f"phase__{p}" for p in PHASES]
    change = pd.DataFrame({c: selected.groupby("parameter_id")[c].apply(lambda x: (x.max()-x.min())/(baseline["cumulative_total_tco2e"] or 1)*100) for c in cols})
    fig, ax = plt.subplots(figsize=(11, max(5, .4*len(change)))); im=ax.imshow(change, aspect="auto", cmap="viridis"); ax.set_yticks(range(len(change)), change.index); ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right"); fig.colorbar(im, ax=ax, label="Low-high span / baseline total [%]")
    save_figure(fig, outdir/"sensitivity_heatmap.png", sens.get("save_pdf", False))

    phase = ranking[ranking.scope == "phase"].groupby("category").head(5)
    fig, ax = plt.subplots(figsize=(11, 6)); phase.pivot_table(index="parameter_id", columns="category", values="max_absolute_change_tco2e", aggfunc="max", fill_value=0).plot.bar(ax=ax); ax.set_ylabel("Maximum absolute change [t CO2e]")
    save_figure(fig, outdir/"sensitivity_lifecycle_phases.png", sens.get("save_pdf", False))

    # Only one configured scenario is calculated in v1; keep scenario in chart/data model.
    fig, ax = plt.subplots(figsize=(10, 6)); overall.head(15).pivot_table(index="parameter_id", columns="scenario", values="max_absolute_change_tco2e").plot.barh(ax=ax); ax.set_xlabel("Maximum absolute change [t CO2e]")
    save_figure(fig, outdir/"sensitivity_ranking_by_policy_scenario.png", sens.get("save_pdf", False))

    top5 = overall.head(5).parameter_id
    years = sorted(int(y) for y in baseline["annual"])
    fig, axes = plt.subplots(len(top5), 1, figsize=(11, 3.2*max(1, len(top5))), sharex=True, squeeze=False)
    for ax, pid in zip(axes[:,0], top5):
        ax.plot(years, [baseline["annual"].get(y, 0) for y in years], label="Baseline", color="black")
        for variation in ["low", "high"]:
            row = results[(results.parameter_id == pid) & (results.variation == variation)].iloc[0]
            annual = {int(k): v for k, v in json.loads(row.annual_emissions_json).items()}
            ax.plot(years, [annual.get(y, 0) for y in years], label=variation)
        ax.set_title(pid); ax.set_ylabel("t CO2e"); ax.legend()
    axes[-1,0].set_xlabel("Year"); save_figure(fig, outdir/"top5_time_series.png", sens.get("save_pdf", False))


def write_summary(ranking: pd.DataFrame, results: pd.DataFrame, baseline: dict, outdir: Path, scenario: str):
    overall = ranking[(ranking.scope == "overall") & (ranking.category == "total")]
    lines = ["# Sensitivity analysis summary", "", f"Scenario: `{scenario}`. Baseline cumulative emissions: {baseline['cumulative_total_tco2e']:,.0f} t CO₂e.", "",
             "## Five most influential parameters", ""]
    for _, item in overall.head(5).iterrows():
        lines.append(f"- `{item.parameter_id}`: maximum change {item.max_absolute_change_tco2e:,.0f} t CO₂e; span {item.sensitivity_span_tco2e:,.0f} t CO₂e; maximum elasticity {item.max_absolute_elasticity:.3g}.")
    for phase, title in [("vehicle_production", "Production"), ("use", "Use"), ("end_of_life", "End of life")]:
        lines += ["", f"## {title} emissions", ""]
        subset = ranking[(ranking.scope == "phase") & (ranking.category == phase)].head(5)
        lines += [f"- `{r.parameter_id}`: maximum change {r.max_absolute_change_tco2e:,.0f} t CO₂e." for _, r in subset.iterrows()]
    low = overall[overall.max_percent_change < .01]
    lines += ["", "## Low-priority parameters", "", f"{len(low)} parameter(s) change cumulative emissions by less than 0.01%." if len(low) else "No parameter is below the 0.01% threshold.", "",
              "## Interpretation limits", "", "Only the centrally configured reference scenario is evaluated in this first version. Cross-scenario robustness and scenario dependence therefore cannot yet be inferred.",
              "Battery-production emissions are zero because the supplied LCA file contains battery capacity but no battery-production intensity (t CO₂e/kWh). No value was invented.",
              "Emission-saving uncertainty versus another policy scenario cannot be calculated from a baseline-only analysis; the output schema remains scenario-aware for later runs."]
    (outdir/"sensitivity_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here/"sensitivity_config.json")
    parser.add_argument("--model-config", type=Path, default=here/"fleet_model_config.json")
    parser.add_argument("--lca", type=Path, default=here/"LCA_Emissions_Python.csv")
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    sens = load_json(args.config.resolve()); model_config_path=args.model_config.resolve()
    model_config = load_json(model_config_path); model_config["_config_path"] = str(model_config_path)
    lca = load_lca_wide(args.lca.resolve())
    outdir = Path(sens["output_directory"]); outdir = outdir if outdir.is_absolute() else here/outdir
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(outdir/"sensitivity.log", encoding="utf-8"), logging.StreamHandler()])
    overview = discover_parameters(model_config, lca, sens)
    overview.to_csv(outdir/"sensitivity_parameter_overview.csv", index=False, encoding="utf-8-sig")
    LOG.info("Discovered %d numeric candidates; %d enabled", len(overview), overview.enabled.sum())
    if args.discover_only: return

    scenario = sens["scenario"]; target_year=int(sens["target_year"])
    model = import_fleet_model(here/"fleet_model_with_policies.py")
    LOG.info("Running fleet baseline for scenario %s", scenario)
    baseline_fleet = run_fleet(model, model_config, scenario)
    baseline_activity = build_activity(baseline_fleet, scenario)
    baseline_emissions = calculate_emissions(baseline_activity, lca)
    baseline = summarize(baseline_emissions, model_config["years"]["start_year"], model_config["years"]["end_year"], target_year)
    pd.DataFrame([flatten_metrics(baseline)]).to_csv(outdir/"baseline_results.csv", index=False, encoding="utf-8-sig")
    baseline_emissions.to_csv(outdir/"baseline_emissions_detailed.csv", index=False, encoding="utf-8-sig")

    rows, failures = [], []
    for _, meta in overview[overview.enabled].iterrows():
        for variation, value in [("low", meta.lower_value), ("high", meta.upper_value)]:
            try:
                validate_value(meta, value, sens)
                if meta.parameter_id.startswith("lca:"):
                    run_lca = modify_lca(lca, meta.parameter_id, value); activity = baseline_activity
                else:
                    cfg = copy.deepcopy(model_config); set_json_path(cfg, meta.location, value)
                    fleet = run_fleet(model, cfg, scenario); activity = build_activity(fleet, scenario); run_lca = lca
                emissions = calculate_emissions(activity, run_lca)
                metrics = summarize(emissions, model_config["years"]["start_year"], model_config["years"]["end_year"], target_year)
                rows.append(result_row(meta, variation, value, metrics, baseline, scenario))
                LOG.info("Completed %s %s", meta.parameter_id, variation)
            except Exception as exc:
                failures.append(dict(scenario=scenario, parameter_id=meta.parameter_id, variation=variation,
                                     parameter_value=value, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
                LOG.exception("Failed %s %s", meta.parameter_id, variation)
    detailed = pd.DataFrame(rows); detailed.to_csv(outdir/"sensitivity_results_detailed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(failures, columns=["scenario","parameter_id","variation","parameter_value","error_type","error","traceback"]).to_csv(outdir/"sensitivity_failed_runs.csv", index=False, encoding="utf-8-sig")
    if detailed.empty: raise RuntimeError("All sensitivity runs failed; see sensitivity_failed_runs.csv")
    ranking = make_ranking(detailed); ranking.to_csv(outdir/"sensitivity_ranking.csv", index=False, encoding="utf-8-sig")
    plots(detailed, ranking, baseline, outdir, sens); write_summary(ranking, detailed, baseline, outdir, scenario)
    LOG.info("Sensitivity analysis complete: %s", outdir)


if __name__ == "__main__":
    main()
