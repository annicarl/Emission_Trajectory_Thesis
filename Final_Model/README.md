# Final Model

This directory contains reproducible command-line versions of the final fleet and
GHG models. Plotting and publication analyses remain in separate notebooks.

## Combined pipeline

Run the fleet and GHG models sequentially with one config:

```powershell
uv run python ".\Final_Model\src\run_pipeline.py" --config ".\Final_Model\configs\baseline.json"
```

Both model directories use the same `<name>__<timestamp>` identifier. The GHG
model automatically receives the fleet run created immediately before it; the
config file is not rewritten. A pipeline manifest linking both model manifests is
saved below `outputs/pipeline/<name>__<timestamp>`.

## Fleet model

Run the model from the repository root:

```powershell
uv run python ".\Final_Model\src\fleet_model_with_policies.py" --config ".\Final_Model\configs\baseline.json"
```

The `--config` argument is required. Each config must contain a `name` field. The
name may contain letters, numbers, dots, underscores, and hyphens and is used as
the prefix of every result CSV.

Input data are read from `Final_Model/data`. The default config reproduces the
settings used by `Fleet Model/Fleet_Model_runner.ipynb`.

## Outputs

Each invocation creates an immutable run directory:

```text
outputs/fleet_model/<config-name>__<YYYY-MM-DD_HHMMSS>/
```

For example:

```text
outputs/fleet_model/baseline__2026-08-27_143052/
├── config.snapshot.json
├── run_manifest.json
├── default__fleet_total.csv
├── default__policy_summary_all.csv
├── default__policy_final_stock_all.csv
└── ...
```

`config.snapshot.json` is the exact config used for the run. `run_manifest.json`
records timestamps in the Europe/Berlin timezone, source paths and hashes, runtime
information, completion status, and the list of generated result files. Existing
run directories are never overwritten.

## GHG model

The GHG model uses the fleet run selected by `ghg.fleet_run` in the shared config.
Run it from the repository root after the selected fleet run has completed:

```powershell
uv run python ".\Final_Model\src\ghg_model.py" --config ".\Final_Model\configs\baseline.json"
```

For a one-off fleet selection without editing the config:

```powershell
uv run python ".\Final_Model\src\ghg_model.py" --config ".\Final_Model\configs\baseline.json" --fleet-run "baseline__2026-08-28_101500"
```

Each invocation creates `outputs/ghg_model/<config-name>__<timestamp>`. It stores
the prepared fleet/LCA tables, detailed emissions by life-cycle phase, three
summary tables, a config snapshot, and a manifest linking the exact fleet run and
emission-parameter file. Plots are deliberately excluded and belong in analysis
notebooks.
