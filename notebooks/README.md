# notebooks/

**Deliberately empty.** Every analysis in this project runs from a script, not a notebook.

The reason is reproducibility. A figure that exists only as the output of a cell someone ran in
an order they no longer remember cannot be regenerated from a clean clone, which is exactly what
the assignment asks for. So each artefact has one command that rebuilds it:

| artefact | command |
|---|---|
| the processed store | `make ingest` |
| memory evidence (`reports/tables/memory_benchmark.md`) | `make bench` |
| time-series diagnostics (`reports/tables/tsa_5161.md`, three figures) | `python scripts/run_tsa.py` |
| the five exploratory figures | `python scripts/make_figures.py` |
| the reference forecasters | `python scripts/run_baselines.py` |
| one training run (`experiments/runs/EXP-NNN/`) | `python scripts/run_experiment.py` |
| the result tables, the nine panels, error by hour | `python scripts/make_results.py --run EXP-008` |
| the significance tests (`reports/tables/significance_dm.md`, one figure) | `python scripts/make_significance.py --run EXP-008` |
| the per-period failure evidence (`reports/tables/per_period_*.md`, one figure) | `python scripts/make_period_breakdown.py --run EXP-008` |
| the capacity grid (`reports/tables/capacity_grid.md`) | `python scripts/make_capacity_table.py` |
| Section IV's descriptive statistics (`reports/tables/eda_stats.md`) | `python scripts/make_eda_stats.py` |
| the experiment log's numbers, re-derived from `metrics.json` | `python scripts/check_log_numbers.py` |
| the model-selection checks (`reports/tables/model_selection_checks.md`) | `python scripts/check_model_selection.py` |

Exploratory work happened in a Python REPL against the same modules and left nothing behind that
the scripts above do not reproduce. If you want a notebook, start one here and import from
`src/`: the analysis logic lives in the package, so a notebook is only ever a viewer.

```python
from milan_traffic import dataio, features, metrics, tsa, viz
viz.use_style()
```
