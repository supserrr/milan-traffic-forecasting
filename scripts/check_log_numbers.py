"""Re-derive the runs-table cells of experiments/EXPERIMENT_LOG.md from each run's metrics.json.

The log's numeric columns were transcribed by hand and several rows drifted from their own
artefacts. This script is the standing check: it reads the table, re-derives the five numeric
columns from the run directory the row points at, prints log value against derived value, and
exits non-zero while any cell still disagrees. Each row names its model and square in its own
prose; that mapping is spelled out in ROW_SOURCE below so it can be audited against the text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "experiments" / "EXPERIMENT_LOG.md"

# log row id -> (run directory, model, area named by that row's prose)
ROW_SOURCE = {
    "EXP-000": ("EXP-000", "linear_ar_144", 5161),  # "row shows linear_ar_144 on square 5161"
    "EXP-001": ("EXP-001", "tcn", 5161),  # "Row shows the TCN on 5161"
    "EXP-002": ("EXP-002", "tcn", 5161),  # "Row is the TCN on 5161 again"
    "EXP-003": ("EXP-003", "tcn", 5161),  # same configuration, for comparability
    "EXP-004": ("EXP-004", "tcn", 5059),  # "tcn on 5059, 3 seeds, calendar on"
    "EXP-005": ("EXP-005", "tcn", 5059),  # "tcn on 5059, 3 seeds, calendar off"
    "EXP-006": ("EXP-006", "tcn", 5059),  # "repeat of EXP-004 with the seed leak fixed"
    "EXP-007": ("EXP-007", "tcn", 5059),  # "control for EXP-006"
    "EXP-008": ("EXP-008", "tcn", 5161),  # "Row is the TCN on 5161, mean over seeds"
    "EXP-010 to EXP-018": ("EXP-010", "tcn", 5161),  # "the validation-selected TCN"
}
COLUMNS = ("Val MAE", "Test MAE", "Test RMSE", "Train (s)", "Infer (ms)")


def derive(run: str, model: str, area: int) -> list[float]:
    """Seed means of the five logged quantities for one model on one square."""
    path = ROOT / "experiments" / "runs" / run / "metrics.json"
    rows = [
        r
        for r in json.loads(path.read_text())["results"]
        if r["model"] == model and r["area"] == area
    ]
    if not rows:
        raise SystemExit(f"{path}: no result for {model} on {area}")
    # EXP-000 predates the repeated-timing protocol and stores one wall time only.
    getters = (
        lambda r: r["val"]["MAE"],
        lambda r: r["test"]["MAE"],
        lambda r: r["test"]["RMSE"],
        lambda r: r["fit_s"],
        lambda r: r.get("predict_ms_batch_median", r["predict_s"] * 1000.0),
    )
    return [sum(map(g, rows)) / len(rows) for g in getters]


def main() -> int:
    bad = 0
    print(f"{'row':<19} {'column':<10} {'logged':>9} {'derived':>9}")
    for line in LOG.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) != 10 or cells[0] not in ROW_SOURCE:
            continue
        logged, (run, model, area) = cells[4:9], ROW_SOURCE[cells[0]]
        derived = derive(run, model, area)
        for name, text, value in zip(COLUMNS, logged, derived, strict=True):
            places = len(text.split(".")[1]) if "." in text else 0
            mismatch = round(value, places) != float(text)
            bad += mismatch
            print(
                f"{cells[0]:<19} {name:<10} {text:>9} {value:>9.4f}"
                f"{'  MISMATCH' if mismatch else ''}"
            )
    print(f"\n{bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
