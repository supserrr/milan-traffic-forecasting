"""Render the capacity grid of EXP-010 to EXP-018 as one table.

The per-area tables report one configuration per architecture. They say nothing about whether
that configuration was a choice, which is the question the "Experimentation and Hyperparameter
Tuning" criterion asks. This script collects the one-variable capacity runs into a single grid and
marks, per architecture, the configuration that *validation* selects. Selection is on validation
by construction: the evaluation week decides nothing here.

Thin CLI. The comparison and the auto-detection of which hyperparameter each run moved live in
``milan_traffic.evaluate.compare_runs``. Re-running overwrites its own output.

    python scripts/make_capacity_table.py

Output: ``reports/tables/capacity_grid.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milan_traffic.config import RUNS_DIR, TABLES_DIR  # noqa: E402
from milan_traffic.evaluate import compare_runs  # noqa: E402
from milan_traffic.models.tcn import receptive_field  # noqa: E402
from milan_traffic.periods import to_markdown  # noqa: E402
from milan_traffic.utils.logging import get_logger  # noqa: E402

LOG = get_logger("make_capacity_table")

#: EXP-008 supplies the incumbent configuration for all three architectures on this cell, at the
#: same seeds and the same everything-else, so it belongs in the grid: without it the selection
#: marker would be choosing from an incomplete set and could name a sweep point that is worse than
#: the configuration already reported. EXP-010 is a control that reproduces EXP-008's TCN on this
#: cell bit-for-bit, which is what licenses mixing the two runs in one table.
INCUMBENT_RUN = "EXP-008"
DEFAULT_RUNS = (INCUMBENT_RUN, *(f"EXP-{n:03d}" for n in range(10, 19)))

DISPLAY = {"lstm": "LSTM", "tcn": "TCN", "gbt": "GBT"}


def _resolve_area(frame: pd.DataFrame) -> int | None:
    """The cell the grid was run on, read off the runs rather than typed into the script.

    Every sweep run holds exactly one area and the incumbent holds the three evaluated cells, so
    the area common to all of them is unique, and it is by construction the cell the sweep was
    run on. Deriving it keeps the project's rule that square ids are never written down, and it
    is the more faithful of the two ways of doing that: ``dataio.top_squares(1)`` answers which
    cell tops the traffic ranking today, which is the right question for a new run and the wrong
    one for a table describing runs that already happened. It also keeps this script readable
    from the run directories alone, with no dependency on the processed store.

    ``None`` when the runs share no single cell, which is a question for the caller rather than
    something to guess at.
    """
    per_run = [set(group["area"].astype(int)) for _, group in frame.groupby("run")]
    common = set.intersection(*per_run) if per_run else set()
    return int(next(iter(common))) if len(common) == 1 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument(
        "--area",
        type=int,
        default=None,
        help="cell the grid covers; default: the one area every run in --runs shares",
    )
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    args = ap.parse_args(argv)

    present = [r for r in args.runs if (Path(args.runs_dir) / r / "metrics.json").exists()]
    missing = sorted(set(args.runs) - set(present))
    if missing:
        LOG.warning("skipping runs with no metrics.json: %s", ", ".join(missing))
    if not present:
        LOG.error("no runs to compare")
        return 1

    frame = compare_runs(present, runs_dir=args.runs_dir, select_on="val")
    area = args.area if args.area is not None else _resolve_area(frame)
    if area is None:
        LOG.error(
            "%s share no single area, so the grid's cell cannot be derived; name it with --area",
            ", ".join(present),
        )
        return 1
    LOG.info("capacity grid on square %d", area)
    frame = frame[(frame["area"] == area) & frame["model"].isin(DISPLAY)].copy()
    # EXP-010 is a bit-for-bit reproduction of EXP-008's TCN on this cell, so keeping both would
    # put a duplicate row in the grid and split the selection between two identical entries.
    dup = (frame["run"] == "EXP-010") & (frame["model"] == "tcn")
    if (frame["run"] == INCUMBENT_RUN).any() and dup.any():
        frame = frame[~dup]
    frame["model"] = frame["model"].map(lambda m: DISPLAY.get(m, m))
    frame["run"] = frame.apply(
        lambda r: f"{r['run']} (reported)" if r["run"] == INCUMBENT_RUN else r["run"], axis=1
    )

    def field(row) -> str:
        if row["model"] != "TCN":
            return ""
        levels = 6
        for part in str(row["changed"]).split(","):
            if "levels=" in part:
                levels = int(part.split("=")[1])
        return str(receptive_field(3, levels))

    frame["receptive_field"] = frame.apply(field, axis=1)
    frame["pick"] = frame["selected"].map({True: "<-- validation selects", False: ""})

    columns = [
        "run",
        "model",
        "changed",
        "n_params",
        "receptive_field",
        "val_MAE",
        "val_MAE_sd",
        "test_MAE",
        "test_MAE_sd",
        "s_per_epoch",
        "pick",
    ]
    lines = [
        f"# Capacity grid, square {area}",
        "",
        "Runs EXP-010 to EXP-018, three seeds each (42, 1337, 2024), one variable moved per run",
        "from the reported configuration, everything else held at EXP-008's values (seq_len 144,",
        "log1p-standard scaling, calendar features on). The EXP-008 rows are the reported",
        "configuration itself, at the same seeds on the same cell. EXP-010 re-ran that",
        "configuration for the TCN as a control and reproduced it bit-for-bit on all three seeds,",
        "which is what licenses reading the two runs as one grid; its duplicate row is dropped.",
        "",
        "`changed` is derived by diffing each run's resolved model config against the others in",
        "the set, so a row cannot claim a variable its run did not move. The arrow marks the",
        "configuration **validation** selects per architecture; the evaluation-week column is",
        "shown for completeness and decides nothing.",
        "",
        "The TCN receptive field is `1 + 2(k-1) * sum(2^i)` at kernel 3. Five levels reach 125",
        "steps and so cannot see the whole 144-slot window; six is the first depth that covers it.",
        "",
        to_markdown(frame[columns], floatfmt="{:.2f}"),
    ]
    out = Path(args.tables_dir) / "capacity_grid.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
