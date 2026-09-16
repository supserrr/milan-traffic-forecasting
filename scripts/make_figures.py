"""Generate the exploratory figures for the report.

Thin CLI. Data access goes through ``milan_traffic.dataio`` (memory-mapped), the figure
inputs are built by the helpers in ``milan_traffic.viz`` and every figure is written as
PDF and PNG under ``reports/figures/``. Nothing here reads the raw ``.txt`` files.

    python scripts/make_figures.py
    python scripts/make_figures.py --only grid_heatmap five_cells_first_two_weeks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from milan_traffic import viz  # noqa: E402
from milan_traffic.config import (  # noqa: E402
    EDA_WINDOW_END,
    EDA_WINDOW_START,
    EVAL_WEEK_START,
    FIGURES_DIR,
    REQUIRED_EDA_SQUARES,
    TIMEZONE,
)
from milan_traffic.dataio import (  # noqa: E402
    column_of,
    load_index,
    load_matrix,
    load_square_totals,
    top_squares,
)

FIGURES = (
    "total_traffic_distribution",
    "grid_heatmap",
    "five_cells_first_two_weeks",
    "weekly_weekday_drift",
    "grid_spike_dec06",
)

#: The single-slot grid-wide spike found in the EDA (largest jump in the series).
SPIKE_TIME = "2013-12-06 20:50"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--only", nargs="+", choices=FIGURES, default=None)
    parser.add_argument(
        "--chunk-rows", type=int, default=288, help="rows per chunk for grid totals"
    )
    args = parser.parse_args(argv)
    wanted = set(args.only or FIGURES)

    viz.use_style()
    totals = load_square_totals()
    tot = totals["internet"].to_numpy(dtype=np.float64)
    ids = totals["square_id"].to_numpy(dtype=int)
    rank = totals["internet"].rank(ascending=False, method="min").astype(int).to_numpy()
    rank_of = dict(zip(ids.tolist(), rank.tolist(), strict=True))
    five = [*top_squares(3), *REQUIRED_EDA_SQUARES]

    X = load_matrix("internet")
    idx = load_index()

    def emit(fig: plt.Figure, name: str) -> None:
        for path in viz.save_figure(fig, name, formats=args.formats, directory=args.out_dir):
            print(path)
        plt.close(fig)

    if "total_traffic_distribution" in wanted:
        emit(viz.plot_total_traffic_distribution(tot), "total_traffic_distribution")

    if "grid_heatmap" in wanted:
        emit(viz.plot_grid_heatmap(tot, ids, highlight=five), "grid_heatmap")

    if "five_cells_first_two_weeks" in wanted:
        lo = idx.get_loc(pd.Timestamp(EDA_WINDOW_START, tz=TIMEZONE))
        hi = idx.get_loc(pd.Timestamp(EDA_WINDOW_END, tz=TIMEZONE)) + 1
        series = {s: np.asarray(X[lo:hi, column_of(s)], dtype=np.float64) for s in five}
        n_cells = f"{ids.size:,}".replace(",", " ")
        labels = {s: f"square {s} (rank {rank_of[s]} of {n_cells} by total)" for s in five}
        emit(
            viz.plot_five_cells_two_weeks(series, idx[lo:hi], labels=labels),
            "five_cells_first_two_weeks",
        )

    if wanted & {"weekly_weekday_drift", "grid_spike_dec06"}:
        grid = viz.grid_totals(X, chunk_rows=args.chunk_rows)

    if "weekly_weekday_drift" in wanted:
        frame = pd.DataFrame(
            {
                "grid": grid,
                **{str(s): np.asarray(X[:, column_of(s)], dtype=np.float64) for s in five},
            },
            index=idx,
        )
        table = viz.weekday_means_by_iso_week(frame)
        eval_week = int(
            idx[idx.get_loc(pd.Timestamp(EVAL_WEEK_START, tz=TIMEZONE))].isocalendar().week
        )
        table = table.loc[:eval_week]  # the models never see the Christmas week
        fig = viz.plot_weekly_weekday_means(
            table, base_week=int(table.index[0]), highlight_week=eval_week
        )
        emit(fig, "weekly_weekday_drift")

    if "grid_spike_dec06" in wanted:
        center = int(idx.get_loc(pd.Timestamp(SPIKE_TIME, tz=TIMEZONE)))
        emit(viz.plot_grid_spike(grid, idx, center_row=center), "grid_spike_dec06")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
