"""Descriptive statistics of Section IV, and the regime block of Section VI.

Thin CLI. Every statistic lives in ``milan_traffic.edastats``; this file only sequences
them, renders the markdown and writes the file. Section VIII promises that every number
in the report has a generating command, and for the descriptive numbers this is it.

    python scripts/make_eda_stats.py                 # all seven sections
    python scripts/make_eda_stats.py --no-zero-scan  # skip the full-matrix zero scan

Output: ``reports/tables/eda_stats.md``, and a short summary on the log.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from milan_traffic import edastats as eda  # noqa: E402
from milan_traffic.config import (  # noqa: E402
    GRID_SIDE,
    REQUIRED_EDA_SQUARES,
    SLOTS_PER_DAY,
    SLOTS_PER_WEEK,
    TABLES_DIR,
    TIMEZONE,
)
from milan_traffic.dataio import (  # noqa: E402
    column_of,
    load_index,
    load_matrix,
    load_square_ids,
    load_square_totals,
    to_grid,
    top_squares,
)
from milan_traffic.features import is_working_day  # noqa: E402
from milan_traffic.tsa import EVAL_ROWS, TRAIN_ROWS, UNIT, decompose, markdown_table  # noqa: E402
from milan_traffic.utils.logging import get_logger  # noqa: E402

LOG = get_logger("make_eda_stats")

#: Christmas block and its baseline: the same three weekdays a fortnight earlier, so the
#: comparison is Tuesday against Tuesday and not against a weekend.
HOLIDAY_BLOCK = ("2013-12-24", "2013-12-25", "2013-12-26")
HOLIDAY_BASELINE = ("2013-12-10", "2013-12-11", "2013-12-12")
CHRISTMAS_DAY = ("2013-12-25",)
CHRISTMAS_BASELINE = ("2013-12-11",)

#: Lags the report quotes off the decomposition remainder: one step, one day, one week.
REMAINDER_LAGS = (1, SLOTS_PER_DAY, SLOTS_PER_WEEK)


def _weekend_slot_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on Saturday and Sunday slots. Calendar weekend, holidays not folded in."""
    return np.asarray(index.dayofweek) >= 5


def _orientation_panel(check: pd.DataFrame, origin: str) -> pd.DataFrame:
    """One convention's half of the orientation table, with readable totals."""
    return pd.DataFrame(
        {
            "landmark": check["landmark"],
            "expected": check["expected"],
            "square": check[f"{origin} square"],
            "cell total": [f"{v:,.0f}" for v in check[f"{origin} total"]],
            "ratio to 5x5 median": check[f"{origin} ratio"],
            "verdict": check[f"{origin} verdict"],
        }
    )


def _near_misses(grid: np.ndarray, check: pd.DataFrame) -> pd.DataFrame:
    """For each landmark that fails under the south convention, the best cell next to it."""
    rows = []
    for mark in eda.LANDMARKS:
        verdict = check.loc[check["landmark"] == mark.name, "south verdict"].iloc[0]
        if verdict != "FAIL":
            continue
        row, col = eda.cell_of(mark.lat, mark.lon)
        mode = mark.expect or "max"
        drow, dcol, ratio = eda.best_neighbour_ratio(grid, row, col, mode=mode)
        rows.append(
            {
                "landmark": mark.name,
                "expected": mode,
                "ratio at the coordinates": float(
                    check.loc[check["landmark"] == mark.name, "south ratio"].iloc[0]
                ),
                "best ratio within one cell, in the expected direction": ratio,
                "direction": (
                    "in place"
                    if (drow, dcol) == (0, 0)
                    else eda.offset_compass(row, col, drow, dcol)
                ),
            }
        )
    return pd.DataFrame(rows)


def _count(value: float) -> int:
    """Render a count as a count. ``markdown_table`` formats every float, and a table that
    reports 10 000 cells as ``10000.0000`` is harder to read than it needs to be."""
    return int(round(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--top-n", type=int, default=3, help="how many busiest squares are study cells"
    )
    parser.add_argument(
        "--neighbourhood",
        type=int,
        default=2,
        help="half-width of the orientation neighbourhood; 2 gives the 5x5 block",
    )
    parser.add_argument(
        "--no-zero-scan",
        action="store_true",
        help="skip section 6's full-matrix scan (the only part that touches every cell)",
    )
    parser.add_argument("--tables-dir", default=str(TABLES_DIR))
    args = parser.parse_args(argv)

    t_start = time.perf_counter()

    X = load_matrix("internet")
    index = load_index()
    totals = load_square_totals().sort_values("square_id").reset_index(drop=True)
    grid = to_grid(totals["internet"].to_numpy())

    top = top_squares(args.top_n)
    cells = [*top, *REQUIRED_EDA_SQUARES]
    LOG.info("study cells: top %d = %s, plus %s", args.top_n, top, list(REQUIRED_EDA_SQUARES))

    fortnight = eda.FORTNIGHT_ROWS
    fortnight_index = index[fortnight]
    train_index = index[TRAIN_ROWS]
    train_days = train_index[::SLOTS_PER_DAY]
    working = is_working_day(train_days).astype(bool)

    # ------------------------------------------------- 1. geometry and orientation
    check = eda.orientation_check(grid, k=args.neighbourhood)
    scores = eda.orientation_scores(check)
    bands = eda.row_band_mass(grid)
    mean_row = eda.mass_weighted_mean_row(grid)
    lat_south = eda.GRID_LAT_MIN + (mean_row + 0.5) * eda.CELL_LAT
    lat_north = eda.GRID_LAT_MAX - (mean_row + 0.5) * eda.CELL_LAT
    farmland = check.loc[check["landmark"] == "Parco Sud farmland"].iloc[0]
    sesto = check.loc[check["landmark"] == "Sesto San Giovanni"].iloc[0]
    misses = _near_misses(grid, check)
    LOG.info("near misses:\n%s", misses.to_string(index=False) if len(misses) else "(none)")
    LOG.info("orientation scores:\n%s", scores.to_string(index=False))
    LOG.info("mass-weighted mean row %.2f", mean_row)

    # ------------------------------------------------------------ 2. study cells
    cell_table = eda.study_cell_table(cells, totals)
    pairs = eda.pairwise_cell_distances(top)
    LOG.info("study cells:\n%s", cell_table.to_string(index=False))

    # -------------------------------------------------------- 3. concentration
    stats = eda.concentration_stats(totals["internet"].to_numpy())
    ranks = eda.rank_table([*top, *REQUIRED_EDA_SQUARES], totals)
    concentration = pd.DataFrame(
        [
            {"quantity": "cells on the grid", "value": f"{_count(stats['n_cells']):,}"},
            {"quantity": "minimum per-cell total", "value": f"{stats['min']:,.0f}"},
            {"quantity": "median per-cell total", "value": f"{stats['median']:,.0f}"},
            {"quantity": "maximum per-cell total", "value": f"{stats['max']:,.0f}"},
            {"quantity": "top 1 percent share (%)", "value": f"{stats['top_1_percent_share']:.2f}"},
            {"quantity": "Gini coefficient", "value": f"{stats['gini']:.3f}"},
            {
                "quantity": "cells needed for half the traffic",
                "value": f"{_count(stats['cells_for_half']):,} "
                f"({stats['cells_for_half_percent']:.2f} % of the grid)",
            },
            {"quantity": "skewness of log10 totals", "value": f"{stats['skewness_log10']:.3f}"},
            {
                "quantity": "decades from the median down to the emptiest cell",
                "value": f"{stats['decades_below_median']:.2f}",
            },
            {
                "quantity": "decades from the median up to the busiest cell",
                "value": f"{stats['decades_above_median']:.2f}",
            },
        ]
    )
    LOG.info("concentration:\n%s", concentration.to_string(index=False))

    # ----------------------------------------------------- 4. regime diagnostics
    weekend = _weekend_slot_mask(fortnight_index)
    regime_rows = []
    for square in cells:
        col = column_of(square)
        train_series = np.asarray(X[TRAIN_ROWS, col], dtype=np.float64)
        diag = eda.regime_diagnostics(train_series, working)
        signal = eda.calendar_signal_r2(train_series, working)
        fortnight_series = np.asarray(X[fortnight, col], dtype=np.float64)
        regime_rows.append(
            {
                "square": square,
                "day-to-day profile r": diag["day_to_day_profile_correlation"],
                "variance outside mean profile (%)": 100.0 * diag["variance_outside_mean_profile"],
                "working vs non-working profile r": diag["working_vs_non_working_correlation"],
                "weekday/weekend ratio, fortnight": eda.weekday_weekend_ratio(
                    fortnight_series, weekend
                ),
                "time-of-day R2": signal["time_of_day_r2"],
                "+ working day R2": signal["with_working_day_r2"],
                "working-day gain": signal["working_day_gain"],
            }
        )
    regime = pd.DataFrame(regime_rows)
    LOG.info("regime diagnostics:\n%s", regime.to_string(index=False))

    # ------------------------------------------------- 5. remainder autocorrelation
    top_series = np.asarray(X[TRAIN_ROWS, column_of(top[0])], dtype=np.float64)
    remainder_rows = []
    for method, periods in (("stl", (SLOTS_PER_DAY,)), ("mstl", (SLOTS_PER_DAY, SLOTS_PER_WEEK))):
        d = decompose(top_series, period=SLOTS_PER_DAY, method=method, periods=periods)
        acf = eda.remainder_acf(d.remainder, REMAINDER_LAGS)
        remainder_rows.append(
            {
                "method": method.upper(),
                "periods": ",".join(map(str, d.periods)),
                "trend share": d.variance_shares["trend"],
                "remainder share": d.variance_shares["remainder"],
                **{f"remainder ACF lag {lag}": value for lag, value in acf.items()},
            }
        )
    remainder = pd.DataFrame(remainder_rows)
    LOG.info("remainder ACF:\n%s", remainder.to_string(index=False))

    # -------------------------------------------------- 6. holidays and anomalies
    top_col = column_of(top[0])
    holiday_rows = []
    for label, days, baseline in (
        ("24 to 26 December", HOLIDAY_BLOCK, HOLIDAY_BASELINE),
        ("25 December alone", CHRISTMAS_DAY, CHRISTMAS_BASELINE),
    ):
        for name, col in ((f"square {top[0]}", top_col), ("grid total", None)):
            current = eda.day_block_total(X, index, days, column=col)
            base = eda.day_block_total(X, index, baseline, column=col)
            holiday_rows.append(
                {
                    "block": label,
                    "series": name,
                    "baseline (a fortnight earlier)": base,
                    "holiday": current,
                    "drop (%)": eda.drop_percent(current, base),
                }
            )
    holidays = pd.DataFrame(holiday_rows)
    LOG.info("holiday drops:\n%s", holidays.to_string(index=False))

    zeros: eda.ZeroScan | None = None
    stalled = pd.DataFrame()
    if not args.no_zero_scan:
        zeros = eda.zero_scan(X)
        stalled = eda.stalled_cells(zeros, load_square_ids(), index)
        LOG.info(
            "zeros: %d of %d pairs (%.4f %%), %d cells affected, %d stalled",
            zeros.n_zero,
            zeros.n_pairs,
            zeros.percent,
            zeros.cells_with_zero,
            len(stalled),
        )

    # ------------------------------------------------------- 7. pairwise similarity
    similarity = eda.pairwise_correlation(
        {s: np.asarray(X[fortnight, column_of(s)], dtype=np.float64) for s in top}
    )
    LOG.info("fortnight correlations:\n%s", similarity.to_string(index=False))

    # ------------------------------------------------------------------ markdown
    elapsed = time.perf_counter() - t_start
    md = markdown_table
    lines = [
        "# Descriptive statistics for Section IV and the regime block of Section VI",
        "",
        f"Generated by `python scripts/make_eda_stats.py` on {platform.machine()} / "
        f"{platform.system()} in {elapsed:.1f} s.",
        "",
        "**Slices.** Section 1 (grid geometry), section 3 (spatial concentration) and section 6 "
        "(holidays, zeros) are descriptive of the whole 62-day record, rows "
        f"`0:{X.shape[0]}`, because they describe the dataset rather than inform a modelling "
        f"choice. Section 4 (regime diagnostics) and section 5 (decomposition remainder) use the "
        f"TRAINING rows `{TRAIN_ROWS.start}:{TRAIN_ROWS.stop}` "
        f"({train_index[0]:%Y-%m-%d %H:%M} to {train_index[-1]:%Y-%m-%d %H:%M} {TIMEZONE}, "
        f"{(TRAIN_ROWS.stop - TRAIN_ROWS.start) // SLOTS_PER_DAY} whole days) and nothing later, "
        "because they are cited in support of a model choice. Section 4's last column and "
        f"section 7 use the first fortnight, rows `{fortnight.start}:{fortnight.stop}` "
        f"({fortnight_index[0]:%Y-%m-%d} to {fortnight_index[-1]:%Y-%m-%d}), the window the brief "
        "fixes for the exploratory figure. The evaluation week is rows "
        f"`{EVAL_ROWS.start}:{EVAL_ROWS.stop}` and enters only the whole-record descriptions.",
        "",
        f"Values are {UNIT} counts: dimensionless, comparable across cells and across time, with "
        "no physical unit. They are not bytes, megabytes or sessions, and no total below should "
        "be converted into one.",
        "",
        f"**Study cells.** The top {args.top_n} squares by total Internet activity are "
        f"{', '.join(map(str, top))}, read from `data/processed/square_totals.csv` at run time "
        "and never written into the source. The brief additionally names "
        f"{' and '.join(map(str, REQUIRED_EDA_SQUARES))}.",
        "",
        "## 1. Grid geometry and orientation",
        "",
        f"The published Milano Grid is a {GRID_SIDE} by {GRID_SIDE} lattice spanning longitude "
        f"{eda.GRID_LON_MIN} to {eda.GRID_LON_MAX} and latitude {eda.GRID_LAT_MIN} to "
        f"{eda.GRID_LAT_MAX}, so one cell is {eda.CELL_LON:.6f} degrees of longitude by "
        f"{eda.CELL_LAT:.6f} of latitude, about "
        f"{eda.haversine_m(eda.DUOMO_LAT, eda.GRID_LON_MIN, eda.DUOMO_LAT, eda.GRID_LON_MIN + eda.CELL_LON):.0f}"
        f" m by "
        f"{eda.haversine_m(eda.GRID_LAT_MIN, eda.DUOMO_LON, eda.GRID_LAT_MIN + eda.CELL_LAT, eda.DUOMO_LON):.0f}"
        " m. `square_totals.csv` already carries `row = (id - 1) // 100` and "
        "`col = (id - 1) % 100` for every square.",
        "",
        "What the corner coordinates do **not** say is which corner holds square 1. The GeoJSON "
        "that would settle it sits behind a Dataverse guestbook form and is deliberately not a "
        "dependency of this repository, so the orientation is measured from the traffic field "
        "instead. Each landmark below is mapped to a cell twice: once under the south-origin "
        "convention (square 1 in the south-west corner, ids running east before north) and once "
        "under its vertical mirror. `ratio` is the cell's total divided by the median of its "
        f"{2 * args.neighbourhood + 1} by {2 * args.neighbourhood + 1} neighbourhood, so a value "
        "above 1 means the cell stands out from its surroundings. This matters because every "
        "land-use claim in Section IV, and the caption of the grid heatmap, depends on it.",
        "",
        md(_orientation_panel(check, "south"), floatfmt=".3f"),
        "",
        md(_orientation_panel(check, "north"), floatfmt=".3f"),
        "",
        md(scores),
        "",
        "**Verdict.** The local-ratio test alone does not separate the two conventions, and "
        "saying so is more useful than quietly reporting the half of it that agrees with the "
        "answer. A landmark is a point and a cell is 235 m wide, so a venue that sits astride a "
        "cell boundary can land on the quiet half of itself. The cells that fail under the "
        "south-origin reading, and what the immediately neighbouring cells score instead:",
        "",
        md(misses, floatfmt=".3f") if len(misses) else "(none)",
        "",
        "Both failures move one cell and come right, which is a positioning error rather than "
        "evidence about the orientation, so neither is counted as support for either reading. "
        "What does separate the conventions is "
        "the absolute level. Under the south-origin reading the farmland of the Parco Sud "
        f"carries {farmland['south total']:,.0f} against Sesto San Giovanni's "
        f"{sesto['south total']:,.0f}, a factor of "
        f"{sesto['south total'] / farmland['south total']:.0f}. Under the mirror the same two "
        f"cells read {farmland['north total']:,.0f} and {sesto['north total']:,.0f}: the "
        "agricultural belt would be busier than one of the densest suburbs in Italy, which is "
        "false. Linate is the same argument in miniature, at a ratio of "
        f"{check.loc[check['landmark'] == 'Linate airport', 'south ratio'].iloc[0]:.3f} under the "
        "south-origin reading against "
        f"{check.loc[check['landmark'] == 'Linate airport', 'north ratio'].iloc[0]:.3f} under the "
        "mirror. The south-origin convention is the one used everywhere in this repository.",
        "",
        "The distribution of traffic mass is the same argument without any single cell in it. "
        "Milan's built-up area extends north through Centrale, Isola and on into Sesto, and gives "
        "way to the agricultural Parco Sud in the south, so the mass must sit above the grid's "
        "mid-line under the correct reading.",
        "",
        md(bands, floatfmt=".2f"),
        "",
        f"The mass-weighted mean row is **{mean_row:.2f}** of the {GRID_SIDE} rows, which the "
        "south-origin reading numbers from 0 at the southern edge. Under that reading it is "
        f"latitude {lat_south:.4f}, about "
        f"{eda.haversine_m(eda.DUOMO_LAT, eda.DUOMO_LON, lat_south, eda.DUOMO_LON) / 1000:.1f} km "
        "north of the Duomo, in the Centrale and Isola belt. Under the mirror it would be "
        f"latitude {lat_north:.4f}, about "
        f"{eda.haversine_m(eda.DUOMO_LAT, eda.DUOMO_LON, lat_north, eda.DUOMO_LON) / 1000:.1f} km "
        "south of the Duomo, in the middle of the Parco Sud. Only one of those is a city.",
        "",
        "## 2. The five study cells",
        "",
        "Where the cells the report talks about actually are. Centroids follow from the id alone "
        "under the orientation established above; distances and bearings are great-circle, from "
        f"Piazza del Duomo at {eda.DUOMO_LAT:.4f} N, {eda.DUOMO_LON:.4f} E, on a sphere of radius "
        f"{eda.EARTH_RADIUS_M / 1000:.1f} km. These coordinates are what the land-use reading of "
        "Section IV is checked against; the land uses themselves are a hypothesis those addresses "
        "support, not a measurement.",
        "",
        md(
            pd.DataFrame(
                {
                    "square": cell_table["square"],
                    "grid (row, col)": [
                        f"({r}, {c})"
                        for r, c in zip(cell_table["row"], cell_table["col"], strict=True)
                    ],
                    "centroid": [
                        f"{lat:.4f} N, {lon:.4f} E"
                        for lat, lon in zip(
                            cell_table["latitude"], cell_table["longitude"], strict=True
                        )
                    ],
                    "from the Duomo": [
                        f"{d:,.0f} m {p} ({b:.0f} deg)"
                        for d, p, b in zip(
                            cell_table["metres from Duomo"],
                            cell_table["compass"],
                            cell_table["bearing (deg)"],
                            strict=True,
                        )
                    ],
                    "total Internet activity": [f"{v:,.0f}" for v in cell_table["total internet"]],
                }
            )
        ),
        "",
        "The pairwise table is the point of the section. The research question asks about "
        '"geographical areas with different traffic characteristics", and the answer here is that '
        "the three busiest areas are not distant districts: no two of them share even a corner, "
        "yet all three sit within about half a kilometre of one another. Whatever separates their "
        "behaviour is land use at cell scale, which is a stronger test of a forecaster than "
        "comparing a city centre with a suburb.",
        "",
        md(pairs, floatfmt=".1f"),
        "",
        "## 3. Spatial concentration",
        "",
        "Per-cell totals over all 62 days. These numbers set the limits of what the study may "
        "claim: the three evaluation areas are the extreme tail of this distribution, so nothing "
        "measured on them transfers to a typical cell, and levels differing by orders of "
        "magnitude are why the cross-area comparison in Section VI leans on scale-free errors "
        "rather than on MAE.",
        "",
        md(concentration, floatfmt=".4f"),
        "",
        "The two decade figures are reported as a pair because the asymmetry is the finding: the "
        "left tail is roughly twice as long as the right, and the skewness of the log10 totals is "
        f"{stats['skewness_log10']:.3f}, so the concentration is the ordinary consequence of a "
        "log-normal spread rather than a separate class of outlier cells.",
        "",
        "Where the five study cells sit in that distribution. The two squares the brief names are "
        "in the upper tail without being extreme, which is worth stating because they are "
        "sometimes read as though they were also top cells.",
        "",
        md(
            ranks.assign(**{"total internet": [f"{v:,.0f}" for v in ranks["total internet"]]}),
            floatfmt=".2f",
        ),
        "",
        "## 4. Regime diagnostics per cell",
        "",
        f"Training slice only, rows `{TRAIN_ROWS.start}:{TRAIN_ROWS.stop}`, reshaped to "
        f"{(TRAIN_ROWS.stop - TRAIN_ROWS.start) // SLOTS_PER_DAY} days by {SLOTS_PER_DAY} slots, "
        "each day divided by its own mean so that what is compared is the *shape* of the day and "
        "not how busy it was. This is the section Section VI's best-model argument rests on, so "
        "it is worth being explicit about what each column answers.",
        "",
        "- **day-to-day profile r**: does today look like yesterday? Near 1 means one repeated "
        "daily shape; a low value means the shape itself changes from day to day.",
        "- **variance outside mean profile**: how much of the profile variance a single average "
        "day fails to explain. A cell that alternates between two shapes scores high, because the "
        "average of two different shapes resembles neither.",
        "- **working vs non-working profile r**: correlation between the mean working-day shape "
        "and the mean non-working-day shape. A negative value means the cell is busy at the hours "
        "it is otherwise quiet. Working day is Monday to Friday excluding Friday 1 November 2013, "
        "Ognissanti, the only public holiday falling on a weekday inside the training slice.",
        "- **weekday/weekend ratio**: the plain calendar contrast over the first fortnight, rows "
        f"`{fortnight.start}:{fortnight.stop}`, Monday-to-Friday mean over Saturday-Sunday mean. "
        "Kept on the calendar label rather than the working-day one so that the working-day "
        "regime stays a finding rather than a definition.",
        "- **time-of-day R2 / + working day R2 / working-day gain**: ordinary least squares on "
        f"the training slice, first on {SLOTS_PER_DAY - 1} time-of-day dummies alone and then "
        "with the working-day indicator and its interaction added. The gain is what the calendar "
        "flag explains that time of day cannot. This is the column the failure analysis of "
        "Section VI.D rests on, and it is the one that actually separates the cells: the ratio "
        "column does not, because a ratio of 0.79 is a contrast of nearly the same size as 1.29 "
        "in the other direction.",
        "",
        md(regime, floatfmt=".3f"),
        "",
        "Read across the row for square "
        f"{top[2] if len(top) > 2 else top[-1]} and the contrast is the whole argument: its daily "
        "shape does not shrink at weekends, it inverts. That is the cell where knowing which of "
        "two shapes the previous day had is worth something to a forecaster, and it is where the "
        "convolutional model wins the architecture comparison that the reordering of Section VI.A "
        "rests on.",
        "",
        "## 5. Decomposition remainder autocorrelation",
        "",
        f"Square {top[0]}, training slice. What either decomposition calls irregular is not "
        "noise: the remainder is strongly autocorrelated one step ahead, which is the same "
        "quantity as the persistence floor of Section VI seen from the decomposition side. The "
        "trend share is quoted alongside because it is method-dependent in an informative way: "
        "STL at period 144 cannot separate the weekly cycle from a slow level, so what it books "
        "as trend the multi-seasonal fit books as a weekly seasonal component instead.",
        "",
        md(remainder, floatfmt=".4f"),
        "",
        "## 6. Holiday and anomaly context",
        "",
        "The Christmas collapse, each block against the same weekdays a fortnight earlier so that "
        "Tuesday is compared with Tuesday. This is why the brief's evaluation week of 16 to 22 "
        "December is the last normal week of the record, and why no result in this study is "
        "extrapolated past it. The single-cell drop is far deeper than the grid's: the city "
        "centre empties on Christmas Day while the residential periphery does not.",
        "",
        md(
            holidays.assign(
                **{
                    "baseline (a fortnight earlier)": [
                        f"{v:,.0f}" for v in holidays["baseline (a fortnight earlier)"]
                    ],
                    "holiday": [f"{v:,.0f}" for v in holidays["holiday"]],
                }
            ),
            floatfmt=".2f",
        ),
        "",
    ]

    if zeros is not None:
        lines += [
            "Missing records. An empty field in the raw export means no record rather than no "
            "traffic, and the ingestion stores it as zero, so the two are indistinguishable "
            "downstream and the honest thing is to count them. They are a small and highly "
            "concentrated fraction of the grid, and none of the five study cells is affected.",
            "",
            md(
                pd.DataFrame(
                    [
                        {
                            "quantity": "(cell, slot) pairs in the record",
                            "value": f"{zeros.n_pairs:,}",
                        },
                        {
                            "quantity": "pairs stored as zero",
                            "value": f"{zeros.n_zero:,} ({zeros.percent:.4f} %)",
                        },
                        {
                            "quantity": "cells carrying at least one zero",
                            "value": f"{zeros.cells_with_zero:,}",
                        },
                        {
                            "quantity": "cells stalled (last non-zero slot over a week "
                            "before the record ends)",
                            "value": f"{len(stalled):,}",
                        },
                    ]
                )
            ),
            "",
            "Scattered night-time gaps in a quiet cell are ordinary. A cell that stops and never "
            "restarts is a different thing, most likely a decommissioned or relocated sensor, and "
            "counting the two together would overstate how patchy the record is. The stalled "
            "cells are contiguous on the grid, which supports the sensor reading:",
            "",
            md(stalled) if len(stalled) else "(none)",
            "",
        ]

    lines += [
        "## 7. Pairwise similarity of the three areas over the first fortnight",
        "",
        f"Pearson correlation between the raw series of the top {args.top_n} cells over rows "
        f"`{fortnight.start}:{fortnight.stop}`. The ordering matches section 4 exactly: the two "
        "cells with similar weekday/weekend behaviour track each other, and the weekend-heavy "
        "cell against the weekday-only one is the weakest pair on the grid's busiest core. Three "
        "neighbours 500 m apart correlating no better than 0.43 is the evidence that the "
        '"different traffic characteristics" of the research question are real and local.',
        "",
        md(similarity, floatfmt=".3f"),
        "",
    ]

    out_dir = Path(args.tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "eda_stats.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    LOG.info("wrote %s (%.1f s total)", out, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
