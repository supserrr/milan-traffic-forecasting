"""Criterion-3 time-series diagnostics for one square, on the training slice only.

Thin CLI. All statistics live in ``milan_traffic.tsa``; this file only sequences them,
renders the markdown tables and saves the figures.

    python scripts/run_tsa.py                        # top square, extras = rest of top 3
    python scripts/run_tsa.py --square 5161 --nlags 1100 --no-figures

Outputs: ``reports/tables/tsa_<square>.md`` and ``reports/figures/{acf_pacf,stl,rolling_stats}_<square>.{pdf,png}``.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from milan_traffic import tsa  # noqa: E402
from milan_traffic.config import SLOTS_PER_DAY, SLOTS_PER_WEEK, TABLES_DIR, TIMEZONE  # noqa: E402
from milan_traffic.dataio import column_of, load_index, load_matrix, top_squares  # noqa: E402
from milan_traffic.utils.logging import get_logger  # noqa: E402
from milan_traffic.viz import save_figure, use_style  # noqa: E402

LOG = get_logger("run_tsa")

#: Lags quoted in the report: the first few, the daily neighbourhood, two days, one week.
REPORT_LAGS = (1, 6, 36, 143, 144, 145, 288, 1008)
#: ISO weeks 45 to 51 of 2013: five training weeks, the validation week, the evaluation week.
REPORT_WEEKS = tuple(range(45, 52))
TRAIN_WEEKS = tuple(range(45, 50))


def _train_series(X: np.ndarray, square: int) -> np.ndarray:
    return np.asarray(X[tsa.TRAIN_ROWS, column_of(square)], dtype=np.float64)


def _week_split(index: pd.DatetimeIndex, week_start) -> str:
    """Split of the first row on or after the week's Monday (partial first week included)."""
    monday = pd.Timestamp(week_start).tz_localize(TIMEZONE)
    return tsa.split_label(int(index.searchsorted(monday)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--square", type=int, default=None, help="square id; default: top square by total"
    )
    parser.add_argument(
        "--extra-squares",
        type=int,
        nargs="*",
        default=None,
        help="squares for the cheap comparisons; default: the rest of the top 3",
    )
    parser.add_argument("--nlags", type=int, default=1100)
    parser.add_argument("--practical-threshold", type=float, default=0.05)
    parser.add_argument("--anomaly-threshold", type=float, default=1.5)
    parser.add_argument("--window", type=int, default=SLOTS_PER_DAY, help="rolling window in slots")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    top3 = top_squares(3)
    square = args.square if args.square is not None else top3[0]
    extras = (
        args.extra_squares if args.extra_squares is not None else [s for s in top3 if s != square]
    )
    squares = [square, *extras]

    X = load_matrix("internet")
    index = load_index()
    train_index = index[tsa.TRAIN_ROWS]
    x = _train_series(X, square)
    n = x.size
    band = tsa.confidence_band(n)
    LOG.info("square %d, training rows %s, n=%d, band=%.4f", square, tsa.TRAIN_ROWS, n, band)

    # ---------------------------------------------------------------- ACF and PACF
    table = tsa.acf_pacf(x, args.nlags, pacf_method="ldb")
    pacf_vals = table["pacf"].to_numpy()
    selected = table.loc[table["lag"].isin(REPORT_LAGS), ["lag", "acf", "pacf"]].copy()
    selected["|pacf| > band"] = np.abs(selected["pacf"]) > band
    selected[f"|pacf| > {args.practical_threshold:.2f}"] = (
        np.abs(selected["pacf"]) > args.practical_threshold
    )

    cutoff_rows = []
    for max_lag in (200, 300, args.nlags):
        cutoff_rows.append(
            {
                "search range": f"1..{max_lag}",
                f"last |pacf| > band ({band:.4f})": tsa.last_significant_lag(
                    pacf_vals, band, max_lag=max_lag
                ),
                f"last |pacf| > {args.practical_threshold:.2f}": tsa.practical_cutoff(
                    pacf_vals, args.practical_threshold, max_lag=max_lag
                ),
            }
        )
    cutoffs = pd.DataFrame(cutoff_rows)
    exceed_rows = []
    for lo, hi in ((1, 300), (301, args.nlags)):
        s = tsa.exceedance_summary(pacf_vals, band, lo=lo, hi=hi)
        exceed_rows.append(
            {
                "lag range": f"{s['lo']}..{s['hi']}",
                "lags": s["n_lags"],
                "count |pacf| > band": s["count"],
                "fraction": s["fraction"],
                "expected by chance (5%)": s["expected_by_chance"],
            }
        )
    exceedances = pd.DataFrame(exceed_rows)
    LOG.info("cutoffs:\n%s", cutoffs.to_string(index=False))

    # --------------------------------------------------------------- decomposition
    xl = np.log1p(x)
    decomps: list[tuple[str, tsa.Decomposition]] = []
    timings: dict[str, float] = {}
    for label, series in (("raw", x), ("log1p", xl)):
        for method in ("stl", "mstl", "classical"):
            t0 = time.perf_counter()
            d = tsa.decompose(
                series, period=SLOTS_PER_DAY, method=method, periods=(SLOTS_PER_DAY, SLOTS_PER_WEEK)
            )
            timings[f"{method} {label}"] = time.perf_counter() - t0
            decomps.append((label, d))
            LOG.info("%s: %s", label, d.describe())
    shares = tsa.variance_share_table(decomps)
    stl_raw = next(d for label, d in decomps if label == "raw" and d.method == "stl")

    # ---------------------------------------------------------------- stationarity
    stationarity = tsa.stationarity_tests(x, seasonal_lag=SLOTS_PER_DAY, kpss_regression="c")
    stationarity_tbl = tsa.stationarity_table(stationarity)
    LOG.info("stationarity:\n%s", stationarity_tbl.to_string(index=False))

    # ------------------------------------------------------------ rolling mean/std
    r_raw = tsa.rolling_mean_std_correlation(x, window=args.window)
    r_log = tsa.rolling_mean_std_correlation(xl, window=args.window)
    rolling_tbl = pd.DataFrame(
        {"series": ["raw", "log1p"], f"Pearson r (window {args.window})": [r_raw, r_log]}
    )
    LOG.info("rolling mean/std r: raw %.3f, log1p %.3f", r_raw, r_log)

    # ------------------------------------------------------ weekly weekday means
    span = slice(0, tsa.EVAL_ROWS.stop)
    span_index = index[span]
    columns: dict[str, np.ndarray] = {
        str(s): np.asarray(X[span, column_of(s)], dtype=np.float64) for s in squares
    }
    columns["grid"] = tsa.grid_totals(X, span)
    weekly_parts = {name: tsa.weekly_weekday_means(v, span_index) for name, v in columns.items()}
    base = next(iter(weekly_parts.values()))
    base = base[base.index.isin(REPORT_WEEKS) & (base["iso_year"] == 2013)]
    weekly_parts = {k: v.loc[base.index] for k, v in weekly_parts.items()}
    weekly = pd.DataFrame(
        {
            "iso_week": base.index,
            "week_start": base["week_start"].to_numpy(),
            "split": [_week_split(index, d) for d in base["week_start"]],
            "n_days": base["n_days"].to_numpy(),
        }
    )
    for name, part in weekly_parts.items():
        weekly[name] = part["mean"].to_numpy()
    weekly = weekly.reset_index(drop=True)
    train_mean = weekly[weekly["iso_week"].isin(TRAIN_WEEKS)][list(columns)].mean()
    drift_rows = [{"quantity": "mean of training weeks 45-49", **train_mean.to_dict()}]
    for wk in (50, 51):
        row = weekly.loc[weekly["iso_week"] == wk, list(columns)].iloc[0]
        drift_rows.append(
            {
                "quantity": f"week {wk} vs training mean (%)",
                **((row / train_mean - 1.0) * 100.0).to_dict(),
            }
        )
    drift = pd.DataFrame(drift_rows)
    LOG.info("weekly weekday means:\n%s", weekly.to_string(index=False))

    # ------------------------------------------------------------- anomaly scan
    grid_train = columns["grid"][tsa.TRAIN_ROWS]
    grid_scan = tsa.same_slot_anomaly_scan(
        grid_train, period=SLOTS_PER_WEEK, threshold=args.anomaly_threshold, index=train_index
    )
    order = np.argsort(-np.nan_to_num(grid_scan.ratio, nan=-np.inf))[:5]
    grid_top = pd.DataFrame(
        {
            "row": order,
            "time": [train_index[i].strftime("%a %Y-%m-%d %H:%M") for i in order],
            "ratio": grid_scan.ratio[order],
        }
    )
    square_scan = tsa.same_slot_anomaly_scan(
        x, period=SLOTS_PER_WEEK, threshold=args.anomaly_threshold, index=train_index
    )
    LOG.info("grid anomalies:\n%s", grid_scan.flagged.to_string(index=False))

    # -------------------------------------------------------------- daily peaks
    peak_rows = []
    for s in squares:
        peaks = tsa.daily_peak_times(_train_series(X, s), train_index)
        summary = tsa.summarise_daily_peaks(peaks)
        summary.insert(0, "square", s)
        peak_rows.append(summary)
    peaks_tbl = pd.concat(peak_rows, ignore_index=True)

    # ------------------------------------------------------------------ figures
    figure_paths: list[Path] = []
    if not args.no_figures:
        import matplotlib.pyplot as plt

        use_style()
        title = f"square {square}, training slice ({n} slots, {train_index[0]:%d %b} to {train_index[-1]:%d %b %Y})"
        fig = tsa.plot_acf_pacf(table, practical_threshold=args.practical_threshold, title=title)
        figure_paths += save_figure(fig, f"acf_pacf_{square}")
        plt.close(fig)
        fig = tsa.plot_decomposition(
            stl_raw, x, train_index, title=f"STL, period {SLOTS_PER_DAY}: {title}"
        )
        figure_paths += save_figure(fig, f"stl_{square}")
        plt.close(fig)
        fig = tsa.plot_rolling_scatter(x, window=args.window, title=title)
        figure_paths += save_figure(fig, f"rolling_stats_{square}")
        plt.close(fig)
        for p in figure_paths:
            LOG.info("wrote %s", p)

    # ---------------------------------------------------------------- markdown
    elapsed = time.perf_counter() - t_start
    md = tsa.markdown_table
    lines = [
        f"# Time-series diagnostics, square {square} (Internet traffic)",
        "",
        f"Generated by `python scripts/run_tsa.py --square {square} --nlags {args.nlags}` on "
        f"{platform.machine()} / {platform.system()} in {elapsed:.1f} s.",
        "",
        f"**Slice.** Every statistic below is computed on the TRAINING rows `{tsa.TRAIN_ROWS.start}:{tsa.TRAIN_ROWS.stop}` "
        f"({train_index[0]:%Y-%m-%d %H:%M} to {train_index[-1]:%Y-%m-%d %H:%M} CET, n = {n} slots), "
        f"except the weekly weekday means (section 6), which span ISO weeks {REPORT_WEEKS[0]} to {REPORT_WEEKS[-1]} "
        "and are labelled by split. Validation is rows "
        f"`{tsa.VAL_ROWS.start}:{tsa.VAL_ROWS.stop}` (Dec 9 to 15) and the evaluation week rows "
        f"`{tsa.EVAL_ROWS.start}:{tsa.EVAL_ROWS.stop}` (Dec 16 to 22); neither enters sections 1 to 5, 7 or 8. "
        "The earlier version of these numbers used the full 8928-slot series including the held-out week.",
        "",
        f"Values are {tsa.UNIT} counts (dimensionless, comparable across cells and time), not bytes.",
        "",
        "## 1. ACF and PACF at selected lags",
        "",
        f"PACF by Levinson-Durbin (`statsmodels.tsa.stattools.pacf(method='ldb')`), ACF by FFT. "
        f"95 percent band = 1.96 / sqrt({n}) = **{band:.4f}** (the old log quoted 0.0207, which is the band for n = 8928). "
        f"The practical cutoff is a magnitude of {args.practical_threshold:.2f}; the two criteria are reported separately.",
        "",
        md(selected, floatfmt=".4f"),
        "",
        "## 2. Last lag above each criterion",
        "",
        "The answer depends on how far one searches, so the search range is stated:",
        "",
        md(cutoffs),
        "",
        "Band exceedances by lag range. Under white noise about 5 percent of lags exceed the band by chance; "
        "a range whose fraction sits near 0.05 carries no evidence of structure.",
        "",
        md(exceedances, floatfmt=".3f"),
        "",
        "## 3. Variance shares by decomposition method",
        "",
        "`var(component) / var(series)` over the rows where every component is defined "
        "(`n_valid`; the classical filter loses 72 rows at each end). Shares need not sum to one because "
        "the components are not orthogonal. STL and classical use period 144 (one day); MSTL fits periods 144 and 1008.",
        "",
        md(shares, floatfmt=".4f"),
        "",
        "Fit times (s): " + ", ".join(f"{k} {v:.2f}" for k, v in timings.items()) + ".",
        "",
        "## 4. Stationarity tests",
        "",
        f"ADF null: {tsa.ADF_NULL}; a small p rejects the unit root. "
        f"KPSS null (regression `c`): {tsa.KPSS_NULL}; a small p rejects stationarity. "
        "The two nulls point in opposite directions, so agreement is two pieces of evidence and disagreement "
        "is itself informative. KPSS p-values are read from a lookup table capped at [0.01, 0.10]; capped values are "
        "shown as inequalities. Lag orders: ADF by AIC, KPSS by the automatic Newey-West rule.",
        "",
        md(stationarity_tbl, floatfmt=".4g"),
        "",
        "## 5. Rolling mean against rolling standard deviation",
        "",
        f"Pearson r between the trailing {args.window}-slot rolling mean and rolling std. A high r on the raw series "
        "means the noise scales with the level (multiplicative noise); the drop after `log1p` is the case for "
        "modelling the transformed series.",
        "",
        md(rolling_tbl, floatfmt=".3f"),
        "",
        "## 6. Weekly Monday-to-Friday means per slot, ISO weeks 45 to 51",
        "",
        f"Mean per 10-minute slot over weekday slots, for squares {', '.join(map(str, squares))} and the grid total "
        "(all 10 000 cells). Weeks 45 to 49 are training, week 50 validation, week 51 the evaluation week. "
        "This table is descriptive context on drift across the splits; no modelling decision uses the "
        "validation or evaluation rows.",
        "",
        md(weekly, floatfmt=".1f"),
        "",
        md(drift, floatfmt=".1f"),
        "",
        "## 7. Same-slot-of-week anomaly scan",
        "",
        f"Ratio of each slot to the median of the same slot of the week (period {SLOTS_PER_WEEK}) over the training "
        f"rows; flagged when the ratio exceeds {args.anomaly_threshold:.2f}.",
        "",
        f"Grid totals: {len(grid_scan.flagged)} slot(s) flagged.",
        "",
        md(grid_scan.flagged, floatfmt=".3f") if len(grid_scan.flagged) else "(none)",
        "",
        "Five largest grid-total ratios, for context:",
        "",
        md(grid_top, floatfmt=".3f"),
        "",
        f"Square {square} alone: {len(square_scan.flagged)} slot(s) flagged"
        + (
            f"; the largest ratio is {np.nanmax(square_scan.ratio):.3f} at row "
            f"{int(np.nanargmax(square_scan.ratio))} "
            f"({train_index[int(np.nanargmax(square_scan.ratio))]:%a %Y-%m-%d %H:%M})."
        ),
        "",
        "Rows 0 to 143 are Friday 2013-11-01, All Saints' Day, a public holiday in Italy, so that day is "
        "compared with working Fridays; the other single-cell hits are mostly night slots with small "
        "denominators. The grid-wide scan above is the one that identifies shared events. Ten largest ratios:",
        "",
        (
            md(square_scan.flagged.sort_values("ratio", ascending=False).head(10), floatfmt=".3f")
            if len(square_scan.flagged)
            else "(none)"
        ),
        "",
        "## 8. Daily peak times, weekday against weekend",
        "",
        "Per-day time of the maximum over the training slice. A weekend median peak in the small hours means the "
        "daytime pattern is absent on those days rather than merely smaller (the regime finding on square 5259).",
        "",
        md(peaks_tbl, floatfmt=".1f"),
        "",
    ]
    if figure_paths:
        lines += (
            ["## Figures", ""] + [f"- `{p.relative_to(p.parents[2])}`" for p in figure_paths] + [""]
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / f"tsa_{square}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("wrote %s (%.1f s total)", out, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
