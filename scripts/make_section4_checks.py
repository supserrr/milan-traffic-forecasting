"""Put a script behind the Section 4 claims that no other table carries.

    python scripts/make_section4_checks.py        # -> reports/tables/section4_checks.md

`scripts/run_tsa.py` and `scripts/make_eda_stats.py` between them cover most of the
exploratory analysis, but each is scoped: the diagnostics run on square 5161 and the
descriptive statistics stop short of a few claims the prose makes. The numbers below
were otherwise sourced to an interactive session, which is the same defect
`scripts/check_model_selection.py` was written to close for the model-selection claims.

Six blocks, each named for the sentence it backs:

1. the partial autocorrelation of square 5259 read on the training slice and on the
   full record, which is the example Section 4 opens with;
2. the KPSS trend-stationary variant, which `run_tsa.py` reports only under regression
   `c`;
3. daily peak times over the first fortnight, split by working day rather than by
   weekend, for the five study cells;
4. the 6 December spike measured across cells rather than on the grid total alone;
5. the minimum of the evaluation week per study cell, which is what makes MAPE
   well defined;
6. the Monday-to-Friday level drift between the first full training week and the
   evaluation week;
7. the isolated partial autocorrelations beyond lag 300 on the highest-traffic cell;
8. the weekday and weekend levels of square 5259 and the profile correlation between
   squares 4159 and 5259, under both conventions the report could mean;
9. where absolute error sits when slots are ranked by traffic level rather than by
   error, which is the concentration claim of Section 6.2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from milan_traffic import dataio, tsa
from milan_traffic.config import (
    EDA_WINDOW_END,
    EDA_WINDOW_START,
    EVAL_WEEK_END,
    EVAL_WEEK_START,
    REQUIRED_EDA_SQUARES,
    RUNS_DIR,
    SLOTS_PER_DAY,
    TABLES_DIR,
)
from milan_traffic.features import is_working_day

TRAIN_END = 5472
PRACTICAL = 0.05
OUT = TABLES_DIR / "section4_checks.md"


def series(square: int, start: str | None = None, end: str | None = None) -> np.ndarray:
    return dataio.load_series(square, start=start, end=end).to_numpy(dtype=float)


def pacf_slices(square: int, max_lag: int) -> pd.DataFrame:
    """Last lag above the practical cutoff, training slice against the full record."""
    full = series(square)
    rows = []
    for name, x in (("training rows 0:5472", full[:TRAIN_END]), ("full record", full)):
        frame = tsa.acf_pacf(x, nlags=max_lag)
        pacf = frame["pacf"].to_numpy()[1:]
        above = np.flatnonzero(np.abs(pacf[:max_lag]) > PRACTICAL)
        rows.append(
            {
                "slice": name,
                "n": len(x),
                "search range": f"1..{max_lag}",
                f"last |pacf| > {PRACTICAL}": int(above[-1] + 1) if above.size else None,
                "band (95%)": round(tsa.confidence_band(len(x)), 4),
            }
        )
    return pd.DataFrame(rows)


def kpss_both_regressions(square: int) -> pd.DataFrame:
    """KPSS under a constant and under a constant plus trend, on the training slice."""
    x = series(square)[:TRAIN_END]
    rows = []
    for regression, label in (("c", "level stationary"), ("ct", "trend stationary")):
        res = tsa.stationarity_tests(np.asarray(x, dtype=float), kpss_regression=regression)["raw"]
        rows.append(
            {
                "KPSS regression": f"`{regression}` ({label} null)",
                "statistic": round(float(res["kpss_stat"]), 5),
                "p": res["kpss_p_display"],
                "lags": int(res["kpss_lags"]),
                "rejects the null at 5%": "yes" if float(res["kpss_p"]) < 0.05 else "no",
            }
        )
    return pd.DataFrame(rows)


def tightest_arc(minutes: np.ndarray) -> tuple[int, int]:
    """The narrowest window on the 24-hour clock containing every one of these times.

    A day's peaks can straddle midnight, so the window that covers 22:50, 00:50 and 02:10
    is 22:50 to 02:10 and not 00:50 to 22:50. Sorting the times and taking the widest gap
    between neighbours, wrapping, gives the arc to exclude; what remains is the answer.
    """
    day = 24 * 60
    m = np.unique(minutes)
    if m.size == 1:
        return int(m[0]), int(m[0])
    gaps = np.diff(np.append(m, m[0] + day))
    widest = int(np.argmax(gaps))
    return int(m[(widest + 1) % m.size]), int(m[widest])


def fortnight_peaks(squares: list[int]) -> pd.DataFrame:
    """Window containing every daily maximum over the first fortnight, by day type."""
    sub_index = dataio.load_index()[:0].append(
        dataio.load_series(squares[0], start=EDA_WINDOW_START, end=EDA_WINDOW_END).index
    )
    # One flag per calendar day, keyed by date, so the join cannot silently mis-align
    # if a day is short.
    working_by_date = (
        pd.Series(is_working_day(sub_index).astype(bool), index=sub_index.date)
        .groupby(level=0)
        .first()
    )

    out = []
    for square in squares:
        x = series(square, EDA_WINDOW_START, EDA_WINDOW_END)
        peaks = tsa.daily_peak_times(x, sub_index)
        day_working = peaks["date"].map(working_by_date).to_numpy(dtype=bool)
        for label, mask in (("working", day_working), ("non-working", ~day_working)):
            if not mask.any():
                continue
            times = pd.to_datetime(peaks.loc[mask, "peak_time"], format="%H:%M")
            minutes = (times.dt.hour * 60 + times.dt.minute).to_numpy()
            lo, hi = tightest_arc(minutes)
            out.append(
                {
                    "square": square,
                    "day type": label,
                    "days": int(mask.sum()),
                    "peaks fall between": f"{lo // 60:02d}:{lo % 60:02d} and "
                    f"{hi // 60:02d}:{hi % 60:02d}",
                }
            )
    return pd.DataFrame(out)


def spike_across_cells(when: str = "2013-12-06 20:50") -> pd.DataFrame:
    """The 6 December event measured cell by cell, not only on the grid total."""
    X = dataio.load_matrix()
    index = dataio.load_index()
    row = int(np.flatnonzero(index == pd.Timestamp(when, tz=index.tz))[0])

    before, at, after = X[row - 1], X[row], X[row + 1]
    neighbours = (before.astype(float) + after.astype(float)) / 2.0
    totals = X.sum(axis=1, dtype=np.float64)
    rises = np.diff(totals[:TRAIN_END])
    order = np.argsort(rises)[::-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(neighbours > 0, at / neighbours, np.nan)

    excess = totals[row] - neighbours.sum()
    # If the spike were traffic moved forward rather than added, the slot after it would
    # sit below the slot before it by the whole excess. The shortfall it actually shows is
    # the part that was moved; the rest was additional.
    shortfall = totals[row - 1] - totals[row + 1]

    return pd.DataFrame(
        [
            (
                "grid total at 20:40, 20:50, 21:00",
                f"{totals[row - 1]:,.0f}, {totals[row]:,.0f}, {totals[row + 1]:,.0f}",
            ),
            (
                "grid total at 20:50 over the mean of its neighbours",
                f"{totals[row] / neighbours.sum():.2f} times",
            ),
            ("cells higher at 20:50 than at 20:40", f"{100 * np.mean(at > before):.1f} %"),
            (
                "cells above 1.5 times their own neighbouring slots",
                f"{100 * np.nanmean(ratio > 1.5):.1f} %",
            ),
            (
                "largest single-step rise in the grid total, training rows",
                f"{rises[order[0]]:,.0f} at {index[order[0] + 1]:%Y-%m-%d %H:%M}",
            ),
            ("second largest", f"{rises[order[1]]:,.0f} at {index[order[1] + 1]:%Y-%m-%d %H:%M}"),
            ("excess over the neighbouring slots", f"{excess:,.0f}"),
            (
                "share of that excess given back in the following slot",
                f"{100 * shortfall / excess:.1f} %",
            ),
        ],
        columns=["quantity", "value"],
    )


def evaluation_week_minima(squares: list[int]) -> pd.DataFrame:
    """MAPE is only well defined because none of these is near zero."""
    rows = []
    for square in squares:
        x = series(square, EVAL_WEEK_START, EVAL_WEEK_END)
        rows.append(
            {
                "square": square,
                "minimum over the evaluation week": f"{x.min():.0f}",
                "slots": int(x.size),
            }
        )
    return pd.DataFrame(rows)


def weekday_drift(squares: list[int]) -> pd.DataFrame:
    """Monday-to-Friday mean per slot, first full training week against the test week."""
    index = dataio.load_index()
    first = (index >= pd.Timestamp("2013-11-04 00:00", tz=index.tz)) & (
        index <= pd.Timestamp("2013-11-08 23:50", tz=index.tz)
    )
    last = (index >= pd.Timestamp(EVAL_WEEK_START, tz=index.tz)) & (
        index <= pd.Timestamp("2013-12-20 23:50", tz=index.tz)
    )

    def drift(values: np.ndarray) -> tuple[float, float, float]:
        a, b = float(values[first].mean()), float(values[last].mean())
        return a, b, 100.0 * (b - a) / a

    rows = []
    grid_a, grid_b, grid_pct = drift(dataio.load_matrix().sum(axis=1, dtype=np.float64))
    rows.append(
        {
            "series": "grid total",
            "week of 4 Nov": f"{grid_a:,.0f}",
            "evaluation week": f"{grid_b:,.0f}",
            "change (%)": f"{grid_pct:+.1f}",
        }
    )
    for square in squares:
        a, b, pct = drift(series(square))
        rows.append(
            {
                "series": f"square {square}",
                "week of 4 Nov": f"{a:,.0f}",
                "evaluation week": f"{b:,.0f}",
                "change (%)": f"{pct:+.1f}",
            }
        )
    return pd.DataFrame(rows)


def isolated_partials(square: int, lo: int = 301, hi: int = 1100) -> pd.DataFrame:
    """Section 4.3 claims exactly two partials past lag 300 clear the practical cutoff."""
    x = series(square)[:TRAIN_END]
    pacf = tsa.acf_pacf(x, nlags=hi)["pacf"].to_numpy()[1:]
    lags = np.arange(1, pacf.size + 1)
    window = (lags >= lo) & (lags <= hi)
    hits = lags[window & (np.abs(pacf) > PRACTICAL)]
    return pd.DataFrame(
        [{"lag": int(lag), "pacf": round(float(pacf[lag - 1]), 4)} for lag in hits]
        or [{"lag": None, "pacf": None}]
    )


def regime_levels(weekday_cell: int, echo_cell: int) -> pd.DataFrame:
    """The 5259 weekday and weekend means, and how closely 4159 reproduces its shape."""
    index = dataio.load_index()
    fortnight = (index >= pd.Timestamp(EDA_WINDOW_START, tz=index.tz)) & (
        index <= pd.Timestamp(EDA_WINDOW_END, tz=index.tz)
    )
    weekend = index[fortnight].dayofweek >= 5

    rows = []
    for square in (weekday_cell, echo_cell):
        x = series(square)[fortnight]
        rows.append(
            {
                "square": square,
                "weekday mean, fortnight": f"{x[~weekend].mean():.0f}",
                "weekend mean, fortnight": f"{x[weekend].mean():.0f}",
                "ratio": f"{x[~weekend].mean() / x[weekend].mean():.2f}",
            }
        )
    frame = pd.DataFrame(rows)

    # The report calls 4159 an amplitude echo of 5259 rather than a second regime. The
    # correlation depends on which profile is compared, so both conventions are printed.
    def profile(square: int, normalise: bool) -> np.ndarray:
        days = series(square)[:TRAIN_END].reshape(-1, SLOTS_PER_DAY)
        if normalise:
            days = days / days.mean(axis=1, keepdims=True)
        return days.mean(axis=0)

    raw_days = series(echo_cell)[fortnight].reshape(-1, SLOTS_PER_DAY).mean(axis=0)
    ref_days = series(weekday_cell)[fortnight].reshape(-1, SLOTS_PER_DAY).mean(axis=0)
    frame.attrs["correlations"] = pd.DataFrame(
        [
            {
                "convention": "mean daily profile, raw, first fortnight",
                f"r({echo_cell}, {weekday_cell})": round(
                    float(np.corrcoef(raw_days, ref_days)[0, 1]), 3
                ),
            },
            {
                "convention": "mean daily profile, each day divided by its own mean, "
                "training slice",
                f"r({echo_cell}, {weekday_cell})": round(
                    float(np.corrcoef(profile(echo_cell, True), profile(weekday_cell, True))[0, 1]),
                    3,
                ),
            },
        ]
    )
    return frame


def peak_window_counts(squares: list[int], lo: str = "19:00", hi: str = "23:40") -> pd.DataFrame:
    """How many of the fortnight's daily maxima fall inside a named window."""
    sub_index = dataio.load_index()[:0].append(
        dataio.load_series(squares[0], start=EDA_WINDOW_START, end=EDA_WINDOW_END).index
    )
    lo_m = int(lo[:2]) * 60 + int(lo[3:])
    hi_m = int(hi[:2]) * 60 + int(hi[3:])
    rows = []
    for square in squares:
        peaks = tsa.daily_peak_times(series(square, EDA_WINDOW_START, EDA_WINDOW_END), sub_index)
        minutes = peaks["peak_time"].map(lambda t: int(t[:2]) * 60 + int(t[3:]))
        inside = int(((minutes >= lo_m) & (minutes <= hi_m)).sum())
        rows.append(
            {
                "square": square,
                f"days peaking within {lo} to {hi}": f"{inside} of {len(peaks)}",
            }
        )
    return pd.DataFrame(rows)


def error_by_traffic_level(run: str = "EXP-008", top_frac: float = 0.05) -> pd.DataFrame:
    """Section 6.2 ranks slots by LEVEL; per_period_stats.md ranks them by error.

    The two answer different questions and give different numbers, so the level-ranked
    version is computed here rather than left to be confused with the other one.
    """
    meta = json.loads((RUNS_DIR / run / "metrics.json").read_text())
    preds = np.load(RUNS_DIR / run / "predictions.npy")
    axes = meta["predictions_axes"]
    index = dataio.load_index()
    window = (index >= pd.Timestamp(EVAL_WEEK_START, tz=index.tz)) & (
        index <= pd.Timestamp(EVAL_WEEK_END, tz=index.tz)
    )

    rows = []
    for ai, area in enumerate(meta["areas"]):
        actual = series(area)[window]
        k = max(1, int(round(top_frac * actual.size)))
        busiest = np.argsort(actual)[::-1][:k]
        for mi, model in enumerate(meta["models"]):
            if model not in ("lstm", "tcn", "gbt", "naive", "linear_ar_144"):
                continue
            err = np.abs(preds[ai, mi].mean(axis=0) - actual)
            rows.append(
                {
                    "area": area,
                    "model": model,
                    "share of slots": f"{100 * k / actual.size:.1f} %",
                    "share of absolute error": f"{100 * err[busiest].sum() / err.sum():.1f} %",
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["axes"] = axes
    return frame


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--pacf-square", type=int, default=5259)
    p.add_argument("--pacf-max-lag", type=int, default=400)
    args = p.parse_args()

    started = time.perf_counter()
    top = list(dataio.top_squares(3))
    study = top + list(REQUIRED_EDA_SQUARES)
    regimes = regime_levels(top[2], REQUIRED_EDA_SQUARES[0])

    parts = [
        "# Section 4 claims that no other table carries",
        "",
        f"Generated by `python scripts/make_section4_checks.py`. Study cells: "
        f"{', '.join(str(s) for s in study)}, the top three by total Internet activity "
        "followed by the two the brief names.",
        "",
        f"## 1. Partial autocorrelation of square {args.pacf_square}, two slices",
        "",
        "Section 4's opening claim is that a sequence length read off the full record "
        "would have been set partly by the week the study is evaluated on. The two rows "
        "are the same calculation on the training rows and on all 8928 slots.",
        "",
        tsa.markdown_table(pacf_slices(args.pacf_square, args.pacf_max_lag), floatfmt=".4f"),
        "",
        f"## 2. KPSS under both regressions, square {top[0]}",
        "",
        "`scripts/run_tsa.py` reports KPSS under regression `c`. Section 4.4 also claims "
        "the trend-stationary variant does not reject, which is this table.",
        "",
        tsa.markdown_table(kpss_both_regressions(top[0]), floatfmt=".5f"),
        "",
        "## 3. Daily peak times over the first fortnight, by working day",
        "",
        "`scripts/run_tsa.py` section 8 splits the training slice by weekend. Section 4.2 "
        "makes a stronger claim over the fortnight and splits by working day, so that "
        "Friday 1 November, Ognissanti, counts as non-working. A window given as 22:50 to "
        "02:10 wraps past midnight.",
        "",
        tsa.markdown_table(fortnight_peaks(study)),
        "",
        "## 4. The 6 December spike measured across cells",
        "",
        "Section 4.5 argues the event is grid-wide rather than local, which needs the "
        "per-cell shares below and not only the grid total.",
        "",
        tsa.markdown_table(spike_across_cells()),
        "",
        "## 5. Minimum of the evaluation week per study cell",
        "",
        "Section 5.10 keeps MAPE on the grounds that traffic never approaches zero in the "
        "scored week, so the epsilon guard never binds.",
        "",
        tsa.markdown_table(evaluation_week_minima(top)),
        "",
        "## 6. Monday-to-Friday level drift",
        "",
        "Section 4.4 separates level stationarity from an unchanging level. The comparison "
        "is the mean per slot over Monday to Friday of the week of 4 November against the "
        "same weekdays of the evaluation week, before any holiday.",
        "",
        tsa.markdown_table(weekday_drift(study)),
        "",
        f"## 7. Partial autocorrelations past lag 300, square {top[0]}",
        "",
        "Section 4.3 claims exactly two partials beyond lag 300 clear the practical cutoff "
        "of 0.05 anywhere in the first 1100 lags. `scripts/run_tsa.py` reports the last such "
        "lag and the band-exceedance fraction, not the individual lags.",
        "",
        tsa.markdown_table(isolated_partials(top[0]), floatfmt=".4f"),
        "",
        f"## 8. Regime levels, and how closely square {REQUIRED_EDA_SQUARES[0]} echoes "
        f"square {top[2]}",
        "",
        "Section 4.2 quotes the weekday and weekend means behind the 2.98 ratio, and calls "
        f"square {REQUIRED_EDA_SQUARES[0]} the same shape at a fifth of the amplitude rather "
        "than a second regime. The correlation depends on which profile is compared, so both "
        "conventions are given; the report quotes the first.",
        "",
        tsa.markdown_table(regimes),
        "",
        tsa.markdown_table(regimes.attrs["correlations"], floatfmt=".3f"),
        "",
        "Section 4.2 also counts the days on which square 4556 peaks after dinner. Section 3 "
        "above gives the windows; this is the count inside the narrower one.",
        "",
        tsa.markdown_table(peak_window_counts(study)),
        "",
        "## 9. Where absolute error sits when slots are ranked by traffic level",
        "",
        "Section 6.2 claims the busiest 5 percent of the evaluation week's slots carry about a "
        "tenth of all absolute error, twice their share of the week. "
        "`reports/tables/per_period_stats.md` reports a different quantity, the share carried "
        "by the slots with the largest *errors*, so the two are easy to confuse. This table "
        "ranks by level, as Section 6.2 does. Forecasts are the seed mean of run EXP-008.",
        "",
        tsa.markdown_table(error_by_traffic_level()),
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts))
    print(f"wrote {args.out} in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
