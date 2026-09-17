"""Reproduce the two model-selection claims that had no script behind them.

Two sentences in the report rest on numbers that were computed interactively and never
committed: that the three evaluated cells come first under alternative definitions of
"busiest", and that SARIMA was rejected because one likelihood evaluation at seasonal
period 144 is expensive enough that a fit does not finish in a sensible budget. Section
VIII claims every number in the report was produced by a script in the repository, so
those two need one. This is it.

The checks live here rather than in ``src/`` on purpose: they verify claims about
choices already made, they are not steps of the modelling pipeline, and nothing else
imports them. All data access goes through ``milan_traffic.dataio``, and the square ids
are never written down: they are read out of the processed store every time.

    python scripts/check_model_selection.py                 # both checks
    python scripts/check_model_selection.py --no-sarimax    # ranking only, a few seconds
    python scripts/check_model_selection.py --budget 300

Output: ``reports/tables/model_selection_checks.md``, and the same text on stdout.
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

from milan_traffic.config import SLOTS_PER_DAY, TABLES_DIR, TARGET  # noqa: E402
from milan_traffic.dataio import (  # noqa: E402
    column_of,
    load_index,
    load_matrix,
    load_square_ids,
)
from milan_traffic.periods import to_markdown  # noqa: E402
from milan_traffic.tsa import EVAL_ROWS, TRAIN_ROWS  # noqa: E402
from milan_traffic.utils.logging import get_logger  # noqa: E402

LOG = get_logger("check_model_selection")

#: The holiday days dropped by the third definition. The record contains Christmas, and a
#: ranking that only survives because of it would not be a ranking of normal-week demand.
HOLIDAY_DAYS = ("2013-12-24", "2013-12-25", "2013-12-26")

#: Rows per chunk when scanning the matrix. 288 rows x 10 000 cells in float64 is 23 MB,
#: so the whole-grid scan never holds more than a fiftieth of the matrix resident.
CHUNK_ROWS = 2 * SLOTS_PER_DAY

Order = tuple[int, int, int]

#: Candidate (order, seasonal order) pairs, seasonal period supplied at run time. The first
#: is the headline configuration and the cheapest of the four: no differencing at all, which
#: is the smallest state vector a seasonal ARIMA on this series can have. The others are the
#: orders the ACF and PACF would actually send you to, and they are what makes the cost a
#: range rather than a single number.
SARIMA_CANDIDATES: tuple[tuple[Order, Order], ...] = (
    ((1, 0, 1), (1, 0, 1)),
    ((2, 0, 2), (1, 0, 1)),
    ((1, 0, 1), (0, 1, 1)),
    ((1, 1, 1), (1, 1, 1)),
)


class _BudgetExceeded(RuntimeError):
    """Raised inside the likelihood to stop a fit that has used up its time budget."""


class TimedLoglike:
    """Instrumented stand-in for ``SARIMAX.loglike`` that also enforces a time budget.

    Timing the optimiser from the outside says nothing about why a fit is slow. Counting
    the likelihood evaluations and timing each one does, and raising from inside the
    objective is the one interruption point that cannot be missed: the optimiser cannot
    take another step without calling it.
    """

    def __init__(self, inner, budget_s: float) -> None:
        self._inner = inner
        self._budget_s = budget_s
        self.started_at = time.perf_counter()
        self.durations: list[float] = []

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at

    def __call__(self, *args, **kwargs):
        if self.elapsed_s > self._budget_s:
            raise _BudgetExceeded
        t0 = time.perf_counter()
        out = self._inner(*args, **kwargs)
        self.durations.append(time.perf_counter() - t0)
        return out


def column_statistics(
    matrix: np.ndarray, mask: np.ndarray, chunk_rows: int = CHUNK_ROWS
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell sum and non-zero slot count over the rows ``mask`` selects.

    Scans the memory-mapped matrix in row blocks, so a whole-grid statistic costs one
    block of RAM rather than the 353 MB the matrix occupies on disk.
    """
    n_rows, n_cols = matrix.shape
    if mask.shape != (n_rows,):
        raise ValueError(f"mask has shape {mask.shape}, expected ({n_rows},)")
    totals = np.zeros(n_cols, dtype=np.float64)
    counts = np.zeros(n_cols, dtype=np.int64)
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        selected = mask[start:stop]
        if not selected.any():
            continue
        block = np.asarray(matrix[start:stop][selected], dtype=np.float64)
        totals += np.nansum(block, axis=0)
        counts += (np.nan_to_num(block) > 0.0).sum(axis=0)
    return totals, counts


def ranking_definitions(
    matrix: np.ndarray, index: pd.DatetimeIndex
) -> dict[str, tuple[np.ndarray, str]]:
    """The four definitions of "busiest cell", each as a per-cell score and its unit."""
    n_rows = matrix.shape[0]
    all_rows = np.ones(n_rows, dtype=bool)

    pre_eval = np.zeros(n_rows, dtype=bool)
    pre_eval[: EVAL_ROWS.start] = True

    holidays = pd.DatetimeIndex([pd.Timestamp(d) for d in HOLIDAY_DAYS])
    not_holiday = ~np.isin(index.normalize().tz_localize(None), holidays)

    totals_all, counts_all = column_statistics(matrix, all_rows)
    totals_pre, _ = column_statistics(matrix, pre_eval)
    totals_work, _ = column_statistics(matrix, not_holiday)
    mean_nonzero = np.divide(
        totals_all, counts_all, out=np.zeros_like(totals_all), where=counts_all > 0
    )
    return {
        "total, all 62 days": (totals_all, "sum"),
        "total, rows before the evaluation week": (totals_pre, "sum"),
        "total, excluding 24 to 26 December": (totals_work, "sum"),
        "mean over non-zero slots": (mean_nonzero, "mean"),
    }


def ranking_tables(
    definitions: dict[str, tuple[np.ndarray, str]], square_ids: np.ndarray, top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Rank table, score table for the cells that appear in it, and top-3 agreement."""
    order = {
        name: square_ids[np.argsort(-score, kind="stable")]
        for name, (score, _) in definitions.items()
    }
    ranks = pd.DataFrame(
        {name: ids[:top_n].astype(int) for name, ids in order.items()},
        index=pd.RangeIndex(1, top_n + 1, name="rank"),
    ).reset_index()

    agree = len({tuple(ids[:3]) for ids in order.values()}) == 1

    # Ordered by the definition the project uses, so the score table reads down the same
    # ranking as the table above it; ids are strings so they are not formatted as decimals.
    primary = next(iter(order.values()))
    shown = [int(s) for s in primary[:top_n]]
    shown += sorted({int(s) for ids in order.values() for s in ids[:top_n]} - set(shown))
    positions = {s: int(np.flatnonzero(square_ids == s)[0]) for s in shown}
    scores = pd.DataFrame({"square": [str(s) for s in shown]})
    for name, (score, kind) in definitions.items():
        column = [score[positions[s]] for s in shown]
        scores[name] = [v / 1e6 if kind == "sum" else v for v in column]
    return ranks, scores, agree


def _build(series: np.ndarray, order: Order, seasonal: Order, period: int):
    """A SARIMAX at ``period``, with the parameter constraints off so nothing is refitted."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    return SARIMAX(
        series,
        order=order,
        seasonal_order=(*seasonal, period),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )


def likelihood_costs(series: np.ndarray, period: int, repeats: int) -> pd.DataFrame:
    """Seconds per likelihood evaluation for each candidate order, at ``start_params``.

    The rejection is of SARIMA as a family, so the cost that matters is the cost of the
    orders one would actually consider, not only of the cheapest one. Differencing is
    what makes the difference: a seasonal difference at period 144 doubles the state
    vector and the Kalman filter carries a matrix of that size across every observation.
    """
    rows = []
    for order, seasonal in SARIMA_CANDIDATES:
        model = _build(series, order, seasonal, period)
        params = model.start_params
        model.loglike(params)  # untimed warm-up: the first call allocates the filter
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.loglike(params)
            times.append(time.perf_counter() - t0)
        rows.append(
            {
                "order": str(order),
                "seasonal order": str((*seasonal, period)),
                # Strings: these are labels, and one float format governs the whole table.
                "state dim": str(model.k_states),
                "free params": str(len(params)),
                "min (s)": min(times),
                "median (s)": float(np.median(times)),
                "max (s)": max(times),
            }
        )
    return pd.DataFrame(rows)


def fit_attempt(series: np.ndarray, period: int, budget_s: float, maxiter: int) -> pd.DataFrame:
    """How far a single fit of the headline order gets before ``budget_s`` runs out."""
    order, seasonal = SARIMA_CANDIDATES[0]
    model = _build(series, order, seasonal, period)
    timed = TimedLoglike(model.loglike, budget_s)
    model.loglike = timed  # instrument the objective the optimiser actually calls
    try:
        result = model.fit(disp=False, maxiter=maxiter)
        outcome = f"finished inside the budget, converged={bool(result.mle_retvals['converged'])}"
    except _BudgetExceeded:
        outcome = f"stopped by the {budget_s:.0f} s budget, unconverged"
    calls = np.asarray(timed.durations, dtype=float)
    rows = [
        ("order", f"{order} x {(*seasonal, period)}"),
        ("observations in the training slice", str(series.size)),
        ("state dimension", str(model.k_states)),
        ("time budget (s)", f"{budget_s:.0f}"),
        ("iteration cap", str(maxiter)),
        ("likelihood evaluations completed", str(calls.size)),
        ("wall time of the attempt (s)", f"{timed.elapsed_s:.1f}"),
        ("outcome", outcome),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--top-n", type=int, default=5, help="rows of the ranking table")
    parser.add_argument("--activity", default=TARGET)
    parser.add_argument(
        "--sarimax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="time the SARIMAX likelihood as well as the ranking",
    )
    parser.add_argument("--period", type=int, default=SLOTS_PER_DAY)
    parser.add_argument("--repeats", type=int, default=3, help="timed likelihood evaluations")
    parser.add_argument("--budget", type=float, default=120.0, help="fit time budget in seconds")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args(argv)

    if args.top_n < 3:
        parser.error("--top-n must be at least 3; the claim under test is about the top three")

    matrix = load_matrix(args.activity)
    index = load_index()
    square_ids = load_square_ids()
    LOG.info("scanning %s, shape %s", args.activity, matrix.shape)

    definitions = ranking_definitions(matrix, index)
    ranks, scores, agree = ranking_tables(definitions, square_ids, args.top_n)
    top3 = [int(v) for v in ranks[ranks.columns[1]].tolist()[:3]]

    lines = [
        "# Model selection checks",
        "",
        "Produced by `scripts/check_model_selection.py`, which reads `data/processed/` and",
        "writes this file. It exists to put a script behind two claims that were otherwise",
        "sourced to an interactive session: the stability of the three evaluated cells, and",
        "the measured cost of a seasonal ARIMA at period 144.",
        "",
        "## 1. Does the same top three come first under other definitions of busiest?",
        "",
        "Four definitions over the same processed matrix. All 62 days is the definition the",
        "project uses. Rows before the evaluation week is the leakage-free version of it:",
        "nothing after 15 December contributes. Excluding 24 to 26 December removes the",
        "holiday collapse. The mean over non-zero slots divides out how often a cell is",
        "active at all, so it cannot be won on volume alone.",
        "",
        to_markdown(ranks, floatfmt="{:.0f}"),
        (
            f"**The top three agree under all four definitions: {agree}.** "
            f"They are {top3[0]}, {top3[1]} and {top3[2]}."
        ),
        "",
        "Scores for every cell that appears above. Sums are in millions of activity units,",
        "the mean is in activity units per active slot. The values are scaled CDR counts and",
        "carry no physical unit.",
        "",
        to_markdown(scores, floatfmt="{:.2f}"),
    ]

    if args.sarimax:
        top_square = top3[0]
        series = np.asarray(matrix[TRAIN_ROWS, column_of(top_square)], dtype=np.float64)
        LOG.info("timing SARIMAX on square %d, %d rows", top_square, series.size)
        costs = likelihood_costs(series, args.period, args.repeats)
        budget = fit_attempt(series, args.period, args.budget, args.maxiter)
        lo, hi = costs["min (s)"].min(), costs["max (s)"].max()
        dims = costs["state dim"].astype(int)
        heavy = dims > dims.min() + 5  # the differenced orders, whose state vector doubles
        cost_ratio = (
            costs.loc[heavy, "median (s)"].mean() / costs.loc[~heavy, "median (s)"].mean()
            if heavy.any() and (~heavy).any()
            else float("nan")
        )
        differencing = (
            ""
            if np.isnan(cost_ratio)
            else (
                " Differencing is what moves it: the undifferenced orders carry a state vector"
                f" of {dims[~heavy].min()} to {dims[~heavy].max()}, a difference at the seasonal"
                f" period takes it to {dims[heavy].min()} to {dims[heavy].max()}, and the"
                f" likelihood then costs {cost_ratio:.1f} times as much."
            )
        )
        lines += [
            "",
            f"## 2. What does one SARIMA likelihood cost at period {args.period}?",
            "",
            f"Square {top_square}, training slice rows {TRAIN_ROWS.start}:{TRAIN_ROWS.stop}",
            "(1 November to 8 December), the same slice every model is fitted on. A daily",
            f"seasonal term at period {args.period} puts about {args.period} lags into the",
            "state vector, and the Kalman filter carries a matrix of that size across every",
            "observation, once per likelihood evaluation. Each row is the median of",
            f"{args.repeats} evaluations at `start_params` after one untimed warm-up.",
            "",
            to_markdown(costs),
            (
                f"**One likelihood evaluation costs {lo:.2f} to {hi:.2f} s** across the"
                f" {len(costs)} candidate orders.{differencing}"
            ),
            "",
            "A fit needs many of those. Below is what a single fit of the cheapest order does",
            "with a fixed budget. It cannot hang: the likelihood refuses to run once the budget",
            "is spent, and the optimiser is capped at a fixed number of iterations besides.",
            "",
            to_markdown(budget),
            f"**Hardware.** {platform.system()} | {platform.release()} | {platform.machine()}",
        ]

    text = "\n".join(lines).rstrip() + "\n"
    out = Path(args.tables_dir) / "model_selection_checks.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    LOG.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
