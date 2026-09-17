"""Where in the evaluation week a model fails, as opposed to how large its weekly average is.

One MAE per model per area cannot answer the brief's requirement for a period on which a model
performs poorly, because the worst such period here is two days wide and invisible in the weekly
aggregate. On square 5059 the convolutional model scores 78.88 against the linear autoregression's
66.25 over the week; outside Thursday 19 and Friday 20 December the two are level at 70.14 against
67.43, and on those two days the network loses to persistence. A reader given only the weekly
number would conclude the model is uniformly mediocre on that cell, which is the wrong diagnosis
and points at the wrong fix.

So this module cuts the 1008 slots by calendar period and scores each one with the same functions
as the headline tables, which is why a per-day number and a weekly number in the report cannot
disagree. It also carries the three diagnostics that discriminate between the candidate
explanations for a bad period, because naming a day is worth little and measuring what is
different about it is worth a lot: level bias against dispersion (:func:`residual_diagnostics`
reports the share of MSE that is squared bias), the volatility of the period itself, and the
correlation of the residual with the series' first difference, which is exactly 1 for a
persistence forecast and so detects a model that is merely following the series one step late.
:func:`calendar_penalty` is the same cut applied to a control run rather than to a baseline: it
sizes what an input feature costs on the bad days against what it costs over the week.

It sits beside ``evaluate.py`` rather than inside it because ``evaluate.py`` runs models and this
only reads their stored output. The seed convention is the one the results tables use throughout:
the mean over seeds of the per-seed statistic, never the statistic of the averaged forecast.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from . import metrics as M
from .features import is_working_day
from .significance import dm_test, seed_averaged_loss

#: Day labels in the figures and tables, e.g. "Thu 19 Dec".
DAY_LABEL_FMT: Final[str] = "%a %d %b"

#: Restated in every generated caption, because a per-day number computed the other way would
#: silently disagree with the report's per-area tables.
SEED_CONVENTION: Final[str] = (
    "Values are the mean over seeds of each per-seed statistic, the same convention as the "
    "per-area results tables. Deterministic reference forecasters have a seed spread of zero."
)


def _stack(pred: ArrayLike, n: int) -> np.ndarray:
    """``(n_seeds, n)`` float64. A 1-D deterministic forecast becomes a single seed."""
    a = np.asarray(pred, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"Expected (n_seeds, n) or (n,), got {a.shape}.")
    if a.shape[1] != n:
        raise ValueError(f"Prediction has {a.shape[1]} steps against {n} in the index.")
    return a


def _seed_reduce(values: Sequence[float]) -> tuple[float, float]:
    """``(mean, sample sd)`` over seeds; the sd is 0.0 for a single deterministic series."""
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def period_masks(index: pd.DatetimeIndex, period: str = "day") -> list[tuple[str, np.ndarray]]:
    """``[(label, boolean mask), ...]`` in chronological order.

    ``period="daytype"`` splits on :func:`milan_traffic.features.is_working_day` rather than on
    the weekday number, so a public holiday inside the window would group with the weekend. There
    is none between 16 and 22 December 2013, but the evaluation week is a configuration value and
    the grouping should not silently become wrong if it moves.
    """
    idx = pd.DatetimeIndex(index)
    if period == "day":
        days = idx.normalize()
        out = []
        for day in pd.unique(days):
            mask = np.asarray(days == day)
            out.append((pd.Timestamp(day).strftime(DAY_LABEL_FMT), mask))
        return out
    if period == "daytype":
        working = is_working_day(idx).astype(bool)
        return [("Mon-Fri", working), ("Sat-Sun", ~working)]
    raise ValueError("period must be 'day' or 'daytype'.")


def per_period_errors(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    index: pd.DatetimeIndex,
    *,
    period: str = "day",
) -> pd.DataFrame:
    """One row per (period, model): MAE, RMSE, WAPE and the **signed** mean error, with seed sds.

    ``ME`` is truth minus prediction, so a negative value means the model over-predicts. It is
    here because it is what separates a biased forecast from a noisy one, and the two call for
    different fixes.
    """
    t = np.asarray(y_true, dtype=np.float64).ravel()
    rows: list[dict[str, object]] = []
    for label, mask in period_masks(index, period):
        level = float(t[mask].mean())
        for name, pred in predictions.items():
            stack = _stack(pred, t.size)[:, mask]
            truth = t[mask]
            mae_m, mae_s = _seed_reduce([M.mae(truth, row) for row in stack])
            rmse_m, rmse_s = _seed_reduce([M.rmse(truth, row) for row in stack])
            wape_m, wape_s = _seed_reduce([M.wape(truth, row) for row in stack])
            me_m, me_s = _seed_reduce([float(np.mean(truth - row)) for row in stack])
            rows.append(
                {
                    "period": label,
                    "model": name,
                    "n_slots": int(mask.sum()),
                    "n_seeds": int(stack.shape[0]),
                    "mean_level": level,
                    "MAE": mae_m,
                    "MAE_sd": mae_s,
                    "RMSE": rmse_m,
                    "RMSE_sd": rmse_s,
                    "WAPE": wape_m,
                    "WAPE_sd": wape_s,
                    "ME": me_m,
                    "ME_sd": me_s,
                }
            )
    return pd.DataFrame(rows)


def excess_error_share(
    y_true: ArrayLike,
    pred: ArrayLike,
    reference: ArrayLike,
    index: pd.DatetimeIndex,
    *,
    period: str = "day",
) -> pd.DataFrame:
    """Where a model's total deficit to ``reference`` comes from, period by period.

    Two denominators are reported and both are named, because they differ materially: 85.6 percent
    of the *net* excess against 77.6 percent of the *positive* excess on the square 5059
    convolutional case. Quoting one without saying which is how an unreproducible "88 percent"
    gets into a draft. ``share_of_net_min`` and ``share_of_net_max`` are taken across seeds, so
    the artefact itself answers whether a concentration holds per seed or only on the seed mean.
    """
    t = np.asarray(y_true, dtype=np.float64).ravel()
    a = _stack(pred, t.size)
    b = _stack(reference, t.size)
    masks = period_masks(index, period)
    per_seed = np.empty((a.shape[0], len(masks)), dtype=np.float64)
    for s in range(a.shape[0]):
        ref = b[s % b.shape[0]]
        for j, (_, mask) in enumerate(masks):
            per_seed[s, j] = float(
                np.abs(t[mask] - a[s][mask]).sum() - np.abs(t[mask] - ref[mask]).sum()
            )
    net = per_seed.sum(axis=1)
    positive = np.where(per_seed > 0, per_seed, 0.0).sum(axis=1)
    rows: list[dict[str, object]] = []
    for j, (label, mask) in enumerate(masks):
        col = per_seed[:, j]
        share_net = 100.0 * col / np.where(net == 0, np.nan, net)
        share_pos = 100.0 * col / np.where(positive == 0, np.nan, positive)
        rows.append(
            {
                "period": label,
                "n_slots": int(mask.sum()),
                "excess": float(col.mean()),
                "excess_sd": float(col.std(ddof=1)) if col.size > 1 else 0.0,
                "share_of_net": float(np.nanmean(share_net)),
                "share_of_positive": float(np.nanmean(share_pos)),
                "share_of_net_min": float(np.nanmin(share_net)),
                "share_of_net_max": float(np.nanmax(share_net)),
            }
        )
    rows.append(
        {
            "period": "TOTAL",
            "n_slots": int(t.size),
            "excess": float(net.mean()),
            "excess_sd": float(net.std(ddof=1)) if net.size > 1 else 0.0,
            "share_of_net": 100.0,
            "share_of_positive": 100.0,
            "share_of_net_min": 100.0,
            "share_of_net_max": 100.0,
        }
    )
    return pd.DataFrame(rows)


def calendar_penalty(
    y_true: ArrayLike,
    pred: ArrayLike,
    reference: ArrayLike,
    index: pd.DatetimeIndex,
    zoom: ArrayLike,
    *,
    seeds: Sequence[int] | None = None,
    period: str = "day",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What an input feature costs, period by period and seed by seed.

    The penalty at slot *t* is ``|y - pred| - |y - reference|``: the pointwise absolute error of
    the model that was given the feature minus that of the same architecture trained without it,
    so a positive value means the feature hurt at that slot. ``zoom`` is the boolean mask of the
    named period whose share is being quoted, and need not line up with the calendar periods.

    The share to quote is the one computed on the **seed-mean pointwise penalty**, which is the
    ``zoom`` row of the first frame. It is deliberately not the mean of the per-seed shares in the
    second, and the two differ by a lot: a seed whose week total is small because it gains on the
    excluded days can put well over 100 percent of its own penalty on the zoom days, and averaging
    ratios like that weights it as heavily as every other seed. Both frames are returned so the
    spread stays visible in the artefact rather than being asserted in prose.

    Returns ``(per-period frame, per-seed frame)``. The per-period frame carries one row per
    calendar period, then the ``zoom`` row, then ``whole week``.
    """
    t = np.asarray(y_true, dtype=np.float64).ravel()
    sel = np.asarray(zoom, dtype=bool)
    on = np.abs(t - _stack(pred, t.size))
    off = np.abs(t - _stack(reference, t.size))
    pooled = on.mean(axis=0) - off.mean(axis=0)
    per_seed = on - off

    week = float(pooled.sum())
    masks = period_masks(index, period)
    zoom_label = " + ".join(label for label, mask in masks if bool((mask & sel).any()))
    rows = [
        {
            "period": label,
            "n_slots": int(mask.sum()),
            "penalty_per_slot": float(pooled[mask].mean()),
            "penalty_total": float(pooled[mask].sum()),
            "share_of_week_pct": 100.0 * float(pooled[mask].sum()) / week,
        }
        for label, mask in [*masks, (zoom_label, sel), ("whole week", np.ones_like(sel))]
    ]

    named = seeds is not None and len(seeds) == per_seed.shape[0]
    seed_rows = [
        {
            "seed": str(seeds[s]) if named else f"#{s}",
            "penalty_zoom": float(per_seed[s, sel].sum()),
            "penalty_week": float(per_seed[s].sum()),
            "share_of_week_pct": 100.0 * float(per_seed[s, sel].sum() / per_seed[s].sum()),
        }
        for s in range(per_seed.shape[0])
    ]
    return pd.DataFrame(rows), pd.DataFrame(seed_rows)


def _bias_share(resid: np.ndarray) -> float:
    """Percentage of MSE attributable to the mean error rather than to dispersion.

    NaN for an identically zero residual: there is no error to attribute, and returning 0.0
    would read as "none of this model's error is bias" for a model that has none.
    """
    mse = float(np.mean(resid**2))
    if mse == 0.0:
        return float("nan")
    return 100.0 * float(resid.mean()) ** 2 / mse


def _is_constant(v: np.ndarray) -> bool:
    """Whether ``v`` is constant to within floating-point noise at its own scale.

    An exact equality test is not enough: ``y - (y + 7.0)`` is -7 everywhere in exact arithmetic
    and varies in the last bit in practice, which is sufficient to give a constant residual a
    spurious correlation of a few percent instead of no correlation at all.
    """
    scale = max(abs(float(np.mean(v))), 1.0)
    return bool(float(np.ptp(v)) <= 1e-9 * scale)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, NaN when either series is constant, and without a numpy warning.

    A constant residual (a perfect or uniformly offset forecast) has no correlation with
    anything, and ``np.corrcoef`` reports that by dividing by zero.
    """
    if a.size < 2 or _is_constant(a) or _is_constant(b):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def residual_diagnostics(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    *,
    mask: ArrayLike | None = None,
) -> pd.DataFrame:
    """Bias against dispersion against lag, over the slots selected by ``mask``.

    ``bias2_share`` is the percentage of MSE attributable to the squared mean error rather than
    to dispersion: it separates a model that sits systematically high from one that is merely
    noisy. ``corr_resid_diff`` is the correlation of the residual with the first difference of
    the series, which is 1.0 by construction for persistence and therefore tests whether a model
    is simply following the series one step late. ``mean_abs_diff`` is the period's own
    volatility, so an explanation appealing to a turbulent period can be ruled in or out.
    """
    t = np.asarray(y_true, dtype=np.float64).ravel()
    sel = np.ones(t.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    diff = np.diff(t, prepend=t[0])
    rows: list[dict[str, object]] = []
    for name, pred in predictions.items():
        stack = _stack(pred, t.size)
        me, resid_sd, bias2, corr = [], [], [], []
        for row in stack:
            r = t[sel] - row[sel]
            me.append(float(r.mean()))
            resid_sd.append(float(r.std(ddof=1)))
            bias2.append(_bias_share(r))
            corr.append(_safe_corr(r, diff[sel]))
        me_m, me_s = _seed_reduce(me)
        rows.append(
            {
                "model": name,
                "n_slots": int(sel.sum()),
                "ME": me_m,
                "ME_sd": me_s,
                "resid_sd": float(np.mean(resid_sd)),
                "bias2_share": float(np.mean(bias2)),
                "corr_resid_diff": float(np.mean(corr)),
                "mean_abs_diff": float(np.mean(np.abs(diff[sel]))),
            }
        )
    return pd.DataFrame(rows)


def error_concentration(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    *,
    counts: Sequence[int] = (10, 50, 101),
) -> pd.DataFrame:
    """Share of the week's pooled absolute error carried by the worst ``k`` slots.

    The reason the failure case has to be a period rather than an outlier: the worst 1 percent of
    slots carry only about 5 percent of the error here, so no handful of slots explains a model's
    ranking. The pooled loss is the mean of the seed-averaged absolute error across the models
    given.
    """
    t = np.asarray(y_true, dtype=np.float64).ravel()
    pooled = np.mean([seed_averaged_loss(t, pred) for pred in predictions.values()], axis=0)
    order = np.sort(pooled)[::-1]
    total = float(order.sum())
    rows = [
        {
            "n_slots": int(k),
            "pct_of_week": 100.0 * k / t.size,
            "pct_of_error": 100.0 * float(order[:k].sum()) / total if total > 0 else float("nan"),
        }
        for k in counts
    ]
    return pd.DataFrame(rows)


def worst_slots(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    index: pd.DatetimeIndex,
    *,
    k: int = 8,
) -> pd.DataFrame:
    """The ``k`` slots with the largest pooled absolute error, worst first.

    ``delta`` is the first difference of the truth at that slot. Every one of the worst slots on
    this dataset is a large step that all models miss on the same side, which is what says the
    remaining headroom is in the turning points rather than in the level.
    """
    t = np.asarray(y_true, dtype=np.float64).ravel()
    idx = pd.DatetimeIndex(index)
    means = {name: seed_averaged_loss(t, pred) for name, pred in predictions.items()}
    pooled = np.mean(list(means.values()), axis=0)
    pick = np.argsort(pooled)[::-1][:k]
    diff = np.diff(t, prepend=np.nan)
    rows = []
    for i in pick:
        row: dict[str, object] = {
            "timestamp": idx[i],
            "truth": float(t[i]),
            "delta": float(diff[i]),
            "pooled_ae": float(pooled[i]),
        }
        for name, pred in predictions.items():
            row[name] = float(_stack(pred, t.size)[:, i].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def compare_on_mask(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    index: pd.DatetimeIndex,
    masks: Mapping[str, ArrayLike],
    *,
    lag: int | None = None,
) -> pd.DataFrame:
    """Diebold-Mariano on each named subset of the week, via :func:`significance.dm_test`.

    The test is the one the results section uses; only the slots change. Restricting to a short
    period shrinks n (144 slots for a single day), so the asymptotic p-value there is indicative
    rather than exact, and the generated caption says so.
    """
    t = np.asarray(y_true, dtype=np.float64).ravel()
    la = seed_averaged_loss(t, pred_a)
    lb = seed_averaged_loss(t, pred_b)
    rows = []
    for label, mask in masks.items():
        sel = np.asarray(mask, dtype=bool)
        res = dm_test(la[sel] - lb[sel], lag=lag)
        rows.append(
            {
                "period": label,
                "n": res.n,
                "d": res.mean,
                "se_hac": res.se,
                "t": res.t,
                "p": res.p,
                "lag": res.lag,
            }
        )
    return pd.DataFrame(rows)


def to_markdown(frame: pd.DataFrame, *, floatfmt: str = "{:.2f}") -> str:
    """Pipe-delimited markdown, right-aligned numerics, no tabulate dependency."""
    cols = list(frame.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(frame[c])]
    # Columns may be non-strings: a frame keyed by square id has integer column labels.
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    rule = "|" + "|".join("--:" if c in numeric else ":--" for c in cols) + "|"
    lines = [header, rule]
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in numeric and pd.notna(v):
                cells.append(floatfmt.format(float(v)))
            else:
                cells.append("n/a" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
