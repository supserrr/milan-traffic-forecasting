"""Forecast error metrics.

MAE, RMSE and MAPE are required by the brief. MAPE is included with a documented
guard because it is genuinely fragile here: overnight traffic in low-activity cells
approaches zero, and an unguarded percentage error there produces numbers in the
thousands that say more about the denominator than about the model. sMAPE and WAPE are
provided as the stable companions to report alongside it.

Every metric here refuses to score a forecast that contains non-finite values. Dropping
the bad points and averaging the rest is the intuitive thing to do and is exactly wrong:
it rewards the failure it should expose, because a model that diverges over part of the
evaluation week is then scored only on the part it survived. See :func:`_prep`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | Sequence[float]
MapeMode = Literal["epsilon", "mask", "raw"]


_EMPTY = np.empty(0, dtype=np.float64)


def _as_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    if t.shape != p.shape:
        raise ValueError(f"Shape mismatch: y_true {t.shape} vs y_pred {p.shape}.")
    return t, p


def finite_counts(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[int, int]:
    """``(n_scored, n_dropped)`` - pairs where both values are finite, and pairs where not.

    Reported alongside every metric so a run's artefacts record how many of the 1008
    evaluation slots actually produced a number.
    """
    t, p = _as_pair(y_true, y_pred)
    n_scored = int(np.count_nonzero(np.isfinite(t) & np.isfinite(p)))
    return n_scored, int(t.size - n_scored)


def _prep(
    y_true: ArrayLike, y_pred: ArrayLike, *, allow_missing: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned pairs, or nothing at all if any pair is non-finite.

    A NaN loss is the commonest way a sequence model fails, and the tempting response -
    mask the non-finite points out and average what remains - inverts the result. Scored
    that way, a persistence forecast with NaN from step 200 onward returns MAE 69.04
    against the intact forecast's 92.80, so the broken model wins the comparison table.

    So unless ``allow_missing`` is set, a single non-finite value empties the arrays and
    every metric returns NaN. A NaN in a results table is unmissable; a flattering MAE is
    not. Pass ``allow_missing=True`` only when scoring a deliberately partial forecast,
    and report ``finite_counts`` beside it.
    """
    t, p = _as_pair(y_true, y_pred)
    finite = np.isfinite(t) & np.isfinite(p)
    if not allow_missing and not bool(finite.all()):
        return _EMPTY, _EMPTY
    return t[finite], p[finite]


def mae(y_true: ArrayLike, y_pred: ArrayLike, *, allow_missing: bool = False) -> float:
    """Mean absolute error, in the units of the series."""
    t, p = _prep(y_true, y_pred, allow_missing=allow_missing)
    if t.size == 0:
        return float("nan")
    return float(np.mean(np.abs(t - p)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike, *, allow_missing: bool = False) -> float:
    """Root mean squared error - penalises the large misses at traffic peaks."""
    t, p = _prep(y_true, y_pred, allow_missing=allow_missing)
    if t.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mape(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    mode: MapeMode = "epsilon",
    epsilon: float = 1e-3,
    floor: float | None = None,
    allow_missing: bool = False,
) -> float:
    """Mean absolute percentage error (%).

    ``mode``
        ``"epsilon"`` - divide by ``max(|y|, epsilon)`` (default; nothing is dropped).
        ``"mask"``    - exclude points with ``|y| <= floor`` and report on the rest.
        ``"raw"``     - textbook definition, will return ``inf`` if any target is zero.

    Whichever mode is used, **state it in the report**: MAPE values are not comparable
    across different guards.
    """
    t, p = _prep(y_true, y_pred, allow_missing=allow_missing)
    if t.size == 0:
        return float("nan")
    if mode == "raw":
        # Deliberately returns inf when a target is zero: that is the textbook
        # definition failing, and seeing it fail is the point of offering the mode.
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.mean(np.abs((t - p) / t)) * 100.0)
    if mode == "mask":
        thresh = epsilon if floor is None else floor
        keep = np.abs(t) > thresh
        if not keep.any():
            return float("nan")
        return float(np.mean(np.abs((t[keep] - p[keep]) / t[keep])) * 100.0)
    denom = np.maximum(np.abs(t), epsilon)
    return float(np.mean(np.abs((t - p) / denom)) * 100.0)


def smape(y_true: ArrayLike, y_pred: ArrayLike, *, allow_missing: bool = False) -> float:
    """Symmetric MAPE (%), bounded at 200 % and defined when the target is zero."""
    t, p = _prep(y_true, y_pred, allow_missing=allow_missing)
    denom = (np.abs(t) + np.abs(p)) / 2.0
    keep = denom > 0
    if not keep.any():
        return float("nan")
    return float(np.mean(np.abs(t[keep] - p[keep]) / denom[keep]) * 100.0)


def wape(y_true: ArrayLike, y_pred: ArrayLike, *, allow_missing: bool = False) -> float:
    """Weighted absolute percentage error (%): total error over total actual.

    Scale-free like MAPE but stable at low traffic, because the normalisation is done
    once over the whole window instead of point by point.
    """
    t, p = _prep(y_true, y_pred, allow_missing=allow_missing)
    total = np.sum(np.abs(t))
    return float(np.sum(np.abs(t - p)) / total * 100.0) if total > 0 else float("nan")


def r2(y_true: ArrayLike, y_pred: ArrayLike, *, allow_missing: bool = False) -> float:
    """Coefficient of determination against the mean of the evaluation window.

    Note the benchmark is a constant, which discriminates weakly on a series this
    autocorrelated: on square 5161's evaluation week a forecast two hours stale still
    scores 0.62. Lead with MASE and WAPE; quote R2 with its reference point stated.
    """
    t, p = _prep(y_true, y_pred, allow_missing=allow_missing)
    if t.size == 0:
        return float("nan")
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def seasonal_naive_scale(y: ArrayLike, *, season: int = 144) -> float:
    """Mean absolute seasonal-naive error of ``y``, the denominator :func:`mase` divides by.

    Compute this on the **training** period and hand it to :func:`mase` as ``scale``.
    """
    v = np.asarray(y, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size <= season:
        return float("nan")
    return float(np.mean(np.abs(v[season:] - v[:-season])))


def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    season: int = 144,
    scale: float | None = None,
    allow_missing: bool = False,
) -> float:
    """Mean absolute scaled error against a seasonal-naive forecast.

    ``scale`` is the mean absolute seasonal-naive error to divide by. Pass the value
    :func:`seasonal_naive_scale` returns for the **training** period: that is the standard
    (Hyndman) definition, and only then does "below 1" mean "beats seasonal persistence".

    When ``scale`` is omitted the denominator is taken from ``y_true`` itself, i.e. from
    the evaluation window. That variant is self-contained but it is **not** comparable
    across windows, and on this dataset it inflates the score by a factor of 1.38 to 1.69, because
    the fixed evaluation week is calmer than the training period. Report which was used.
    """
    t, p = _prep(y_true, y_pred, allow_missing=allow_missing)
    if t.size == 0:
        return float("nan")
    if scale is None:
        if t.size <= season:
            return float("nan")
        scale = float(np.mean(np.abs(t[season:] - t[:-season])))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(np.mean(np.abs(t - p)) / scale)


def evaluate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    season: int = 144,
    mape_mode: MapeMode = "epsilon",
    mase_scale: float | None = None,
    allow_missing: bool = False,
) -> dict[str, float]:
    """All metrics for one model on one window, plus how many points were scored.

    ``mase_scale`` is forwarded to :func:`mase`; pass the training-period value from
    :func:`seasonal_naive_scale` so MASE keeps its usual interpretation.

    ``n_scored`` and ``n_dropped`` are always reported. They belong in the artefacts even
    when nothing went wrong: ``n_scored`` confirms the window count a run actually used,
    and a non-zero ``n_dropped`` beside a row of NaNs says which model diverged and how
    far, rather than leaving a gap to be explained from memory later.
    """
    n_scored, n_dropped = finite_counts(y_true, y_pred)
    opt = {"allow_missing": allow_missing}
    return {
        "MAE": mae(y_true, y_pred, **opt),
        "RMSE": rmse(y_true, y_pred, **opt),
        "MAPE": mape(y_true, y_pred, mode=mape_mode, **opt),
        "sMAPE": smape(y_true, y_pred, **opt),
        "WAPE": wape(y_true, y_pred, **opt),
        "MASE": mase(y_true, y_pred, season=season, scale=mase_scale, **opt),
        "R2": r2(y_true, y_pred, **opt),
        "n_scored": n_scored,
        "n_dropped": n_dropped,
    }


def metrics_table(
    results: Mapping[str, tuple[ArrayLike, ArrayLike]],
    *,
    season: int = 144,
    mape_mode: MapeMode = "epsilon",
    mase_scale: float | None = None,
    sort_by: str = "MAE",
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Comparison table across models: ``{model_name: (y_true, y_pred)}``."""
    rows = {
        name: evaluate(
            t,
            p,
            season=season,
            mape_mode=mape_mode,
            mase_scale=mase_scale,
            allow_missing=allow_missing,
        )
        for name, (t, p) in results.items()
    }
    df = pd.DataFrame(rows).T
    df.index.name = "model"
    return df.sort_values(sort_by)
