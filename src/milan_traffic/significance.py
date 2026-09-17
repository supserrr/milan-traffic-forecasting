"""Whether two forecasters can be told apart, as opposed to which one scored lower.

A per-area MAE table says which model had the smallest number. It does not say whether that
number could have come out the other way. This module turns the 1008 paired forecast errors of
the evaluation week into an interval and a p-value, so the report can separate the one margin
that survives a test from the several that do not.

It answers a different question from the seed standard deviations already in the results tables.
Those measure training stochasticity: what a fourth seed might have produced. These measure
sampling error over the scored week: what a different week might have produced. Neither bounds
the other, and a fully unconditional interval would be wider than both, so the report quotes them
side by side rather than letting one look like a correction of the other.

One convention matters more than the choice of kernel. Each cell of the report's tables is the
mean over seeds of a per-seed metric, so the quantity to test is the difference of *seed-averaged
pointwise losses*, whose sample mean is exactly the printed margin (see
:func:`seed_averaged_loss`). Scoring the mean of the three forecasts instead would test a
three-member ensemble that appears in no table: on square 5161 that ensemble is 2.68 MAE units
better than the tabulated TCN, which is 94 percent of the 2.85-unit margin under test, and it
moves that comparison from p = 0.36 to p = 0.10. Ensembling helps a stochastic model and cannot
help a deterministic one, so testing it biases every trained-versus-linear comparison one way.

No scipy. It is installed here only as a transitive dependency of statsmodels and is imported
nowhere else in the project, and ``requirements.txt`` states that every package it lists is
imported somewhere. The Diebold-Mariano statistic is obtained instead by regressing the loss
differential on a constant with statsmodels' HAC covariance, which reproduces the textbook
Bartlett Newey-West form exactly, and the two normal quantiles needed for the power calculation
are hard-coded and pinned against ``math.erf`` in the tests.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import ArrayLike
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

#: The bandwidth rule, named here so the report's stated lag cannot drift from the code.
NW_LAG_RULE: Final[str] = "floor(1.5 * n ** (1/3))"

#: Standard normal quantiles, for the minimum-detectable-difference calculation. Hard-coded
#: because scipy is not a declared dependency; :func:`tests.test_significance` re-derives both
#: by bisection on ``math.erf`` so they cannot rot.
Z_975: Final[float] = 1.959963985
Z_80: Final[float] = 0.841621234


def newey_west_lag(n: int) -> int:
    """Bartlett bandwidth by the ``floor(1.5 * n ** (1/3))`` rule. 15 at n = 1008.

    A fixed rule rather than a per-comparison plug-in. Andrews' AR(1) bandwidth chooses between
    0 and 8 across the comparisons reported here, which would make a table's rows incomparable
    and invite the charge of tuning each test; 15 exceeds every one of those estimates, so it is
    the conservative choice relative to the data.
    """
    if n < 1:
        raise ValueError("n must be positive.")
    return int(math.floor(1.5 * n ** (1.0 / 3.0)))


def andrews_ar1_lag(d: ArrayLike) -> int:
    """Andrews (1991) AR(1) plug-in Bartlett bandwidth. Diagnostic only.

    Reported so that :func:`newey_west_lag` can be shown to be conservative relative to a
    data-driven choice. Never used to select the lag of a reported test.
    """
    v = np.asarray(d, dtype=np.float64).ravel()
    v = v - v.mean()
    if v.size < 3:
        return 0
    rho = float(np.dot(v[1:], v[:-1]) / np.dot(v[:-1], v[:-1]))
    rho = min(max(rho, -0.97), 0.97)
    alpha = 4.0 * rho**2 / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    return int(math.floor(1.1447 * (alpha * v.size) ** (1.0 / 3.0)))


def _stack(pred: ArrayLike) -> np.ndarray:
    """``(n_seeds, n_steps)`` float64, accepting a 1-D deterministic forecast as one seed."""
    a = np.asarray(pred, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"Expected predictions of shape (n_seeds, n) or (n,), got {a.shape}.")
    return a


def loss_series(y_true: ArrayLike, y_pred: ArrayLike, *, power: int = 1) -> np.ndarray:
    """Pointwise ``|y_true - y_pred| ** power`` for one forecast series.

    Returns an all-NaN array of the right length if any pair is non-finite, mirroring
    :func:`milan_traffic.metrics._prep`. Dropping the bad slots and scoring the rest rewards the
    failure it should expose: a model that diverges partway through the week would otherwise be
    scored only on the part it survived.
    """
    if power not in (1, 2):
        raise ValueError("power must be 1 (absolute) or 2 (squared).")
    t = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    if t.shape != p.shape:
        raise ValueError(f"Shape mismatch: y_true {t.shape} vs y_pred {p.shape}.")
    out = np.abs(t - p) ** power
    if not np.isfinite(out).all():
        return np.full(t.size, np.nan)
    return out


def seed_averaged_loss(y_true: ArrayLike, y_pred: ArrayLike, *, power: int = 1) -> np.ndarray:
    """Mean over seeds of the pointwise loss. **The unit of test.**

    ``y_pred`` is ``(n_seeds, n_steps)``. The mean of this series is exactly the seed-mean metric
    the report's tables print, which is why the differential between two of them tests the
    tabulated margin and nothing else.
    """
    stack = _stack(y_pred)
    losses = np.stack([loss_series(y_true, row, power=power) for row in stack])
    return np.asarray(losses.mean(axis=0), dtype=np.float64)


def ensemble_loss(y_true: ArrayLike, y_pred: ArrayLike, *, power: int = 1) -> np.ndarray:
    """Pointwise loss of the seed-mean *forecast*. Provided to quantify what it is not.

    By Jensen's inequality this never exceeds :func:`seed_averaged_loss` on average, so it is a
    different and better forecaster. It appears in no results table, and the gap between the two
    is reported in the reproduction artefact so the choice of unit is auditable.
    """
    stack = _stack(y_pred)
    return loss_series(y_true, stack.mean(axis=0), power=power)


@dataclass(frozen=True)
class DMResult:
    """One Diebold-Mariano comparison. ``mean`` is negative when the first model is better.

    ``p`` and the ``ci_low``/``ci_high`` interval take a **standard normal** reference, for the
    reason given in :func:`dm_test`. ``df`` is recorded because it is part of the run's record,
    not because anything here uses it as a reference distribution.
    """

    mean: float
    se: float
    t: float
    p: float
    ci_low: float
    ci_high: float
    lag: int
    n: int
    df: int
    hln: bool
    loss_power: int

    def one_sided_p(self, *, favours_first: bool) -> float:
        """``p / 2`` when the observed sign agrees with the stated direction, else ``1 - p / 2``.

        A claim that disagrees with its own point estimate must not pass quietly, which is why
        the disagreeing case returns the complement rather than the halved value.
        """
        if not math.isfinite(self.p):
            return float("nan")
        agrees = (self.mean < 0) if favours_first else (self.mean > 0)
        return self.p / 2.0 if agrees else 1.0 - self.p / 2.0


def dm_test(
    d: ArrayLike, *, lag: int | None = None, hln: bool = True, loss_power: int = 1
) -> DMResult:
    """Diebold-Mariano test on a loss differential ``d``, by HAC regression on a constant.

    ``lag`` defaults to :func:`newey_west_lag`. ``hln`` applies statsmodels'
    ``use_correction``, which for h = 1 is algebraically the Harvey, Leybourne and Newbold (1997)
    factor ``sqrt((n - 1) / n)``. At n = 1008 the correction moves a p-value by at most 2.4e-04,
    so it is applied because it is free rather than because it matters here.

    The reference distribution is the **standard normal**, not Student t. statsmodels forces
    ``use_t=False`` for any robust ``cov_type``, so ``fit.pvalues`` are normal tail probabilities
    and the interval below is built from :data:`Z_975`. Pairing the Harvey-Leybourne-Newbold
    factor with a t reference on ``n - 1`` = 1007 degrees of freedom would be the small-sample
    form of the test; across the fifteen comparisons the report makes it moves a p-value by at
    most 3.0e-04 (0.18956 to 0.18986 on square 5161, TCN versus LSTM) and changes no conclusion.
    ``df`` is stored on the result for the record and is not used.

    At ``lag=0`` with ``hln=True`` the statistic collapses to the ordinary one-sample t-test,
    which is what the unit tests pin.
    """
    v = np.asarray(d, dtype=np.float64).ravel()
    n = v.size
    if n < 2:
        raise ValueError("Need at least two observations.")
    maxlags = newey_west_lag(n) if lag is None else int(lag)
    if maxlags < 0:
        raise ValueError("lag must be non-negative.")
    nan = float("nan")
    if not np.isfinite(v).all():
        return DMResult(nan, nan, nan, nan, nan, nan, maxlags, n, n - 1, hln, loss_power)
    fit = sm.OLS(v, np.ones((n, 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags, "kernel": "bartlett", "use_correction": hln},
    )
    mean = float(fit.params[0])
    se = float(fit.bse[0])
    return DMResult(
        mean=mean,
        se=se,
        t=float(fit.tvalues[0]),
        p=float(fit.pvalues[0]),
        ci_low=mean - Z_975 * se,
        ci_high=mean + Z_975 * se,
        lag=maxlags,
        n=n,
        df=n - 1,
        hln=hln,
        loss_power=loss_power,
    )


def differential_autocorrelation(d: ArrayLike, *, nlags: int = 24) -> tuple[np.ndarray, float]:
    """``(acf[0..nlags], Ljung-Box p at nlags)``: the evidence that lag 0 is not available.

    Diebold-Mariano theory allows an uncorrected variance for a one-step-ahead forecast. On this
    data the loss differentials are serially correlated at every cell: Ljung-Box p at 24 lags is
    below 1e-07 for every trained-versus-linear comparison (largest 1.11e-08, square 5259 LSTM)
    and below 0.002 for all fifteen the report makes (largest 1.06e-03, square 5259 linear
    AR(144) versus persistence). Every significant coefficient is positive, so omitting the
    correction would understate the variance in all cases rather than in some.
    """
    v = np.asarray(d, dtype=np.float64).ravel()
    values = np.asarray(acf(v, nlags=nlags, fft=True), dtype=np.float64)
    lb = acorr_ljungbox(v, lags=[nlags], return_df=True)
    return values, float(lb["lb_pvalue"].iloc[-1])


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Preferred over Bonferroni, which it uniformly dominates, and over an uncorrected family:
    the comparison set here was chosen after the point estimates were seen, so the multiplicity
    is real. Holm is valid under arbitrary dependence between the tests, which matters because
    the three areas share one evaluation window and so are not independent replications.
    """
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    scaled = (m - np.arange(m)) * p[order]
    # Step-down: each adjusted value is the running maximum, so an adjusted p can never fall
    # below one reported for a smaller raw p.
    adjusted = np.maximum.accumulate(scaled)
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def intersection_union_p(results: Sequence[DMResult], favours_first: Sequence[bool]) -> float:
    """The largest one-sided component p-value. No multiplicity correction applies.

    A reordering claim is a *conjunction*: model A beats B on one area **and** loses to it on
    another. Under any parameter in the null at least one component null holds, so the size of
    the joint test is bounded by the largest component size (Berger, 1982). Correcting it would
    be conservative for no reason, and the joint claim is exactly what the report asserts.

    The components are **one-sided**, via :meth:`DMResult.one_sided_p`, because each leg asserts
    a direction and not merely a difference: a two-sided component would charge the claim for a
    tail it does not use. The lag table in ``reports/tables/significance_dm.md`` prints two-sided
    p, so a component read off it is twice the value entering this maximum.
    """
    if len(results) != len(favours_first):
        raise ValueError("results and favours_first must be the same length.")
    if not results:
        raise ValueError("Need at least one result.")
    return max(r.one_sided_p(favours_first=f) for r, f in zip(results, favours_first, strict=True))


def block_bootstrap_ci(
    d: ArrayLike,
    *,
    block: int = 144,
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Circular moving-block bootstrap percentile interval for ``mean(d)``, plus its std error.

    Block 144 is one day: longer than the measured dependence range of about 25 slots, and it
    keeps a whole diurnal cycle inside a block. Circular rather than plain moving-block so the
    ends are not under-sampled. Reported as a check on the HAC interval rather than instead of
    it; the two agree on whether zero is contained for every comparison in the report.
    """
    v = np.asarray(d, dtype=np.float64).ravel()
    n = v.size
    if block < 1 or block > n:
        raise ValueError(f"block must lie in 1..{n}.")
    if not np.isfinite(v).all():
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    wrapped = np.concatenate([v, v[: block - 1]]) if block > 1 else v
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = (starts[i, :, None] + offsets[None, :]).ravel()[:n]
        means[i] = wrapped[idx % wrapped.size].mean()
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi), float(means.std(ddof=1))


def min_detectable_difference(se: float, *, power: float = 0.80) -> float:
    """Smallest true difference this evaluation could detect, ``(z_.975 + z_.80) * se``.

    The single most useful number for reading the results tables: it says what the week could
    ever have resolved, independently of what it happened to show. Only 80 percent power is
    offered, because only the two quantiles above are available without scipy, and silently
    interpolating a third would be worse than refusing.
    """
    if power != 0.80:
        raise ValueError("Only power=0.80 is supported; Z_80 is the only quantile available.")
    if not math.isfinite(se):
        return float("nan")
    return (Z_975 + Z_80) * se


def required_n(se: float, observed_margin: float, n: int, *, power: float = 0.80) -> float:
    """Evaluation length needed to detect ``observed_margin`` at ``power``, in slots."""
    mdd = min_detectable_difference(se, power=power)
    if not math.isfinite(mdd) or observed_margin == 0:
        return float("nan")
    return float(n * (mdd / abs(observed_margin)) ** 2)


def compare(
    y_true: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    pairs: Sequence[tuple[str, str]],
    *,
    lag: int | None = None,
    lag_grid: Sequence[int] = (0, 6, 15, 36, 72, 144),
    loss_powers: Sequence[int] = (1, 2),
    block: int = 144,
    n_boot: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per (pair, loss power): the test, its sensitivities and its detection floor.

    ``predictions`` maps a model name to a ``(n_seeds, n_steps)`` or ``(n_steps,)`` array. Holm
    adjustment is left to the caller, because only the caller knows where one pre-specified
    family ends and the next begins.
    """
    missing = [m for pair in pairs for m in pair if m not in predictions]
    if missing:
        raise KeyError(f"No predictions for {sorted(set(missing))}.")
    rows: list[dict[str, object]] = []
    for power in loss_powers:
        losses = {
            name: seed_averaged_loss(y_true, pred, power=power)
            for name, pred in predictions.items()
        }
        for a, b in pairs:
            d = losses[a] - losses[b]
            res = dm_test(d, lag=lag, hln=True, loss_power=power)
            acf_values, lb_p = differential_autocorrelation(d)
            boot_lo, boot_hi, boot_se = block_bootstrap_ci(d, block=block, n_boot=n_boot, seed=seed)
            per_seed = _stack(predictions[a]).shape[0]
            row: dict[str, object] = {
                "model_a": a,
                "model_b": b,
                "loss": "absolute" if power == 1 else "squared",
                "d": res.mean,
                "se_hac": res.se,
                "t": res.t,
                "p": res.p,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "lag": res.lag,
                "n": res.n,
                "mdd": min_detectable_difference(res.se),
                "required_slots": required_n(res.se, res.mean, res.n),
                "boot_ci_low": boot_lo,
                "boot_ci_high": boot_hi,
                "boot_se": boot_se,
                "acf_lag1": float(acf_values[1]) if acf_values.size > 1 else float("nan"),
                "ljung_box_p": lb_p,
                "andrews_lag": andrews_ar1_lag(d),
                "n_seeds_a": per_seed,
                "ens_d": float(
                    np.mean(ensemble_loss(y_true, predictions[a], power=power))
                    - np.mean(ensemble_loss(y_true, predictions[b], power=power))
                ),
            }
            for grid_lag in lag_grid:
                row[f"p_lag_{grid_lag}"] = dm_test(d, lag=grid_lag, hln=True).p
            rows.append(row)
    return pd.DataFrame(rows)
