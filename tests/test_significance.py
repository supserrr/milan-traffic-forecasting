"""Diebold-Mariano machinery, especially the choices the report's prose commits to.

Three of these tests exist because a future editor could plausibly get them wrong in a way that
would leave the report's numbers unchanged but its claims unsupported: the unit of test, the
algebra of the small-sample correction, and the direction handling of the conjunction test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from milan_traffic import metrics as M
from milan_traffic import significance as S


def _bisect_normal_quantile(target: float) -> float:
    """Standard-normal quantile by bisection on ``math.erf``, so the hard-coded constants cannot
    drift without a test noticing."""
    cdf = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))  # noqa: E731
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def test_hardcoded_normal_quantiles_match_erf():
    assert pytest.approx(_bisect_normal_quantile(0.975), abs=1e-9) == S.Z_975
    assert pytest.approx(_bisect_normal_quantile(0.800), abs=1e-9) == S.Z_80


def test_newey_west_lag_rule():
    assert S.newey_west_lag(1008) == 15
    assert S.newey_west_lag(100) == 6
    sizes = [10, 50, 100, 500, 1008, 5000]
    lags = [S.newey_west_lag(n) for n in sizes]
    assert lags == sorted(lags)
    with pytest.raises(ValueError):
        S.newey_west_lag(0)


def test_seed_averaged_margin_equals_difference_of_seed_mean_maes():
    """The identity the whole choice of unit rests on.

    Each cell of the report's tables is a mean over seeds of a per-seed MAE. The mean of the
    seed-averaged loss differential is exactly the difference of two such cells, so the test and
    the table describe the same forecaster. Computed here through ``metrics.mae`` so the two
    modules are bound together rather than merely consistent today.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(1000, 200, size=240)
    a = y[None, :] + rng.normal(0, 40, size=(3, 240))
    b = y[None, :] + rng.normal(0, 55, size=(3, 240))

    d = S.seed_averaged_loss(y, a) - S.seed_averaged_loss(y, b)
    table_a = float(np.mean([M.mae(y, row) for row in a]))
    table_b = float(np.mean([M.mae(y, row) for row in b]))
    assert float(np.mean(d)) == pytest.approx(table_a - table_b, abs=1e-12)


def test_ensemble_loss_is_a_different_and_better_forecaster():
    """Two seeds erring in opposite directions cancel in the ensemble and do not in the average.

    The extreme case of the bias the report discloses: here the ensemble is perfect and the
    seed-average is not, so swapping one for the other would change the reported effect entirely.
    """
    y = np.zeros(4)
    pred = np.stack([np.full(4, 10.0), np.full(4, -10.0)])
    assert float(np.mean(S.seed_averaged_loss(y, pred))) == pytest.approx(10.0)
    assert float(np.mean(S.ensemble_loss(y, pred))) == pytest.approx(0.0)

    rng = np.random.default_rng(1)
    for _ in range(10):
        truth = rng.normal(0, 1, size=50)
        preds = truth[None, :] + rng.normal(0, 1, size=(3, 50))
        assert np.mean(S.ensemble_loss(truth, preds)) <= np.mean(S.seed_averaged_loss(truth, preds))


def test_dm_at_lag_zero_with_hln_is_the_one_sample_t_test():
    """At lag 0 the long-run variance is gamma_0 and ``use_correction`` rescales it by n/(n-1),
    so the statistic must collapse to the textbook paired t-test."""
    d = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    res = S.dm_test(d, lag=0, hln=True)
    expected_t = d.mean() / (d.std(ddof=1) / math.sqrt(d.size))
    assert res.t == pytest.approx(expected_t, rel=1e-12)
    assert res.t == pytest.approx(3.71993, abs=1e-5)
    assert res.df == 4


def test_dm_matches_hand_computed_bartlett_newey_west():
    """That the statsmodels HAC route is the textbook Bartlett estimator and not another one."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=300)
    d = np.empty(300)
    d[0] = u[0]
    for i in range(1, 300):
        d[i] = 0.5 * d[i - 1] + u[i]
    dev = d - d.mean()
    n = d.size
    for lag in (0, 3, 15):
        v = float(np.dot(dev, dev) / n)
        for k in range(1, lag + 1):
            gamma = float(np.dot(dev[k:], dev[:-k]) / n)
            v += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
        expected = d.mean() / math.sqrt(v / n)
        assert S.dm_test(d, lag=lag, hln=False).t == pytest.approx(expected, rel=1e-10)


def test_hln_correction_is_exactly_sqrt_n_minus_one_over_n():
    """The claim the report's methodology section makes in prose about ``use_correction``."""
    rng = np.random.default_rng(2)
    d = rng.normal(size=1008)
    ratio = S.dm_test(d, lag=15, hln=True).t / S.dm_test(d, lag=15, hln=False).t
    assert ratio == pytest.approx(math.sqrt(1007 / 1008), rel=1e-12)
    assert ratio == pytest.approx(0.9995038451691601, rel=1e-12)


def test_holm_hand_computed_capped_and_order_preserving():
    assert S.holm([0.01, 0.02, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert S.holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    raw = [0.0021, 0.1896, 0.0085]
    adjusted = S.holm(raw)
    assert adjusted == pytest.approx([0.0063, 0.1896, 0.0170], abs=1e-4)
    assert np.all(adjusted >= np.asarray(raw))
    perm = [2, 0, 1]
    assert S.holm([raw[i] for i in perm]) == pytest.approx([adjusted[i] for i in perm], abs=1e-12)


def test_min_detectable_difference_constant_and_linearity():
    assert S.min_detectable_difference(1.0) == pytest.approx(2.801585, abs=1e-6)
    assert S.min_detectable_difference(3.807) == pytest.approx(2.801585219 * 3.807, rel=1e-9)
    with pytest.raises(ValueError):
        S.min_detectable_difference(1.0, power=0.9)


def test_required_n_scales_with_the_square_of_the_shortfall():
    se, n = 3.0, 1008
    mdd = S.min_detectable_difference(se)
    assert S.required_n(se, mdd, n) == pytest.approx(float(n))
    assert S.required_n(se, mdd / 2.0, n) == pytest.approx(4.0 * n)


def test_non_finite_prediction_yields_nan_rather_than_a_dropped_slot():
    """The house rule from ``metrics._prep``: a diverged forecast must not be rewarded by being
    scored only on the slots it survived."""
    y = np.linspace(100, 200, 1008)
    pred = y + 5.0
    pred[200] = np.nan
    losses = S.loss_series(y, pred)
    assert losses.shape == (1008,)
    assert np.isnan(losses).all()
    res = S.dm_test(losses - S.loss_series(y, y + 1.0))
    assert all(math.isnan(v) for v in (res.mean, res.se, res.t, res.p))
    assert res.n == 1008


def test_intersection_union_p_is_the_max_of_one_sided_components():
    """The conjunction that carries the study's contribution, including direction handling."""
    worse = S.dm_test(np.full(400, 2.0) + np.random.default_rng(3).normal(0, 1, 400))
    better = S.dm_test(np.full(400, -2.0) + np.random.default_rng(4).normal(0, 1, 400))
    assert worse.mean > 0 and better.mean < 0
    joint = S.intersection_union_p([worse, better], [False, True])
    assert joint == pytest.approx(
        max(worse.one_sided_p(favours_first=False), better.one_sided_p(favours_first=True))
    )
    flipped = S.intersection_union_p([worse, better], [True, True])
    assert flipped > 0.5


def test_dm_sign_convention_is_negative_when_the_first_model_is_better():
    rng = np.random.default_rng(5)
    d = -2.0 + rng.normal(0, 0.5, size=300)
    res = S.dm_test(d)
    assert res.mean < 0 and res.t < 0 and res.ci_high < 0


def test_block_bootstrap_is_reproducible_and_recovers_the_iid_standard_error():
    rng = np.random.default_rng(6)
    d = rng.normal(size=2000)
    first = S.block_bootstrap_ci(d, block=1, n_boot=2000, seed=42)
    assert S.block_bootstrap_ci(d, block=1, n_boot=2000, seed=42) == first
    lo, hi, se = first
    assert se == pytest.approx(1.0 / math.sqrt(2000), rel=0.10)
    assert lo < d.mean() < hi
    with pytest.raises(ValueError):
        S.block_bootstrap_ci(d, block=0)


def test_compare_returns_one_row_per_pair_and_loss_and_reproduces_dm_test():
    rng = np.random.default_rng(7)
    y = rng.normal(1000, 150, size=300)
    preds = {
        "a": y[None, :] + rng.normal(0, 30, size=(3, 300)),
        "b": y[None, :] + rng.normal(0, 45, size=(3, 300)),
        "det": y + rng.normal(0, 38, size=300),
    }
    frame = S.compare(y, preds, [("a", "b"), ("a", "det")], lag_grid=(0, 15), n_boot=200, block=24)
    assert len(frame) == 4
    assert set(frame["loss"]) == {"absolute", "squared"}
    row = frame[(frame.model_a == "a") & (frame.loss == "absolute")].iloc[0]
    direct = S.dm_test(S.seed_averaged_loss(y, preds["a"]) - S.seed_averaged_loss(y, preds["b"]))
    assert row["d"] == pytest.approx(direct.mean) and row["p"] == pytest.approx(direct.p)
    assert row["lag"] == S.newey_west_lag(300) == 10
    with pytest.raises(KeyError):
        S.compare(y, preds, [("a", "nope")])
