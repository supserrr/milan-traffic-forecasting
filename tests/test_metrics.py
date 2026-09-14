"""Metric definitions, especially the edge cases that matter on this dataset."""

from __future__ import annotations

import numpy as np
import pytest

from milan_traffic import metrics as M


def test_perfect_prediction_is_zero_error():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.mae(y, y) == 0.0
    assert M.rmse(y, y) == 0.0
    assert M.mape(y, y) == pytest.approx(0.0)
    assert M.smape(y, y) == pytest.approx(0.0)
    assert M.r2(y, y) == pytest.approx(1.0)


def test_mae_and_rmse_known_values():
    y = np.array([0.0, 0.0, 0.0])
    p = np.array([1.0, 2.0, 3.0])
    assert M.mae(y, p) == pytest.approx(2.0)
    assert M.rmse(y, p) == pytest.approx(np.sqrt(14 / 3))


def test_rmse_punishes_outliers_more_than_mae():
    y = np.zeros(100)
    spread = np.full(100, 1.0)
    spike = np.zeros(100)
    spike[0] = 100.0
    assert M.mae(y, spread) == pytest.approx(M.mae(y, spike))  # identical total error...
    assert M.rmse(y, spread) < M.rmse(y, spike)  # ...opposite RMSE ordering


def test_mape_modes_disagree_by_design_near_zero():
    """Overnight near-zero traffic is exactly where MAPE misbehaves."""
    y = np.array([0.0, 10.0, 20.0])
    p = np.array([1.0, 11.0, 19.0])

    assert np.isinf(M.mape(y, p, mode="raw"))
    guarded = M.mape(y, p, mode="epsilon", epsilon=1e-3)
    assert np.isfinite(guarded) and guarded > 1000  # huge, but finite and explicable
    masked = M.mape(y, p, mode="mask", epsilon=1e-3)
    assert masked == pytest.approx((1 / 10 + 1 / 20) / 2 * 100)


def test_smape_and_wape_are_finite_when_target_is_zero():
    y = np.array([0.0, 10.0, 20.0])
    p = np.array([1.0, 11.0, 19.0])
    assert np.isfinite(M.smape(y, p))
    assert M.wape(y, p) == pytest.approx(3 / 30 * 100)


def test_mase_below_one_means_better_than_seasonal_naive(seasonal_series):
    y = seasonal_series[144 * 2 :]
    seasonal_naive = seasonal_series[144:-144]
    perfect = y.copy()
    assert M.mase(y, perfect, season=144) == pytest.approx(0.0)
    assert M.mase(y, seasonal_naive, season=144) == pytest.approx(1.0, rel=0.35)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.mae(np.zeros(3), np.zeros(4))


def test_metrics_table_sorts_and_covers_all_models():
    y = np.arange(50, dtype=float)
    table = M.metrics_table(
        {
            "good": (y, y + 0.1),
            "bad": (y, y + 5.0),
        },
        season=10,
    )
    assert list(table.index) == ["good", "bad"]
    assert {"MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "MASE", "R2"} <= set(table.columns)


def test_mase_scale_comes_from_the_training_period_when_given(seasonal_series):
    """A seasonal-naive forecast must score ~1.0 against its own training scale.

    Without an explicit scale the denominator is taken from the evaluation window, which
    on a calmer window inflates the score. The two must therefore differ, and only the
    training-scaled value carries the "below 1 beats seasonal persistence" meaning.
    """
    train, test = seasonal_series[:1440], seasonal_series[1440:]
    y = test[144:]
    seasonal_naive = test[:-144]

    scale = M.seasonal_naive_scale(train, season=144)
    assert scale > 0
    assert M.mase(y, seasonal_naive, season=144, scale=scale) == pytest.approx(1.0, rel=0.15)
    assert M.mase(y, y, season=144, scale=scale) == pytest.approx(0.0)


def test_mase_is_nan_when_the_scale_is_degenerate():
    y = np.arange(300.0)
    assert np.isnan(M.mase(y, y, season=144, scale=0.0))
    assert np.isnan(M.seasonal_naive_scale(np.arange(10.0), season=144))
