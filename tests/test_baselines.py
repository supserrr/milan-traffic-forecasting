"""The reference forecasters. Cheap to test, and everything else is measured against them.

A baseline that is quietly off by one lag would shift every comparison in the report
without ever looking wrong, so each model is checked against a window whose correct answer
is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from milan_traffic.models import REGISTRY, build
from milan_traffic.models.baselines import (
    LinearAR,
    Persistence,
    SeasonalNaive,
    TimeOfDayMean,
)

SEQ_LEN = 1008


@pytest.fixture
def windows() -> np.ndarray:
    """Three windows of consecutive integers, so any lag is identifiable by value."""
    return np.arange(3 * SEQ_LEN, dtype=np.float64).reshape(3, SEQ_LEN)


def test_persistence_returns_the_last_observation(windows):
    np.testing.assert_array_equal(Persistence().predict(windows), windows[:, -1])
    assert Persistence().n_params == 0


@pytest.mark.parametrize("season", [144, 1008])
def test_seasonal_naive_returns_the_right_lag(windows, season):
    model = SeasonalNaive(season=season)
    np.testing.assert_array_equal(model.predict(windows), windows[:, -season])
    assert model.name == f"seasonal_naive_{season}"


def test_seasonal_naive_rejects_a_window_shorter_than_its_lag():
    short = np.zeros((2, 100))
    with pytest.raises(ValueError, match="at least 1008 steps"):
        SeasonalNaive(season=1008).predict(short)


def test_time_of_day_mean_averages_the_requested_daily_lags(windows):
    pred = TimeOfDayMean(season=144, n_seasons=4).predict(windows)
    expected = np.mean(
        [windows[:, -144], windows[:, -288], windows[:, -432], windows[:, -576]], axis=0
    )
    np.testing.assert_allclose(pred, expected)


def test_linear_ar_recovers_an_exact_linear_relationship():
    """y = 2*x[-1] - 0.5*x[-2] + 3 is recoverable, so the fit is not merely plausible."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 10))
    y = 2.0 * X[:, -1] - 0.5 * X[:, -2] + 3.0

    model = LinearAR(order=2).fit(X, y)
    np.testing.assert_allclose(model.coef_, [-0.5, 2.0], atol=1e-8)
    assert model.intercept_ == pytest.approx(3.0)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-8)
    assert model.n_params == 3


def test_linear_ar_uses_only_the_last_order_lags():
    """Columns older than `order` must not reach the fit, or seq_len stops meaning anything."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 20))
    y = X[:, -1] * 1.5
    model = LinearAR(order=3).fit(X, y)
    assert model.coef_.shape == (3,)

    tampered = X.copy()
    tampered[:, :-3] = 999.0
    np.testing.assert_allclose(model.predict(tampered), model.predict(X), atol=1e-8)


def test_linear_ar_refuses_to_predict_before_fit():
    with pytest.raises(RuntimeError, match="before fit"):
        LinearAR(order=6).predict(np.zeros((2, 10)))


def test_baselines_are_registered_and_buildable():
    for name in (
        "naive",
        "seasonal_naive_daily",
        "seasonal_naive_weekly",
        "time_of_day_mean",
        "linear_ar",
    ):
        assert name in REGISTRY
    assert build("seasonal_naive_weekly").season == 1008
    assert build("naive").n_params == 0
