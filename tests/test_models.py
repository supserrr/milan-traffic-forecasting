"""The model registry and the tree model.

Two properties matter beyond "it runs". First, every name a run config can contain must
resolve through :func:`build`: a stored config that cannot rebuild its own run is not a
reproducible artefact, and six of EXP-000's seven names once failed this. Second, the
calendar features must actually reach each model, because the comparison claims all three
are fed identically and an ignored ``exog`` argument would silently break that claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from milan_traffic.models import REGISTRY, Forecaster, build
from milan_traffic.models.gbt import GBTForecaster
from milan_traffic.models.tcn import TCNForecaster

SEQ_LEN = 144

#: Every name that appears in a run config in this repository, including EXP-000's.
CONFIG_NAMES = [
    "naive",
    "seasonal_naive_144",
    "seasonal_naive_1008",
    "seasonal_naive_daily",
    "seasonal_naive_weekly",
    "time_of_day_mean_4d",
    "time_of_day_mean",
    "linear_ar_6",
    "linear_ar_36",
    "linear_ar_144",
    "linear_ar",
    "lstm",
    "tcn",
    "gbt",
]


@pytest.fixture
def windows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, SEQ_LEN)).astype(np.float32)
    y = (0.8 * X[:, -1] + 0.2 * X[:, -144]).astype(np.float32)
    exog = np.zeros((300, 5), dtype=np.float32)
    exog[:150, 4] = 1.0
    return X, y, exog


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_every_name_a_config_can_contain_resolves(name):
    """A run's config.yaml must be enough to rebuild the run."""
    assert name in REGISTRY
    model = build(name)
    assert isinstance(model, Forecaster)
    assert isinstance(model.name, str) and model.name


def test_presets_accept_an_explicit_override():
    """`build(name, season=...)` once raised TypeError, so a config could name a model
    but not configure it."""
    assert build("seasonal_naive_daily").season == 144
    assert build("seasonal_naive_weekly").season == 1008
    assert build("seasonal_naive_daily", season=288).season == 288
    assert build("linear_ar", order=12).order == 12


def test_gbt_fits_and_predicts(windows):
    X, y, exog = windows
    model = GBTForecaster(max_iter=30).fit(X, y, X_val=X[:60], y_val=y[:60])
    pred = model.predict(X[:25])
    assert pred.shape == (25,)
    assert np.isfinite(pred).all()
    assert model.n_params > 0
    assert model.history["device"] == "cpu"


def test_calendar_features_reach_the_tree_model(windows):
    """Passing exog must change the prediction, or the ablation run measures nothing."""
    X, y, exog = windows
    y_cal = y + 5.0 * exog[:, 4]  # a signal only the calendar column carries
    with_exog = GBTForecaster(max_iter=60).fit(X, y_cal, exog=exog).predict(X[:40], exog[:40])
    without = GBTForecaster(max_iter=60).fit(X, y_cal).predict(X[:40])
    assert not np.allclose(with_exog, without)
    # The model that can see the flag should track the jump it creates.
    assert abs(with_exog[:40].mean() - y_cal[:40].mean()) < abs(
        without[:40].mean() - y_cal[:40].mean()
    )


def test_gbt_design_matrix_width_matches_the_features_given(windows):
    X, y, exog = windows
    model = GBTForecaster(max_iter=10).fit(X, y, exog=exog)
    assert model.history["n_features"] == SEQ_LEN + exog.shape[1]
    assert GBTForecaster(max_iter=10).fit(X, y).history["n_features"] == SEQ_LEN


def test_tcn_receptive_field_covers_the_window():
    """A receptive field shorter than the window would silently truncate the history the
    Methodology section says the model reads."""
    model = TCNForecaster()
    assert model.receptive_field >= SEQ_LEN


@pytest.mark.parametrize("name", ["naive", "seasonal_naive_daily", "linear_ar_6", "gbt"])
def test_baselines_accept_and_ignore_the_calendar_block(windows, name):
    """The harness calls every model the same way, so exog must never be a TypeError."""
    X, y, exog = windows
    model = build(name)
    model.fit(X, y, X_val=X[:50], y_val=y[:50], exog=exog, exog_val=exog[:50])
    pred = model.predict(X[:10], exog[:10])
    assert pred.shape == (10,)
