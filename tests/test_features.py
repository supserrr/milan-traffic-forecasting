"""Windowing, splitting and scaling - the places leakage hides."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from milan_traffic.features import (
    Scaler,
    calendar_features,
    chronological_split,
    context_windows,
    eval_week_split,
    make_windows,
    split_windows,
)


def test_make_windows_shapes_and_alignment():
    series = np.arange(20, dtype=np.float32)
    X, y = make_windows(series, seq_len=5, horizon=1)
    assert X.shape == (15, 5)
    assert y.shape == (15,)
    np.testing.assert_array_equal(X[0], [0, 1, 2, 3, 4])
    assert y[0] == 5.0  # target is the step after the window
    np.testing.assert_array_equal(X[-1], [14, 15, 16, 17, 18])
    assert y[-1] == 19.0


def test_make_windows_rejects_too_short_series():
    with pytest.raises(ValueError):
        make_windows(np.arange(4, dtype=np.float32), seq_len=10)


def test_chronological_split_is_ordered_and_contiguous():
    split = chronological_split(1000, val_frac=0.2, test_slice=slice(800, 1000))
    assert split.train.stop == split.val.start
    assert split.val.stop == split.test.start
    assert split.train.start == 0
    assert split.val.stop - split.val.start == 160  # 20% of the pre-test period


def test_eval_week_split_selects_the_right_window():
    index = pd.date_range("2013-11-01", periods=8928, freq="10min", tz="Europe/Rome")
    split = eval_week_split(index, "2013-12-16 00:00", "2013-12-22 23:50")
    assert split.test.stop - split.test.start == 1008  # exactly one week
    assert index[split.test.start].strftime("%Y-%m-%d %H:%M") == "2013-12-16 00:00"
    assert index[split.test.stop - 1].strftime("%Y-%m-%d %H:%M") == "2013-12-22 23:50"
    assert split.val.stop <= split.test.start  # no overlap


def test_scaler_refuses_to_transform_before_fit():
    with pytest.raises(RuntimeError):
        Scaler("standard").transform(np.array([1.0, 2.0]))


@pytest.mark.parametrize("bad", ["standrd", "zscore", "LOG1P", "log1p_standard", ""])
def test_scaler_rejects_an_unknown_method(bad):
    """A typo must fail, not quietly become the identity.

    The unrecognised name used to fall through to the no-op branch, so a misspelled
    --scaling produced a complete run with scaling silently disabled and nothing in the
    artefacts recording it.
    """
    with pytest.raises(ValueError, match="Unknown scaling method"):
        Scaler(bad)


@pytest.mark.parametrize("method", ["standard", "minmax", "log1p", "log1p-standard", "none"])
def test_scaler_roundtrip(method):
    x = np.abs(np.random.default_rng(0).lognormal(0, 1, size=500))
    sc = Scaler(method).fit(x)
    np.testing.assert_allclose(sc.inverse_transform(sc.transform(x)), x, rtol=1e-6, atol=1e-6)


def test_scaler_is_fit_on_train_only(seasonal_series):
    """The whole point: test-period statistics must not reach the training transform."""
    split = chronological_split(len(seasonal_series), test_slice=slice(1500, 2016))
    sc = Scaler("standard").fit(seasonal_series[split.train])
    train_mean = float(np.mean(seasonal_series[split.train]))
    full_mean = float(np.mean(seasonal_series))
    assert sc.center_ == pytest.approx(train_mean)
    assert sc.center_ != pytest.approx(full_mean)


def test_split_windows_never_cross_a_boundary(seasonal_series):
    split = chronological_split(len(seasonal_series), test_slice=slice(1500, 2016))
    seq_len = 36
    parts = split_windows(seasonal_series, split, seq_len=seq_len)
    for name, sl in zip(("train", "val", "test"), split, strict=True):
        n = (sl.stop - sl.start) - seq_len
        assert parts[name][0].shape == (n, seq_len)
        assert parts[name][1].shape == (n,)


def test_context_windows_cover_every_test_point(seasonal_series):
    split = chronological_split(len(seasonal_series), test_slice=slice(1500, 2016))
    sc = Scaler("standard").fit(seasonal_series[split.train])
    X, y = context_windows(seasonal_series, split, seq_len=36, scaler=sc)
    assert len(y) == split.test.stop - split.test.start  # nothing dropped
    np.testing.assert_allclose(
        sc.inverse_transform(y), seasonal_series[split.test], rtol=1e-4, atol=1e-3
    )


def test_calendar_features_are_cyclical():
    index = pd.date_range("2013-11-01", periods=288, freq="10min", tz="Europe/Rome")
    feats = calendar_features(index)
    assert set(feats.columns) == {
        "sin_day",
        "cos_day",
        "sin_week",
        "cos_week",
        "is_weekend",
        "is_working_day",
    }
    # Fri 2013-11-01 is Ognissanti, a public holiday: a weekday that is not a working
    # day. Square 5259 peaks at night on it exactly as it does at weekends (F-02), which
    # is why the regime indicator is the working calendar and not the day-of-week label.
    assert feats["is_weekend"].iloc[0] == 0.0
    assert feats["is_working_day"].iloc[0] == 0.0
    # 23:50 and the following 00:00 must be neighbours, not opposites
    gap = np.hypot(
        feats["sin_day"].iloc[143] - feats["sin_day"].iloc[144],
        feats["cos_day"].iloc[143] - feats["cos_day"].iloc[144],
    )
    assert gap < 0.1


def test_context_windows_can_target_a_short_validation_period(seasonal_series):
    """A validation slice shorter than one window still gets one prediction per step.

    Inputs reach back into the training period, which is exactly what is available when
    selecting a model; the targets stay strictly inside validation.
    """
    split = chronological_split(len(seasonal_series), val_frac=0.1, test_slice=slice(1800, 2016))
    n_val = split.val.stop - split.val.start
    seq_len = n_val + 50  # deliberately longer than the val slice
    sc = Scaler("standard").fit(seasonal_series[split.train])

    X, y = context_windows(seasonal_series, split, seq_len, scaler=sc, target_slice=split.val)
    assert len(y) == n_val
    np.testing.assert_allclose(
        sc.inverse_transform(y), seasonal_series[split.val], rtol=1e-4, atol=1e-3
    )
