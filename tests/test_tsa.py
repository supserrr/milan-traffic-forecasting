"""Time-series diagnostics on synthetic data: a period-24 sine plus noise, nothing real."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from milan_traffic import tsa

PERIOD = 24
N_DAYS = 21


@pytest.fixture
def sine() -> np.ndarray:
    """Positive hourly series: 21 days of a period-24 sine plus small noise."""
    t = np.arange(N_DAYS * PERIOD)
    base = 10.0 + 5.0 * np.sin(2 * np.pi * t / PERIOD)
    noise = np.random.default_rng(0).normal(0.0, 0.4, size=t.size)
    return base + noise


@pytest.fixture
def hourly_index() -> pd.DatetimeIndex:
    """Hourly stamps starting Monday 2013-11-04 00:00 local time, three ISO weeks."""
    return pd.date_range("2013-11-04 00:00", periods=N_DAYS * PERIOD, freq="h", tz="Europe/Rome")


# ------------------------------------------------------------------ split constants


def test_split_rows_match_the_fixed_convention():
    assert slice(0, 5472) == tsa.TRAIN_ROWS
    assert slice(5472, 6480) == tsa.VAL_ROWS
    assert slice(6480, 7488) == tsa.EVAL_ROWS
    assert tsa.split_label(0) == "train"
    assert tsa.split_label(5472) == "val"
    assert tsa.split_label(7487) == "eval"
    assert tsa.split_label(7488) == "post-eval"


# --------------------------------------------------------------------- ACF and PACF


def test_band_formula_is_1_96_over_sqrt_n(sine):
    assert tsa.confidence_band(400) == pytest.approx(1.96 / 20)
    table = tsa.acf_pacf(sine, nlags=48)
    expected = 1.96 / np.sqrt(sine.size)
    assert table["band"].nunique() == 1
    assert table["band"].iloc[0] == pytest.approx(expected)
    assert table.attrs["band"] == pytest.approx(expected)
    assert table.attrs["n"] == sine.size
    assert table.attrs["pacf_method"] == "ldb"


def test_acf_pacf_shape_and_seasonal_lag(sine):
    table = tsa.acf_pacf(sine, nlags=48)
    assert list(table.columns) == [
        "lag",
        "acf",
        "pacf",
        "band",
        "acf_above_band",
        "pacf_above_band",
    ]
    assert len(table) == 49
    assert table["acf"].iloc[0] == pytest.approx(1.0)
    # A clean sine has a strong autocorrelation at its period and a sign flip at half of it.
    assert table["acf"].iloc[PERIOD] > 0.9
    assert table["acf"].iloc[PERIOD // 2] < -0.9


def test_pacf_methods_agree(sine):
    ldb = tsa.acf_pacf(sine, nlags=48, pacf_method="ldb")["pacf"]
    ywm = tsa.acf_pacf(sine, nlags=48, pacf_method="ywm")["pacf"]
    np.testing.assert_allclose(ldb.to_numpy()[1:], ywm.to_numpy()[1:], atol=1e-6)


def test_acf_pacf_rejects_bad_nlags(sine):
    with pytest.raises(ValueError):
        tsa.acf_pacf(sine, nlags=sine.size // 2)


def test_cutoff_helpers_use_their_own_level_and_skip_lag_zero():
    pacf_values = np.array([1.0, 0.50, 0.010, 0.060, 0.000, -0.030, 0.005])
    band = 0.02
    assert tsa.last_significant_lag(pacf_values, band) == 5  # |-0.030| > 0.02
    assert tsa.practical_cutoff(pacf_values, 0.05) == 3  # 0.060 > 0.05, lag 5 does not qualify
    assert tsa.last_significant_lag(pacf_values, band, max_lag=2) == 1
    assert tsa.practical_cutoff(pacf_values, 0.9) is None  # lag 0 never counts
    summary = tsa.exceedance_summary(pacf_values, band, lo=1, hi=6)
    assert summary["n_lags"] == 6
    assert summary["count"] == 3
    assert summary["expected_by_chance"] == pytest.approx(0.3)


# ------------------------------------------------------------------- decomposition


def test_classical_variance_shares_sum_to_about_one(sine):
    d = tsa.decompose(sine, period=PERIOD, method="classical")
    assert d.method == "classical"
    assert d.periods == (PERIOD,)
    assert set(d.variance_shares) == {"trend", f"seasonal_{PERIOD}", "remainder", "sum"}
    assert d.variance_shares["sum"] == pytest.approx(1.0, abs=0.05)
    assert d.variance_shares[f"seasonal_{PERIOD}"] > 0.9
    # the centred moving average leaves period // 2 undefined rows at each end
    assert d.n_valid == sine.size - 2 * (PERIOD // 2)


def test_stl_and_mstl_return_the_same_structure(sine):
    stl = tsa.decompose(sine, period=PERIOD, method="stl")
    assert stl.method == "stl" and stl.n_valid == sine.size
    assert stl.variance_shares[f"seasonal_{PERIOD}"] > 0.9
    assert stl.variance_shares["remainder"] < 0.05
    mstl = tsa.decompose(sine, method="mstl", periods=(PERIOD, 7 * PERIOD))
    assert mstl.method == "mstl" and mstl.periods == (PERIOD, 7 * PERIOD)
    assert set(mstl.seasonal) == {f"seasonal_{PERIOD}", f"seasonal_{7 * PERIOD}"}
    assert mstl.variance_shares[f"seasonal_{PERIOD}"] > 0.85
    table = tsa.variance_share_table([("raw", stl), ("raw", mstl)])
    assert list(table["method"]) == ["stl", "mstl"]
    assert "describe" in dir(stl) and "stl" in stl.describe()


def test_decompose_rejects_unknown_method(sine):
    with pytest.raises(ValueError):
        tsa.decompose(sine, method="fourier")  # type: ignore[arg-type]


# --------------------------------------------------------------------- stationarity


def test_stationarity_tests_cover_four_series_and_emit_no_warnings(sine):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = tsa.stationarity_tests(sine, seasonal_lag=PERIOD)
    assert list(results) == ["raw", "log1p", "diff1", "sdiff"]
    required = {
        "transform",
        "n",
        "adf_stat",
        "adf_p",
        "adf_lags",
        "kpss_stat",
        "kpss_p",
        "kpss_p_capped",
        "kpss_p_display",
        "kpss_lags",
        "verdict",
    }
    for name, r in results.items():
        assert required <= set(r), name
        assert 0.0 <= r["adf_p"] <= 1.0
        assert 0.0 <= r["kpss_p"] <= 1.0
    assert results["diff1"]["n"] == sine.size - 1
    assert results["sdiff"]["n"] == sine.size - PERIOD
    assert results["sdiff"]["transform"] == f"x[t] - x[t-{PERIOD}]"
    table = tsa.stationarity_table(results)
    assert len(table) == 4 and "KPSS p" in table.columns


def test_interpret_stationarity_reads_the_two_nulls_correctly():
    assert tsa.interpret_stationarity(0.01, 0.10).startswith("stationary")
    assert tsa.interpret_stationarity(0.60, 0.01).startswith("unit root")
    assert tsa.interpret_stationarity(0.01, 0.01).startswith("both reject")
    assert tsa.interpret_stationarity(0.60, 0.10).startswith("inconclusive")


# ------------------------------------------------------------ level-dependent noise


def test_rolling_correlation_is_high_for_multiplicative_noise():
    rng = np.random.default_rng(3)
    t = np.arange(40 * PERIOD)
    level = 50.0 + 45.0 * np.sin(2 * np.pi * t / PERIOD)
    multiplicative = level * (1.0 + rng.normal(0.0, 0.15, size=t.size))
    r_raw = tsa.rolling_mean_std_correlation(multiplicative, window=PERIOD)
    r_log = tsa.rolling_mean_std_correlation(
        np.log1p(np.clip(multiplicative, 0, None)), window=PERIOD
    )
    assert r_raw > 0.5
    assert r_log < r_raw
    frame = tsa.rolling_mean_std(multiplicative, window=PERIOD)
    assert len(frame) == t.size - PERIOD + 1


# ------------------------------------------------------------------ calendar views


def test_weekly_weekday_means_shape_and_values(hourly_index):
    # a distinct constant per ISO week, so the weekday mean must equal it exactly
    week_of_row = np.repeat([1.0, 2.0, 3.0], 7 * PERIOD)
    out = tsa.weekly_weekday_means(week_of_row, hourly_index)
    assert out.shape == (3, 5)
    assert list(out.columns) == ["iso_year", "week_start", "n_days", "n_slots", "mean"]
    assert list(out.index) == [45, 46, 47]
    assert list(out["n_days"]) == [5, 5, 5]
    assert list(out["n_slots"]) == [5 * PERIOD] * 3
    np.testing.assert_allclose(out["mean"].to_numpy(), [1.0, 2.0, 3.0])
    assert str(out["week_start"].iloc[0]) == "2013-11-04"


def test_weekly_weekday_means_ignores_weekends(hourly_index):
    x = np.where(hourly_index.dayofweek >= 5, 1000.0, 1.0)
    out = tsa.weekly_weekday_means(x, hourly_index)
    np.testing.assert_allclose(out["mean"].to_numpy(), 1.0)


def test_daily_peak_times_find_the_injected_hour(hourly_index):
    x = np.ones(hourly_index.size)
    hours = hourly_index.hour
    x[hours == 15] = 5.0  # weekday-style afternoon peak
    weekend = hourly_index.dayofweek >= 5
    x[weekend & (hours == 15)] = 1.0
    x[weekend & (hours == 1)] = 5.0  # weekend-style small-hours peak
    peaks = tsa.daily_peak_times(x, hourly_index)
    assert len(peaks) == N_DAYS
    assert set(peaks.loc[~peaks["is_weekend"], "peak_time"]) == {"15:00"}
    assert set(peaks.loc[peaks["is_weekend"], "peak_time"]) == {"01:00"}
    summary = tsa.summarise_daily_peaks(peaks)
    assert list(summary["day_type"]) == ["weekday", "weekend"]
    assert list(summary["n_days"]) == [15, 6]
    assert list(summary["n_peaks_before_06:00"]) == [0, 6]


# ------------------------------------------------------------------- anomaly scan


def test_anomaly_scan_flags_injected_spike_and_nothing_else(sine, hourly_index):
    x = sine.copy()
    x[100] *= 3.0
    scan = tsa.same_slot_anomaly_scan(x, period=PERIOD, threshold=1.5, index=hourly_index)
    assert list(scan.flagged["row"]) == [100]
    assert scan.flagged["ratio"].iloc[0] > 2.5
    assert scan.flagged["time"].iloc[0].startswith("Fri 2013-11-08 04:00")
    assert scan.ratio.shape == (x.size,)
    assert scan.reference_median.shape == (PERIOD,)
    assert scan.period == PERIOD and scan.threshold == 1.5


def test_anomaly_scan_reduces_a_matrix_to_grid_totals(sine):
    matrix = np.stack([sine, sine, sine], axis=1)
    matrix[200, :] *= 4.0
    scan = tsa.same_slot_anomaly_scan(matrix, period=PERIOD, threshold=1.5)
    assert list(scan.flagged["row"]) == [200]
    totals = tsa.grid_totals(matrix, chunk=7)
    np.testing.assert_allclose(totals, matrix.sum(axis=1))


def test_anomaly_scan_reference_rows_can_exclude_the_spike(sine):
    x = sine.copy()
    x[-1] *= 3.0
    scan = tsa.same_slot_anomaly_scan(
        x, period=PERIOD, threshold=1.5, reference_rows=slice(0, x.size - 1)
    )
    assert list(scan.flagged["row"]) == [x.size - 1]


# ---------------------------------------------------------------------- reporting


def test_markdown_table_formats_floats_bools_and_headers():
    df = pd.DataFrame({"lag": [1, 2], "pacf": [0.98234, -0.03], "above": [True, False]})
    text = tsa.markdown_table(df, floatfmt=".3f")
    lines = text.splitlines()
    assert lines[0] == "| lag | pacf | above |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| 1 | 0.982 | yes |"
    assert lines[3] == "| 2 | -0.030 | no |"


def test_markdown_table_escapes_pipes_and_shows_nan_as_dash():
    df = pd.DataFrame({"|pacf| > band": [True], "share": [float("nan")]})
    lines = tsa.markdown_table(df).splitlines()
    assert lines[0] == "| \\|pacf\\| > band | share |"
    assert lines[2] == "| yes | - |"


def test_variance_share_table_orders_seasonal_columns(sine):
    stl = tsa.decompose(sine, period=PERIOD, method="stl")
    mstl = tsa.decompose(sine, method="mstl", periods=(PERIOD, 7 * PERIOD))
    table = tsa.variance_share_table([("raw", stl), ("raw", mstl)])
    assert list(table.columns) == [
        "series",
        "method",
        "periods",
        "trend",
        f"seasonal_{PERIOD}",
        f"seasonal_{7 * PERIOD}",
        "remainder",
        "sum",
        "n_valid",
    ]
    assert np.isnan(table.loc[0, f"seasonal_{7 * PERIOD}"])
