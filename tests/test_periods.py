"""Per-period error breakdown, and the three diagnostics the failure section's argument rests on.

Synthetic data throughout: these must run without the processed store so a clean clone can check
them. The diagnostics have exact values on constructed inputs (a persistence forecast correlates
with the first difference at exactly 1.0, a constant offset puts 100 percent of MSE in bias),
which is what makes them worth pinning rather than eyeballing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from milan_traffic import metrics as M
from milan_traffic import periods as PD


def _index(n: int = 288, start: str = "2013-12-19 00:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="10min", tz="Europe/Rome")


def test_day_masks_are_chronological_and_partition_the_window():
    idx = _index(432)  # three days
    masks = PD.period_masks(idx, "day")
    assert [label for label, _ in masks] == ["Thu 19 Dec", "Fri 20 Dec", "Sat 21 Dec"]
    stacked = np.stack([m for _, m in masks])
    assert stacked.sum(axis=0).tolist() == [1] * 432
    assert [int(m.sum()) for _, m in masks] == [144, 144, 144]


def test_daytype_masks_group_a_public_holiday_with_the_weekend():
    """Not a weekday/weekend split: Sant'Ambrogio is a Saturday in 2013, but Immacolata on the
    Sunday and Ognissanti on a Friday are the cases that matter, so the split goes through
    ``features.is_working_day`` rather than through ``dayofweek``."""
    idx = pd.date_range("2013-11-01", periods=288, freq="10min", tz="Europe/Rome")
    (_, working), (_, rest) = PD.period_masks(idx, "daytype")
    assert not working.any()  # 1 Nov is Ognissanti, 2 Nov is a Saturday
    assert rest.all()
    with pytest.raises(ValueError):
        PD.period_masks(idx, "hour")


def test_per_period_errors_agree_with_the_metrics_module_on_each_day():
    rng = np.random.default_rng(0)
    idx = _index(288)
    y = rng.normal(2000, 300, size=288)
    pred = np.stack([y + rng.normal(0, 50, size=288) for _ in range(3)])
    frame = PD.per_period_errors(y, {"m": pred}, idx)
    assert list(frame["period"]) == ["Thu 19 Dec", "Fri 20 Dec"]
    for label, mask in PD.period_masks(idx, "day"):
        row = frame[frame.period == label].iloc[0]
        expected = float(np.mean([M.mae(y[mask], p[mask]) for p in pred]))
        assert row["MAE"] == pytest.approx(expected, rel=1e-12)
        assert row["n_slots"] == 144 and row["n_seeds"] == 3
        assert row["mean_level"] == pytest.approx(float(y[mask].mean()))


def test_signed_mean_error_is_negative_when_the_model_over_predicts():
    """The sign convention the failure diagnosis depends on, stated in the docstring and pinned
    here because getting it backwards would invert the whole argument about the afternoon
    plateau."""
    idx = _index(144)
    y = np.full(144, 1000.0)
    frame = PD.per_period_errors(y, {"high": y + 50.0, "low": y - 50.0}, idx)
    assert frame[frame.model == "high"].iloc[0]["ME"] == pytest.approx(-50.0)
    assert frame[frame.model == "low"].iloc[0]["ME"] == pytest.approx(+50.0)


def test_excess_error_share_separates_its_two_denominators():
    """One period better and two worse: the net and positive denominators then differ, which is
    why both are reported and named."""
    idx = _index(432)
    y = np.zeros(432)
    pred = np.zeros(432)
    ref = np.zeros(432)
    pred[:144] = 1.0  # day 1: model worse by 144
    pred[144:288] = 3.0  # day 2: model worse by 432
    ref[288:] = 2.0  # day 3: model better by 288
    frame = PD.excess_error_share(y, pred, ref, idx)
    by = frame.set_index("period")
    assert by.loc["Thu 19 Dec", "excess"] == pytest.approx(144.0)
    assert by.loc["Sat 21 Dec", "excess"] == pytest.approx(-288.0)
    assert by.loc["TOTAL", "excess"] == pytest.approx(288.0)
    # net denominator 288, positive denominator 576
    assert by.loc["Fri 20 Dec", "share_of_net"] == pytest.approx(150.0)
    assert by.loc["Fri 20 Dec", "share_of_positive"] == pytest.approx(75.0)


def test_excess_error_share_reports_the_spread_across_seeds():
    """So the artefact itself answers whether a concentration holds per seed or only on the mean."""
    idx = _index(288)
    y = np.zeros(288)
    pred = np.stack([np.concatenate([np.full(144, k), np.zeros(144)]) for k in (1.0, 5.0)])
    frame = PD.excess_error_share(y, pred, np.zeros(288), idx).set_index("period")
    assert frame.loc["Thu 19 Dec", "share_of_net"] == pytest.approx(100.0)
    assert frame.loc["Thu 19 Dec", "excess"] == pytest.approx(144.0 * 3.0)
    assert frame.loc["Thu 19 Dec", "excess_sd"] > 0.0


def _penalty_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Three days, zoom on the middle one, truth at zero so the penalty is read off directly.

    With ``y = 0`` the penalty at a slot is ``|pred| - |reference|``, so every quantity below is
    arithmetic on the numbers written here rather than on a fitted forecast.
    """
    idx = _index(432)
    zoom = np.zeros(432, dtype=bool)
    zoom[144:288] = True
    return np.zeros(432), np.zeros(432), np.zeros(432), idx, zoom


def test_calendar_penalty_pooled_share_is_the_penalty_on_the_zoom_days():
    """Day 1 costs 1 per slot and day 2 costs 3, so the zoom day carries 432 of 576, i.e. 75."""
    y, pred, reference, idx, zoom = _penalty_case()
    pred[:144] = 1.0
    pred[144:288] = 3.0
    per_day, per_seed = PD.calendar_penalty(y, pred, reference, idx, zoom)

    assert list(per_day["period"]) == [
        "Thu 19 Dec",
        "Fri 20 Dec",
        "Sat 21 Dec",
        "Fri 20 Dec",
        "whole week",
    ]
    by_row = per_day.iloc[[0, 2]].set_index("period")
    assert by_row.loc["Thu 19 Dec", "penalty_total"] == pytest.approx(144.0)
    assert by_row.loc["Sat 21 Dec", "penalty_total"] == pytest.approx(0.0)
    zoom_row, week_row = per_day.iloc[-2], per_day.iloc[-1]
    assert zoom_row["n_slots"] == 144
    assert zoom_row["penalty_per_slot"] == pytest.approx(3.0)
    assert zoom_row["penalty_total"] == pytest.approx(432.0)
    assert zoom_row["share_of_week_pct"] == pytest.approx(75.0)
    assert week_row["penalty_total"] == pytest.approx(576.0)
    assert week_row["share_of_week_pct"] == pytest.approx(100.0)
    # A 1-D forecast is one seed, and that seed's share is the pooled one.
    assert list(per_seed["seed"]) == ["#0"]
    assert per_seed.iloc[0]["share_of_week_pct"] == pytest.approx(75.0)


def test_calendar_penalty_zoom_row_names_every_day_it_covers():
    y, pred, reference, idx, zoom = _penalty_case()
    pred[:] = 1.0
    zoom[:144] = True  # Thursday and Friday, the pair the report quotes
    per_day, _ = PD.calendar_penalty(y, pred, reference, idx, zoom)
    assert per_day.iloc[-2]["period"] == "Thu 19 Dec + Fri 20 Dec"
    assert per_day.iloc[-2]["n_slots"] == 288


def test_calendar_penalty_share_exceeds_100_percent_for_a_seed_that_gains_elsewhere():
    """The behaviour that made the first reported figure wrong.

    Seed 1 is better off without the feature on the days outside the zoom, so its week total is
    small and its own share is unbounded above. A share above 100 percent is not a bug to clip
    away, it is the reason the per-seed shares must not be averaged into a headline number.
    """
    y, _, reference, idx, zoom = _penalty_case()
    reference[~zoom] = 6.0
    pred = np.zeros((2, 432))
    pred[:, zoom] = 10.0
    pred[0, ~zoom] = 12.0  # penalty +6 per slot outside the zoom
    pred[1, ~zoom] = 2.0  # penalty -4 per slot outside the zoom
    per_day, per_seed = PD.calendar_penalty(y, pred, reference, idx, zoom, seeds=[42, 1337])

    assert list(per_seed["seed"]) == ["42", "1337"]
    assert list(per_seed["penalty_zoom"]) == pytest.approx([1440.0, 1440.0])
    assert list(per_seed["penalty_week"]) == pytest.approx([3168.0, 288.0])
    assert list(per_seed["share_of_week_pct"]) == pytest.approx([1440 / 3168 * 100, 500.0])
    assert (per_seed["share_of_week_pct"] > 100.0).any()
    # The pooled share stays inside 100 while one seed sits at 500.
    assert per_day.iloc[-2]["share_of_week_pct"] == pytest.approx(1440 / 1728 * 100)


def test_calendar_penalty_pooled_share_is_not_the_mean_of_the_per_seed_shares():
    """The distinction the generated caption rests on, pinned so it cannot quietly collapse."""
    y, _, reference, idx, zoom = _penalty_case()
    reference[~zoom] = 6.0
    pred = np.zeros((2, 432))
    pred[:, zoom] = 10.0
    pred[0, ~zoom] = 12.0
    pred[1, ~zoom] = 2.0
    per_day, per_seed = PD.calendar_penalty(y, pred, reference, idx, zoom)

    pooled = per_day.iloc[-2]["share_of_week_pct"]
    assert pooled == pytest.approx(83.333333, rel=1e-6)
    assert per_seed["share_of_week_pct"].mean() == pytest.approx(272.727272, rel=1e-6)
    assert pooled < per_seed["share_of_week_pct"].mean() / 3
    # Unnamed seeds fall back to positional labels rather than borrowing the wrong ids.
    assert list(per_seed["seed"]) == ["#0", "#1"]


def test_calendar_penalty_ignores_a_seed_list_that_does_not_match_the_forecast():
    y, pred, reference, idx, zoom = _penalty_case()
    pred[:] = 1.0
    _, per_seed = PD.calendar_penalty(y, pred, reference, idx, zoom, seeds=[42, 1337, 2024])
    assert list(per_seed["seed"]) == ["#0"]


def test_persistence_residual_correlates_with_the_first_difference_at_exactly_one():
    """The lag test. A forecast that simply repeats the last observation has residual equal to
    the series' first difference, so a correlation near 1 means a model is following one step
    late rather than predicting."""
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(size=300)) + 100.0
    persistence = np.concatenate([[y[0]], y[:-1]])
    frame = PD.residual_diagnostics(y, {"naive": persistence}).set_index("model")
    assert frame.loc["naive", "corr_resid_diff"] == pytest.approx(1.0, abs=1e-12)


def test_constant_offset_puts_all_of_mse_in_bias_and_noise_puts_almost_none():
    y = np.full(500, 1000.0)
    rng = np.random.default_rng(2)
    frame = PD.residual_diagnostics(
        y, {"biased": y + 30.0, "noisy": y + rng.normal(0, 30, size=500)}
    ).set_index("model")
    assert frame.loc["biased", "bias2_share"] == pytest.approx(100.0)
    assert frame.loc["noisy", "bias2_share"] < 2.0
    assert frame.loc["biased", "ME"] == pytest.approx(-30.0)


def test_residual_diagnostics_respects_the_mask_and_reports_period_volatility():
    rng = np.random.default_rng(4)
    y = np.concatenate([np.full(100, 500.0), np.arange(100) * 50.0 + 500.0])
    pred = y + rng.normal(0, 10, size=200)
    mask = np.zeros(200, dtype=bool)
    mask[:100] = True
    calm = PD.residual_diagnostics(y, {"m": pred}, mask=mask).iloc[0]
    busy = PD.residual_diagnostics(y, {"m": pred}, mask=~mask).iloc[0]
    assert calm["n_slots"] == 100 and busy["n_slots"] == 100
    assert calm["mean_abs_diff"] < busy["mean_abs_diff"]


def test_degenerate_residuals_return_nan_rather_than_dividing_by_zero():
    """A perfect forecast has no MSE to attribute and a constant residual has no correlation
    with anything. Both are reported as undefined instead of crashing or reading as zero."""
    y = np.linspace(100.0, 200.0, 50)
    frame = PD.residual_diagnostics(y, {"perfect": y, "offset": y + 7.0}).set_index("model")
    assert np.isnan(frame.loc["perfect", "bias2_share"])
    assert np.isnan(frame.loc["perfect", "corr_resid_diff"])
    assert frame.loc["offset", "bias2_share"] == pytest.approx(100.0)
    assert np.isnan(frame.loc["offset", "corr_resid_diff"])


def test_error_concentration_is_flat_for_a_uniform_error():
    y = np.zeros(1000)
    frame = PD.error_concentration(y, {"m": np.ones(1000)}, counts=(10, 100))
    assert frame.iloc[0]["pct_of_error"] == pytest.approx(1.0)
    assert frame.iloc[1]["pct_of_error"] == pytest.approx(10.0)
    assert frame.iloc[1]["pct_of_week"] == pytest.approx(10.0)


def test_error_concentration_detects_a_concentrated_error():
    y = np.zeros(1000)
    pred = np.zeros(1000)
    pred[:10] = 100.0
    frame = PD.error_concentration(y, {"m": pred}, counts=(10,))
    assert frame.iloc[0]["pct_of_error"] == pytest.approx(100.0)


def test_worst_slots_ranks_by_pooled_error_and_reports_the_step():
    idx = _index(20)
    y = np.zeros(20)
    y[5] = 400.0
    pred = np.zeros(20)
    frame = PD.worst_slots(y, {"m": pred}, idx, k=2)
    assert frame.iloc[0]["timestamp"] == idx[5]
    assert frame.iloc[0]["truth"] == pytest.approx(400.0)
    assert frame.iloc[0]["delta"] == pytest.approx(400.0)
    assert frame.iloc[0]["pooled_ae"] >= frame.iloc[1]["pooled_ae"]


def test_compare_on_mask_reduces_to_the_shared_dm_test():
    from milan_traffic.significance import dm_test, seed_averaged_loss

    rng = np.random.default_rng(3)
    idx = _index(288)
    y = rng.normal(1000, 100, size=288)
    a = np.stack([y + rng.normal(0, 40, size=288) for _ in range(3)])
    b = np.stack([y + rng.normal(0, 60, size=288) for _ in range(3)])
    mask = np.zeros(288, dtype=bool)
    mask[:144] = True
    frame = PD.compare_on_mask(y, a, b, idx, {"half": mask})
    direct = dm_test((seed_averaged_loss(y, a) - seed_averaged_loss(y, b))[mask])
    assert frame.iloc[0]["d"] == pytest.approx(direct.mean)
    assert frame.iloc[0]["p"] == pytest.approx(direct.p)
    assert frame.iloc[0]["n"] == 144


def test_to_markdown_right_aligns_numerics_and_labels_missing_values():
    frame = pd.DataFrame({"period": ["Thu"], "MAE": [98.657], "note": [None]})
    text = PD.to_markdown(frame)
    lines = text.strip().split("\n")
    assert lines[0] == "| period | MAE | note |"
    assert lines[1] == "|:--|--:|:--|"
    assert "98.66" in lines[2] and "n/a" in lines[2]


def test_prediction_with_the_wrong_step_count_is_rejected():
    idx = _index(144)
    with pytest.raises(ValueError, match="steps"):
        PD.per_period_errors(np.zeros(144), {"m": np.zeros(143)}, idx)
