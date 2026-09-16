"""The plotting layer, on synthetic data.

These tests do not judge how a figure looks; that is done by eye on the rendered PNGs.
They check the contract the report generation relies on: every function returns a
Figure with the expected panel structure, every annotated number is computed from the
data passed in (a fixed string would survive a data change unnoticed), and the markdown
table bolds the right cell. A non-interactive backend keeps the suite headless.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from milan_traffic import viz  # noqa: E402
from milan_traffic.config import SLOTS_PER_DAY, SLOTS_PER_WEEK  # noqa: E402

FIVE = (5161, 5059, 5259, 4159, 4556)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def two_weeks_index() -> pd.DatetimeIndex:
    return pd.date_range("2013-11-01", periods=2 * SLOTS_PER_WEEK, freq="10min", tz="Europe/Rome")


@pytest.fixture
def eval_week_index() -> pd.DatetimeIndex:
    return pd.date_range("2013-12-16", periods=SLOTS_PER_WEEK, freq="10min", tz="Europe/Rome")


def _daily(n: int, level: float, seed: int) -> np.ndarray:
    t = np.arange(n)
    base = level * (1 + 0.8 * np.sin(2 * np.pi * t / SLOTS_PER_DAY - np.pi / 2))
    return base + np.random.default_rng(seed).normal(0, 0.05 * level, n)


def _texts(fig) -> list[str]:
    out = []
    for ax in fig.axes:
        out += [t.get_text() for t in ax.texts]
        out += [t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() else []
    out += [t.get_text() for t in fig.texts]
    for leg in fig.legends:
        out += [t.get_text() for t in leg.get_texts()]
    return out


# ------------------------------------------------------------- concentration numbers


def test_gini_and_shares_on_known_vectors():
    assert viz.gini(np.ones(1000)) == pytest.approx(0.0, abs=1e-9)
    one_hot = np.zeros(1000)
    one_hot[0] = 5.0
    assert viz.gini(one_hot) == pytest.approx(1 - 1 / 1000, abs=1e-9)
    assert viz.top_share(one_hot, 0.01) == pytest.approx(1.0)
    assert viz.cells_to_share(one_hot, 0.5) == 1
    assert viz.cells_to_share(np.ones(1000), 0.5) == 500


def test_total_traffic_distribution_two_panels_annotated_from_data():
    totals = np.random.default_rng(0).lognormal(mean=12.5, sigma=1.0, size=10_000)
    fig = viz.plot_total_traffic_distribution(totals)
    assert len(fig.axes) == 2
    joined = " ".join(_texts(fig))
    assert f"Gini = {viz.gini(totals):.3f}" in joined
    assert f"{viz.top_share(totals) * 100:.2f} % of traffic" in joined
    assert f"{viz.cells_to_share(totals):,} cells" in joined


def test_grid_heatmap_full_zoom_and_colourbar():
    totals = np.random.default_rng(1).lognormal(mean=12.0, sigma=1.0, size=10_000)
    ids = np.arange(1, 10_001)
    fig = viz.plot_grid_heatmap(totals, ids, highlight=FIVE)
    assert len(fig.axes) == 3  # full grid, zoom, colourbar
    zoom_labels = {t.get_text() for t in fig.axes[1].texts}
    assert {str(s) for s in FIVE} <= zoom_labels
    with pytest.raises(ValueError):
        viz.plot_grid_heatmap(totals[:-1], ids)


def test_five_cells_two_weeks_one_panel_per_cell(two_weeks_index):
    n = two_weeks_index.size
    series = {s: _daily(n, 1000.0 * (i + 1), i) for i, s in enumerate(FIVE)}
    fig = viz.plot_five_cells_two_weeks(
        series, two_weeks_index, labels={5161: "square 5161 (rank 1)"}
    )
    assert len(fig.axes) == 5
    assert fig.axes[0].get_title(loc="left") == "square 5161 (rank 1)"
    assert any("Ognissanti" in t for t in _texts(fig))  # Fri 1 Nov is labelled
    with pytest.raises(ValueError):
        viz.plot_five_cells_two_weeks({5161: np.ones(10)}, two_weeks_index)


def test_weekly_weekday_means_normalises_and_reports_change():
    weeks = list(range(45, 52))
    table = pd.DataFrame(
        {
            "grid": np.linspace(100.0, 82.0, len(weeks)),
            "5161": np.full(len(weeks), 50.0),
            "4159": np.linspace(10.0, 7.5, len(weeks)),
        },
        index=weeks,
    )
    fig = viz.plot_weekly_weekday_means(table, base_week=45, highlight_week=51)
    assert len(fig.axes) == 1
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "all 10 000 cells (-18.0 % by week 51)" in legend
    assert "square 5161 (+0.0 % by week 51)" in legend
    assert "square 4159 (-25.0 % by week 51)" in legend
    with pytest.raises(ValueError):
        viz.plot_weekly_weekday_means(table, base_week=40)


def test_grid_spike_annotates_ratio_from_data():
    n = 62 * SLOTS_PER_DAY
    index = pd.date_range("2013-11-01", periods=n, freq="10min", tz="Europe/Rome")
    g = np.full(n, 100.0)
    g[5165] = 186.0
    g[5166] = 86.0
    fig = viz.plot_grid_spike(g, index, center_row=5165, halo=9)
    assert len(fig.axes) == 1
    texts = _texts(fig)
    expected = g[5165] / ((g[5164] + g[5166]) / 2)
    assert any(f"{expected:.2f}x" in t for t in texts)
    assert any("-14 %" in t for t in texts)
    assert any("Fri 06 Dec" in t for t in texts)
    with pytest.raises(ValueError):
        viz.plot_grid_spike(g, index, center_row=3, halo=9)


# ---------------------------------------------------------------- experiment figures


def test_actual_vs_predicted_grid_is_areas_by_models(eval_week_index):
    n = eval_week_index.size
    y = np.stack([_daily(n, 1000.0 * (i + 1), i) for i in range(3)])
    preds = {
        m: y + np.random.default_rng(k).normal(0, 30, y.shape)
        for k, m in enumerate(("A", "B", "C"))
    }
    fig = viz.plot_actual_vs_predicted(y, preds, eval_week_index, area_ids=[5161, 5059, 5259])
    assert len(fig.axes) == 9
    titles = [fig.axes[j].get_title(loc="left") for j in range(3)]
    assert titles == ["A", "B", "C"]
    assert len(fig.legends) == 1
    with pytest.raises(ValueError):
        viz.plot_actual_vs_predicted(y, preds, eval_week_index, area_ids=[1, 2])


def test_actual_vs_predicted_single_with_zoom_inset(eval_week_index):
    n = eval_week_index.size
    y = _daily(n, 1000.0, 0)
    p = y + 20
    fig = viz.plot_actual_vs_predicted_single(
        y, p, eval_week_index, area_id=5161, model_name="LSTM"
    )
    assert len(fig.axes) == 1 and len(fig.axes[0].child_axes) == 0
    fig2 = viz.plot_actual_vs_predicted_single(
        y,
        p,
        eval_week_index,
        area_id=5161,
        model_name="LSTM",
        zoom=("2013-12-18 06:00", "2013-12-18 12:00"),
    )
    assert len(fig2.axes) == 1 and len(fig2.axes[0].child_axes) == 1
    fig3 = viz.plot_actual_vs_predicted_single(
        y, p, eval_week_index, area_id=5161, model_name="LSTM", zoom=slice(100, 200)
    )
    assert len(fig3.axes[0].child_axes) == 1


def test_error_by_hour_panels(eval_week_index):
    n = eval_week_index.size
    res = {
        "A": np.random.default_rng(0).normal(0, 1, n),
        "B": np.random.default_rng(1).normal(0, 2, (2, n)),
    }
    fig = viz.plot_error_by_hour(res, eval_week_index)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].get_lines()) == 2
    fig2 = viz.plot_error_by_hour(res, eval_week_index, y_true=_daily(n, 500.0, 2))
    assert len(fig2.axes) == 2


def test_daily_peak_times_and_table(two_weeks_index):
    s = pd.Series(_daily(two_weeks_index.size, 1000.0, 3), index=two_weeks_index)
    table = viz.daily_peak_table(s)
    assert list(table.columns) == ["date", "peak_hour", "peak_value", "kind"]
    assert len(table) == 14
    assert table.loc[0, "kind"] == "holiday"  # Fri 1 Nov 2013, Ognissanti
    assert (table["kind"] == "weekend").sum() == 4
    assert table["peak_hour"].between(0, 24).all()
    fig = viz.plot_daily_peak_times(table, square_id=5259)
    assert len(fig.axes) == 1
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["working day", "Saturday / Sunday", "public holiday"]


# -------------------------------------------------------------------------- tables


def test_metrics_table_bolds_min_mae_and_max_r2():
    df = pd.DataFrame(
        {
            "area": [1, 1, 1, 2, 2, 2],
            "model": ["naive", "lstm", "gru"] * 2,
            "MAE": [10.0, 8.0, 9.0, 5.0, 6.0, 4.5],
            "RMSE": [15.0, 12.0, 13.0, 7.0, 8.0, 6.0],
            "R2": [0.80, 0.90, 0.85, 0.70, 0.60, 0.75],
            "MASE": [1.0, 0.8, 0.9, np.nan, 0.7, 0.8],
            "n_params": [0, 1200, 900, 0, 1200, 900],
        }
    )
    md = viz.metrics_table_markdown(
        df, metrics=("MAE", "RMSE", "MASE", "R2"), extra_cols=("n_params",)
    )
    assert "### Square 1" in md and "### Square 2" in md
    block1, block2 = md.split("### Square 2")
    assert "| lstm | **8.00** | **12.00** | **0.800** | **0.900** | 1,200 |" in block1
    assert "| naive | 10.00 | 15.00 | 1.000 | 0.800 | 0 |" in block1
    assert "| gru | **4.50** | **6.00** | 0.800 | **0.750** | 900 |" in block2
    assert "| naive | 5.00 | 7.00 | n/a | 0.700 | 0 |" in block2
    assert block1.count("**") == 8  # four bold cells, all on the lstm row
    with pytest.raises(ValueError):
        viz.metrics_table_markdown(df.drop(columns="area"))


def test_metrics_table_averages_repeated_seeds():
    df = pd.DataFrame(
        {
            "area": [1, 1, 1],
            "model": ["lstm", "lstm", "naive"],
            "MAE": [8.0, 10.0, 9.5],
            "R2": [0.9, 0.8, 0.85],
        }
    )
    md = viz.metrics_table_markdown(df, metrics=("MAE", "R2"))
    assert "| lstm | **9.00** | **0.850** |" in md
    assert "| naive | 9.50 | 0.850 |" in md or "| naive | 9.50 | **0.850** |" in md


# --------------------------------------------------------------- input helpers


def test_grid_totals_chunked_matches_full_sum():
    m = np.random.default_rng(0).random((1000, 50)).astype(np.float32)
    out = viz.grid_totals(m, chunk_rows=288)
    np.testing.assert_allclose(out, m.astype(np.float64).sum(axis=1), rtol=1e-6)
    assert out.shape == (1000,)


def test_weekday_means_by_iso_week_drops_partial_weeks(two_weeks_index):
    frame = pd.DataFrame({"a": np.ones(two_weeks_index.size)}, index=two_weeks_index)
    frame.loc[two_weeks_index.dayofweek >= 5, "a"] = 100.0  # weekends must be excluded
    table = viz.weekday_means_by_iso_week(frame)
    assert list(table.index) == [45]  # week 44 has only Fri 1 Nov; week 46 has Mon to Thu
    assert table.loc[45, "a"] == pytest.approx(1.0)
    everything = viz.weekday_means_by_iso_week(frame, full_weeks_only=False)
    assert list(everything.index) == [44, 45, 46]


def test_colour_for_is_fixed_and_bounded():
    assert viz.colour_for(["x", "y"]) == {"x": viz.PALETTE[0], "y": viz.PALETTE[1]}
    with pytest.raises(ValueError):
        viz.colour_for(list(range(9)))


def _dm_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "area": [5059, 5161, 5259],
            "model_a": ["TCN", "TCN", "TCN"],
            "model_b": ["linear AR(144)"] * 3,
            "d": [12.63, -2.85, -2.25],
            "ci_low": [5.17, -9.01, -6.72],
            "ci_high": [20.09, 3.31, 2.21],
            "p": [0.0009, 0.3648, 0.3223],
        }
    )


def test_dm_intervals_draws_a_row_per_comparison_and_a_zero_line():
    fig = viz.plot_dm_intervals(_dm_frame())
    ax = fig.axes[0]
    assert len(ax.get_yticklabels()) == 3
    assert [t.get_text() for t in ax.get_yticklabels()][0].startswith("5059")
    assert any(ln.get_linestyle() == "--" for ln in ax.get_lines())


def test_dm_intervals_annotates_the_p_value_from_the_data():
    """A hard-coded label would survive a data change unnoticed, so the text must track ``p``."""
    frame = _dm_frame()
    texts = {t.get_text() for t in viz.plot_dm_intervals(frame).axes[0].texts}
    assert "p<0.001" in texts  # 0.0009
    assert "p=0.365" in texts
    frame.loc[1, "p"] = 0.0421
    assert "p=0.042" in {t.get_text() for t in viz.plot_dm_intervals(frame).axes[0].texts}


def test_dm_intervals_reserves_room_to_the_right_of_the_widest_interval():
    """Without the margin the widest interval's point estimate sits under its own p label."""
    frame = _dm_frame()
    ax = viz.plot_dm_intervals(frame).axes[0]
    assert ax.get_xlim()[1] > float(frame["ci_high"].max())


def test_dm_intervals_rejects_a_frame_missing_a_required_column():
    with pytest.raises(ValueError, match="needs columns"):
        viz.plot_dm_intervals(_dm_frame().drop(columns=["p"]))


def test_per_period_error_has_three_panels_and_one_colour_per_entity():
    """A model must keep its colour between the residual panel and the per-day panel: the two
    are read together, and ``colour_for`` assigns by position, so each panel's own list would
    give the same model two colours."""
    idx = pd.date_range("2013-12-19", periods=2 * SLOTS_PER_DAY, freq="10min", tz="Europe/Rome")
    truth = np.full(idx.size, 1000.0)
    zoom = {"TCN": truth + 120.0, "TCN, no calendar": truth + 40.0, "Linear AR(144)": truth - 20.0}
    days = ["Mon 16 Dec", "Tue 17 Dec", "Sat 21 Dec"]
    mae = pd.DataFrame(
        {
            "TCN": [62.0, 82.0, 66.0],
            "TCN, no calendar": [60.0, 80.0, 66.0],
            "Persistence": [86.0, 102.0, 76.0],
            "Linear AR(144)": [67.0, 82.0, 66.0],
        },
        index=days,
    )
    wape = pd.DataFrame(
        {5059: [5.0, 6.1, 5.7], 5161: [6.7, 6.3, 5.4], 5259: [4.6, 4.4, 9.2]}, index=days
    )
    fig = viz.plot_per_period_error(
        zoom_truth=truth, zoom_predictions=zoom, zoom_index=idx, daily_mae=mae, daily_wape=wape
    )
    assert len(fig.axes) == 3
    resid, per_day = fig.axes[0], fig.axes[1]
    a = {ln.get_label(): ln.get_color() for ln in resid.get_lines() if ln.get_label() in zoom}
    b = {ln.get_label(): ln.get_color() for ln in per_day.get_lines() if ln.get_label() in mae}
    shared = set(a) & set(b)
    assert shared == {"TCN", "TCN, no calendar", "Linear AR(144)"}
    assert all(a[k] == b[k] for k in shared)


def test_per_period_error_plots_the_signed_error_not_the_level():
    """Panel (a)'s whole purpose is a bias of about 8 percent on a plateau, which is invisible on
    an axis that spans the diurnal cycle. An over-predicting model must appear below zero."""
    idx = pd.date_range("2013-12-19", periods=SLOTS_PER_DAY, freq="10min", tz="Europe/Rome")
    truth = np.full(idx.size, 2500.0)
    fig = viz.plot_per_period_error(
        zoom_truth=truth,
        zoom_predictions={"TCN": truth + 200.0},
        zoom_index=idx,
        daily_mae=pd.DataFrame({"TCN": [1.0]}, index=["Thu 19 Dec"]),
        daily_wape=pd.DataFrame({5059: [1.0]}, index=["Thu 19 Dec"]),
    )
    drawn = [ln for ln in fig.axes[0].get_lines() if ln.get_label() == "TCN"][0]
    assert np.allclose(drawn.get_ydata(), -200.0)
    assert fig.axes[0].get_ylim()[0] < 0.0
