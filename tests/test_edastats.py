"""Descriptive statistics on synthetic data: a known grid and two made-up cells.

These statistics are quoted directly in Section IV and underpin the best-model argument
of Section VI, which makes them the numbers in the project with the shortest path from a
silent error to a wrong claim in the report. Nothing here reads ``data/processed/``: every
fixture is constructed so that the right answer can be written down independently, which
is the only way a test of a descriptive statistic is worth anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from milan_traffic import edastats as eda
from milan_traffic.config import GRID_SIDE, rowcol_to_square, square_to_rowcol

PERIOD = 24
N_DAYS = 14


# --------------------------------------------------------------------- geometry


def test_cell_size_is_about_235_metres_each_way():
    lat, lon = eda.centroid(50, 50)
    east = eda.haversine_m(lat, lon, lat, lon + eda.CELL_LON)
    north = eda.haversine_m(lat, lon, lat + eda.CELL_LAT, lon)
    assert 230.0 < east < 240.0
    assert 230.0 < north < 240.0


def test_centroids_stay_inside_the_published_bounding_box():
    for row, col in (
        (0, 0),
        (0, GRID_SIDE - 1),
        (GRID_SIDE - 1, 0),
        (GRID_SIDE - 1, GRID_SIDE - 1),
    ):
        lat, lon = eda.centroid(row, col)
        assert eda.GRID_LAT_MIN < lat < eda.GRID_LAT_MAX
        assert eda.GRID_LON_MIN < lon < eda.GRID_LON_MAX


def test_south_origin_puts_row_zero_at_the_bottom_and_north_flips_it():
    south_lat, south_lon = eda.centroid(0, 7, origin="south")
    north_lat, north_lon = eda.centroid(0, 7, origin="north")
    assert south_lat < north_lat
    assert south_lon == north_lon
    # Row r under one convention is row (99 - r) under the other, exactly.
    assert eda.centroid(0, 7, origin="north") == pytest.approx(
        eda.centroid(GRID_SIDE - 1, 7, origin="south")
    )


def test_cell_of_inverts_centroid_for_every_probe():
    for row, col in ((0, 0), (13, 87), (51, 60), (99, 99)):
        lat, lon = eda.centroid(row, col)
        assert eda.cell_of(lat, lon) == (row, col)
        north_lat, north_lon = eda.centroid(row, col, origin="north")
        assert eda.cell_of(north_lat, north_lon, origin="north") == (row, col)


def test_cell_of_rejects_a_point_outside_the_grid():
    with pytest.raises(ValueError, match="bounding box"):
        eda.cell_of(45.0, 9.19)


def test_centroid_rejects_an_unknown_origin():
    with pytest.raises(ValueError, match="unknown origin"):
        eda.centroid(0, 0, origin="up")  # type: ignore[arg-type]


def test_square_centroid_agrees_with_the_row_col_mapping():
    square = rowcol_to_square(51, 60)
    assert eda.square_centroid(square) == eda.centroid(*square_to_rowcol(square))


def test_haversine_is_zero_for_a_point_and_symmetric():
    assert eda.haversine_m(45.0, 9.0, 45.0, 9.0) == pytest.approx(0.0)
    assert eda.haversine_m(45.0, 9.0, 45.1, 9.1) == pytest.approx(
        eda.haversine_m(45.1, 9.1, 45.0, 9.0)
    )


def test_one_degree_of_latitude_is_about_111_kilometres():
    assert eda.haversine_m(45.0, 9.0, 46.0, 9.0) == pytest.approx(111_195.0, rel=1e-3)


@pytest.mark.parametrize(
    "dlat,dlon,expected",
    [(1.0, 0.0, "N"), (0.0, 1.0, "E"), (-1.0, 0.0, "S"), (0.0, -1.0, "W")],
)
def test_bearings_point_the_way_the_offset_does(dlat, dlon, expected):
    bearing = eda.initial_bearing_deg(45.0, 9.0, 45.0 + dlat * 0.01, 9.0 + dlon * 0.01)
    assert eda.compass_point(bearing) == expected


def test_compass_covers_the_full_circle_and_wraps():
    assert eda.compass_point(0.0) == "N"
    assert eda.compass_point(359.9) == "N"
    assert eda.compass_point(360.0 + 90.0) == "E"
    assert len({eda.compass_point(b) for b in np.arange(0, 360, 1.0)}) == 16


# ------------------------------------------------------------------ orientation


@pytest.fixture
def planted_grid() -> np.ndarray:
    """A grid built to satisfy every landmark's expectation under the south convention.

    Each landmark expected to be a maximum gets a bright cell at its south-origin
    position and each expected minimum gets a dark one, over a north-heavy background so
    that the mass statistics also have the right sign. The mirrored positions are left at
    the background level, so the reflected reading has to fail on all six.
    """
    grid = np.full((GRID_SIDE, GRID_SIDE), 100.0)
    grid[GRID_SIDE // 2 :, :] += 100.0  # the built-up half is the northern half
    for mark in eda.LANDMARKS:
        if mark.expect is None:
            continue
        row, col = eda.cell_of(mark.lat, mark.lon)
        grid[row, col] = 10_000.0 if mark.expect == "max" else 1.0
    return grid


def test_neighbourhood_median_uses_the_block_and_clips_at_the_edge():
    grid = np.arange(49, dtype=np.float64).reshape(7, 7)
    assert eda.neighbourhood_median(grid, 3, 3) == pytest.approx(24.0)
    # A corner sees only the 3x3 block that exists.
    assert eda.neighbourhood_median(grid, 0, 0) == pytest.approx(np.median(grid[0:3, 0:3]))


def test_neighbourhood_median_includes_the_centre_cell():
    grid = np.zeros((5, 5))
    grid[2, 2] = 1.0
    # 25 cells, 24 zeros and one 1.0, so the median is still 0 but the cell is counted.
    assert eda.neighbourhood_median(grid, 2, 2) == pytest.approx(0.0)
    assert eda.neighbourhood_median(np.ones((5, 5)), 2, 2) == pytest.approx(1.0)


def test_orientation_check_recovers_a_planted_south_origin_grid(planted_grid):
    check = eda.orientation_check(planted_grid)
    south = check.set_index("landmark")["south verdict"]
    assert (
        list(south[south.index.isin([m.name for m in eda.LANDMARKS if m.expect == "max"])])
        == ["pass"] * 4
    )
    scores = eda.orientation_scores(check).set_index("convention")
    assert scores.loc["south", "FAIL"] == 0
    assert scores.loc["south", "pass"] == 6
    assert scores.loc["north", "pass"] == 0


def test_best_neighbour_ratio_finds_a_peak_one_cell_away():
    grid = np.full((9, 9), 100.0)
    grid[4, 3] = 10_000.0  # the bright cell is one column west of (4, 4)
    drow, dcol, ratio = eda.best_neighbour_ratio(grid, 4, 4, mode="max")
    assert (drow, dcol) == (0, -1)
    assert ratio == pytest.approx(100.0)
    assert eda.offset_compass(4, 4, drow, dcol) == "W"


def test_best_neighbour_ratio_in_min_mode_finds_the_quietest_cell():
    grid = np.full((9, 9), 100.0)
    grid[5, 4] = 1.0  # one row north of (4, 4) under the south-origin convention
    drow, dcol, ratio = eda.best_neighbour_ratio(grid, 4, 4, mode="min")
    assert (drow, dcol) == (1, 0)
    assert ratio == pytest.approx(0.01)
    assert eda.offset_compass(4, 4, drow, dcol) == "N"


def test_best_neighbour_ratio_stays_put_when_the_cell_itself_is_the_extreme():
    grid = np.full((9, 9), 100.0)
    grid[4, 4] = 10_000.0
    assert eda.best_neighbour_ratio(grid, 4, 4, mode="max")[:2] == (0, 0)


def test_best_neighbour_ratio_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        eda.best_neighbour_ratio(np.ones((9, 9)), 4, 4, mode="sideways")  # type: ignore[arg-type]


def test_orientation_check_reports_the_square_ids_it_used(planted_grid):
    check = eda.orientation_check(planted_grid).set_index("landmark")
    row, col = eda.cell_of(*[(m.lat, m.lon) for m in eda.LANDMARKS if m.name == "Duomo"][0])
    assert check.loc["Duomo", "south square"] == rowcol_to_square(row, col)
    assert check.loc["Duomo", "north square"] == rowcol_to_square(GRID_SIDE - 1 - row, col)


def test_landmarks_without_an_expectation_get_no_verdict(planted_grid):
    check = eda.orientation_check(planted_grid).set_index("landmark")
    assert check.loc["Duomo", "south verdict"] == "-"
    assert check.loc["Duomo", "north verdict"] == "-"


def test_row_band_mass_shares_sum_to_one_hundred(planted_grid):
    bands = eda.row_band_mass(planted_grid)
    assert len(bands) == GRID_SIDE // 10
    assert bands["share of traffic (%)"].sum() == pytest.approx(100.0)


def test_row_band_mass_rejects_a_band_that_does_not_divide_the_grid():
    with pytest.raises(ValueError, match="do not divide"):
        eda.row_band_mass(np.ones((10, 10)), band=3)


def test_mass_weighted_mean_row_sits_where_the_mass_is(planted_grid):
    assert eda.mass_weighted_mean_row(planted_grid) > (GRID_SIDE - 1) / 2
    flat = np.ones((GRID_SIDE, GRID_SIDE))
    assert eda.mass_weighted_mean_row(flat) == pytest.approx((GRID_SIDE - 1) / 2)


# ------------------------------------------------------------------ study cells


@pytest.fixture
def fake_totals() -> pd.DataFrame:
    """Totals for the whole grid: cell k carries ``k + 1`` units, so every rank is known."""
    ids = np.arange(1, GRID_SIDE * GRID_SIDE + 1)
    rows, cols = np.divmod(ids - 1, GRID_SIDE)
    return pd.DataFrame(
        {"square_id": ids, "row": rows, "col": cols, "internet": ids.astype(np.float64)}
    )


def test_study_cell_table_carries_position_distance_and_total(fake_totals):
    table = eda.study_cell_table([5161, 5059], fake_totals).set_index("square")
    assert (table.loc[5161, "row"], table.loc[5161, "col"]) == (51, 60)
    assert table.loc[5161, "total internet"] == pytest.approx(5161.0)
    # 5161 is north and east of the Duomo reference point, so the bearing is in quadrant 1.
    assert 0.0 < table.loc[5161, "bearing (deg)"] < 90.0
    assert table.loc[5161, "compass"] == "ENE"


def test_pairwise_distances_are_symmetric_in_cells_and_positive_in_metres():
    pairs = eda.pairwise_cell_distances([5161, 5059, 5259]).set_index("pair")
    assert len(pairs) == 3
    assert set(pairs["Chebyshev (cells)"]) == {2}
    assert pairs["centre to centre (m)"].min() > 0.0
    # The two pairs with the same row/col offset agree to within the change in the
    # longitude scale over two grid rows, which is centimetres at this latitude.
    assert pairs.loc["5161 / 5059", "centre to centre (m)"] == pytest.approx(
        pairs.loc["5161 / 5259", "centre to centre (m)"], abs=0.1
    )


def test_pairwise_distance_matches_the_cell_size_for_one_step_apart():
    a, b = rowcol_to_square(50, 50), rowcol_to_square(50, 51)
    step = eda.pairwise_cell_distances([a, b])["centre to centre (m)"].iloc[0]
    assert 230.0 < step < 240.0


# -------------------------------------------------------------- concentration


def test_gini_is_zero_for_a_flat_vector_and_near_one_for_a_single_holder():
    assert eda.gini(np.ones(500)) == pytest.approx(0.0, abs=1e-12)
    spike = np.zeros(500)
    spike[0] = 1.0
    assert eda.gini(spike) == pytest.approx(1.0 - 1.0 / 500)


def test_gini_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        eda.gini(np.array([-1.0, 2.0]))


def test_skewness_matches_a_hand_computed_moment_ratio():
    v = np.array([1.0, 2.0, 3.0, 10.0])
    d = v - v.mean()
    assert eda.skewness(v) == pytest.approx((d**3).mean() / ((d**2).mean() ** 1.5))
    assert eda.skewness(np.array([-2.0, -1.0, 0.0, 1.0, 2.0])) == pytest.approx(0.0)


def test_skewness_is_undefined_for_a_constant():
    with pytest.raises(ValueError, match="constant"):
        eda.skewness(np.full(10, 3.0))


def test_cells_to_share_counts_the_largest_first():
    # Ten cells: one of 50 and nine of 50/9, so the single cell is exactly half.
    v = np.array([50.0] + [50.0 / 9] * 9)
    assert eda.cells_to_share(v, 0.5) == 1
    assert eda.cells_to_share(np.ones(100), 0.5) == 50


def test_top_share_of_a_flat_vector_is_the_fraction_itself():
    assert eda.top_share(np.ones(1000), 0.01) == pytest.approx(1.0)
    assert eda.top_share(np.ones(1000), 0.10) == pytest.approx(10.0)


def test_concentration_stats_report_the_decade_spread(fake_totals):
    stats = eda.concentration_stats(fake_totals["internet"].to_numpy())
    assert stats["n_cells"] == GRID_SIDE * GRID_SIDE
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(10_000.0)
    assert stats["decades_below_median"] == pytest.approx(np.log10(stats["median"]))
    assert stats["decades_above_median"] == pytest.approx(np.log10(10_000.0 / stats["median"]))
    assert 0.0 < stats["gini"] < 1.0


def test_rank_table_ranks_descending_by_total(fake_totals):
    ranks = eda.rank_table([10_000, 1], fake_totals).set_index("square")
    assert ranks.loc[10_000, "rank of 10 000"] == 1
    assert ranks.loc[1, "rank of 10 000"] == GRID_SIDE * GRID_SIDE
    median = float(np.median(fake_totals["internet"]))
    assert ranks.loc[10_000, "multiple of the median cell"] == pytest.approx(10_000.0 / median)


# ---------------------------------------------------------- regime diagnostics


def _two_regime_series(flip: bool) -> tuple[np.ndarray, np.ndarray]:
    """Fourteen days: a day-peaking shape on working days and, if ``flip``, a night-peaking
    shape on the rest. Working days are the first five of each seven."""
    slots = np.arange(PERIOD)
    day_shape = 10.0 + 8.0 * np.sin(2 * np.pi * (slots - 6) / PERIOD)
    night_shape = 10.0 - 8.0 * np.sin(2 * np.pi * (slots - 6) / PERIOD)
    working = np.array([(d % 7) < 5 for d in range(N_DAYS)])
    days = np.stack([day_shape if (w or not flip) else night_shape for w in working])
    return days.ravel(), working


def test_daily_profiles_remove_the_level_but_keep_the_shape():
    x, _ = _two_regime_series(flip=False)
    profiles = eda.daily_profiles(x * np.repeat(np.arange(1, N_DAYS + 1), PERIOD), period=PERIOD)
    assert profiles.shape == (N_DAYS, PERIOD)
    # Every day scaled differently, yet every profile is identical.
    np.testing.assert_allclose(profiles, np.tile(profiles[0], (N_DAYS, 1)), rtol=1e-12)
    assert profiles.mean(axis=1) == pytest.approx(np.ones(N_DAYS))


def test_daily_profiles_reject_a_ragged_series_and_an_empty_day():
    with pytest.raises(ValueError, match="whole number of periods"):
        eda.daily_profiles(np.ones(25), period=PERIOD)
    ragged = np.ones(2 * PERIOD)
    ragged[PERIOD:] = 0.0
    with pytest.raises(ValueError, match="zero mean"):
        eda.daily_profiles(ragged, period=PERIOD)


def test_one_repeated_shape_scores_a_stable_regime():
    x, working = _two_regime_series(flip=False)
    diag = eda.regime_diagnostics(x, working, period=PERIOD)
    assert diag["day_to_day_profile_correlation"] == pytest.approx(1.0)
    assert diag["variance_outside_mean_profile"] == pytest.approx(0.0, abs=1e-20)
    assert diag["working_vs_non_working_correlation"] == pytest.approx(1.0)


def test_an_inverted_weekend_shape_is_the_5259_signature():
    x, working = _two_regime_series(flip=True)
    diag = eda.regime_diagnostics(x, working, period=PERIOD)
    # Mirrored shapes correlate at -1, which is the sign the report leans on.
    assert diag["working_vs_non_working_correlation"] == pytest.approx(-1.0)
    # Most of the profile variance now sits outside a single mean profile.
    assert diag["variance_outside_mean_profile"] > 0.5
    # Consecutive days mostly repeat, but the two regime boundaries per week pull it down.
    assert 0.0 < diag["day_to_day_profile_correlation"] < 1.0


def test_working_profile_correlation_needs_both_kinds_of_day():
    profiles = np.ones((4, PERIOD))
    with pytest.raises(ValueError, match="both working and non-working"):
        eda.working_profile_correlation(profiles, np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="one entry per day"):
        eda.working_profile_correlation(profiles, np.array([True, False]))


def test_consecutive_profile_correlation_needs_two_days():
    with pytest.raises(ValueError, match="at least two days"):
        eda.consecutive_profile_correlation(np.ones((1, PERIOD)))


def test_variance_outside_mean_profile_is_undefined_for_constant_profiles():
    with pytest.raises(ValueError, match="constant"):
        eda.variance_outside_mean_profile(np.ones((5, PERIOD)))


def test_weekday_weekend_ratio_is_the_ratio_of_the_two_means():
    weekend = np.array([False] * 10 + [True] * 10)
    x = np.array([3.0] * 10 + [1.5] * 10)
    assert eda.weekday_weekend_ratio(x, weekend) == pytest.approx(2.0)


def test_weekday_weekend_ratio_rejects_a_mask_that_covers_everything():
    with pytest.raises(ValueError, match="both weekday and weekend"):
        eda.weekday_weekend_ratio(np.ones(10), np.ones(10, dtype=bool))
    with pytest.raises(ValueError, match="one entry per slot"):
        eda.weekday_weekend_ratio(np.ones(10), np.ones(5, dtype=bool))


def test_pairwise_correlation_covers_every_unordered_pair():
    base = np.sin(np.arange(200) / 7.0)
    series = {1: base, 2: base.copy(), 3: -base}
    table = eda.pairwise_correlation(series).set_index("pair")
    assert list(table.index) == ["1 / 2", "1 / 3", "2 / 3"]
    assert table.loc["1 / 2", "Pearson r"] == pytest.approx(1.0)
    assert table.loc["1 / 3", "Pearson r"] == pytest.approx(-1.0)


# ------------------------------------------------------------ remainder, drops


def test_remainder_acf_of_white_noise_is_near_zero_and_lag_zero_is_excluded():
    noise = np.random.default_rng(0).normal(size=4000)
    acf = eda.remainder_acf(noise, lags=(1, 10, 100))
    assert set(acf) == {1, 10, 100}
    assert max(abs(v) for v in acf.values()) < 0.1


def test_remainder_acf_finds_a_planted_ar1_coefficient():
    rng = np.random.default_rng(1)
    x = np.zeros(20_000)
    for i in range(1, x.size):
        x[i] = 0.8 * x[i - 1] + rng.normal()
    assert eda.remainder_acf(x, lags=(1,))[1] == pytest.approx(0.8, abs=0.05)


def test_drop_percent_is_positive_for_a_fall_and_negative_for_a_rise():
    assert eda.drop_percent(50.0, 200.0) == pytest.approx(75.0)
    assert eda.drop_percent(400.0, 200.0) == pytest.approx(-100.0)
    with pytest.raises(ValueError, match="zero baseline"):
        eda.drop_percent(1.0, 0.0)


# ------------------------------------------------------------------ day blocks


@pytest.fixture
def toy_matrix() -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Three days of a three-cell grid at a period of 24, with a known total per day."""
    index = pd.date_range(
        "2013-12-10 00:00", periods=3 * PERIOD, freq="h", tz="Europe/Rome", name="time"
    )
    X = np.zeros((3 * PERIOD, 3), dtype=np.float32)
    for day in range(3):
        X[day * PERIOD : (day + 1) * PERIOD, :] = float(day + 1)
    return X, index


def test_day_rows_finds_the_block_that_starts_at_midnight(toy_matrix):
    _, index = toy_matrix
    assert eda.day_rows(index, "2013-12-11", period=PERIOD) == slice(PERIOD, 2 * PERIOD)
    with pytest.raises(KeyError, match="not a row"):
        eda.day_rows(index, "2013-12-20", period=PERIOD)


def test_day_block_total_sums_whole_days_for_a_column_and_for_the_grid(toy_matrix):
    X, index = toy_matrix
    assert eda.day_block_total(X, index, ["2013-12-11"], column=0, period=PERIOD) == pytest.approx(
        2.0 * PERIOD
    )
    assert eda.day_block_total(X, index, ["2013-12-11"], period=PERIOD) == pytest.approx(
        3 * 2.0 * PERIOD
    )
    assert eda.day_block_total(
        X, index, ["2013-12-10", "2013-12-12"], column=1, period=PERIOD
    ) == pytest.approx((1.0 + 3.0) * PERIOD)


# ---------------------------------------------------------------- zero scanning


def test_zero_scan_counts_zeros_and_finds_each_cell_s_last_record():
    X = np.ones((10 * PERIOD, 4), dtype=np.float32)
    X[5, 0] = 0.0  # one isolated gap
    X[3 * PERIOD :, 1] = 0.0  # cell 1 stops after three days
    scan = eda.zero_scan(X, chunk=PERIOD)
    assert scan.n_zero == 1 + 7 * PERIOD
    assert scan.n_pairs == 10 * PERIOD * 4
    assert scan.percent == pytest.approx(100.0 * scan.n_zero / scan.n_pairs)
    assert scan.cells_with_zero == 2
    assert scan.last_nonzero_row.tolist() == [
        10 * PERIOD - 1,
        3 * PERIOD - 1,
        10 * PERIOD - 1,
        10 * PERIOD - 1,
    ]


def test_zero_scan_is_independent_of_the_chunk_size():
    rng = np.random.default_rng(3)
    X = (rng.random((5 * PERIOD, 6)) > 0.2).astype(np.float32)
    a = eda.zero_scan(X, chunk=PERIOD)
    b = eda.zero_scan(X, chunk=7)
    assert (a.n_zero, a.cells_with_zero) == (b.n_zero, b.cells_with_zero)
    assert a.last_nonzero_row.tolist() == b.last_nonzero_row.tolist()


def test_zero_scan_marks_an_all_zero_cell_as_never_recording():
    X = np.ones((2 * PERIOD, 2), dtype=np.float32)
    X[:, 1] = 0.0
    scan = eda.zero_scan(X, chunk=PERIOD)
    assert scan.last_nonzero_row[1] == -1


def test_stalled_cells_lists_only_the_cells_that_stop_early():
    X = np.ones((10 * PERIOD, 3), dtype=np.float32)
    X[5, 0] = 0.0
    X[4 * PERIOD :, 2] = 0.0
    index = pd.date_range(
        "2013-11-01 00:00", periods=10 * PERIOD, freq="h", tz="Europe/Rome", name="time"
    )
    scan = eda.zero_scan(X, chunk=PERIOD)
    stalled = eda.stalled_cells(scan, np.array([11, 22, 33]), index, quiet_rows=2 * PERIOD)
    assert stalled["square"].tolist() == [33]
    assert stalled["last non-zero slot"].iloc[0] == index[4 * PERIOD - 1].strftime("%Y-%m-%d %H:%M")


def test_stalled_cells_is_empty_when_every_cell_reports_to_the_end():
    X = np.ones((5 * PERIOD, 3), dtype=np.float32)
    index = pd.date_range(
        "2013-11-01 00:00", periods=5 * PERIOD, freq="h", tz="Europe/Rome", name="time"
    )
    scan = eda.zero_scan(X, chunk=PERIOD)
    assert len(eda.stalled_cells(scan, np.array([1, 2, 3]), index, quiet_rows=PERIOD)) == 0


# -------------------------------------------------------------------- constants


def test_the_fortnight_slice_is_the_first_two_weeks_of_ten_minute_slots():
    assert slice(0, 2016) == eda.FORTNIGHT_ROWS


def test_every_landmark_falls_inside_the_grid_under_both_conventions():
    for mark in eda.LANDMARKS:
        for origin in ("south", "north"):
            row, col = eda.cell_of(mark.lat, mark.lon, origin=origin)
            assert 0 <= row < GRID_SIDE and 0 <= col < GRID_SIDE


def test_calendar_signal_r2_isolates_what_the_working_day_flag_adds():
    """The gain is the variance the flag explains that time of day alone cannot.

    Built so the answer is known: a pure time-of-day wave gets a gain of zero, and the same
    wave with a working-day offset on top gets a gain equal to that offset's share.
    """
    period, days = 8, 12
    slot = np.tile(np.arange(period), days)
    wave = np.sin(2 * np.pi * slot / period)

    flat = eda.calendar_signal_r2(wave, np.ones(days), period=period)
    assert flat["time_of_day_r2"] == pytest.approx(1.0, abs=1e-9)
    assert flat["working_day_gain"] == pytest.approx(0.0, abs=1e-9)

    working = np.array([1.0, 1.0, 0.0] * (days // 3))
    shifted = wave + 5.0 * np.repeat(working, period)
    d = eda.calendar_signal_r2(shifted, working, period=period)
    assert d["time_of_day_r2"] < 0.2  # time of day alone cannot see the level shift
    assert d["with_working_day_r2"] == pytest.approx(1.0, abs=1e-9)
    assert d["working_day_gain"] > 0.8


def test_calendar_signal_r2_takes_a_per_slot_flag_too():
    """The script passes one flag per day; other callers hold one per slot."""
    period, days = 8, 9
    rng = np.random.default_rng(0)
    x = rng.normal(size=period * days)
    per_day = np.array([1.0, 0.0, 1.0] * (days // 3))
    assert eda.calendar_signal_r2(x, per_day, period=period) == pytest.approx(
        eda.calendar_signal_r2(x, np.repeat(per_day, period), period=period)
    )


def test_calendar_signal_r2_rejects_a_flag_of_the_wrong_length():
    with pytest.raises(ValueError, match="working-day flag"):
        eda.calendar_signal_r2(np.zeros(16), np.ones(5), period=8)
