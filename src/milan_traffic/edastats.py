"""Descriptive statistics quoted in Section IV, and the regime diagnostics of Section VI.

Why this module exists. Section VIII of the report claims that every figure and every
table cell can be regenerated from a clean clone. That claim was false for one block of
numbers: the spatial concentration figures, the cell coordinates, the regime contrasts
between the three evaluation areas and the holiday and zero counts were all computed
interactively while the exploratory notebook was open, and never committed. A number
whose only provenance is a dead kernel cannot be checked, by a marker or by the author,
and a report that says otherwise is making a claim it cannot support. Everything Section
IV asserts descriptively is therefore computed here, once, with tests, and written to
``reports/tables/eda_stats.md`` by ``scripts/make_eda_stats.py``.

The load-bearing part is the regime block. Section VI names the dilated convolutional
network as the best architecture on the grounds that its only statistically resolved win
sits on square 5259, the cell whose daily shape *changes* between working and
non-working days rather than merely shrinking. That argument rests on four numbers per
cell (:func:`regime_diagnostics`), and if they are wrong the conclusion is wrong. They
are computed on the training slice alone, for the same reason every other diagnostic in
this project is: choosing or justifying a model from the evaluation week is a leak.

Orientation is verified, not assumed. The Milano Grid GeoJSON sits behind a Dataverse
guestbook form and is not a dependency of this repository, so the published corner
coordinates are all this code has, and they do not by themselves say which corner holds
square 1. :func:`orientation_check` settles it from the traffic field instead: it maps
landmarks with known coordinates onto cells under both candidate conventions and asks
which reading puts airports, stations and dense suburbs on busy cells and farmland and
open water on empty ones. That is a measurement, and it is reported with its failures.

Nothing here reads the raw ``.txt`` files. Inputs are arrays: a totals vector, a grid of
totals, a single cell's column of the memmap, or the memmap itself for the scans that
have to touch every cell, which stream one day of rows at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf as sm_acf

from .config import GRID_SIDE, SLOTS_PER_DAY, SLOTS_PER_WEEK, rowcol_to_square, square_to_rowcol

# ------------------------------------------------------------------- grid geometry

#: Bounding box of the published Milano Grid, in degrees (Barlacchi et al. 2015). The
#: lattice is GRID_SIDE x GRID_SIDE cells of equal angular size inside this box.
GRID_LON_MIN = 9.011533
GRID_LON_MAX = 9.312688
GRID_LAT_MIN = 45.356261
GRID_LAT_MAX = 45.568161

#: Angular size of one cell. About 235 m each way at this latitude.
CELL_LON = (GRID_LON_MAX - GRID_LON_MIN) / GRID_SIDE
CELL_LAT = (GRID_LAT_MAX - GRID_LAT_MIN) / GRID_SIDE

#: IUGG mean Earth radius, the sphere every distance below is measured on.
EARTH_RADIUS_M = 6_371_008.8

#: Reference point for the bearings quoted in Section IV: Piazza del Duomo.
DUOMO_LAT = 45.4642
DUOMO_LON = 9.1900

#: Which corner holds square 1. ``south`` is the published convention (square 1 in the
#: south-west corner, ids running east before north); ``north`` is its vertical mirror,
#: the reading a row-major image viewer produces if nobody checks.
Origin = Literal["south", "north"]

#: The 16-point compass, clockwise from north.
COMPASS_POINTS: tuple[str, ...] = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)

#: Rows of the first fortnight, 2013-11-01 to 2013-11-14, the window the brief fixes for
#: the exploratory figure and therefore the window its weekday/weekend contrasts use.
FORTNIGHT_ROWS = slice(0, 14 * SLOTS_PER_DAY)


@dataclass(frozen=True)
class Landmark:
    """A place whose coordinates are known independently of this dataset.

    ``expect`` is what the traffic field should say about the cell if the orientation is
    read correctly: ``max`` for an isolated generator of demand (a terminal, a station),
    ``min`` for farmland or open water, ``None`` where no honest prediction can be made.
    """

    name: str
    lat: float
    lon: float
    expect: Literal["max", "min"] | None = None
    note: str = ""


#: The seven probes. Chosen so that four sit far from the grid's horizontal mid-line,
#: which is the only place a vertical reflection changes anything: a cell near row 50
#: maps onto its own neighbourhood under either convention and cannot discriminate.
LANDMARKS: tuple[Landmark, ...] = (
    Landmark("Linate airport", 45.4451, 9.2767, "max", "isolated terminal"),
    Landmark("San Siro stadium", 45.4781, 9.1240, "max", "isolated venue"),
    Landmark("Milano Centrale", 45.4869, 9.2050, "max", "main railway station"),
    Landmark("Duomo", 45.4642, 9.1900, None, "city centre, near the mirror line"),
    Landmark("Idroscalo lake", 45.4600, 9.2900, "min", "open water"),
    Landmark("Parco Sud farmland", 45.3900, 9.1900, "min", "agricultural belt"),
    Landmark("Sesto San Giovanni", 45.5300, 9.2400, "max", "dense northern suburb"),
)


def centroid(row: int, col: int, *, origin: Origin = "south") -> tuple[float, float]:
    """Latitude and longitude of the centre of grid cell ``(row, col)``.

    ``row`` is the index used by ``square_totals.csv`` and by :func:`dataio.to_grid`,
    that is ``(square_id - 1) // 100``. Under the published ``south`` convention row 0 is
    the southernmost band; under ``north`` it is the northernmost.
    """
    if origin == "south":
        lat = GRID_LAT_MIN + (row + 0.5) * CELL_LAT
    elif origin == "north":
        lat = GRID_LAT_MAX - (row + 0.5) * CELL_LAT
    else:
        raise ValueError(f"unknown origin {origin!r}; expected 'south' or 'north'")
    return lat, GRID_LON_MIN + (col + 0.5) * CELL_LON


def square_centroid(square_id: int, *, origin: Origin = "south") -> tuple[float, float]:
    """Centroid of a square id, via its ``(row, col)`` position."""
    return centroid(*square_to_rowcol(square_id), origin=origin)


def cell_of(lat: float, lon: float, *, origin: Origin = "south") -> tuple[int, int]:
    """The ``(row, col)`` whose cell contains a point. Raises outside the bounding box."""
    col = int((lon - GRID_LON_MIN) // CELL_LON)
    if origin == "south":
        row = int((lat - GRID_LAT_MIN) // CELL_LAT)
    elif origin == "north":
        row = int((GRID_LAT_MAX - lat) // CELL_LAT)
    else:
        raise ValueError(f"unknown origin {origin!r}; expected 'south' or 'north'")
    if not (0 <= row < GRID_SIDE and 0 <= col < GRID_SIDE):
        raise ValueError(f"({lat}, {lon}) falls outside the Milano Grid bounding box")
    return row, col


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres on a sphere of radius :data:`EARTH_RADIUS_M`."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return float(2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a)))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    y = np.sin(dlam) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlam)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def compass_point(bearing: float) -> str:
    """Nearest of the 16 compass points to a bearing in degrees."""
    return COMPASS_POINTS[int(((bearing % 360.0) + 11.25) % 360.0 // 22.5)]


# --------------------------------------------------------- orientation from traffic


def neighbourhood_median(grid: np.ndarray, row: int, col: int, *, k: int = 2) -> float:
    """Median of the ``(2k+1) x (2k+1)`` block centred on ``(row, col)``, clipped at edges.

    The centre cell is part of its own neighbourhood, so a cell that is exactly typical of
    its surroundings scores a ratio of 1.0 rather than something biased away from it.
    """
    g = np.asarray(grid, dtype=np.float64)
    r0, r1 = max(0, row - k), min(g.shape[0], row + k + 1)
    c0, c1 = max(0, col - k), min(g.shape[1], col + k + 1)
    return float(np.median(g[r0:r1, c0:c1]))


def orientation_check(
    grid: np.ndarray,
    landmarks: Sequence[Landmark] = LANDMARKS,
    *,
    k: int = 2,
) -> pd.DataFrame:
    """Score each landmark's cell under both candidate orientations.

    One row per landmark, carrying for each convention the square id the landmark falls
    in, that cell's total, the median of its ``(2k+1)`` square neighbourhood and the
    ratio between them. A ratio above 1 means the cell stands out against its
    surroundings, which is what an airport or a station should do and what farmland
    should not. ``verdict`` columns say whether the ratio agrees with ``expect``.
    """
    g = np.asarray(grid, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for mark in landmarks:
        record: dict[str, Any] = {"landmark": mark.name, "expected": mark.expect or "-"}
        for origin in ("south", "north"):
            row, col = cell_of(mark.lat, mark.lon, origin=origin)
            total = float(g[row, col])
            median = neighbourhood_median(g, row, col, k=k)
            ratio = total / median if median else np.nan
            record[f"{origin} square"] = rowcol_to_square(row, col)
            record[f"{origin} total"] = total
            record[f"{origin} ratio"] = ratio
            record[f"{origin} verdict"] = _verdict(ratio, mark.expect)
        rows.append(record)
    return pd.DataFrame(rows)


def _verdict(ratio: float, expect: str | None) -> str:
    if expect is None or not np.isfinite(ratio):
        return "-"
    if expect == "max":
        return "pass" if ratio > 1.0 else "FAIL"
    return "pass" if ratio < 1.0 else "FAIL"


def best_neighbour_ratio(
    grid: np.ndarray,
    row: int,
    col: int,
    *,
    mode: Literal["max", "min"] = "max",
    search: int = 1,
    k: int = 2,
) -> tuple[int, int, float]:
    """The most extreme cell within ``search`` of ``(row, col)``, by ratio to its own block.

    Exists so that a landmark which fails :func:`orientation_check` can be explained
    rather than merely recorded. A landmark is a point, a cell is 235 m across and a
    stadium or a terminal is neither, so a venue lying astride a boundary can put its
    quiet half under the coordinates and its busy half one cell away. ``mode`` picks which
    extreme is the interesting one: the busiest neighbour for a landmark that should be a
    local maximum, the quietest for one that should be a minimum. Returns the offset and
    the winning ratio, so the caller can say which way and by how much.
    """
    g = np.asarray(grid, dtype=np.float64)
    if mode not in ("max", "min"):
        raise ValueError(f"unknown mode {mode!r}; expected 'max' or 'min'")
    sign = 1.0 if mode == "max" else -1.0
    best = (0, 0, -np.inf * sign)
    for drow in range(-search, search + 1):
        for dcol in range(-search, search + 1):
            r, c = row + drow, col + dcol
            if not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]):
                continue
            median = neighbourhood_median(g, r, c, k=k)
            if not median:
                continue
            ratio = float(g[r, c] / median)
            if sign * ratio > sign * best[2]:
                best = (drow, dcol, ratio)
    return best


def offset_compass(row: int, col: int, drow: int, dcol: int) -> str:
    """Compass direction of a one-or-two-cell step, under the south-origin convention."""
    lat, lon = centroid(row, col)
    lat2, lon2 = centroid(row + drow, col + dcol)
    return compass_point(initial_bearing_deg(lat, lon, lat2, lon2))


def orientation_scores(check: pd.DataFrame) -> pd.DataFrame:
    """Passes and failures per convention, from an :func:`orientation_check` table."""
    rows = []
    for origin in ("south", "north"):
        col = check[f"{origin} verdict"]
        rows.append(
            {
                "convention": origin,
                "landmarks with an expectation": int((col != "-").sum()),
                "pass": int((col == "pass").sum()),
                "FAIL": int((col == "FAIL").sum()),
            }
        )
    return pd.DataFrame(rows)


def row_band_mass(grid: np.ndarray, *, band: int = 10) -> pd.DataFrame:
    """Share of all traffic in each horizontal band of ``band`` grid rows.

    Reported because it is the one orientation statistic that does not depend on a single
    cell being correctly located: Milan's built-up area extends well north of the Duomo
    and gives way to the agricultural Parco Sud in the south, so the mass has to sit
    above the mid-line under the correct reading and below it under the mirror.
    """
    g = np.asarray(grid, dtype=np.float64)
    if g.shape[0] % band:
        raise ValueError(f"{g.shape[0]} rows do not divide into bands of {band}")
    per_band = g.reshape(g.shape[0] // band, band, g.shape[1]).sum(axis=(1, 2))
    total = float(g.sum())
    starts = np.arange(0, g.shape[0], band)
    return pd.DataFrame(
        {
            "grid rows": [f"{s}-{s + band - 1}" for s in starts],
            "share of traffic (%)": 100.0 * per_band / total,
            "south-origin latitude band": [
                f"{GRID_LAT_MIN + s * CELL_LAT:.4f} to {GRID_LAT_MIN + (s + band) * CELL_LAT:.4f}"
                for s in starts
            ],
        }
    )


def mass_weighted_mean_row(grid: np.ndarray) -> float:
    """Traffic-weighted centroid of the grid, as a fractional row index."""
    g = np.asarray(grid, dtype=np.float64)
    mass = g.sum(axis=1)
    return float((np.arange(g.shape[0]) * mass).sum() / mass.sum())


# ------------------------------------------------------------------- study cells


def study_cell_table(
    squares: Sequence[int],
    totals: pd.DataFrame,
    *,
    origin: Origin = "south",
    activity: str = "internet",
) -> pd.DataFrame:
    """Grid position, centroid, offset from the Duomo and total, one row per study cell."""
    lookup = totals.set_index("square_id")[activity]
    rows = []
    for square in squares:
        row, col = square_to_rowcol(square)
        lat, lon = centroid(row, col, origin=origin)
        bearing = initial_bearing_deg(DUOMO_LAT, DUOMO_LON, lat, lon)
        rows.append(
            {
                "square": square,
                "row": row,
                "col": col,
                "latitude": lat,
                "longitude": lon,
                "metres from Duomo": haversine_m(DUOMO_LAT, DUOMO_LON, lat, lon),
                "bearing (deg)": bearing,
                "compass": compass_point(bearing),
                f"total {activity}": float(lookup.loc[square]),
            }
        )
    return pd.DataFrame(rows)


def pairwise_cell_distances(squares: Sequence[int], *, origin: Origin = "south") -> pd.DataFrame:
    """Chebyshev separation in cells and centre-to-centre distance for every pair."""
    rows = []
    for i, a in enumerate(squares):
        for b in squares[i + 1 :]:
            ra, ca = square_to_rowcol(a)
            rb, cb = square_to_rowcol(b)
            lat_a, lon_a = centroid(ra, ca, origin=origin)
            lat_b, lon_b = centroid(rb, cb, origin=origin)
            rows.append(
                {
                    "pair": f"{a} / {b}",
                    "rows apart": abs(ra - rb),
                    "cols apart": abs(ca - cb),
                    "Chebyshev (cells)": max(abs(ra - rb), abs(ca - cb)),
                    "centre to centre (m)": haversine_m(lat_a, lon_a, lat_b, lon_b),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------ spatial concentration


def gini(values: Any) -> float:
    """Gini coefficient of a non-negative vector, by the sorted-rank formula."""
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if v.size == 0 or v.min() < 0:
        raise ValueError("Gini needs a non-empty, non-negative vector")
    n = v.size
    ranks = np.arange(1, n + 1)
    return float((2.0 * (ranks * v).sum()) / (n * v.sum()) - (n + 1.0) / n)


def skewness(values: Any) -> float:
    """Fisher-Pearson moment coefficient ``m3 / m2**1.5``, the uncorrected g1."""
    v = np.asarray(values, dtype=np.float64).ravel()
    d = v - v.mean()
    m2 = float((d**2).mean())
    if m2 == 0.0:
        raise ValueError("skewness is undefined for a constant vector")
    return float((d**3).mean() / m2**1.5)


def cells_to_share(values: Any, share: float = 0.5) -> int:
    """How many of the largest cells are needed to reach ``share`` of the total."""
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())[::-1]
    return int(np.searchsorted(np.cumsum(v), share * v.sum()) + 1)


def top_share(values: Any, fraction: float = 0.01) -> float:
    """Share of the total carried by the busiest ``fraction`` of cells, as a percentage."""
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())[::-1]
    k = max(1, int(round(v.size * fraction)))
    return float(100.0 * v[:k].sum() / v.sum())


def concentration_stats(values: Any) -> dict[str, float]:
    """The spatial-concentration block of Section IV, as one dictionary.

    ``decades_below_median`` and ``decades_above_median`` are the orders of magnitude
    separating the emptiest and busiest cells from the median one. They are reported as a
    pair because the asymmetry is the point: the left tail is roughly twice as long, so
    the concentration is a log-normal spread rather than a separate class of hot cells.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    median = float(np.median(v))
    return {
        "n_cells": float(v.size),
        "min": float(v.min()),
        "max": float(v.max()),
        "median": median,
        "total": float(v.sum()),
        "top_1_percent_share": top_share(v, 0.01),
        "gini": gini(v),
        "cells_for_half": float(cells_to_share(v, 0.5)),
        "cells_for_half_percent": 100.0 * cells_to_share(v, 0.5) / v.size,
        "skewness_log10": skewness(np.log10(v)),
        "decades_below_median": float(np.log10(median / v.min())),
        "decades_above_median": float(np.log10(v.max() / median)),
    }


def rank_table(
    squares: Sequence[int], totals: pd.DataFrame, *, activity: str = "internet"
) -> pd.DataFrame:
    """Rank by total and multiple of the median cell, for named squares."""
    v = totals[activity].to_numpy(dtype=np.float64)
    median = float(np.median(v))
    ranked = totals.assign(_rank=totals[activity].rank(ascending=False, method="min"))
    lookup = ranked.set_index("square_id")
    rows = []
    for square in squares:
        total = float(lookup.loc[square, activity])
        rows.append(
            {
                "square": square,
                f"total {activity}": total,
                "rank of 10 000": int(lookup.loc[square, "_rank"]),
                "multiple of the median cell": total / median,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------- regime diagnostics


def daily_profiles(x: Any, *, period: int = SLOTS_PER_DAY) -> np.ndarray:
    """Reshape a series to ``(n_days, period)`` and divide each day by its own mean.

    Dividing out the daily level is what makes the result a *shape*. Two days that differ
    only in how busy they were have the same profile, so every statistic built on top of
    this answers "does the pattern change?" and never "is the cell busier?".
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size % period:
        raise ValueError(f"{v.size} values are not a whole number of periods of {period}")
    days = v.reshape(-1, period)
    means = days.mean(axis=1, keepdims=True)
    if np.any(means == 0.0):
        raise ValueError("a day with a zero mean has no profile")
    return days / means


def consecutive_profile_correlation(profiles: np.ndarray) -> float:
    """Mean Pearson correlation between each day's profile and the next day's."""
    p = np.asarray(profiles, dtype=np.float64)
    if p.shape[0] < 2:
        raise ValueError("need at least two days")
    return float(np.mean([np.corrcoef(p[i], p[i + 1])[0, 1] for i in range(p.shape[0] - 1)]))


def variance_outside_mean_profile(profiles: np.ndarray) -> float:
    """``var(profile - mean profile) / var(profile)``: how little one shape explains.

    A cell with one repeated daily shape leaves almost nothing outside it. A cell that
    alternates between two shapes leaves most of its profile variance outside, because
    the average of two different shapes resembles neither.
    """
    p = np.asarray(profiles, dtype=np.float64)
    total = float(p.var())
    if total == 0.0:
        raise ValueError("profiles are constant; the ratio is undefined")
    return float((p - p.mean(axis=0)).var() / total)


def working_profile_correlation(profiles: np.ndarray, working: Any) -> float:
    """Correlation between the mean working-day profile and the mean non-working one."""
    p = np.asarray(profiles, dtype=np.float64)
    mask = np.asarray(working, dtype=bool)
    if mask.shape != (p.shape[0],):
        raise ValueError("the working-day mask must have one entry per day")
    if not mask.any() or mask.all():
        raise ValueError("need both working and non-working days")
    return float(np.corrcoef(p[mask].mean(axis=0), p[~mask].mean(axis=0))[0, 1])


def weekday_weekend_ratio(x: Any, weekend: Any) -> float:
    """Mean over Monday-to-Friday slots divided by the mean over Saturday-Sunday slots.

    Calendar weekdays, not working days: this is the number the exploratory figure is
    read against, and keeping it on the plain weekday label is what makes square 5259's
    working-day regime a separate finding rather than a definition.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    mask = np.asarray(weekend, dtype=bool).ravel()
    if mask.shape != v.shape:
        raise ValueError("the weekend mask must have one entry per slot")
    if not mask.any() or mask.all():
        raise ValueError("need both weekday and weekend slots")
    return float(v[~mask].mean() / v[mask].mean())


def regime_diagnostics(
    x: Any,
    working: Any,
    *,
    period: int = SLOTS_PER_DAY,
) -> dict[str, float]:
    """Three of the four per-cell regime numbers Section VI's best-model argument rests on.

    The fourth, the weekday-to-weekend ratio, is computed by :func:`weekday_weekend_ratio`
    because it is read off a different slice: the first fortnight the brief fixes for the
    exploratory figure, not the whole training slice these three use.
    """
    profiles = daily_profiles(x, period=period)
    return {
        "day_to_day_profile_correlation": consecutive_profile_correlation(profiles),
        "variance_outside_mean_profile": variance_outside_mean_profile(profiles),
        "working_vs_non_working_correlation": working_profile_correlation(profiles, working),
    }


def calendar_signal_r2(x: Any, working: Any, *, period: int = SLOTS_PER_DAY) -> dict[str, float]:
    """How much the working-day flag explains beyond time of day, on one cell.

    Section VI.D's failure analysis turns on square 5059 having the weakest calendar signal
    of the three, and the weekday-to-weekend ratio does not show that: 1.29 on 5059 sits
    between 0.79 on 5161 and 2.98 on 5259, and 0.79 is a contrast of almost the same size in
    the other direction. The ratio is the wrong statistic because the models already receive
    144 lags, which carry the time of day; what the calendar block adds is the working-day
    distinction on top of it.

    So this fits two ordinary least-squares models on the training slice and returns the gap
    between them: first the series on 143 time-of-day dummies, then the same plus the
    working-day indicator and its interaction with time of day. The increment is the variance
    the flag explains that time of day alone cannot, which is exactly the quantity the
    argument needs. It is descriptive, not a forecast: nothing here is fitted to predict the
    next slot, and the evaluation week never enters.
    """
    y = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(working, dtype=np.float64).ravel()
    if y.size % period:
        raise ValueError(f"series length {y.size} is not a whole number of periods of {period}")
    n_days = y.size // period
    if w.size == n_days:  # one flag per day, as :func:`regime_diagnostics` takes it
        w = np.repeat(w, period)
    elif w.size != y.size:
        raise ValueError(
            f"working-day flag has {w.size} entries; expected {n_days} (per day) or {y.size} (per slot)"
        )
    slot = np.tile(np.arange(period), y.size // period)
    tod = np.eye(period)[slot][:, 1:]  # drop one level, the intercept carries it

    def _r2(design: np.ndarray) -> float:
        d = np.column_stack([np.ones(y.size), design])
        beta, *_ = np.linalg.lstsq(d, y, rcond=None)
        return float(1.0 - np.var(y - d @ beta) / np.var(y))

    base = _r2(tod)
    full = _r2(np.column_stack([tod, w, tod * w[:, None]]))
    return {
        "time_of_day_r2": base,
        "with_working_day_r2": full,
        "working_day_gain": full - base,
    }


def pairwise_correlation(series: dict[int, np.ndarray]) -> pd.DataFrame:
    """Pearson correlation for every unordered pair of the given cells' series."""
    keys = list(series)
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            va = np.asarray(series[a], dtype=np.float64).ravel()
            vb = np.asarray(series[b], dtype=np.float64).ravel()
            rows.append({"pair": f"{a} / {b}", "Pearson r": float(np.corrcoef(va, vb)[0, 1])})
    return pd.DataFrame(rows)


# --------------------------------------------------------- decomposition remainder


def remainder_acf(
    remainder: Any, lags: Sequence[int] = (1, SLOTS_PER_DAY, SLOTS_PER_WEEK)
) -> dict[int, float]:
    """Autocorrelation of a decomposition remainder at named lags.

    The point of quoting these is that "irregular" is not "unpredictable": whatever STL
    leaves over is still strongly correlated one step ahead, which is the persistence
    floor of Section VI seen from the decomposition side.
    """
    r = np.asarray(remainder, dtype=np.float64).ravel()
    values = sm_acf(r, nlags=int(max(lags)), fft=True)
    return {int(lag): float(values[int(lag)]) for lag in lags}


# -------------------------------------------------------- holidays, zeros, scanning


def drop_percent(current: float, baseline: float) -> float:
    """Percentage fall from ``baseline`` to ``current``. Positive means a fall."""
    if baseline == 0.0:
        raise ValueError("cannot express a drop against a zero baseline")
    return float(100.0 * (1.0 - current / baseline))


def day_rows(index: pd.DatetimeIndex, day: str, *, period: int = SLOTS_PER_DAY) -> slice:
    """The ``period`` matrix rows belonging to a calendar day, by lookup in the index."""
    start = pd.Timestamp(day)
    if index.tz is not None:
        start = start.tz_localize(index.tz) if start.tz is None else start.tz_convert(index.tz)
    i = int(index.searchsorted(start))
    if i >= len(index) or index[i] != start:
        raise KeyError(f"{day} 00:00 is not a row of this index")
    return slice(i, i + period)


def day_block_total(
    X: np.ndarray,
    index: pd.DatetimeIndex,
    days: Sequence[str],
    *,
    column: int | None = None,
    period: int = SLOTS_PER_DAY,
) -> float:
    """Sum of a set of whole days, for one column or for the whole grid.

    Reads one day of rows at a time so the grid-wide variant never holds more than
    ``period x n_squares`` float64 values, about 11 MB, resident.
    """
    total = 0.0
    for day in days:
        block = np.asarray(X[day_rows(index, day, period=period)], dtype=np.float64)
        total += float(block[:, column].sum() if column is not None else block.sum())
    return total


@dataclass(frozen=True)
class ZeroScan:
    """Result of :func:`zero_scan`: the counts, plus where each cell's record stops."""

    n_zero: int
    n_pairs: int
    cells_with_zero: int
    last_nonzero_row: np.ndarray

    @property
    def percent(self) -> float:
        return 100.0 * self.n_zero / self.n_pairs


def zero_scan(X: np.ndarray, *, chunk: int = SLOTS_PER_DAY) -> ZeroScan:
    """Count exact zeros over the whole matrix and find each cell's last non-zero slot.

    An empty field in the raw export means "no record", not "no traffic", and the
    ingestion stores it as zero. Counting them is how the report can say that the missing
    data is 0.17 percent of the grid and that none of it touches a study cell. The scan
    streams by day for the usual reason: the matrix is 353 MB and must not be resident.
    """
    n_rows, n_cols = X.shape
    last = np.full(n_cols, -1, dtype=np.int64)
    seen_zero = np.zeros(n_cols, dtype=bool)
    n_zero = 0
    for i in range(0, n_rows, chunk):
        block = np.asarray(X[i : i + chunk], dtype=np.float64)
        is_zero = block == 0.0
        n_zero += int(is_zero.sum())
        seen_zero |= is_zero.any(axis=0)
        nonzero = ~is_zero
        has_any = nonzero.any(axis=0)
        # argmax on the reversed block is the last True; only trusted where has_any.
        offset = block.shape[0] - 1 - np.argmax(nonzero[::-1], axis=0)
        last = np.where(has_any, i + offset, last)
    return ZeroScan(
        n_zero=n_zero,
        n_pairs=int(n_rows) * int(n_cols),
        cells_with_zero=int(seen_zero.sum()),
        last_nonzero_row=last,
    )


def stalled_cells(
    scan: ZeroScan,
    square_ids: Any,
    index: pd.DatetimeIndex,
    *,
    quiet_rows: int = SLOTS_PER_WEEK,
) -> pd.DataFrame:
    """Cells whose last non-zero slot is more than ``quiet_rows`` before the record ends.

    A cell with scattered night-time gaps is ordinary; a cell that stops and never
    restarts is a decommissioned or relocated sensor, and the two deserve to be counted
    separately before anyone calls either one "missing data".
    """
    ids = np.asarray(square_ids).ravel()
    last = scan.last_nonzero_row
    cutoff = len(index) - 1 - int(quiet_rows)
    hits = np.flatnonzero(last < cutoff)
    return pd.DataFrame(
        {
            "square": ids[hits].astype(int),
            "last non-zero slot": [
                index[int(last[j])].strftime("%Y-%m-%d %H:%M") if last[j] >= 0 else "never"
                for j in hits
            ],
            "matrix row": last[hits],
        }
    )
