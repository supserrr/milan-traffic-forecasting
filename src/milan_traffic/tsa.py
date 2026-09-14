"""Time-series diagnostics behind the model-design decisions (criterion 3 of the brief).

Why this module exists. The brief asks for evidence that the forecasting setup follows
from the data rather than from habit: which lags carry information (sequence length),
how much of the variance is seasonal against irregular (what any model can hope to
explain), whether the series is stationary (whether to difference or transform), and
whether the noise scales with the level (whether to model ``log1p``). Those questions
are answered here, once, with tests, so that notebooks, the experiment log and the
report all quote the same numbers.

Training slice only. Every statistic in this module is meant to be computed on the
training rows, :data:`TRAIN_ROWS` (0:5472, 2013-11-01 00:00 to 2013-12-08 23:50 CET).
An earlier version of this project computed the ACF, PACF and the decomposition on the
full 8928-slot series, which includes the validation week and the held-out evaluation
week. Using test-period observations to choose the sequence length is a leak of
test information into design decisions, and an obvious question in a viva. Two
consequences to keep in mind when comparing with the older numbers: the 95 percent band
is ``1.96 / sqrt(5472) = 0.0265`` rather than ``1.96 / sqrt(8928) = 0.0207``, and the
"last significant PACF lag" is not the same lag.

Two cutoffs, stated separately. The experiment log previously mixed a practical 0.05
magnitude cutoff with the +/-0.0207 statistical band without saying which one produced
which number. :func:`last_significant_lag` takes the band and :func:`practical_cutoff`
takes the threshold as explicit arguments, so the criterion is visible at every call
site. Both also take ``max_lag``: beyond a few hundred lags the number of band
exceedances is what white noise would produce by chance, so "the last lag above the
band" is a function of how far one searches. :func:`exceedance_summary` reports that
count against the chance rate so the reader can judge it.

Nothing here reads the raw data. Inputs are one-dimensional arrays (a memmap column
slice is fine) or, for the grid-wide scan, the ``(n_slots, n_squares)`` memmap itself.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import MSTL, STL, seasonal_decompose
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import pacf as sm_pacf

from .config import EVAL_WEEK_START, SLOTS_PER_DAY, SLOTS_PER_WEEK

#: Values are scaled CDR activity counts, not bytes. Every axis label uses this string.
UNIT = "scaled CDR activity"

#: Width of one text column of the two-column report, in inches. The figures here are
#: drawn at the width they are placed at, so the typesetter scales them 1:1 and the font
#: sizes below are the sizes on the page. ``viz.COLUMN_WIDTH_IN`` is the same number;
#: it is repeated rather than imported so this module keeps its lazy matplotlib import.
COLUMN_WIDTH_IN = 3.4

#: Two-sided 95 percent normal quantile, the constant behind the classical ACF band.
Z_95 = 1.96

#: First row of every matrix in the processed store.
SERIES_START = "2013-11-01 00:00"

# Row ranges of the chronological split. The evaluation week is fixed by the brief,
# validation is the week immediately before it, and training is everything earlier.
# Derived from the constants so they cannot drift from configs/data.yaml.
_EVAL_START_ROW = (pd.Timestamp(EVAL_WEEK_START) - pd.Timestamp(SERIES_START)).days * SLOTS_PER_DAY
EVAL_ROWS = slice(_EVAL_START_ROW, _EVAL_START_ROW + SLOTS_PER_WEEK)
VAL_ROWS = slice(_EVAL_START_ROW - SLOTS_PER_WEEK, _EVAL_START_ROW)
TRAIN_ROWS = slice(0, _EVAL_START_ROW - SLOTS_PER_WEEK)

PacfMethod = Literal["ldb", "ywm"]
DecomposeMethod = Literal["stl", "classical", "mstl"]

ADF_NULL = "unit root (the series is non-stationary)"
KPSS_NULL = "stationary around a constant level"


# ------------------------------------------------------------------ small utilities


def _as_1d(x: Any) -> np.ndarray:
    """Copy ``x`` into a finite float64 vector; a memmap column costs 36 KB, nothing more."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected a one-dimensional series, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("series contains NaN or inf; clean it before running diagnostics")
    return arr


def split_label(row: int) -> str:
    """Name the split a matrix row belongs to, for labelling tables that span splits."""
    if row < TRAIN_ROWS.stop:
        return "train"
    if row < VAL_ROWS.stop:
        return "val"
    if row < EVAL_ROWS.stop:
        return "eval"
    return "post-eval"


def grid_totals(
    X: np.ndarray, rows: slice = slice(None), *, chunk: int = SLOTS_PER_DAY
) -> np.ndarray:
    """Sum the ``(n_slots, n_squares)`` matrix over squares, one day of rows at a time.

    Exists so the grid-wide series can be built from the memmap without ever holding the
    357 MB matrix in RAM: each chunk is ``chunk x 10000`` float32, about 5.8 MB.
    """
    start, stop, step = rows.indices(X.shape[0])
    if step != 1:
        raise ValueError("rows must be a contiguous slice")
    out = np.empty(stop - start, dtype=np.float64)
    for i in range(start, stop, chunk):
        j = min(i + chunk, stop)
        out[i - start : j - start] = np.asarray(X[i:j], dtype=np.float64).sum(axis=1)
    return out


# --------------------------------------------------------------------- ACF and PACF


def confidence_band(n: int, *, z: float = Z_95) -> float:
    """Half-width of the two-sided 95 percent band for a sample (P)ACF: ``z / sqrt(n)``.

    This is the large-sample band for white noise. It is the *statistical* criterion;
    the practical 0.05 magnitude cutoff used elsewhere in the log is a different thing.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return float(z / np.sqrt(n))


def acf_pacf(x: Any, nlags: int, *, pacf_method: PacfMethod = "ldb") -> pd.DataFrame:
    """Sample ACF and PACF to ``nlags`` with the 95 percent band, one row per lag.

    The PACF uses statsmodels' Levinson-Durbin recursion on the biased ACF (``ldb``) by
    default, which is what the earlier log quoted; ``ywm`` (unbiased Yule-Walker) is
    offered for comparison and agrees to about 1e-13 on the training series. The band is
    ``1.96 / sqrt(n)`` for ``n`` the length of the series analysed, so it is only correct
    for the slice actually passed in.

    Returns columns ``lag``, ``acf``, ``pacf``, ``band``, ``acf_above_band`` and
    ``pacf_above_band``; ``n``, ``band`` and ``pacf_method`` are stored in ``.attrs``.
    """
    x = _as_1d(x)
    n = x.size
    if nlags < 1 or nlags >= n // 2:
        raise ValueError(f"nlags must be in [1, n // 2); got {nlags} for n={n}")
    a = sm_acf(x, nlags=nlags, fft=True)
    p = sm_pacf(x, nlags=nlags, method=pacf_method)
    band = confidence_band(n)
    table = pd.DataFrame(
        {
            "lag": np.arange(nlags + 1, dtype=int),
            "acf": a,
            "pacf": p,
            "band": band,
        }
    )
    table["acf_above_band"] = np.abs(table["acf"]) > band
    table["pacf_above_band"] = np.abs(table["pacf"]) > band
    table.attrs.update(n=n, band=band, pacf_method=pacf_method)
    return table


def _last_lag_above(values: Any, level: float, max_lag: int | None) -> int | None:
    v = np.abs(np.asarray(values, dtype=np.float64))
    hi = v.size - 1 if max_lag is None else min(int(max_lag), v.size - 1)
    if hi < 1:
        return None
    lags = np.arange(1, hi + 1)
    hits = lags[v[1 : hi + 1] > level]
    return int(hits.max()) if hits.size else None


def last_significant_lag(
    pacf_values: Any, band: float, *, max_lag: int | None = None
) -> int | None:
    """Largest lag in ``1..max_lag`` whose ``|pacf|`` exceeds the statistical ``band``.

    Lag 0 is skipped (it is 1 by definition). ``None`` means no lag qualifies. The band
    is an argument rather than recomputed so the caller cannot accidentally pair a PACF
    from one slice with the band of another.
    """
    return _last_lag_above(pacf_values, band, max_lag)


def practical_cutoff(
    pacf_values: Any, threshold: float, *, max_lag: int | None = None
) -> int | None:
    """Largest lag in ``1..max_lag`` whose ``|pacf|`` exceeds a *practical* ``threshold``.

    Same search as :func:`last_significant_lag` but the level is a magnitude judged to
    matter for forecasting (the log uses 0.05), not a significance band. Keeping the two
    functions separate is the point: a lag can be statistically significant and
    practically negligible, and the report should say which criterion gave which lag.
    """
    return _last_lag_above(pacf_values, threshold, max_lag)


def exceedance_summary(
    values: Any, level: float, *, lo: int = 1, hi: int | None = None, alpha: float = 0.05
) -> dict[str, float | int]:
    """Count lags in ``lo..hi`` with ``|value| > level`` and compare with the chance rate.

    Under the white-noise null about ``alpha`` of the lags exceed the ``alpha`` band by
    chance. When the observed fraction in a lag range is close to ``alpha`` the
    exceedances there are noise, not structure, and no "last significant lag" should be
    read off from that range.
    """
    v = np.abs(np.asarray(values, dtype=np.float64))
    hi = v.size - 1 if hi is None else min(int(hi), v.size - 1)
    if lo < 1 or hi < lo:
        raise ValueError("need 1 <= lo <= hi within the available lags")
    seg = v[lo : hi + 1]
    count = int(np.sum(seg > level))
    return {
        "lo": int(lo),
        "hi": int(hi),
        "n_lags": int(seg.size),
        "count": count,
        "fraction": count / seg.size,
        "chance_fraction": alpha,
        "expected_by_chance": alpha * seg.size,
    }


# ------------------------------------------------------------------- decomposition


@dataclass(frozen=True)
class Decomposition:
    """Components of an additive decomposition plus each one's share of the variance.

    ``variance_shares`` holds ``var(component) / var(series)`` for the trend, each
    seasonal component and the remainder, computed over ``n_valid`` rows where every
    component is defined (the classical filter has ``period // 2`` NaN rows at each
    end). The shares need not sum to exactly one because the components are not
    orthogonal; ``sum`` is included so the reader can see how far off they are. The
    method is recorded because the remainder share is method dependent: on square 5161
    it is about 0.05 for STL at period 144 and about 0.11 for the classical filter, and
    quoting one without the other invites a wrong comparison.
    """

    method: str
    periods: tuple[int, ...]
    trend: np.ndarray
    seasonal: dict[str, np.ndarray]
    remainder: np.ndarray
    variance_shares: dict[str, float]
    n_valid: int

    def describe(self) -> str:
        parts = ", ".join(f"{k} {v:.3f}" for k, v in self.variance_shares.items())
        periods = ",".join(map(str, self.periods))
        return f"{self.method} (period {periods}, n_valid {self.n_valid}): {parts}"


def _shares(x: np.ndarray, components: dict[str, np.ndarray]) -> tuple[dict[str, float], int]:
    valid = np.ones(x.size, dtype=bool)
    for comp in components.values():
        valid &= np.isfinite(comp)
    total = float(np.var(x[valid]))
    if total == 0.0:
        raise ValueError("series has zero variance; shares are undefined")
    shares = {k: float(np.var(v[valid]) / total) for k, v in components.items()}
    shares["sum"] = float(sum(shares.values()))
    return shares, int(valid.sum())


def decompose(
    x: Any,
    period: int = SLOTS_PER_DAY,
    method: DecomposeMethod = "stl",
    *,
    periods: Sequence[int] = (SLOTS_PER_DAY, SLOTS_PER_WEEK),
    robust: bool = False,
) -> Decomposition:
    """Additive decomposition by ``stl``, ``classical`` moving averages or ``mstl``.

    ``stl`` (Cleveland et al., loess) and ``classical`` (centred moving-average trend,
    period-mean seasonal) use ``period``; ``mstl`` fits one seasonal component per entry
    of ``periods`` (daily and weekly by default) and ignores ``period``. All three return
    the same :class:`Decomposition` so their variance shares can be tabulated together.
    """
    x = _as_1d(x)
    if method == "stl":
        res = STL(x, period=int(period), robust=robust).fit()
        seasonal = {f"seasonal_{period}": np.asarray(res.seasonal, dtype=np.float64)}
        trend, remainder = np.asarray(res.trend), np.asarray(res.resid)
        used = (int(period),)
    elif method == "classical":
        res = seasonal_decompose(x, period=int(period), model="additive")
        seasonal = {f"seasonal_{period}": np.asarray(res.seasonal, dtype=np.float64)}
        trend, remainder = np.asarray(res.trend), np.asarray(res.resid)
        used = (int(period),)
    elif method == "mstl":
        used = tuple(int(p) for p in periods)
        res = MSTL(x, periods=used).fit()
        seas = np.asarray(res.seasonal, dtype=np.float64)
        if seas.ndim == 1:
            seas = seas[:, None]
        seasonal = {f"seasonal_{p}": seas[:, i] for i, p in enumerate(used)}
        trend, remainder = np.asarray(res.trend), np.asarray(res.resid)
    else:
        raise ValueError(f"unknown method {method!r}; expected stl, classical or mstl")

    components = {"trend": trend, **seasonal, "remainder": remainder}
    shares, n_valid = _shares(x, components)
    return Decomposition(
        method=method,
        periods=used,
        trend=trend,
        seasonal=seasonal,
        remainder=remainder,
        variance_shares=shares,
        n_valid=n_valid,
    )


def variance_share_table(decompositions: Iterable[tuple[str, Decomposition]]) -> pd.DataFrame:
    """One row per labelled decomposition, one column per component share, for a report."""
    rows = []
    seasonal_cols: set[str] = set()
    for label, d in decompositions:
        row: dict[str, Any] = {
            "series": label,
            "method": d.method,
            "periods": ",".join(map(str, d.periods)),
        }
        row.update(d.variance_shares)
        row["n_valid"] = d.n_valid
        seasonal_cols.update(d.seasonal)
        rows.append(row)
    ordered_seasonal = sorted(seasonal_cols, key=lambda c: int(c.rsplit("_", 1)[1]))
    columns = [
        "series",
        "method",
        "periods",
        "trend",
        *ordered_seasonal,
        "remainder",
        "sum",
        "n_valid",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


# --------------------------------------------------------------------- stationarity


def interpret_stationarity(adf_p: float, kpss_p: float, *, alpha: float = 0.05) -> str:
    """Read the two tests together; their nulls point in opposite directions.

    ADF: H0 = unit root, so a small p is evidence *for* stationarity.
    KPSS: H0 = level stationary, so a small p is evidence *against* it.
    When both reject, the series is neither a clean unit root nor level stationary,
    which is typical of a changing mean or variance (drift, regime change,
    heteroskedasticity). That disagreement is a finding, not a failure of the tests.
    """
    adf_rejects = adf_p < alpha
    kpss_rejects = kpss_p < alpha
    if adf_rejects and not kpss_rejects:
        return "stationary (ADF rejects unit root; KPSS keeps stationarity)"
    if kpss_rejects and not adf_rejects:
        return "unit root (ADF keeps unit root; KPSS rejects stationarity)"
    if adf_rejects and kpss_rejects:
        return "both reject: no unit root but not level-stationary (drift or changing variance)"
    return "inconclusive: neither test rejects its null"


def _adf(s: np.ndarray) -> dict[str, Any]:
    stat, p, lags, nobs, crit, _ = adfuller(s, regression="c", autolag="AIC")
    return {
        "adf_stat": float(stat),
        "adf_p": float(p),
        "adf_lags": int(lags),
        "adf_nobs": int(nobs),
        "adf_crit_5": float(crit["5%"]),
    }


def _kpss(s: np.ndarray, regression: str) -> dict[str, Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InterpolationWarning)
        stat, p, lags, crit = kpss(s, regression=regression, nlags="auto")
    capped = any(issubclass(w.category, InterpolationWarning) for w in caught)
    # statsmodels clamps p to the edge of its table (0.01 or 0.1) and warns; report the
    # inequality rather than pretend the clamp is an estimate.
    if capped and p >= 0.1:
        display = "> 0.10"
    elif capped and p <= 0.01:
        display = "< 0.01"
    else:
        display = f"{p:.3f}"
    return {
        "kpss_stat": float(stat),
        "kpss_p": float(p),
        "kpss_p_capped": bool(capped),
        "kpss_p_display": display,
        "kpss_lags": int(lags),
        "kpss_crit_5": float(crit["5%"]),
    }


def stationarity_tests(
    x: Any,
    *,
    seasonal_lag: int = SLOTS_PER_DAY,
    kpss_regression: Literal["c", "ct"] = "c",
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """ADF and KPSS on the raw, ``log1p``, first-differenced and seasonally differenced series.

    Keys are ``raw``, ``log1p``, ``diff1`` and ``sdiff``; each value carries the ADF
    statistic, p-value and lag order (null: :data:`ADF_NULL`), the KPSS statistic,
    p-value and lag order (null: :data:`KPSS_NULL`), the transform applied, the sample
    size and a one-line ``verdict`` from :func:`interpret_stationarity`. KPSS p-values
    that hit the edge of statsmodels' lookup table are flagged ``kpss_p_capped`` and
    shown as an inequality; the FutureWarning about the tuple return type is silenced
    because it says nothing about the data.

    Note on power: a strongly seasonal series with a stable mean rejects the ADF null
    easily because the test asks about a unit root at lag one, not about seasonality.
    A "stationary" verdict here does not mean the daily cycle is absent.
    """
    x = _as_1d(x)
    if x.min() <= -1.0:
        raise ValueError("log1p requires values > -1")
    if x.size <= seasonal_lag + 10:
        raise ValueError("series too short for the seasonal difference")
    variants = {
        "raw": ("x[t]", x),
        "log1p": ("log1p(x[t])", np.log1p(x)),
        "diff1": ("x[t] - x[t-1]", np.diff(x)),
        "sdiff": (f"x[t] - x[t-{seasonal_lag}]", x[seasonal_lag:] - x[:-seasonal_lag]),
    }
    out: dict[str, dict[str, Any]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for name, (transform, s) in variants.items():
            entry: dict[str, Any] = {"transform": transform, "n": int(s.size)}
            entry.update(_adf(s))
            entry.update(_kpss(s, kpss_regression))
            entry["verdict"] = interpret_stationarity(entry["adf_p"], entry["kpss_p"], alpha=alpha)
            out[name] = entry
    return out


def stationarity_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Flatten :func:`stationarity_tests` output into one row per series for a report."""
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "series": name,
                "transform": r["transform"],
                "n": r["n"],
                "ADF stat": r["adf_stat"],
                "ADF p": r["adf_p"],
                "ADF lags": r["adf_lags"],
                "KPSS stat": r["kpss_stat"],
                "KPSS p": r["kpss_p_display"],
                "KPSS lags": r["kpss_lags"],
                "verdict": r["verdict"],
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------ level-dependent noise


def rolling_mean_std(x: Any, window: int = SLOTS_PER_DAY) -> pd.DataFrame:
    """Trailing rolling mean and standard deviation, NaN warm-up rows dropped."""
    s = pd.Series(_as_1d(x))
    frame = pd.DataFrame({"mean": s.rolling(window).mean(), "std": s.rolling(window).std()})
    return frame.dropna().reset_index(drop=True)


def rolling_mean_std_correlation(x: Any, window: int = SLOTS_PER_DAY) -> float:
    """Pearson r between the rolling mean and the rolling standard deviation.

    A multiplicative-noise diagnostic: if the spread grows with the level, the raw
    series has level-dependent variance and a variance-stabilising transform (``log1p``,
    given the zeros at night) is justified. On the same series the correlation of the
    transformed values should be markedly lower; the pair of numbers is the evidence.
    """
    frame = rolling_mean_std(x, window)
    return float(frame["mean"].corr(frame["std"]))


# ------------------------------------------------------------------ calendar views


def weekly_weekday_means(
    x: Any, index: pd.DatetimeIndex, *, weekdays: Sequence[int] = (0, 1, 2, 3, 4)
) -> pd.DataFrame:
    """Mean per slot over Monday-to-Friday slots, one row per ISO week.

    The evaluation week (ISO 51) is the last week before the Christmas collapse, and the
    question for the report is whether weekday levels were already drifting through the
    training weeks. A per-week weekday mean answers that without being swayed by the
    weekend regime difference documented in EDA finding F-02.

    Returns a frame indexed by ``iso_week`` with ``iso_year``, ``week_start`` (Monday),
    ``n_days``, ``n_slots`` and ``mean``. Partial weeks are kept, with their ``n_days``
    telling the reader how partial.
    """
    values = _as_1d(x)
    index = pd.DatetimeIndex(index)
    if len(index) != values.size:
        raise ValueError("x and index must have the same length")
    keep = np.isin(index.dayofweek, list(weekdays))
    iso = index.isocalendar()
    frame = pd.DataFrame(
        {
            "iso_year": iso["year"].to_numpy(dtype=int)[keep],
            "iso_week": iso["week"].to_numpy(dtype=int)[keep],
            "date": index.normalize()[keep],
            "value": values[keep],
        }
    )
    grouped = frame.groupby(["iso_year", "iso_week"], sort=True)
    out = grouped.agg(mean=("value", "mean"), n_slots=("value", "size"), n_days=("date", "nunique"))
    out = out.reset_index()
    out["week_start"] = [
        pd.Timestamp.fromisocalendar(int(y), int(w), 1).date()
        for y, w in zip(out["iso_year"], out["iso_week"], strict=True)
    ]
    out = out[["iso_year", "iso_week", "week_start", "n_days", "n_slots", "mean"]]
    return out.set_index("iso_week")


def daily_peak_times(x: Any, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Time and value of each day's maximum.

    Used for the regime finding on square 5259: on weekdays its peak is in office hours,
    on weekends the daily maximum falls in the small hours because the daytime pattern
    is simply absent. One row per calendar day with ``date``, ``weekday``,
    ``is_weekend``, ``peak_time`` (``HH:MM`` local), ``peak_value`` and ``n_slots``.
    """
    values = _as_1d(x)
    index = pd.DatetimeIndex(index)
    if len(index) != values.size:
        raise ValueError("x and index must have the same length")
    s = pd.Series(values, index=index)
    day = index.floor("D")
    grouped = s.groupby(day)
    peak_at = grouped.idxmax()
    days = pd.DatetimeIndex(peak_at.index)
    return pd.DataFrame(
        {
            "date": days.date,
            "weekday": days.day_name(),
            "is_weekend": days.dayofweek >= 5,
            "peak_time": (
                pd.DatetimeIndex(peak_at.to_numpy()).tz_convert(index.tz).strftime("%H:%M")
                if index.tz is not None
                else pd.DatetimeIndex(peak_at.to_numpy()).strftime("%H:%M")
            ),
            "peak_value": grouped.max().to_numpy(),
            "n_slots": grouped.size().to_numpy(),
        }
    ).reset_index(drop=True)


def summarise_daily_peaks(peaks: pd.DataFrame, *, early_cutoff: str = "06:00") -> pd.DataFrame:
    """Weekday against weekend: number of days, median peak time, days peaking before dawn."""
    minutes = peaks["peak_time"].map(lambda t: int(t[:2]) * 60 + int(t[3:]))
    cutoff = int(early_cutoff[:2]) * 60 + int(early_cutoff[3:])
    frame = peaks.assign(_minutes=minutes, _early=minutes < cutoff)
    rows = []
    for is_weekend, sub in frame.groupby("is_weekend", sort=True):
        med = int(round(float(sub["_minutes"].median())))
        rows.append(
            {
                "day_type": "weekend" if is_weekend else "weekday",
                "n_days": int(len(sub)),
                "median_peak_time": f"{med // 60:02d}:{med % 60:02d}",
                f"n_peaks_before_{early_cutoff}": int(sub["_early"].sum()),
                "median_peak_value": float(sub["peak_value"].median()),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- anomaly scan


@dataclass(frozen=True)
class AnomalyScan:
    """Result of :func:`same_slot_anomaly_scan`.

    ``ratio[i]`` is the value at row ``i`` divided by the median of the reference rows
    sharing its slot-of-period; ``reference_median`` is that median per slot;
    ``flagged`` lists rows whose ratio exceeds ``threshold``.
    """

    ratio: np.ndarray
    reference_median: np.ndarray
    flagged: pd.DataFrame
    period: int
    threshold: float


def same_slot_anomaly_scan(
    data: Any,
    *,
    period: int = SLOTS_PER_WEEK,
    threshold: float = 1.5,
    index: pd.DatetimeIndex | None = None,
    reference_rows: slice | None = None,
) -> AnomalyScan:
    """Flag slots far above the typical value for the same slot of the week.

    Exists to separate a grid-wide event from a local blip before either is blamed on a
    model. A one-dimensional input is scanned as is; a two-dimensional
    ``(n_slots, n_squares)`` input (the memmap) is first reduced to grid totals with
    :func:`grid_totals`. The reference median for each slot-of-period is taken over
    ``reference_rows`` (default: every row passed), so the caller can restrict it to the
    training slice. With about five weeks of training data each slot-of-week median is
    over five or six values, which a single spike cannot move.

    On the grid totals of the training slice, period 1008 and threshold 1.5 flag exactly
    one row: 5165, Friday 2013-12-06 20:50 CET, at 1.58 times its slot median, with the
    next largest ratio at 1.26.
    """
    arr = np.asarray(data) if not hasattr(data, "ndim") else data
    x = grid_totals(arr) if arr.ndim == 2 else _as_1d(arr)
    n = x.size
    if period < 2 or period > n:
        raise ValueError(f"period must be in [2, n]; got {period} for n={n}")
    phase = np.arange(n) % period
    ref = np.arange(n) if reference_rows is None else np.arange(n)[reference_rows]
    med = pd.Series(x[ref]).groupby(phase[ref]).median().reindex(range(period)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(med[phase] > 0, x / med[phase], np.nan)
    hits = np.flatnonzero(ratio > threshold)
    flagged = pd.DataFrame(
        {
            "row": hits,
            "value": x[hits],
            "slot_median": med[phase[hits]],
            "ratio": ratio[hits],
        }
    )
    if index is not None:
        index = pd.DatetimeIndex(index)
        if len(index) != n:
            raise ValueError("index must align with the scanned rows")
        flagged.insert(1, "time", [index[i].strftime("%a %Y-%m-%d %H:%M") for i in hits])
    return AnomalyScan(
        ratio=ratio,
        reference_median=med,
        flagged=flagged,
        period=int(period),
        threshold=float(threshold),
    )


# ------------------------------------------------------------------------ figures


def plot_acf_pacf(
    table: pd.DataFrame,
    *,
    mark_lags: Sequence[int] = (SLOTS_PER_DAY, SLOTS_PER_WEEK),
    practical_threshold: float | None = 0.05,
    pacf_ylim: tuple[float, float] = (-0.2, 0.2),
    title: str | None = None,
    figsize: tuple[float, float] = (COLUMN_WIDTH_IN, 4.4),
):
    """Two panels, ACF and PACF, with the band shaded and the seasonal lags marked.

    The PACF axis is clipped to ``pacf_ylim`` because lag 1 (about 0.98 on traffic) would
    otherwise flatten everything else; clipped lags are named in an annotation.

    The panels are stacked and share the lag axis. Side by side at column width each one
    would be 1.6 in across, which is narrower than the PACF legend.
    """
    import matplotlib.pyplot as plt

    band = float(table.attrs.get("band", table["band"].iloc[0]))
    method = table.attrs.get("pacf_method", "ldb")
    lags = table["lag"].to_numpy()
    fig, (ax_a, ax_p) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax_a.plot(lags, table["acf"], lw=0.9)
    ax_a.axhspan(-band, band, color="grey", alpha=0.25, lw=0)
    ax_a.set_ylim(min(-0.1, float(table["acf"].min()) - 0.05), 1.05)
    ax_a.set_title("ACF")
    ax_a.set_ylabel("autocorrelation")

    pacf_vals = table["pacf"].to_numpy()
    ax_p.vlines(lags[1:], 0, pacf_vals[1:], lw=0.6, color="C0")
    ax_p.axhspan(-band, band, color="grey", alpha=0.25, lw=0, label=f"95% band +/-{band:.4f}")
    if practical_threshold is not None:
        ax_p.axhline(
            practical_threshold,
            color="C1",
            ls=":",
            lw=0.9,
            label=f"practical +/-{practical_threshold:.2f}",
        )
        ax_p.axhline(-practical_threshold, color="C1", ls=":", lw=0.9)
    ax_p.set_ylim(*pacf_ylim)
    label_method = "Levinson-Durbin" if method == "ldb" else "Yule-Walker"
    ax_p.set_title(f"PACF ({label_method})")
    ax_p.set_ylabel("partial autocorrelation")
    clipped = [
        (int(lag), float(v))
        for lag, v in zip(lags[1:], pacf_vals[1:], strict=True)
        if v > pacf_ylim[1] or v < pacf_ylim[0]
    ]
    if clipped:
        text = "off scale: " + ", ".join(f"lag {lag} = {v:.2f}" for lag, v in clipped[:3])
        if len(clipped) > 3:
            text += f" (+{len(clipped) - 3} more)"
        # wrap=True keeps the wording and lets the line break where the axis ends.
        ax_p.text(
            0.02,
            0.88,
            text,
            transform=ax_p.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            wrap=True,
        )
    ax_p.legend(loc="lower right", fontsize=7)

    for ax in (ax_a, ax_p):
        for lag in mark_lags:
            ax.axvline(lag, color="C2", ls="--", lw=0.8)
            ax.text(lag, ax.get_ylim()[1], f" {lag}", color="C2", fontsize=7, va="top", ha="left")
        ax.set_xlim(0, lags[-1])
    ax_p.set_xlabel("lag (10-minute slots)")
    if title:
        fig.suptitle(title, fontsize=7.5, wrap=True)
    fig.tight_layout()
    return fig


def plot_decomposition(
    decomp: Decomposition,
    x: Any,
    index: pd.DatetimeIndex,
    *,
    unit: str = UNIT,
    title: str | None = None,
    figsize: tuple[float, float] = (COLUMN_WIDTH_IN, 4.9),
):
    """Observed, trend, seasonal and remainder panels sharing the time axis.

    Panel titles carry each component's variance share so the figure states its own
    headline number. For MSTL the seasonal panel overlays every seasonal component.

    All four panels are kept: the remainder share only means something beside the
    seasonal panel it is a remainder from. What is cut at column width is the repetition.
    The four panels carried the same y label four times; one figure-level label says it
    once and gives the height back to the panels, and the Monday date labels drop to the
    6.5 pt floor because six of them at 7 pt are wider than the axis.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    values = _as_1d(x)
    index = pd.DatetimeIndex(index)
    shares = decomp.variance_shares
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, layout="constrained")
    axes[0].plot(index, values, lw=0.6)
    axes[0].set_title("observed")
    axes[1].plot(index, decomp.trend, lw=0.9)
    axes[1].set_title(f"trend ({100 * shares['trend']:.1f}% of variance)")
    for name, comp in decomp.seasonal.items():
        share = 100 * shares[name]
        axes[2].plot(index, comp, lw=0.6, label=f"{name.replace('_', ' ')} ({share:.1f}%)")
    axes[2].set_title(f"seasonal, {decomp.method.upper()}")
    if len(decomp.seasonal) > 1:
        axes[2].legend(loc="upper right", fontsize=7, ncol=len(decomp.seasonal))
    axes[3].plot(index, decomp.remainder, lw=0.5)
    axes[3].set_title(f"remainder ({100 * shares['remainder']:.1f}% of variance)")
    axes[3].axhline(0, color="black", lw=0.5)
    fig.supylabel(unit, fontsize=8)
    axes[3].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    # Six weekly labels side by side are wider than the column even at 6.5 pt, so every
    # second Monday is labelled. The tick marks stay weekly, so no week is lost.
    stamp = mdates.DateFormatter("%a %d %b")
    axes[3].xaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: stamp(v) if (mdates.num2date(v).toordinal() // 7) % 2 == 0 else ""
        )
    )
    axes[3].tick_params(axis="x", labelsize=6.5)
    axes[3].set_xlabel("date (CET)")
    if title:
        fig.suptitle(title, fontsize=7.5, wrap=True)
    return fig


def plot_rolling_scatter(
    x: Any,
    *,
    window: int = SLOTS_PER_DAY,
    unit: str = UNIT,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 3.2),
):
    """Rolling std against rolling mean for the raw and the ``log1p`` series, r in the legend."""
    import matplotlib.pyplot as plt

    values = _as_1d(x)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (label, series, unit_label) in zip(
        axes,
        [("raw", values, unit), ("log1p", np.log1p(values), f"log1p({unit})")],
        strict=True,
    ):
        frame = rolling_mean_std(series, window)
        r = float(frame["mean"].corr(frame["std"]))
        ax.scatter(frame["mean"], frame["std"], s=3, alpha=0.3, lw=0, label=f"{label}: r = {r:.3f}")
        ax.set_xlabel(f"rolling mean, {window} slots ({unit_label})")
        ax.set_ylabel(f"rolling std ({unit_label})")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=7, markerscale=3)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- reporting


def markdown_table(df: pd.DataFrame, *, floatfmt: str = ".4f", index: bool = False) -> str:
    """Render a frame as a GitHub-flavoured markdown table without needing ``tabulate``."""
    frame = df.reset_index() if index else df

    def fmt(v: Any) -> str:
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return format(float(v), floatfmt) if np.isfinite(v) else "-"
        return str(v).replace("|", "\\|")

    header = "| " + " | ".join(fmt(c) for c in frame.columns) + " |"
    sep = "|" + "|".join(" --- " for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, sep, *body])
