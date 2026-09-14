"""Splitting, scaling and windowing - the preprocessing every model shares.

Two things in this module exist to prevent silent mistakes rather than to add
capability, and both are worth a sentence in the report:

*Leakage through scaling.* ``Scaler`` refuses to transform before it has been fit, and
is only ever fit on the training slice. Fitting on the full series leaks the test-period
mean and variance into training and quietly flatters every model equally, which is worse
than a visible bug because the comparison still looks internally consistent.

*Leakage through windowing.* A supervised window spans ``seq_len + horizon`` steps. If
windows are built over the whole series and split afterwards, windows straddling a
boundary put training observations into a test input. :func:`split_windows` builds
windows **inside** each split, so a boundary costs ``seq_len`` unusable windows instead
of leaking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, NamedTuple, get_args

import numpy as np
import pandas as pd

ScaleMethod = Literal["standard", "minmax", "log1p", "log1p-standard", "none"]


class Split(NamedTuple):
    """Row ranges of a chronological train/validation/test partition."""

    train: slice
    val: slice
    test: slice

    def describe(self, index: pd.DatetimeIndex | None = None) -> str:
        def part(name: str, s: slice) -> str:
            n = (s.stop or 0) - (s.start or 0)
            if index is None:
                return f"{name}={n}"
            return f"{name}={n} [{index[s.start]:%Y-%m-%d %H:%M} .. {index[s.stop - 1]:%Y-%m-%d %H:%M}]"

        return "  ".join(part(n, s) for n, s in zip(("train", "val", "test"), self, strict=True))


def chronological_split(
    n: int,
    *,
    val_frac: float = 0.15,
    test_slice: slice | None = None,
    test_frac: float = 0.10,
) -> Split:
    """Split ``n`` observations in time order.

    If ``test_slice`` is given (the usual case here - the brief fixes the evaluation
    week), the test period is taken from it and train/validation are carved out of
    everything strictly before it. Nothing after the test window is used at all, which
    matches the deployment story: you forecast forward, not into a gap.
    """
    if test_slice is None:
        n_test = int(round(n * test_frac))
        test_slice = slice(n - n_test, n)

    end_train_val = test_slice.start
    if end_train_val is None or end_train_val <= 0:
        raise ValueError("test_slice must start after at least one observation.")

    n_val = int(round(end_train_val * val_frac))
    return Split(
        train=slice(0, end_train_val - n_val),
        val=slice(end_train_val - n_val, end_train_val),
        test=test_slice,
    )


def eval_week_split(
    index: pd.DatetimeIndex, start: str, end: str, *, val_frac: float = 0.15
) -> Split:
    """Chronological split whose test period is the fixed evaluation week."""
    tz = index.tz
    mask = (index >= pd.Timestamp(start, tz=tz)) & (index <= pd.Timestamp(end, tz=tz))
    pos = np.flatnonzero(mask)
    if pos.size == 0:
        raise ValueError(f"No observations between {start} and {end}.")
    return chronological_split(
        len(index), val_frac=val_frac, test_slice=slice(int(pos[0]), int(pos[-1]) + 1)
    )


@dataclass
class Scaler:
    """Fit-on-train-only scaler with an exact inverse.

    ``log1p`` variants are worth trying on this data: the per-cell traffic distribution
    is strongly right-skewed, and a model trained on raw values spends its capacity on
    the peaks. Metrics are always computed after :meth:`inverse_transform`, so the
    choice of scaling never changes the units a result is reported in.
    """

    method: ScaleMethod = "standard"
    center_: float = 0.0
    scale_: float = 1.0
    fitted_: bool = False

    def __post_init__(self) -> None:
        # An unrecognised name used to fall through to the identity branch in fit(),
        # so a typo ("standrd", "zscore", "LOG1P") silently disabled scaling instead of
        # failing. That is the worst kind of bug here: the run completes, the numbers
        # look reasonable, and nothing in the artefacts records that no scaling happened.
        if self.method not in get_args(ScaleMethod):
            raise ValueError(
                f"Unknown scaling method {self.method!r}; expected one of "
                f"{list(get_args(ScaleMethod))}."
            )

    def fit(self, x: np.ndarray) -> Scaler:
        v = np.asarray(x, dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ValueError("Cannot fit a scaler on an empty array.")
        if self.method in ("log1p", "log1p-standard"):
            if v.min() < -1:
                raise ValueError("log1p scaling requires values > -1.")
            v = np.log1p(v)

        if self.method in ("standard", "log1p-standard"):
            self.center_, self.scale_ = float(v.mean()), float(v.std())
        elif self.method == "minmax":
            self.center_, self.scale_ = float(v.min()), float(v.max() - v.min())
        else:  # "log1p" without standardisation, or "none" - both leave values as they are
            self.center_, self.scale_ = 0.0, 1.0

        if self.scale_ == 0.0:
            self.scale_ = 1.0
        self.fitted_ = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError(
                "Scaler.transform called before fit - this is how "
                "test-set statistics leak into training."
            )
        v = np.asarray(x, dtype=np.float64)
        if self.method in ("log1p", "log1p-standard"):
            v = np.log1p(v)
        return (v - self.center_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Scaler.inverse_transform called before fit.")
        v = np.asarray(x, dtype=np.float64) * self.scale_ + self.center_
        if self.method in ("log1p", "log1p-standard"):
            v = np.expm1(v)
        return v


def make_windows(
    series: np.ndarray,
    seq_len: int,
    *,
    horizon: int = 1,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Supervised windows for one-step (or ``horizon``-step) ahead forecasting.

    Returns ``X`` of shape ``(n_windows, seq_len)`` and ``y`` of shape ``(n_windows,)``,
    where ``y[i]`` is the value ``horizon`` steps after the end of ``X[i]``. Built as a
    strided view where possible, so no data is copied until a model asks for it.
    """
    v = np.asarray(series, dtype=np.float32).ravel()
    n_windows = v.size - seq_len - horizon + 1
    if n_windows <= 0:
        raise ValueError(
            f"Series of length {v.size} is too short for seq_len={seq_len}, " f"horizon={horizon}."
        )
    windows = np.lib.stride_tricks.sliding_window_view(v, seq_len)[:n_windows]
    targets = v[seq_len + horizon - 1 :]
    return windows[::stride].copy(), targets[::stride].copy()


def split_windows(
    series: np.ndarray,
    split: Split,
    seq_len: int,
    *,
    horizon: int = 1,
    scaler: Scaler | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Windows built **within** each split, so no window crosses a boundary.

    The scaler is fit on the raw training slice only and applied to all three. Targets
    are returned in scaled space for training; invert with ``scaler.inverse_transform``
    before computing any reported metric.
    """
    v = np.asarray(series, dtype=np.float64).ravel()
    sc = scaler or Scaler("standard")
    if not sc.fitted_:
        sc.fit(v[split.train])

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, sl in zip(("train", "val", "test"), split, strict=True):
        segment = sc.transform(v[sl])
        out[name] = make_windows(segment, seq_len, horizon=horizon)
    return out


def context_windows(
    series: np.ndarray,
    split: Split,
    seq_len: int,
    *,
    horizon: int = 1,
    scaler: Scaler,
    target_slice: slice | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Windows that may reach back into the *past* for their inputs.

    Discarding the first ``seq_len`` test points is honest but wasteful: at forecast
    time the history immediately before the evaluation week is legitimately available.
    This builds windows starting ``seq_len + horizon - 1`` steps early, so every point of
    the period gets a prediction while the *targets* stay strictly inside it. Use this for
    the reported Dec 16-22 results and say which convention was used; use
    :func:`split_windows` when a clean, context-free comparison is wanted.

    ``target_slice`` defaults to ``split.test``. Pass ``split.val`` to apply the same
    convention to validation, which is necessary whenever the validation period is shorter
    than one window: the inputs then come from training data, which is exactly what is
    available when selecting a model, and the targets still lie strictly inside the
    validation period.
    """
    v = np.asarray(series, dtype=np.float64).ravel()
    target = split.test if target_slice is None else target_slice
    start = target.start - seq_len - horizon + 1
    if start < 0:
        raise ValueError("Not enough history before the window for this seq_len.")
    segment = scaler.transform(v[start : target.stop])
    return make_windows(segment, seq_len, horizon=horizon)


#: Public holidays observed in Milan that fall inside the 62-day observation window:
#: Ognissanti (Nov 1), Sant'Ambrogio (Dec 7), Immacolata (Dec 8), Natale and Santo Stefano
#: (Dec 25-26) and Capodanno (Jan 1). Square 5259's daily maximum falls at night on every
#: one of the first fortnight's non-working days, holiday included, so "working day" and
#: not "weekday" is the regime label the models are given.
ITALIAN_PUBLIC_HOLIDAYS: frozenset[date] = frozenset(
    {
        date(2013, 11, 1),
        # Sant'Ambrogio, Milan's patron saint and a city-only public holiday. It fell on a
        # Saturday in 2013, so every flag below is unchanged by its presence; it is listed
        # because the calendar encoding should be right about the city it describes, not
        # only about the days where being wrong would have shown up in the metrics.
        date(2013, 12, 7),
        date(2013, 12, 8),
        date(2013, 12, 25),
        date(2013, 12, 26),
        date(2014, 1, 1),
    }
)

#: Column order of :func:`target_calendar`. Fixed here so every model, and the ablation
#: that switches the features off, refers to the same five columns.
TARGET_CALENDAR_COLUMNS: tuple[str, ...] = (
    "sin_tod",
    "cos_tod",
    "sin_dow",
    "cos_dow",
    "is_working_day",
)


def is_working_day(index: pd.DatetimeIndex) -> np.ndarray:
    """1.0 for Monday to Friday that is not a public holiday, else 0.0 (float32)."""
    weekday = np.asarray(index.dayofweek) < 5
    dates = np.asarray([ts.date() for ts in index])
    holiday = np.isin(dates, list(ITALIAN_PUBLIC_HOLIDAYS))
    return (weekday & ~holiday).astype(np.float32)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical time-of-day / day-of-week encodings, for the feature-based models.

    Sine-cosine pairs rather than integer hour and weekday, so that 23:50 and 00:00 are
    adjacent in feature space instead of maximally distant. ``is_working_day`` is the
    weekend flag corrected for :data:`ITALIAN_PUBLIC_HOLIDAYS`.
    """
    slot = index.hour * 6 + index.minute // 10
    dow = index.dayofweek
    return pd.DataFrame(
        {
            "sin_day": np.sin(2 * np.pi * slot / 144),
            "cos_day": np.cos(2 * np.pi * slot / 144),
            "sin_week": np.sin(2 * np.pi * (dow * 144 + slot) / 1008),
            "cos_week": np.cos(2 * np.pi * (dow * 144 + slot) / 1008),
            "is_weekend": (dow >= 5).astype(np.float32),
            "is_working_day": is_working_day(index),
        },
        index=index,
    )


def target_calendar(index: pd.DatetimeIndex, target_rows: np.ndarray) -> np.ndarray:
    """Calendar features of the *target* slot of each window, shape ``(n, 5)`` float32.

    Columns are :data:`TARGET_CALENDAR_COLUMNS`: sine and cosine of the time of day, sine
    and cosine of the day of week, and the working-day flag. ``target_rows`` are absolute
    row positions into ``index`` (see :func:`window_target_rows`). The calendar of the slot
    being predicted is known at forecast time, so this leaks nothing.
    """
    rows = np.asarray(target_rows, dtype=np.int64)
    sub = index[rows]
    slot = np.asarray(sub.hour * 6 + sub.minute // 10, dtype=np.float64)
    dow = np.asarray(sub.dayofweek, dtype=np.float64)
    out = np.column_stack(
        [
            np.sin(2 * np.pi * slot / 144),
            np.cos(2 * np.pi * slot / 144),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            is_working_day(sub),
        ]
    ).astype(np.float32)
    assert out.shape == (rows.size, len(TARGET_CALENDAR_COLUMNS))
    return out


def window_target_rows(split: Split, seq_len: int, *, horizon: int = 1) -> dict[str, np.ndarray]:
    """Absolute row of the target of window ``i`` in each of the three window sets.

    Two conventions are in play and they align differently:

    * training windows come from :func:`make_windows` over ``series[split.train]``, so
      window ``i`` targets row ``split.train.start + seq_len + horizon - 1 + i``;
    * validation and test windows come from :func:`context_windows` with
      ``target_slice=s``, whose window ``i`` targets row ``s.start + i``.

    Exogenous features of the target slot must be built from these rows, or they are
    silently offset from the values they describe. ``tests/test_features.py`` proves the
    alignment by windowing a series equal to its own row index.
    """
    if seq_len < 1 or horizon < 1:
        raise ValueError("seq_len and horizon must be positive.")
    first_train = split.train.start + seq_len + horizon - 1
    if first_train >= split.train.stop:
        raise ValueError("Training slice is too short for this seq_len and horizon.")
    return {
        "train": np.arange(first_train, split.train.stop, dtype=np.int64),
        "val": np.arange(split.val.start, split.val.stop, dtype=np.int64),
        "test": np.arange(split.test.start, split.test.stop, dtype=np.int64),
    }
