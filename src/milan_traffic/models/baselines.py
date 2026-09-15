"""Reference forecasters: the error floor a trained model has to clear.

One-step-ahead traffic forecasting is easy. On the fixed evaluation week the previous
observation alone explains between 94 and 99 percent of the variance of every cell
studied here, so a sequential model that merely matches persistence has learned nothing
the last sample did not already carry. These baselines exist to state that floor in
numbers *before* any model is trained, which is what lets the comparison in the report
read as evidence rather than as a table.

Three design points worth a sentence in the write-up:

*Same input representation.* Every baseline consumes the identical windowed inputs the
trained models will receive (``features.make_windows``), so no comparison here turns on a
difference in data format. The calendar features the trained models receive are accepted
and ignored (``LinearAR`` can opt in), so the harness calls every model the same way.

*Scaling cuts across these four differently.* ``Persistence`` and ``SeasonalNaive`` copy a
single past value out of the window, and any monotone transform commutes with copying, so
whatever space the run scales into, the central inverse in ``evaluate.run_model`` returns
exactly the raw-scale observation: EXP-000 (``scaling: none``) and EXP-008
(``scaling: log1p-standard``) agree on the naive test MAE of all three reported cells to
within 7e-7, which is float32 round-trip noise. ``TimeOfDayMean`` averages four lags, so it
survives an affine scaler but not ``log1p``, under which the mean of the transformed lags is
not the transform of their mean. ``LinearAR`` is *fitted* in whatever space it is handed:
``evaluate.prepare_area`` scales every window with the train-fitted scaler before the models
see it, so under ``log1p-standard`` the least-squares solution is linear in ``log1p(x)`` and
not in the raw series. That is a different model, not the same model in different units. On
square 5161 the raw-scale AR(144) of EXP-000 reached a test MAE of 84.64 against 84.98 for
the log1p-standard fit of EXP-008, and on 5059 the transformed fit was 5.03 MAE better; the
two runs also differ in window length, so neither gap is a clean single-factor contrast, and
no linear-baseline number should be quoted without naming the run it came from.

*Seasonal lags are not free.* ``SeasonalNaive`` at the weekly lag needs a window of 1008
steps, and ``TimeOfDayMean`` needs four daily lags. Running every baseline from one
1008-step window keeps them comparable at the cost of discarding a week of windows at the
start of each split, which is the honest trade and is recorded in the run config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import preset, register


def _as2d(X: np.ndarray) -> np.ndarray:
    """Windows as a float64 ``(n_windows, seq_len)`` array, oldest step first."""
    a = np.asarray(X, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"Expected windows of shape (n, seq_len), got {a.shape}.")
    return a


def _require_lag(X: np.ndarray, lag: int, who: str) -> None:
    if X.shape[1] < lag:
        raise ValueError(f"{who} needs a window of at least {lag} steps, got seq_len={X.shape[1]}.")


@dataclass
class Persistence:
    """``x̄(t+1) = x(t)``. The last observation, repeated.

    The reference the brief's one-step-ahead task makes unavoidable: if a model cannot
    beat this, its architecture is not buying anything on this data.
    """

    name: str = "naive"

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> Persistence:
        return self

    def predict(self, X: np.ndarray, exog: np.ndarray | None = None) -> np.ndarray:
        return _as2d(X)[:, -1]

    @property
    def n_params(self) -> int:
        return 0


@dataclass
class SeasonalNaive:
    """``x̄(t+1) = x(t + 1 - season)``. The same slot one season ago.

    At ``season=144`` this is yesterday at the same time of day, at ``season=1008`` the
    same weekday and time last week. It isolates how much of the signal is pure calendar
    repetition, which on this data turns out to be much less than the daily cycle in the
    variance decomposition suggests.
    """

    season: int = 144
    name: str = ""

    def __post_init__(self) -> None:
        if self.season < 1:
            raise ValueError("season must be positive.")
        if not self.name:
            self.name = f"seasonal_naive_{self.season}"

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> SeasonalNaive:
        return self

    def predict(self, X: np.ndarray, exog: np.ndarray | None = None) -> np.ndarray:
        a = _as2d(X)
        _require_lag(a, self.season, self.name)
        return a[:, -self.season]

    @property
    def n_params(self) -> int:
        return 0


@dataclass
class TimeOfDayMean:
    """Mean of the same time-of-day slot over the previous ``n_seasons`` days.

    A smoothed seasonal naive: averaging several days suppresses the day-to-day noise
    that makes a single lagged day a poor forecast. Whether the smoothing helps is an
    empirical question about how stable the daily profile is, and the answer here is a
    result worth reporting.
    """

    season: int = 144
    n_seasons: int = 4
    name: str = "time_of_day_mean"

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> TimeOfDayMean:
        return self

    def predict(self, X: np.ndarray, exog: np.ndarray | None = None) -> np.ndarray:
        a = _as2d(X)
        lags = [self.season * k for k in range(1, self.n_seasons + 1)]
        _require_lag(a, max(lags), self.name)
        return np.mean(np.stack([a[:, -lag] for lag in lags], axis=1), axis=1)

    @property
    def n_params(self) -> int:
        return 0


@dataclass
class LinearAR:
    """Ordinary least squares on the last ``order`` lags, with an intercept.

    The cheapest model that is actually *fitted*, and therefore the reference that
    matters most: it separates "the previous sample is informative" from "a nonlinear
    sequence model is informative". Coefficients come from ``np.linalg.lstsq`` on the
    training windows only, so nothing from the validation or evaluation period reaches
    the fit.

    ``use_exog`` appends the calendar features of the target slot to the design matrix.
    It is off by default so that EXP-000 stays reproducible from its config; switched on
    it is the linear answer to "what are the calendar features worth", which is a
    useful reference for the same ablation on the nonlinear models.
    """

    order: int = 144
    use_exog: bool = False
    name: str = ""
    coef_: np.ndarray | None = field(default=None, repr=False)
    intercept_: float = 0.0
    n_exog_: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be positive.")
        if not self.name:
            self.name = f"linear_ar_{self.order}"

    def _design(self, X: np.ndarray, exog: np.ndarray | None) -> np.ndarray:
        a = _as2d(X)
        _require_lag(a, self.order, self.name)
        cols = [a[:, -self.order :]]
        if self.use_exog:
            if exog is None:
                raise ValueError(
                    f"{self.name} was configured with use_exog=True but got exog=None."
                )
            cols.append(np.asarray(exog, dtype=np.float64))
        return np.hstack(cols)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        exog: np.ndarray | None = None,
        **kwargs: object,
    ) -> LinearAR:
        design = self._design(X, exog)
        self.n_exog_ = design.shape[1] - self.order
        design = np.hstack([design, np.ones((design.shape[0], 1))])
        target = np.asarray(y, dtype=np.float64).ravel()
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        self.coef_, self.intercept_ = solution[:-1], float(solution[-1])
        return self

    def predict(self, X: np.ndarray, exog: np.ndarray | None = None) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError(f"{self.name}.predict called before fit.")
        return self._design(X, exog) @ self.coef_ + self.intercept_

    @property
    def n_params(self) -> int:
        return self.order + self.n_exog_ + 1


register("naive")(Persistence)
register("seasonal_naive_daily")(preset(SeasonalNaive, season=144))
register("seasonal_naive_144")(preset(SeasonalNaive, season=144))
register("seasonal_naive_weekly")(preset(SeasonalNaive, season=1008))
register("seasonal_naive_1008")(preset(SeasonalNaive, season=1008))
register("time_of_day_mean")(preset(TimeOfDayMean, season=144, n_seasons=4))
register("time_of_day_mean_4d")(
    preset(TimeOfDayMean, season=144, n_seasons=4, name="time_of_day_mean_4d")
)
register("linear_ar")(LinearAR)
register("linear_ar_6")(preset(LinearAR, order=6))
register("linear_ar_36")(preset(LinearAR, order=36))
register("linear_ar_144")(preset(LinearAR, order=144))

__all__ = ["LinearAR", "Persistence", "SeasonalNaive", "TimeOfDayMean"]
