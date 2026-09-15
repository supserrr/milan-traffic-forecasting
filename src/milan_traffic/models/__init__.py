"""Forecasting models.

Every model implements :class:`Forecaster` so that the training loop, the evaluation
code and the results tables stay model-agnostic. A comparison is only meaningful if the
three models differ in architecture and in nothing else about how they are fed, trained
and scored - the shared interface is what enforces that.

Add one module per model here and register it in :data:`REGISTRY`. The three models are
chosen and justified in writing *before* any of this is implemented, so that the
comparison tests a prediction rather than rationalising a result.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Forecaster(Protocol):
    """Minimal contract for a one-step-ahead forecaster.

    Implementations predict in the **same space they were fed**. The harness scales the
    inputs, and ``evaluate.run_model`` inverts the transform before scoring, so metrics
    are always reported in the original units of the series without any model having to
    know which normalisation was chosen. A model is never handed the scaler, so it could
    not invert internally even if it wanted to.

    ``exog`` is the ``(n, 5)`` float32 matrix of calendar features of each window's
    *target* slot (``features.target_calendar``), or ``None`` when the run switches the
    calendar off. Models that cannot use it accept and ignore it, so the harness calls
    every model the same way.

    Optional attributes the harness reads when present: ``history`` (a dict with
    ``n_epochs``, ``best_epoch``, ``s_per_epoch``, ``device`` and per-epoch loss traces)
    and ``device`` (``"cpu"``, ``"mps"``, ...), used to synchronise before a timer stops.
    """

    name: str

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        exog: np.ndarray | None = None,
        exog_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Forecaster:
        """Train on windowed inputs. Returns ``self``."""
        ...

    def predict(self, X: np.ndarray, exog: np.ndarray | None = None) -> np.ndarray:
        """One-step-ahead predictions, shape ``(len(X),)``, in the scale of ``X``."""
        ...

    @property
    def n_params(self) -> int:
        """Number of trainable parameters - reported alongside timing."""
        ...


#: name -> factory accepting keyword arguments. Populated by the model modules below.
REGISTRY: dict[str, Callable[..., Forecaster]] = {}


def register(name: str) -> Callable[[Callable[..., Forecaster]], Callable[..., Forecaster]]:
    """Decorator registering a model factory under ``name``."""

    def wrap(factory: Callable[..., Forecaster]) -> Callable[..., Forecaster]:
        if name in REGISTRY:
            raise KeyError(f"Model {name!r} is already registered.")
        REGISTRY[name] = factory
        return factory

    return wrap


def preset(factory: Callable[..., Forecaster], **defaults: Any) -> Callable[..., Forecaster]:
    """A factory with baked-in defaults that explicit keyword arguments may override.

    ``lambda **kw: SeasonalNaive(season=144, **kw)`` raised ``TypeError`` the moment a
    caller passed ``season=`` itself, so a run config could name the model but not
    configure it. Merging the two dicts, caller last, is the fix.
    """

    def build_preset(**kwargs: Any) -> Forecaster:
        return factory(**{**defaults, **kwargs})

    build_preset.__name__ = getattr(factory, "__name__", "factory")
    build_preset.__doc__ = f"{build_preset.__name__} with defaults {defaults!r}."
    return build_preset


def build(name: str, **kwargs: Any) -> Forecaster:
    """Instantiate a registered model, forwarding ``kwargs`` to its factory."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


# Model modules register themselves on import. They are imported here, after `register`
# exists, so that `import milan_traffic.models` alone makes every name a run config can
# contain resolvable through `build`.
from . import baselines, gbt, lstm, tcn  # noqa: E402, F401

__all__ = ["Forecaster", "REGISTRY", "build", "preset", "register"]
