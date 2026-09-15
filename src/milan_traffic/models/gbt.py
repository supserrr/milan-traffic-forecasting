"""Model 3: gradient-boosted regression trees on the lags and the calendar features.

The non-neural, non-sequential contender. Two of the three lines of evidence in the
model-selection note point away from deep models: the headroom over persistence is a few
percent, and the strongest predictor beyond the last value is a calendar effect, which
trees exploit through explicit feature interactions rather than by learning them from
shape. To the model the 144 scaled lags are 144 exchangeable columns; the sequence
inductive bias is deliberately absent, which is what makes the comparison informative.

Two caveats are stated here because they belong in the report:

*Early stopping uses scikit-learn's internal split.* ``HistGradientBoostingRegressor``
cannot take an external validation set, so ``validation_fraction=0.1`` holds out a random
tenth of the *training* windows. Adjacent windows share 143 of 144 values, so that
held-out tenth is optimistic compared with the chronological validation week the neural
models stop on. The chronological validation loss is still computed after the fit and
recorded in ``history`` so the three models can be compared on the same quantity.

*"Parameters" means leaves.* A boosted ensemble has no weight vector; the reported
``n_params`` is the total number of leaves across all fitted trees, which is the number
of distinct output values the model can produce and the closest analogue to a parameter
count. It is not comparable one-to-one with the neural models' weights, and the tables
should say so.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from . import register


class GBTForecaster:
    """``HistGradientBoostingRegressor`` on ``concat(window, exog)``."""

    name = "gbt"
    device = "cpu"

    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2: float = 0.0,
        use_exog: bool = True,
        seed: int = 42,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 20,
        **ignored: Any,
    ) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.use_exog = use_exog
        self.seed = seed
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.model_: HistGradientBoostingRegressor | None = None
        self.history: dict[str, Any] = {}
        self.n_features_: int = 0

    def _design(self, X: np.ndarray, exog: np.ndarray | None) -> np.ndarray:
        a = np.asarray(X, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(f"Expected windows of shape (n, seq_len), got {a.shape}.")
        if self.use_exog and exog is not None:
            e = np.asarray(exog, dtype=np.float64)
            if e.shape[0] != a.shape[0]:
                raise ValueError(f"exog has {e.shape[0]} rows for {a.shape[0]} windows.")
            a = np.hstack([a, e])
        return a

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
    ) -> GBTForecaster:
        design = self._design(X, exog)
        self.n_features_ = design.shape[1]
        self.model_ = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2,
            early_stopping=True,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.seed,
        )
        t0 = time.perf_counter()
        self.model_.fit(design, np.asarray(y, dtype=np.float64).ravel())
        fit_s = time.perf_counter() - t0

        n_iter = int(self.model_.n_iter_)
        # Scores are negative losses, first entry is the empty ensemble; the best
        # iteration is therefore the argmax and counts trees.
        val_scores = getattr(self.model_, "validation_score_", None)
        train_scores = getattr(self.model_, "train_score_", None)
        best_epoch = int(np.argmax(val_scores)) if val_scores is not None else n_iter
        chrono_val = None
        if X_val is not None and y_val is not None:
            pred = self.predict(X_val, exog=exog_val)
            chrono_val = float(np.mean((pred - np.asarray(y_val, dtype=np.float64).ravel()) ** 2))
        self.history = {
            "train_loss": [] if train_scores is None else (-np.asarray(train_scores)).tolist(),
            "val_loss": [] if val_scores is None else (-np.asarray(val_scores)).tolist(),
            "val_loss_source": f"random {self.validation_fraction:g} of training windows",
            "chronological_val_mse": chrono_val,
            "n_epochs": n_iter,
            "best_epoch": best_epoch,
            "s_per_epoch": fit_s / max(n_iter, 1),
            "device": self.device,
            "n_features": self.n_features_,
            "stopped_early": n_iter < self.max_iter,
        }
        return self

    def predict(self, X: np.ndarray, exog: np.ndarray | None = None) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("gbt.predict called before fit.")
        design = self._design(X, exog)
        if design.shape[1] != self.n_features_:
            raise ValueError(
                f"gbt was fitted on {self.n_features_} features but got {design.shape[1]}; "
                "pass the same exog as at fit time."
            )
        return np.asarray(self.model_.predict(design), dtype=np.float64)

    @property
    def n_params(self) -> int:
        """Total number of leaves across all fitted trees (see the module docstring)."""
        if self.model_ is None:
            return 0
        predictors = getattr(self.model_, "_predictors", None)
        if not predictors:
            return 0
        return int(sum(tree.get_n_leaf_nodes() for stage in predictors for tree in stage))


register("gbt")(GBTForecaster)

__all__ = ["GBTForecaster"]
