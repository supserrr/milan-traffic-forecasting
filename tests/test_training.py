"""The torch training loop.

Every trained model in the comparison runs through :class:`TorchForecaster`, so a defect
here moves two of the three result columns at once and would be invisible in a results
table. These tests pin the properties the report claims: that fitting reduces the loss,
that early stopping restores the best epoch rather than the last one, that nothing
float64 reaches the device (MPS rejects it outright), and that the history the timing
table is built from is actually populated.

Everything runs on the CPU with a tiny synthetic series, so the suite stays fast and
gives the same answer on a machine without an accelerator.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from milan_traffic.models.lstm import LSTMForecaster
from milan_traffic.models.tcn import TCNForecaster
from milan_traffic.training import TorchForecaster, resolve_device, synchronize

SEQ_LEN = 24


@pytest.fixture
def windows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """400 windows of a noisy period-24 sine, plus the five-column exog block."""
    rng = np.random.default_rng(0)
    t = np.arange(400 + SEQ_LEN)
    series = np.sin(2 * np.pi * t / SEQ_LEN) + rng.normal(0, 0.05, t.size)
    X = np.stack([series[i : i + SEQ_LEN] for i in range(400)]).astype(np.float32)
    y = series[SEQ_LEN : SEQ_LEN + 400].astype(np.float32)
    exog = np.zeros((400, 5), dtype=np.float32)
    exog[:, 4] = 1.0
    return X, y, exog


def _model(cls, **kw):
    return cls(max_epochs=3, patience=10, device="cpu", batch_size=64, **kw)


@pytest.mark.parametrize("cls", [LSTMForecaster, TCNForecaster])
def test_fit_reduces_the_training_loss(windows, cls):
    X, y, exog = windows
    model = _model(cls).fit(X, y, X_val=X[:100], y_val=y[:100], exog=exog, exog_val=exog[:100])
    losses = model.history["train_loss"]
    assert len(losses) == 3
    assert losses[-1] < losses[0], f"{cls.__name__} did not learn: {losses}"


@pytest.mark.parametrize("cls", [LSTMForecaster, TCNForecaster])
def test_predict_shape_and_dtype(windows, cls):
    X, y, exog = windows
    model = _model(cls).fit(X, y, X_val=X[:100], y_val=y[:100], exog=exog, exog_val=exog[:100])
    pred = model.predict(X[:37], exog[:37])
    assert pred.shape == (37,)
    assert pred.dtype == np.float64  # scoring happens in float64, on the CPU
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("cls", [LSTMForecaster, TCNForecaster])
def test_history_carries_what_the_timing_table_reports(windows, cls):
    X, y, exog = windows
    model = _model(cls).fit(X, y, X_val=X[:100], y_val=y[:100], exog=exog, exog_val=exog[:100])
    for key in ("n_epochs", "best_epoch", "s_per_epoch", "device", "stopped_early", "val_loss"):
        assert key in model.history, key
    assert model.history["device"] == "cpu"
    assert model.history["n_epochs"] == 3
    assert model.history["s_per_epoch"] > 0
    assert model.n_params > 0


def test_early_stopping_fires_and_restores_the_best_epoch(windows):
    """Patience 1 on a validation set the model cannot improve must stop early.

    The restored weights must be the best epoch's, not the last one's: otherwise early
    stopping selects a model and then reports a different one.
    """
    X, y, exog = windows
    model = LSTMForecaster(max_epochs=40, patience=1, device="cpu", batch_size=64, lr=5e-2).fit(
        X, y, X_val=X[:50], y_val=np.full(50, 5.0, dtype=np.float32), exog=exog, exog_val=exog[:50]
    )
    assert model.history["stopped_early"] is True
    assert model.history["n_epochs"] < 40
    best = model.history["best_epoch"]
    assert model.history["val_loss"][best - 1] == pytest.approx(model.history["best_val_loss"])
    assert min(model.history["val_loss"]) == pytest.approx(model.history["best_val_loss"])


def test_no_float64_tensor_reaches_the_device(windows, monkeypatch):
    """MPS rejects float64 outright, so a float64 window is a crash on the real device.

    The check runs on the CPU, where float64 is legal and would therefore pass silently;
    intercepting ``Tensor.to`` is what makes the omission visible here.
    """
    X, y, exog = windows
    seen: list[torch.dtype] = []
    original = torch.Tensor.to

    def spy(self, *args, **kwargs):
        seen.append(self.dtype)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", spy)
    model = _model(LSTMForecaster).fit(
        X.astype(np.float64), y.astype(np.float64), X_val=X[:50], y_val=y[:50]
    )
    model.predict(X[:10].astype(np.float64))
    assert torch.float64 not in seen, "a float64 tensor was moved to the device"


def test_resolve_device_honours_an_explicit_cpu():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "mps", "cuda"}
    synchronize("cpu")  # must be a no-op rather than an error


def test_a_forecaster_subclass_must_build_a_net():
    """The base class is abstract in behaviour: it cannot fit without a net."""
    with pytest.raises((NotImplementedError, TypeError)):
        TorchForecaster(max_epochs=1, device="cpu").fit(
            np.zeros((8, SEQ_LEN), np.float32), np.zeros(8, np.float32)
        )
