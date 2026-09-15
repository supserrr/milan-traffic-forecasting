"""Model-agnostic training loop for the PyTorch forecasters.

The LSTM and the TCN differ in architecture and in nothing else: same optimiser, same
loss, same batch size, same early-stopping rule, same seed handling, same device. That
only holds if the loop is written once, so both models are thin subclasses of
:class:`TorchForecaster` that supply a network and inherit everything else. A difference
between their rows in the results table is then a difference between recurrence and
dilated convolution, not between two training scripts.

Three details are there for the report rather than for capability:

*float32 everywhere.* The processed store is float32 and MPS rejects float64 tensors, so
inputs are cast once on entry and never promoted. Predictions are returned as float64
numpy arrays because that is what the metrics expect, but no float64 tensor is created.

*Early stopping with best-state restore.* Validation loss is measured in the scaled
space the model trains in, on the chronological validation week, after every epoch.
The weights that produced the best validation loss are restored at the end, so
``history["best_epoch"]`` is the epoch whose weights were actually scored.

*Non-finite loss stops the run.* A NaN loss is the commonest way a sequence model fails.
The loop stops at the first non-finite batch loss, records ``history["diverged"]`` and
restores the best weights seen, so a diverged run produces finite predictions from its
last good state and a flag in the artefacts rather than a table row of NaN with no
explanation.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features import TARGET_CALENDAR_COLUMNS
from .utils.logging import get_logger
from .utils.seed import set_seed

LOG = get_logger(__name__)

LOSSES: dict[str, Callable[[], nn.Module]] = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
}


def resolve_device(device: str = "auto") -> str:
    """``"auto"`` picks ``mps`` when available, then ``cuda``, else ``cpu``.

    An explicit device is honoured only if it is available: silently falling back would
    record ``device: mps`` in a config for a run that trained on the CPU.
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device == "cpu":
        return "cpu"
    if device.startswith("mps") and not torch.backends.mps.is_available():
        raise ValueError("device='mps' requested but MPS is not available on this machine.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available on this machine.")
    return device


def synchronize(device: str | torch.device | None) -> None:
    """Block until queued work on ``device`` has finished. No-op on the CPU."""
    name = str(device or "cpu")
    if name.startswith("mps"):
        torch.mps.synchronize()
    elif name.startswith("cuda"):
        torch.cuda.synchronize()


class TorchForecaster:
    """fit/predict around any ``nn.Module`` mapping ``(window, exog) -> prediction``.

    The network receives the window as ``(B, L, 1)`` float32 and the exogenous features
    as ``(B, E)`` float32 or ``None``, and returns ``(B,)``. Subclasses override
    :meth:`build_net`; alternatively pass ``net_factory(exog_dim) -> nn.Module``.

    ``use_exog=False`` drops the calendar features before they reach the network, which
    is how the calendar ablation is run without changing the harness.
    """

    name: str = "torch"

    def __init__(
        self,
        *,
        net_factory: Callable[[int], nn.Module] | None = None,
        use_exog: bool = True,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 1e-4,
        grad_clip: float | None = 1.0,
        loss: str = "mse",
        weight_decay: float = 0.0,
        device: str = "auto",
        seed: int = 42,
        scheduler: str | None = "reduce_on_plateau",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        predict_batch_size: int = 1024,
    ) -> None:
        if loss not in LOSSES:
            raise ValueError(f"Unknown loss {loss!r}; expected one of {sorted(LOSSES)}.")
        if scheduler not in (None, "reduce_on_plateau"):
            raise ValueError(f"Unknown scheduler {scheduler!r}.")
        self.net_factory = net_factory
        self.use_exog = use_exog
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip = grad_clip
        self.loss_name = loss
        self.weight_decay = weight_decay
        self.device = resolve_device(device)
        self.seed = seed
        self.scheduler_name = scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.predict_batch_size = predict_batch_size
        self.net: nn.Module | None = None
        self.history: dict[str, Any] = {}
        self._exog_dim: int = len(TARGET_CALENDAR_COLUMNS) if use_exog else 0

    # ------------------------------------------------------------------ hooks

    def build_net(self, exog_dim: int) -> nn.Module:
        """Return the network for ``exog_dim`` exogenous columns (0 when none)."""
        if self.net_factory is None:
            raise NotImplementedError("Subclasses must override build_net or pass net_factory.")
        return self.net_factory(exog_dim)

    # ---------------------------------------------------------------- tensors

    @staticmethod
    def _windows(X: np.ndarray) -> torch.Tensor:
        a = np.asarray(X, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"Expected windows of shape (n, seq_len), got {a.shape}.")
        return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(-1)

    def _exog(self, exog: np.ndarray | None) -> torch.Tensor | None:
        if not self.use_exog or exog is None:
            return None
        a = np.asarray(exog, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"Expected exog of shape (n, k), got {a.shape}.")
        return torch.from_numpy(np.ascontiguousarray(a))

    def _forward(self, xb: torch.Tensor, eb: torch.Tensor | None) -> torch.Tensor:
        assert self.net is not None
        return self.net(xb.to(self.device), None if eb is None else eb.to(self.device))

    # -------------------------------------------------------------------- fit

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
    ) -> TorchForecaster:
        set_seed(self.seed)
        Xt = self._windows(X)
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32).ravel())
        Et = self._exog(exog)
        self._exog_dim = 0 if Et is None else int(Et.shape[1])
        self.net = self.build_net(self._exog_dim).to(self.device)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv = self._windows(X_val)
            yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32).ravel())
            Ev = self._exog(exog_val)
            if (Ev is None) != (Et is None):
                raise ValueError("exog and exog_val must both be given or both be None.")

        tensors = [Xt, yt] if Et is None else [Xt, yt, Et]
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )
        loss_fn = LOSSES[self.loss_name]()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = None
        if self.scheduler_name == "reduce_on_plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=self.scheduler_factor, patience=self.scheduler_patience
            )

        train_losses: list[float] = []
        val_losses: list[float] = []
        lrs: list[float] = []
        epoch_s: list[float] = []
        best = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(self.net.state_dict())
        bad = 0
        diverged = False
        stopped_early = False

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.perf_counter()
            self.net.train()
            total, count = 0.0, 0
            for batch in loader:
                xb, yb = batch[0], batch[1]
                eb = batch[2] if len(batch) > 2 else None
                opt.zero_grad(set_to_none=True)
                pred = self._forward(xb, eb)
                loss = loss_fn(pred, yb.to(self.device))
                if not torch.isfinite(loss):
                    diverged = True
                    break
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                opt.step()
                total += float(loss.detach()) * len(yb)
                count += len(yb)
            synchronize(self.device)
            if diverged:
                LOG.warning("%s: non-finite loss at epoch %d, stopping.", self.name, epoch)
                break
            train_loss = total / max(count, 1)
            val_loss = self._loss(Xv, yv, Ev, loss_fn) if has_val else train_loss
            epoch_s.append(time.perf_counter() - t0)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            lrs.append(float(opt.param_groups[0]["lr"]))
            if sched is not None:
                sched.step(val_loss)

            if val_loss < best - self.min_delta:
                best, best_epoch, bad = val_loss, epoch, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    stopped_early = True
                    break

        self.net.load_state_dict(best_state)
        self.net.eval()
        self.history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "lr": lrs,
            "epoch_s": epoch_s,
            "n_epochs": len(train_losses),
            "best_epoch": best_epoch,
            "best_val_loss": best if np.isfinite(best) else None,
            "s_per_epoch": float(np.mean(epoch_s)) if epoch_s else None,
            "device": self.device,
            "loss": self.loss_name,
            "stopped_early": stopped_early,
            "diverged": diverged,
            "monitor": "val_loss" if has_val else "train_loss",
        }
        return self

    @torch.no_grad()
    def _loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        E: torch.Tensor | None,
        loss_fn: nn.Module,
    ) -> float:
        assert self.net is not None
        self.net.eval()
        total = 0.0
        for i in range(0, len(X), self.predict_batch_size):
            xb = X[i : i + self.predict_batch_size]
            eb = None if E is None else E[i : i + self.predict_batch_size]
            pred = self._forward(xb, eb)
            total += float(loss_fn(pred, y[i : i + self.predict_batch_size].to(self.device))) * len(
                xb
            )
        return total / max(len(X), 1)

    # ---------------------------------------------------------------- predict

    @torch.no_grad()
    def predict(self, X: np.ndarray, exog: np.ndarray | None = None) -> np.ndarray:
        if self.net is None:
            raise RuntimeError(f"{self.name}.predict called before fit.")
        self.net.eval()
        Xt = self._windows(X)
        Et = self._exog(exog)
        if self._exog_dim and Et is None:
            raise ValueError(
                f"{self.name} was fitted with {self._exog_dim} exogenous columns; "
                "predict needs the matching exog."
            )
        if self._exog_dim == 0:
            Et = None
        chunks = []
        for i in range(0, len(Xt), self.predict_batch_size):
            xb = Xt[i : i + self.predict_batch_size]
            eb = None if Et is None else Et[i : i + self.predict_batch_size]
            chunks.append(self._forward(xb, eb).float().cpu())
        return torch.cat(chunks).numpy().astype(np.float64)

    @property
    def n_params(self) -> int:
        net = self.net if self.net is not None else self.build_net(self._exog_dim)
        return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


__all__ = ["LOSSES", "TorchForecaster", "resolve_device", "synchronize"]
