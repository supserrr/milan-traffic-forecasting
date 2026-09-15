"""Model 2: a dilated causal temporal convolutional network (Bai, Kolter and Koltun, 2018).

Architecturally distinct from recurrence: the window is read in parallel through a stack
of residual blocks whose dilations double at each level, so the receptive field is an
explicit design choice rather than something the model has to learn to remember. With
the defaults (six levels, kernel 3) the last output sees 253 steps, which covers the
144-step window the EDA argues for with room to spare.

Receptive field. Each block has two causal convolutions of kernel ``k`` at dilation
``d``, each reaching ``(k - 1) * d`` steps into the past, so a block adds
``2 * (k - 1) * d`` and the network as a whole sees ``1 + 2 * (k - 1) * sum(d_i)`` steps
with ``d_i = 2 ** i``. The property :attr:`TCNForecaster.receptive_field` evaluates that
formula and ``fit`` warns if it is shorter than the window it is given, because a window
longer than the receptive field is silently truncated by the architecture.

Weight normalisation uses ``torch.nn.utils.parametrizations.weight_norm`` (the
non-deprecated API), causal padding is applied on the left and chomped on the right, and
a 1x1 convolution carries the residual when the channel count changes (only in the first
block, where the input has one channel).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from ..training import LOG, TorchForecaster
from . import register


class _Chomp1d(nn.Module):
    """Drop the trailing ``n`` steps a symmetric padding added, restoring causality."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.n] if self.n > 0 else x


class _TemporalBlock(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation)),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(c_out, c_out, kernel_size, padding=pad, dilation=dilation)),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TCNNet(nn.Module):
    def __init__(
        self, channels: int, levels: int, kernel_size: int, dropout: float, exog_dim: int
    ) -> None:
        super().__init__()
        blocks = []
        c_in = 1
        for i in range(levels):
            blocks.append(_TemporalBlock(c_in, channels, kernel_size, 2**i, dropout))
            c_in = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(channels + exog_dim, 1)

    def forward(self, x: torch.Tensor, exog: torch.Tensor | None = None) -> torch.Tensor:
        h = self.tcn(x.transpose(1, 2))  # (B, 1, L) -> (B, C, L)
        feat = h[:, :, -1]
        if exog is not None:
            feat = torch.cat([feat, exog], dim=1)
        return self.head(feat).squeeze(-1)


def receptive_field(kernel_size: int, levels: int) -> int:
    """``1 + 2 * (k - 1) * sum(2 ** i for i in range(levels))``."""
    return 1 + 2 * (kernel_size - 1) * sum(2**i for i in range(levels))


class TCNForecaster(TorchForecaster):
    """Causal dilated residual TCN over ``(B, L, 1)``; last-step features + exog -> Linear."""

    name = "tcn"

    def __init__(
        self,
        channels: int = 32,
        levels: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_exog: bool = True,
        **torch_kwargs: Any,
    ) -> None:
        if channels < 1 or levels < 1 or kernel_size < 2:
            raise ValueError("channels and levels must be positive and kernel_size at least 2.")
        self.channels = channels
        self.levels = levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        super().__init__(use_exog=use_exog, **torch_kwargs)

    @property
    def receptive_field(self) -> int:
        return receptive_field(self.kernel_size, self.levels)

    def build_net(self, exog_dim: int) -> nn.Module:
        return _TCNNet(self.channels, self.levels, self.kernel_size, self.dropout, exog_dim)

    def fit(self, X, y, **kwargs):  # type: ignore[override]
        seq_len = int(X.shape[1])
        if self.receptive_field < seq_len:
            LOG.warning(
                "tcn: receptive field %d is shorter than seq_len %d; the oldest %d steps "
                "of every window cannot influence the output.",
                self.receptive_field,
                seq_len,
                seq_len - self.receptive_field,
            )
        return super().fit(X, y, **kwargs)


register("tcn")(TCNForecaster)

__all__ = ["TCNForecaster", "receptive_field"]
