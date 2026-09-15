"""Model 1: a single-layer LSTM over the 144-step window.

The reference deep sequence model in the cellular-traffic literature, including work on
this dataset, so a reader can place the result. Its gating is built for carrying a
regime (the afternoon ramp, the weekend collapse) across many steps without a hand-set
lag structure, which is the property the EDA says matters most on square 5259.

The final hidden state of the last layer is concatenated with the five calendar features
of the target slot and mapped to one output by a linear layer. With the defaults that is
about 17 000 parameters. Training, early stopping and prediction are inherited from
:class:`milan_traffic.training.TorchForecaster`, so the LSTM and the TCN differ in
nothing but the network.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..training import TorchForecaster
from . import register


class _LSTMNet(nn.Module):
    def __init__(self, hidden: int, layers: int, dropout: float, exog_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            # nn.LSTM applies dropout between layers only and warns if asked to with one
            # layer, so the argument is passed through only when it can take effect.
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden + exog_dim, 1)

    def forward(self, x: torch.Tensor, exog: torch.Tensor | None = None) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        feat = h[-1]
        if exog is not None:
            feat = torch.cat([feat, exog], dim=1)
        return self.head(feat).squeeze(-1)


class LSTMForecaster(TorchForecaster):
    """``nn.LSTM(batch_first=True)`` over ``(B, L, 1)``; last hidden state + exog -> Linear."""

    name = "lstm"

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.0,
        use_exog: bool = True,
        **torch_kwargs: Any,
    ) -> None:
        if hidden < 1 or layers < 1:
            raise ValueError("hidden and layers must be positive.")
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        super().__init__(use_exog=use_exog, **torch_kwargs)

    def build_net(self, exog_dim: int) -> nn.Module:
        return _LSTMNet(self.hidden, self.layers, self.dropout, exog_dim)


register("lstm")(LSTMForecaster)

__all__ = ["LSTMForecaster"]
