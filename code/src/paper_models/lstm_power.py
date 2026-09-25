"""
Minimal LSTM baseline for farm-level power forecasting.

References:
- Hochreiter & Schmidhuber (1997), "Long Short-Term Memory", Neural
  Computation. https://doi.org/10.1162/neco.1997.9.8.1735
- Lei et al. (2013), "A review on the forecasting of wind speed and generated
  power", Renewable and Sustainable Energy Reviews.
  https://doi.org/10.1016/j.rser.2013.08.062
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LSTMPowerConfig:
    seq_len: int
    input_dim: int
    horizon: int
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1


class LSTMPowerNet(nn.Module):
    """A minimal stacked LSTM with last-state readout for sequence regression."""

    def __init__(self, cfg: LSTMPowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_norm = nn.LayerNorm(cfg.input_dim)
        self.encoder = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=max(int(cfg.n_layers), 1),
            dropout=float(cfg.dropout) if int(cfg.n_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(cfg.hidden_dim, cfg.horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"LSTMPowerNet expects [B, T, C], got {tuple(x.shape)}")
        x = self.input_norm(x)
        _, (h_n, _) = self.encoder(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden)
