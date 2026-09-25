"""
Plain Transformer baseline for farm-level multi-horizon power forecasting.

References:
- Vaswani et al. (2017), "Attention Is All You Need", NeurIPS.
  https://doi.org/10.48550/arXiv.1706.03762
- Wen et al. (2023), "Transformers in Time Series: A Survey", IJCAI 2023.
  https://doi.org/10.24963/ijcai.2023/759
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SimpleTransformerPowerConfig:
    seq_len: int
    input_dim: int
    horizon: int
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1


class SimpleTransformerPowerNet(nn.Module):
    """A minimal time-token Transformer encoder with last-token readout."""

    def __init__(self, cfg: SimpleTransformerPowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(int(cfg.n_layers), 1))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        if x.dim() != 3:
            raise ValueError(f"SimpleTransformerPowerNet expects [B, T, C], got {tuple(x.shape)}")
        h = self.input_proj(x)
        pos = self.pos_embed[:, : h.shape[1], :]
        h = h + pos
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        return self.head(h)
