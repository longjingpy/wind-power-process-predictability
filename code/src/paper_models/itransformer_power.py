"""
iTransformer baseline for farm-level multi-horizon power forecasting.

Reference:
- Yong Liu, Haixu Wu, Jiehui Xu, et al. "iTransformer: Inverted Transformers Are
  Effective for Time Series Forecasting", ICLR 2024.
  OpenReview: https://openreview.net/forum?id=JePfAI8fah
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class InvertedDataEmbedding(nn.Module):
    """Map input [B, L, C] into token sequence [B, C, D] by inverting L/C axes."""

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # iTransformer core idea: each variable is treated as a token.
        x = x.transpose(1, 2)  # [B, C, L]
        x = self.proj(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


@dataclass
class ITransformerPowerConfig:
    seq_len: int
    input_dim: int
    horizon: int
    d_model: int = 256
    d_ff: int = 512
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    head_hidden: int = 256


class ITransformerPowerNet(nn.Module):
    """Farm-level regressor head on top of iTransformer encoder."""

    def __init__(self, cfg: ITransformerPowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = InvertedDataEmbedding(seq_len=cfg.seq_len, d_model=cfg.d_model, dropout=cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.input_dim * cfg.d_model, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        h = self.embed(x)  # [B, C, D]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        h = h.reshape(h.shape[0], -1)
        return self.head(h)

