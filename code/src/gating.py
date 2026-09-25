"""
Gating utilities for stability-aware mixture-of-experts.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


GATE_FEATURES = [
    "wind_speed",
    "u100",
    "v100",
    "t2m",
    "sp",
    "rho",
    "z0",
]

SPEED_BINS = [0.0, 5.0, 10.0, 15.0]


class GatingNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_experts: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def build_gating_features(
    x_seq: torch.Tensor,
    feature_columns: Iterable[str],
    gate_features: Iterable[str] | None = None,
) -> torch.Tensor:
    feature_columns = list(feature_columns)
    gate_features = list(gate_features or GATE_FEATURES)

    x_last = x_seq[:, -1]
    feat_idx = [feature_columns.index(name) for name in gate_features if name in feature_columns]
    if feat_idx:
        base = torch.nanmean(x_last[:, :, feat_idx], dim=1)
    else:
        base = torch.zeros((x_last.shape[0], 0), device=x_last.device)

    if "l_stability_norm" in feature_columns:
        l_idx = feature_columns.index("l_stability_norm")
    elif "l_stability" in feature_columns:
        l_idx = feature_columns.index("l_stability")
    else:
        l_idx = None
    if l_idx is not None:
        l_mean = torch.nanmean(torch.abs(x_last[:, :, l_idx]), dim=1, keepdim=True)
    else:
        l_mean = torch.zeros((x_last.shape[0], 1), device=x_last.device)

    speed = None
    if "wind_speed_100" in feature_columns:
        idx = feature_columns.index("wind_speed_100")
        speed = torch.nanmean(x_last[:, :, idx], dim=1, keepdim=True)
    elif "wind_speed" in feature_columns:
        idx = feature_columns.index("wind_speed")
        speed = torch.nanmean(x_last[:, :, idx], dim=1, keepdim=True)
    elif "u100" in feature_columns and "v100" in feature_columns:
        u_idx = feature_columns.index("u100")
        v_idx = feature_columns.index("v100")
        u_mean = torch.nanmean(x_last[:, :, u_idx], dim=1, keepdim=True)
        v_mean = torch.nanmean(x_last[:, :, v_idx], dim=1, keepdim=True)
        speed = torch.sqrt(u_mean**2 + v_mean**2 + 1e-8)
    else:
        speed = torch.zeros((x_last.shape[0], 1), device=x_last.device)

    speed_log = torch.log1p(torch.clamp(speed, min=0.0))
    bins = torch.tensor(SPEED_BINS, device=x_last.device)
    bin_idx = torch.bucketize(speed.squeeze(1), bins)
    n_bins = len(SPEED_BINS) + 1
    speed_onehot = torch.zeros((x_last.shape[0], n_bins), device=x_last.device)
    speed_onehot.scatter_(1, bin_idx.unsqueeze(1), 1.0)

    return torch.cat([l_mean, base, speed_log, speed_onehot], dim=1)
