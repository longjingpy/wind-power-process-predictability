"""
GNN utilities for wind forecasting with dynamic graph weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


DEFAULT_FEATURE_COLUMNS = [
    "wind_speed",
    "power",
    "u10",
    "v10",
    "wind_speed_10",
    "u100",
    "v100",
    "t2m",
    "sp",
    "q",
    "rho",
    "z0",
    "elevation_m",
    "slope_deg",
    "landcover",
    "l_stability_norm",
    "is_stable",
    "l_bin",
]


@dataclass
class GraphParams:
    sigma_m: float = 1500.0
    power_p: float = 2.0
    z_scale: float = 200.0
    alpha: float = 0.1
    wake_k: float = 0.05
    rotor_d_m: float = 120.0
    beta: float = 0.7
    dropout: float = 0.1


class WindowedGraphDataset(Dataset):
    """Windowed dataset for GNN training."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Iterable[str],
        t_in: int,
        horizon: int,
        step_out: int,
        max_samples: int = 0,
    ) -> None:
        self.feature_columns = [c for c in feature_columns if c in df.columns]
        self.feature_columns = [c for c in self.feature_columns if not df[c].isna().all()]
        if not self.feature_columns:
            raise ValueError("No feature columns found for dataset.")

        self.turbine_ids = sorted(df["turbine_id"].unique().tolist())
        turbine_index = {tid: idx for idx, tid in enumerate(self.turbine_ids)}

        times = pd.to_datetime(df["datetime"]).sort_values().unique()
        time_index = {t: idx for idx, t in enumerate(times)}

        t_count = len(times)
        n_turbines = len(self.turbine_ids)
        f_count = len(self.feature_columns)

        features = np.full((t_count, n_turbines, f_count), np.nan, dtype=np.float32)
        target_u = "u100"
        target_v = "v100"
        if target_u not in df.columns or target_v not in df.columns:
            if "u140" in df.columns and "v140" in df.columns:
                target_u = "u140"
                target_v = "v140"
                print("Warning: using u140/v140 as uv targets (u100/v100 missing).")
            else:
                raise ValueError("Missing u100/v100 columns for uv targets.")
        wind_uv = np.full((t_count, n_turbines, 2), np.nan, dtype=np.float32)

        for _, row in df.iterrows():
            t_idx = time_index[pd.Timestamp(row["datetime"])]
            n_idx = turbine_index[row["turbine_id"]]
            features[t_idx, n_idx, :] = row[self.feature_columns].to_numpy(dtype=np.float32)
            wind_uv[t_idx, n_idx, 0] = float(row[target_u])
            wind_uv[t_idx, n_idx, 1] = float(row[target_v])

        feature_mean = np.nanmean(features.reshape(-1, f_count), axis=0)
        if np.isnan(feature_mean).any():
            feature_mean = np.nan_to_num(feature_mean, nan=0.0)
        nan_mask = np.isnan(features)
        if nan_mask.any():
            features[nan_mask] = np.take(feature_mean, np.where(nan_mask)[2])

        self.features = torch.from_numpy(features)
        self.wind_uv = torch.from_numpy(wind_uv)
        self.times = times
        self.t_in = t_in
        self.horizon = horizon
        self.step_out = step_out

        max_t = t_count - step_out * horizon
        indices = []
        for t_idx in range(t_in - 1, max_t):
            x_window = features[t_idx - t_in + 1 : t_idx + 1]
            y_steps = [t_idx + (i + 1) * step_out for i in range(horizon)]
            y_window = wind_uv[y_steps]
            if np.isnan(y_window).any():
                continue
            indices.append(t_idx)

        if max_samples and max_samples > 0 and len(indices) > max_samples:
            rng = np.random.default_rng(7)
            indices = rng.choice(indices, size=max_samples, replace=False).tolist()

        self.indices = sorted(indices)
        self.sample_times = [times[idx] for idx in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t_idx = self.indices[idx]
        x_seq = self.features[t_idx - self.t_in + 1 : t_idx + 1]
        y_steps = [t_idx + (i + 1) * self.step_out for i in range(self.horizon)]
        y_seq = self.wind_uv[y_steps].transpose(0, 1)
        return x_seq, y_seq


class GraphBuilder:
    """Build dynamic adjacency matrices."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        distance_m: torch.Tensor,
        bearing_rad: torch.Tensor,
        params: GraphParams,
    ) -> None:
        self.edge_index = edge_index
        self.distance_m = distance_m
        self.bearing_rad = bearing_rad
        self.params = params
        if edge_index.numel() == 0:
            self.n_nodes = 0
        else:
            self.n_nodes = int(edge_index.max().item()) + 1
        self.base_weight = torch.exp(-distance_m / max(params.sigma_m, 1.0))

    def static_adj(self) -> torch.Tensor:
        if self.n_nodes == 0:
            return torch.zeros((0, 0), device=self.distance_m.device)
        adj = torch.zeros((self.n_nodes, self.n_nodes), device=self.distance_m.device)
        src = self.edge_index[0]
        dst = self.edge_index[1]
        adj[src, dst] = self.base_weight
        adj = adj + torch.eye(self.n_nodes, device=adj.device)
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
        return adj

    def mask(self) -> torch.Tensor:
        if self.n_nodes == 0:
            return torch.zeros((0, 0), device=self.distance_m.device)
        mask = torch.zeros((self.n_nodes, self.n_nodes), device=self.distance_m.device)
        src = self.edge_index[0]
        dst = self.edge_index[1]
        mask[src, dst] = 1.0
        mask = mask + torch.eye(self.n_nodes, device=mask.device)
        return (mask > 0).float()

    def build(self, wind_dir: torch.Tensor, z0: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        batch, n_nodes = wind_dir.shape
        src = self.edge_index[0]
        dst = self.edge_index[1]

        base = self.base_weight.unsqueeze(0)
        cos_dir = torch.cos(wind_dir[:, src] - self.bearing_rad)
        cos_term = torch.clamp(cos_dir, min=0.0) ** self.params.power_p
        z0_term = torch.clamp(z0[:, src] * z0[:, dst], min=1e-6) ** self.params.alpha
        dz = torch.abs(elevation[:, src] - elevation[:, dst])
        topo_term = torch.exp(-dz / max(self.params.z_scale, 1.0))

        downwind = torch.clamp(self.distance_m.unsqueeze(0) * torch.clamp(cos_dir, min=0.0), min=0.0)
        rotor_d = max(self.params.rotor_d_m, 1.0)
        wake = 1.0 / (1.0 + 2.0 * self.params.wake_k * downwind / rotor_d) ** 2

        weight = base * cos_term * topo_term * z0_term * wake
        adj = torch.zeros((batch, n_nodes, n_nodes), device=wind_dir.device)
        adj[:, src, dst] = weight
        adj = adj + torch.eye(n_nodes, device=wind_dir.device).unsqueeze(0)
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
        return adj


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        h = torch.bmm(adj, h)
        return self.dropout(h)


class LearnedAdjacency(nn.Module):
    def __init__(self, n_nodes: int, adj_mask: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros((n_nodes, n_nodes)))
        self.register_buffer("adj_mask", adj_mask)

    def forward(self) -> torch.Tensor:
        weights = torch.nn.functional.softplus(self.logits) * self.adj_mask
        weights = weights + torch.eye(weights.shape[0], device=weights.device) * 1e-6
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        return weights


class GnnModelA(nn.Module):
    """Per-turbine output: shape (B, N, H, 2)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        horizon: int,
        lstm_layers: int,
        gnn_layers: int,
        dropout: float,
        n_nodes: int,
        beta: float,
        adj_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.gconvs = nn.ModuleList([GraphConv(hidden_dim, hidden_dim, dropout) for _ in range(gnn_layers)])
        self.learned_adj = LearnedAdjacency(n_nodes, adj_mask)
        self.beta = beta
        self.horizon = horizon
        self.head_uv = nn.Linear(hidden_dim, horizon * 2)

    def forward(self, x_seq: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_learned = self.learned_adj()
        adj = self.beta * adj + (1.0 - self.beta) * adj_learned.unsqueeze(0)
        batch, t_len, n_nodes, feat = x_seq.shape
        x_seq = x_seq.permute(0, 2, 1, 3).reshape(batch * n_nodes, t_len, feat)
        h_seq, _ = self.lstm(x_seq)
        h_last = h_seq[:, -1, :].reshape(batch, n_nodes, -1)
        h_graph = h_last
        for conv in self.gconvs:
            h_next = torch.relu(conv(h_graph, adj))
            h_graph = h_graph + h_next
        uv = self.head_uv(h_graph).reshape(batch, n_nodes, self.horizon, 2)
        return uv


class GnnModelB(nn.Module):
    """Farm-level output: shape (B, H, 2)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        horizon: int,
        lstm_layers: int,
        gnn_layers: int,
        dropout: float,
        n_nodes: int,
        beta: float,
        adj_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.gconvs = nn.ModuleList([GraphConv(hidden_dim, hidden_dim, dropout) for _ in range(gnn_layers)])
        self.learned_adj = LearnedAdjacency(n_nodes, adj_mask)
        self.beta = beta
        self.attn = nn.Linear(hidden_dim, 1)
        self.horizon = horizon
        self.head_uv = nn.Linear(hidden_dim, horizon * 2)

    def forward(self, x_seq: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_learned = self.learned_adj()
        adj = self.beta * adj + (1.0 - self.beta) * adj_learned.unsqueeze(0)
        batch, t_len, n_nodes, feat = x_seq.shape
        x_seq = x_seq.permute(0, 2, 1, 3).reshape(batch * n_nodes, t_len, feat)
        h_seq, _ = self.lstm(x_seq)
        h_last = h_seq[:, -1, :].reshape(batch, n_nodes, -1)
        h_graph = h_last
        for conv in self.gconvs:
            h_next = torch.relu(conv(h_graph, adj))
            h_graph = h_graph + h_next
        scores = self.attn(h_graph).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = (h_graph * weights.unsqueeze(-1)).sum(dim=1)
        uv = self.head_uv(pooled).reshape(batch, self.horizon, 2)
        return uv


def build_graph_tensors(graph_df: pd.DataFrame, turbine_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = {tid: idx for idx, tid in enumerate(turbine_ids)}
    src = graph_df["src"].map(index).to_numpy(dtype=np.int64)
    dst = graph_df["dst"].map(index).to_numpy(dtype=np.int64)
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0))
    distance = torch.from_numpy(graph_df["distance_m"].to_numpy(dtype=np.float32))
    bearing = torch.from_numpy(graph_df["bearing_rad"].to_numpy(dtype=np.float32))
    return edge_index, distance, bearing
