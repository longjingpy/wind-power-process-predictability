"""
Windformer adapters.

Windformer keeps the Rainformer U-shape spatial backbone as the starting point,
then upgrades farm_v2 forecasting with deeper temporal encoders, cross-attention
fusion, graph-logit bias, and gated farm-head fusion.

References:
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
  https://arxiv.org/abs/1706.03762
- Tsai et al., "Multimodal Transformer for Unaligned Multimodal Language Sequences",
  ACL 2019.
  https://arxiv.org/abs/1906.00295
- Ying et al., "Do Transformers Really Perform Bad for Graph Representation?",
  NeurIPS 2021.
  https://arxiv.org/abs/2106.05234
- Arevalo et al., "Gated Multimodal Units for Information Fusion", ICLR Workshop 2017.
  https://arxiv.org/abs/1702.01992
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .rainformer_adapter import (
    RainformerBackbone,
    RainformerPowerFarmNet,
    RainformerPowerFarmV2Net,
    StageModule,
    StageModule_up,
    StageModule_up_final,
    _check_divisible,
    _require_rainformer,
    _validate_stage_hw,
    node_features_to_grid,
)


def febm_to_stage_layers(febm_per_stage: Sequence[int]) -> list[int]:
    """
    Convert FEBM counts to Swin stage `layers`.

    In current StageModule implementation, one FEBM unit corresponds to
    one pair of regular/shifted Swin blocks, so:
    - FEBM count k -> layers 2*k
    """
    out: list[int] = []
    for k in febm_per_stage:
        kk = max(int(k), 1)
        out.append(kk * 2)
    return out


def _pair(value: Sequence[int] | str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, str):
        parts = [int(x.strip()) for x in value.split(",") if x.strip()]
    else:
        parts = [int(x) for x in value]
    if len(parts) != 2:
        raise ValueError(f"Expected 2 integers for pair, got: {value}")
    return int(parts[0]), int(parts[1])


class WindformerPowerFarmNet(RainformerPowerFarmNet):
    """
    Legacy Windformer farm-output model.

    This path intentionally stays lightweight for legacy experiments.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = "windformer_power"
        self.identity = nn.Identity()


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.pos = nn.Parameter(torch.zeros(1, self.max_len, int(dim)))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.max_len:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max_len={self.max_len}.")
        return x + self.pos[:, : x.shape[1]]


class AttentionSummaryPool(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, int(dim)) * 0.02)
        self.attn = nn.MultiheadAttention(int(dim), max(int(heads), 1), batch_first=True)
        self.norm = nn.LayerNorm(int(dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        return self.norm(pooled.squeeze(1))


class LeadQueryReadout(nn.Module):
    """
    Lead-conditioned query readout over temporal tokens.
    """

    def __init__(self, dim: int, heads: int, max_target_step: int = 256) -> None:
        super().__init__()
        self.max_target_step = max(int(max_target_step), 1)
        self.query = nn.Parameter(torch.randn(1, 1, int(dim)) * 0.02)
        self.step_embed = nn.Embedding(self.max_target_step + 1, int(dim))
        self.attn = nn.MultiheadAttention(int(dim), max(int(heads), 1), batch_first=True)
        self.norm = nn.LayerNorm(int(dim))

    def forward(self, tokens: torch.Tensor, target_step: int) -> torch.Tensor:
        step_idx = int(max(min(int(target_step), self.max_target_step), 1))
        base_query = self.query.expand(tokens.shape[0], -1, -1)
        step_query = self.step_embed.weight[step_idx].view(1, 1, -1).expand(tokens.shape[0], -1, -1)
        query = base_query + step_query
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        return self.norm(pooled.squeeze(1))


class TemporalTokenEncoder(nn.Module):
    """
    Transformer encoder for farm or GNSS time tokens.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(model_dim))
        self.pos = LearnedPositionalEncoding(int(model_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(model_dim),
            nhead=max(int(heads), 1),
            dim_feedforward=max(int(model_dim) * 4, 64),
            dropout=float(max(dropout, 0.0)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(int(layers), 1))
        self.summary = AttentionSummaryPool(int(model_dim), max(int(heads), 1))
        self.norm = nn.LayerNorm(int(model_dim))

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.input_proj(seq)
        tokens = self.pos(tokens)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        summary = self.summary(tokens)
        return tokens, summary


class LSTMTemporalTokenEncoder(nn.Module):
    """
    LSTM encoder for farm-level temporal tokens.

    LSTM follows:
    - Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Computation 1997.
      https://doi.org/10.1162/neco.1997.9.8.1735
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(model_dim))
        self.encoder = nn.LSTM(
            input_size=int(model_dim),
            hidden_size=int(model_dim),
            num_layers=max(int(layers), 1),
            dropout=float(max(dropout, 0.0)) if int(layers) > 1 else 0.0,
            batch_first=True,
        )
        self.summary = AttentionSummaryPool(int(model_dim), 1)
        self.norm = nn.LayerNorm(int(model_dim))

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.input_proj(seq)
        tokens, _ = self.encoder(tokens)
        tokens = self.norm(tokens)
        summary = self.summary(tokens)
        return tokens, summary


def _normalize_direction_vectors(x: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
    x = x / norm
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class GNSSFlowEncoder(nn.Module):
    """
    Station-structured GNSS encoder with a multi-step farm-level flow head.

    The station-first then temporal-encoder path follows the GNSS/water-vapor
    flow-prediction idea in:
    - Zhang et al. (2025), arXiv 2509.16068.
    - Lee et al. (2025), ESS Open Archive wind/water-vapor forecasting preprint.
    """

    def __init__(
        self,
        station_input_dim: int,
        station_static_dim: int,
        model_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        horizon: int,
        temporal_readout: str = "summary",
        target_step: int = 1,
        station_weight_idx: int = -1,
        max_target_step: int = 256,
        direction_only: bool = False,
    ) -> None:
        super().__init__()
        self.station_input_dim = int(station_input_dim)
        self.station_static_dim = int(station_static_dim)
        self.model_dim = int(model_dim)
        self.horizon = int(horizon)
        self.temporal_readout = str(temporal_readout)
        self.target_step = int(max(target_step, 1))
        self.station_weight_idx = int(station_weight_idx)
        self.direction_only = bool(direction_only)
        station_hidden = max(int(hidden_dim), int(model_dim), 32)
        self.station_mlp = nn.Sequential(
            nn.Linear(self.station_input_dim + self.station_static_dim + 1, station_hidden),
            nn.GELU(),
            nn.Dropout(float(max(dropout, 0.0))),
            nn.Linear(station_hidden, self.model_dim),
        )
        self.temporal = TemporalTokenEncoder(
            input_dim=self.model_dim,
            model_dim=self.model_dim,
            heads=int(heads),
            layers=int(layers),
            dropout=float(dropout),
        )
        self.lead_readout = (
            LeadQueryReadout(int(self.model_dim), max(int(heads), 1), max_target_step=max_target_step)
            if self.temporal_readout == "lead_query"
            else None
        )
        self.flow_head = nn.Sequential(
            nn.Linear(self.model_dim, max(self.model_dim, 64)),
            nn.GELU(),
            nn.Dropout(float(max(dropout, 0.0))),
            nn.Linear(max(self.model_dim, 64), self.horizon * 2),
        )

    def forward(
        self,
        station_x: torch.Tensor,
        station_mask: torch.Tensor,
        station_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if station_x.dim() != 4:
            raise ValueError("gnss_station_x must have shape (B, T, S, C).")
        batch, steps, station_count, _ = station_x.shape
        if station_mask.dim() == 2:
            station_mask = station_mask.unsqueeze(0).expand(batch, -1, -1)
        if station_mask.dim() != 3:
            raise ValueError("gnss_station_mask must have shape (B, T, S) or (T, S).")
        if station_static.dim() == 2:
            station_static = station_static.unsqueeze(0).expand(batch, -1, -1)
        if station_static.dim() != 3:
            raise ValueError("gnss_station_static must have shape (B, S, M) or (S, M).")
        static_rep = station_static.unsqueeze(1).expand(-1, steps, -1, -1)
        mask = station_mask.to(dtype=station_x.dtype)
        mask_feat = mask.unsqueeze(-1)
        station_input = torch.cat([station_x, static_rep, mask_feat], dim=-1)
        station_tokens = self.station_mlp(station_input)
        station_tokens = station_tokens * mask_feat
        agg_weight = mask
        if 0 <= self.station_weight_idx < station_x.shape[-1]:
            station_weight = torch.clamp(station_x[..., self.station_weight_idx], min=0.0) * mask
            has_weight = station_weight.sum(dim=2, keepdim=True) > 0
            agg_weight = torch.where(has_weight, station_weight, mask)
        agg_weight_feat = agg_weight.unsqueeze(-1)
        denom = agg_weight.sum(dim=2, keepdim=True).clamp(min=1.0)
        time_tokens = (station_tokens * agg_weight_feat).sum(dim=2) / denom
        flow_tokens, flow_summary = self.temporal(time_tokens)
        readout = (
            self.lead_readout(flow_tokens, self.target_step)
            if self.temporal_readout == "lead_query" and self.lead_readout is not None
            else flow_summary
        )
        flow_uv_hat = self.flow_head(readout).view(batch, self.horizon, 2)
        if self.direction_only:
            flow_uv_hat = _normalize_direction_vectors(flow_uv_hat)
        return flow_tokens, readout, flow_uv_hat


class DynamicGraphBiasBuilder(nn.Module):
    """
    Build horizon-wise graph bias maps from predicted GNSS flow direction.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_attr_columns: Sequence[str],
        flat_idx: torch.Tensor,
        count_flat: torch.Tensor,
        grid_h: int,
        grid_w: int,
        coverage_threshold: float = 0.3,
        sigma_m: float = 1500.0,
        z_scale: float = 200.0,
        roughness_alpha: float = 0.1,
        direction_power: float = 1.5,
    ) -> None:
        super().__init__()
        self.register_buffer("edge_index", edge_index.long(), persistent=False)
        self.register_buffer("edge_attr", edge_attr.float(), persistent=False)
        self.register_buffer("flat_idx", flat_idx.long(), persistent=False)
        self.register_buffer("count_flat", count_flat.float(), persistent=False)
        self.edge_attr_columns = tuple(str(x) for x in edge_attr_columns)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.coverage_threshold = float(coverage_threshold)
        self.sigma_m = float(max(sigma_m, 1.0))
        self.z_scale = float(max(z_scale, 1.0))
        self.roughness_alpha = float(max(roughness_alpha, 0.0))
        self.direction_power = float(max(direction_power, 1.0))

    def _base_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.edge_attr.numel() == 0:
            return torch.ones((self.edge_index.shape[1],), dtype=dtype, device=device)
        col_to_idx = {name: idx for idx, name in enumerate(self.edge_attr_columns)}
        weight = torch.ones((self.edge_attr.shape[0],), dtype=dtype, device=device)
        if "distance_m" in col_to_idx:
            dist = self.edge_attr[:, col_to_idx["distance_m"]].to(device=device, dtype=dtype)
            weight = weight * torch.exp(-dist / self.sigma_m)
        if "elevation_diff_m" in col_to_idx:
            dz = self.edge_attr[:, col_to_idx["elevation_diff_m"]].to(device=device, dtype=dtype)
            weight = weight * torch.exp(-torch.abs(dz) / self.z_scale)
        if "roughness_pair" in col_to_idx:
            rough = self.edge_attr[:, col_to_idx["roughness_pair"]].to(device=device, dtype=dtype)
            weight = weight * torch.clamp(rough, min=1e-6).pow(self.roughness_alpha)
        if "wake_prior" in col_to_idx:
            wake = self.edge_attr[:, col_to_idx["wake_prior"]].to(device=device, dtype=dtype)
            weight = weight * torch.clamp(wake, min=0.0)
        return weight

    def forward(
        self,
        last_power: torch.Tensor,
        flow_uv_hat: torch.Tensor,
        gnss_coverage: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, horizon, _ = flow_uv_hat.shape
        if self.edge_index.numel() == 0 or last_power.numel() == 0:
            return torch.zeros(
                (batch, horizon, self.grid_h, self.grid_w),
                dtype=last_power.dtype,
                device=last_power.device,
            )
        src = self.edge_index[0].long()
        dst = self.edge_index[1].long()
        col_to_idx = {name: idx for idx, name in enumerate(self.edge_attr_columns)}
        bearing = None
        if "bearing_rad" in col_to_idx:
            bearing = self.edge_attr[:, col_to_idx["bearing_rad"]].to(device=last_power.device, dtype=last_power.dtype)
        base_weight = self._base_weight(last_power.dtype, last_power.device).view(1, 1, -1)
        power_src = torch.clamp(last_power[:, src], min=0.0)
        power_scale = power_src / power_src.mean(dim=1, keepdim=True).clamp(min=1e-6)
        messages = power_src.unsqueeze(1) * torch.clamp(power_scale.unsqueeze(1), min=0.5, max=2.5)
        if bearing is not None:
            flow_dir = torch.atan2(flow_uv_hat[:, :, 1], flow_uv_hat[:, :, 0])
            delta = flow_dir.unsqueeze(-1) - bearing.view(1, 1, -1)
            directional = ((torch.cos(delta) + 1.0) * 0.5).clamp(min=0.0).pow(self.direction_power)
        else:
            directional = torch.ones_like(base_weight.expand(batch, horizon, -1))
        edge_weight = base_weight * directional
        edge_messages = messages * edge_weight
        influence = torch.zeros(
            (batch, horizon, last_power.shape[1]),
            dtype=last_power.dtype,
            device=last_power.device,
        )
        dst_idx = dst.view(1, 1, -1).expand(batch, horizon, -1)
        influence.scatter_add_(2, dst_idx, edge_messages)
        bias_maps: list[torch.Tensor] = []
        for lead in range(horizon):
            bias_maps.append(
                node_features_to_grid(
                    influence[:, lead, :].unsqueeze(-1),
                    self.flat_idx,
                    self.count_flat,
                    self.grid_h,
                    self.grid_w,
                )
            )
        bias = torch.cat(bias_maps, dim=1)
        if gnss_coverage is not None and gnss_coverage.numel() > 0:
            if gnss_coverage.dim() == 1:
                cov_mean = gnss_coverage.unsqueeze(0)
            else:
                cov_mean = gnss_coverage
            cov_mean = cov_mean.to(dtype=last_power.dtype, device=last_power.device).mean(dim=1, keepdim=True)
            keep = (cov_mean >= self.coverage_threshold).to(dtype=last_power.dtype).view(batch, 1, 1, 1)
            bias = bias * keep
        return bias


class SpatialTokenProjector(nn.Module):
    def __init__(self, in_channels: int, token_dim: int, pool_hw: tuple[int, int]) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_hw)
        self.proj = nn.Linear(int(in_channels), int(token_dim))
        self.norm = nn.LayerNorm(int(token_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x)
        tokens = pooled.flatten(2).transpose(1, 2)
        return self.norm(self.proj(tokens))


class BiasedSelfAttention(nn.Module):
    """
    Multi-head self-attention with additive logits bias.

    The additive bias follows the Graphormer-style idea of injecting structural
    priors into attention logits instead of only concatenating graph channels.
    """

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.heads = max(int(heads), 1)
        if self.dim % self.heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}.")
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.out = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(float(max(dropout, 0.0)))

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attn_bias is not None:
            scores = scores + attn_bias.unsqueeze(1)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, length, self.dim)
        return self.out(out)


class TokenSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(int(dim))
        self.attn = BiasedSelfAttention(int(dim), int(heads), float(dropout))
        self.norm2 = nn.LayerNorm(int(dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(dim), max(int(dim) * 4, 64)),
            nn.GELU(),
            nn.Dropout(float(max(dropout, 0.0))),
            nn.Linear(max(int(dim) * 4, 64), int(dim)),
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention from spatial tokens to conditioning tokens.
    """

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(int(dim))
        self.context_norm = nn.LayerNorm(int(dim))
        self.attn = nn.MultiheadAttention(int(dim), max(int(heads), 1), batch_first=True, dropout=float(max(dropout, 0.0)))
        self.out_norm = nn.LayerNorm(int(dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(dim), max(int(dim) * 4, 64)),
            nn.GELU(),
            nn.Dropout(float(max(dropout, 0.0))),
            nn.Linear(max(int(dim) * 4, 64), int(dim)),
        )

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(query_tokens)
        c = self.context_norm(context_tokens)
        out, _ = self.attn(q, c, c, need_weights=False)
        query_tokens = query_tokens + out
        query_tokens = query_tokens + self.mlp(self.out_norm(query_tokens))
        return query_tokens


class GatedFusionHead(nn.Module):
    """
    Farm-dominant fusion head for farm-level prediction.
    """

    def __init__(self, spatial_dim: int, farm_dim: int, gnss_dim: int, fusion_hidden: int, output_channels: int, dropout: float) -> None:
        super().__init__()
        self.farm_proj = nn.Linear(int(farm_dim), int(fusion_hidden))
        aux_dim = int(spatial_dim + max(int(gnss_dim), 0))
        gate_in = int(farm_dim + aux_dim)
        self.aux_proj = nn.Linear(aux_dim, int(fusion_hidden))
        self.gate = nn.Linear(gate_in, int(fusion_hidden))
        self.out = nn.Sequential(
            nn.Linear(int(fusion_hidden + gate_in), max(int(fusion_hidden), int(output_channels))),
            nn.GELU(),
            nn.Dropout(float(max(dropout, 0.0))),
            nn.Linear(max(int(fusion_hidden), int(output_channels)), int(output_channels)),
        )

    def forward(self, spatial_summary: torch.Tensor, farm_summary: torch.Tensor, gnss_summary: torch.Tensor | None = None) -> torch.Tensor:
        aux_parts = [spatial_summary]
        if gnss_summary is not None:
            aux_parts.append(gnss_summary)
        aux_context = torch.cat(aux_parts, dim=-1)
        farm_hidden = self.farm_proj(farm_summary)
        aux_hidden = self.aux_proj(aux_context)
        aux_gate = torch.sigmoid(self.gate(torch.cat([farm_summary, aux_context], dim=-1)))
        fused = farm_hidden + aux_gate * aux_hidden
        return self.out(torch.cat([fused, farm_summary, aux_context], dim=-1))


@dataclass(frozen=True)
class WindformerBackboneOutputs:
    bottleneck: torch.Tensor
    decoder: torch.Tensor
    final: torch.Tensor


class WindformerMultiscaleBackbone(nn.Module):
    """
    Rainformer-style U-shape backbone that exposes bottleneck and decoder maps.
    """

    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        downscaling_factors: Sequence[int],
        layers: Sequence[int],
        heads: Sequence[int],
        head_dim: int,
        window_size: int,
        relative_pos_embedding: bool,
        input_hw: tuple[int, int],
    ) -> None:
        super().__init__()
        _require_rainformer()
        if len(downscaling_factors) != 4 or len(layers) != 4 or len(heads) != 4:
            raise ValueError("Windformer expects 4-stage configurations.")

        h, w = input_hw
        d0, d1, d2, d3 = map(int, downscaling_factors)
        _check_divisible(h, d0, "input_h")
        _check_divisible(w, d0, "input_w")
        h1, w1 = h // d0, w // d0
        _check_divisible(h1, d1, "stage2_h")
        _check_divisible(w1, d1, "stage2_w")
        h2, w2 = h1 // d1, w1 // d1
        _check_divisible(h2, d2, "stage3_h")
        _check_divisible(w2, d2, "stage3_w")
        h3, w3 = h2 // d2, w2 // d2
        _check_divisible(h3, d3, "stage4_h")
        _check_divisible(w3, d3, "stage4_w")
        h4, w4 = h3 // d3, w3 // d3

        _validate_stage_hw((h1, w1), window_size, "stage1")
        _validate_stage_hw((h2, w2), window_size, "stage2")
        _validate_stage_hw((h3, w3), window_size, "stage3")
        _validate_stage_hw((h4, w4), window_size, "stage4")

        self.stage1 = StageModule(
            in_channels=input_channel,
            hidden_dimension=hidden_dim,
            layers=layers[0],
            downscaling_factor=d0,
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=[h1, w1],
        )
        self.stage2 = StageModule(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim * 2,
            layers=layers[1],
            downscaling_factor=d1,
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=[h2, w2],
        )
        self.stage3 = StageModule(
            in_channels=hidden_dim * 2,
            hidden_dimension=hidden_dim * 4,
            layers=layers[2],
            downscaling_factor=d2,
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=[h3, w3],
        )
        self.stage4 = StageModule(
            in_channels=hidden_dim * 4,
            hidden_dimension=hidden_dim * 8,
            layers=layers[3],
            downscaling_factor=d3,
            num_heads=heads[3],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=[h4, w4],
        )
        self.stage5 = StageModule_up(
            in_channels=hidden_dim * 8,
            hidden_dimension=hidden_dim * 4,
            layers=layers[3],
            upscaling_factor=d3,
            num_heads=heads[3],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=[h3, w3],
        )
        self.stage6 = StageModule_up(
            in_channels=hidden_dim * 8,
            hidden_dimension=hidden_dim * 2,
            layers=layers[2],
            upscaling_factor=d2,
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=[h2, w2],
        )
        self.stage7 = StageModule_up(
            in_channels=hidden_dim * 4,
            hidden_dimension=hidden_dim,
            layers=layers[1],
            upscaling_factor=d1,
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=[h1, w1],
        )
        self.stage8 = StageModule_up_final(
            in_channels=hidden_dim * 2,
            hidden_dimension=input_channel,
            layers=layers[0],
            upscaling_factor=d0,
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=[h, w],
        )

    def forward(self, x: torch.Tensor) -> WindformerBackboneOutputs:
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4, x3)
        x6 = self.stage6(x5, x2)
        x7 = self.stage7(x6, x1)
        x8 = self.stage8(x7)
        return WindformerBackboneOutputs(bottleneck=x4, decoder=x7, final=x8)


class WindformerPowerFarmV2Net(nn.Module):
    """
    Windformer farm_v2 model with:
    - multiscale spatial tokens,
    - temporal Transformer encoders for farm/GNSS,
    - graph logits bias on spatial token self-attention,
    - cross-attention at bottleneck + decoder,
    - gated farm-output head.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        input_hw: tuple[int, int],
        farm_input_dim: int,
        gnss_input_dim: int = 0,
        gnss_mode: str = "raw",
        graph_mode: str = "static",
        horizon: int = 16,
        gnss_station_input_dim: int = 0,
        gnss_station_static_dim: int = 0,
        hidden_dim: int = 64,
        downscaling_factors: Sequence[int] = (2, 2, 2, 2),
        layers: Sequence[int] = (2, 2, 2, 2),
        heads: Sequence[int] = (1, 2, 4, 8),
        head_dim: int = 16,
        window_size: int = 9,
        relative_pos_embedding: bool = True,
        fusion_hidden: int = 128,
        farm_hidden: int = 128,
        gnss_hidden: int = 64,
        head_dropout: float = 0.1,
        token_dim: int = 128,
        temporal_heads: int = 4,
        temporal_layers: int = 2,
        cross_attn_heads: int = 4,
        spatial_pool_hw: tuple[int, int] = (3, 3),
        graph_logit_bias: bool = True,
        farm_temporal_backbone: str = "transformer",
        temporal_readout: str = "summary",
        target_step: int = 1,
        gnss_station_weight_idx: int = -1,
        graph_edge_index: torch.Tensor | None = None,
        graph_edge_attr: torch.Tensor | None = None,
        graph_edge_attr_columns: Sequence[str] = (),
        grid_flat_idx: torch.Tensor | None = None,
        grid_count_flat: torch.Tensor | None = None,
        graph_coverage_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.gnss_mode = str(gnss_mode)
        self.graph_mode = str(graph_mode)
        self.gnss_input_dim = int(gnss_input_dim)
        self.gnss_station_input_dim = int(gnss_station_input_dim)
        self.gnss_station_static_dim = int(gnss_station_static_dim)
        self.horizon = int(horizon)
        self.graph_logit_bias = bool(graph_logit_bias)
        self.spatial_pool_hw = _pair(spatial_pool_hw)
        self.farm_temporal_backbone = str(farm_temporal_backbone)
        self.temporal_readout = str(temporal_readout)
        self.target_step = int(max(target_step, 1))
        self.gnss_station_weight_idx = int(gnss_station_weight_idx)
        if self.farm_temporal_backbone not in {"transformer", "lstm"}:
            raise ValueError("farm_temporal_backbone must be one of {'transformer', 'lstm'}.")
        if self.temporal_readout not in {"summary", "lead_query", "auto"}:
            raise ValueError("temporal_readout must be one of {'summary', 'lead_query', 'auto'}.")
        self.use_lead_query = bool(
            self.temporal_readout == "lead_query"
            or (self.temporal_readout == "auto" and self.target_step >= 5)
        )
        self.effective_temporal_readout = "lead_query" if self.use_lead_query else "summary"
        if self.use_lead_query and self.horizon != 1:
            raise ValueError("lead_query temporal readout requires horizon=1 for per-step windformer forecasting.")
        self.backbone = WindformerMultiscaleBackbone(
            input_channel=input_channels,
            hidden_dim=hidden_dim,
            downscaling_factors=downscaling_factors,
            layers=layers,
            heads=heads,
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            input_hw=input_hw,
        )
        self.bottleneck_tokens = SpatialTokenProjector(hidden_dim * 8, int(token_dim), self.spatial_pool_hw)
        self.decoder_tokens = SpatialTokenProjector(hidden_dim * 2, int(token_dim), self.spatial_pool_hw)
        self.bottleneck_self = TokenSelfAttentionBlock(int(token_dim), int(cross_attn_heads), float(head_dropout))
        self.decoder_self = TokenSelfAttentionBlock(int(token_dim), int(cross_attn_heads), float(head_dropout))
        self.bottleneck_farm = CrossAttentionBlock(int(token_dim), int(cross_attn_heads), float(head_dropout))
        self.decoder_farm = CrossAttentionBlock(int(token_dim), int(cross_attn_heads), float(head_dropout))
        if self.gnss_mode == "flow":
            self.bottleneck_gnss = CrossAttentionBlock(int(token_dim), int(cross_attn_heads), float(head_dropout))
            self.decoder_gnss = CrossAttentionBlock(int(token_dim), int(cross_attn_heads), float(head_dropout))
            self.gnss_encoder = GNSSFlowEncoder(
                station_input_dim=self.gnss_station_input_dim,
                station_static_dim=self.gnss_station_static_dim,
                model_dim=int(token_dim),
                hidden_dim=int(gnss_hidden),
                heads=int(temporal_heads),
                layers=int(temporal_layers),
                dropout=float(head_dropout),
                horizon=self.horizon,
                temporal_readout=self.effective_temporal_readout,
                target_step=self.target_step,
                station_weight_idx=self.gnss_station_weight_idx,
            )
        elif self.gnss_input_dim > 0:
            self.bottleneck_gnss = CrossAttentionBlock(int(token_dim), int(cross_attn_heads), float(head_dropout))
            self.decoder_gnss = CrossAttentionBlock(int(token_dim), int(cross_attn_heads), float(head_dropout))
            self.gnss_encoder = TemporalTokenEncoder(
                input_dim=self.gnss_input_dim,
                model_dim=int(token_dim),
                heads=int(temporal_heads),
                layers=int(temporal_layers),
                dropout=float(head_dropout),
            )
        else:
            self.bottleneck_gnss = None
            self.decoder_gnss = None
            self.gnss_encoder = None
        if self.farm_temporal_backbone == "lstm":
            self.farm_encoder = LSTMTemporalTokenEncoder(
                input_dim=int(farm_input_dim),
                model_dim=int(token_dim),
                layers=int(temporal_layers),
                dropout=float(head_dropout),
            )
        else:
            self.farm_encoder = TemporalTokenEncoder(
                input_dim=int(farm_input_dim),
                model_dim=int(token_dim),
                heads=int(temporal_heads),
                layers=int(temporal_layers),
                dropout=float(head_dropout),
            )
        self.farm_readout = (
            LeadQueryReadout(int(token_dim), max(int(temporal_heads), 1))
            if self.use_lead_query
            else None
        )
        self.gnss_readout = (
            LeadQueryReadout(int(token_dim), max(int(temporal_heads), 1))
            if self.use_lead_query and self.gnss_mode != "flow" and self.gnss_encoder is not None
            else None
        )
        self.spatial_summary = AttentionSummaryPool(int(token_dim), max(int(cross_attn_heads), 1))
        self.head = GatedFusionHead(
            spatial_dim=int(token_dim) * 2,
            farm_dim=int(token_dim),
            gnss_dim=0,
            fusion_hidden=max(int(fusion_hidden), int(farm_hidden), int(token_dim)),
            output_channels=int(output_channels),
            dropout=float(head_dropout),
        )
        if self.graph_mode == "flow_dynamic":
            if graph_edge_index is None or graph_edge_attr is None or grid_flat_idx is None or grid_count_flat is None:
                raise ValueError("flow_dynamic graph mode requires edge tensors and grid mapping buffers.")
            self.dynamic_graph_builder = DynamicGraphBiasBuilder(
                edge_index=graph_edge_index,
                edge_attr=graph_edge_attr,
                edge_attr_columns=graph_edge_attr_columns,
                flat_idx=grid_flat_idx,
                count_flat=grid_count_flat,
                grid_h=int(input_hw[0]),
                grid_w=int(input_hw[1]),
                coverage_threshold=float(graph_coverage_threshold),
            )
        else:
            self.dynamic_graph_builder = None

    def _graph_attn_bias(self, graph_bias_map: torch.Tensor | None, token_count: int) -> torch.Tensor | None:
        if not self.graph_logit_bias or graph_bias_map is None or graph_bias_map.numel() == 0:
            return None
        pooled = nn.functional.adaptive_avg_pool2d(graph_bias_map, self.spatial_pool_hw)
        if pooled.shape[1] > 1:
            pooled = pooled.mean(dim=1, keepdim=True)
        token_bias = pooled.flatten(2).squeeze(1)
        if token_bias.shape[1] != token_count:
            token_bias = token_bias[:, :token_count]
        token_bias = torch.tanh(token_bias)
        return 0.5 * (token_bias.unsqueeze(2) + token_bias.unsqueeze(1))

    def _fuse_spatial_tokens(
        self,
        tokens: torch.Tensor,
        farm_tokens: torch.Tensor,
        gnss_tokens: torch.Tensor | None,
        self_block: TokenSelfAttentionBlock,
        farm_block: CrossAttentionBlock,
        gnss_block: CrossAttentionBlock | None,
        attn_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = self_block(tokens, attn_bias=attn_bias)
        tokens = farm_block(tokens, farm_tokens)
        if gnss_tokens is not None and gnss_block is not None:
            tokens = gnss_block(tokens, gnss_tokens)
        return tokens

    def forward(
        self,
        x_grid: torch.Tensor,
        farm_seq: torch.Tensor,
        gnss_seq: torch.Tensor | None = None,
        graph_bias_map: torch.Tensor | None = None,
        gnss_station_x: torch.Tensor | None = None,
        gnss_station_mask: torch.Tensor | None = None,
        gnss_station_static: torch.Tensor | None = None,
        gnss_coverage: torch.Tensor | None = None,
        turbine_last_power: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        if self.gnss_mode == "flow" and self.dynamic_graph_builder is not None:
            if gnss_station_x is None or gnss_station_mask is None or gnss_station_static is None:
                raise ValueError("flow GNSS mode requires gnss_station_x/mask/static tensors.")
            if turbine_last_power is None:
                raise ValueError("flow_dynamic graph mode requires turbine_last_power.")
            flow_tokens_pre, flow_summary_pre, flow_uv_hat = self.gnss_encoder(
                gnss_station_x,
                gnss_station_mask,
                gnss_station_static,
            )
            graph_bias_map = self.dynamic_graph_builder(
                last_power=turbine_last_power,
                flow_uv_hat=flow_uv_hat,
                gnss_coverage=gnss_coverage,
            )
            x_grid = torch.cat([x_grid, graph_bias_map], dim=1)
            aux["flow_uv_hat"] = flow_uv_hat
            aux["graph_bias_map"] = graph_bias_map
        else:
            flow_tokens_pre = None
            flow_summary_pre = None
        feats = self.backbone(x_grid)
        farm_tokens, farm_summary_base = self.farm_encoder(farm_seq)
        farm_summary = (
            self.farm_readout(farm_tokens, self.target_step)
            if self.use_lead_query and self.farm_readout is not None
            else farm_summary_base
        )
        gnss_tokens = None
        gnss_summary = None
        if self.gnss_encoder is not None:
            if self.gnss_mode == "flow":
                if flow_tokens_pre is None or flow_summary_pre is None:
                    if gnss_station_x is None or gnss_station_mask is None or gnss_station_static is None:
                        raise ValueError("flow GNSS mode requires gnss_station_x/mask/static tensors.")
                    gnss_tokens, gnss_summary, flow_uv_hat = self.gnss_encoder(
                        gnss_station_x,
                        gnss_station_mask,
                        gnss_station_static,
                    )
                    aux["flow_uv_hat"] = flow_uv_hat
                else:
                    gnss_tokens, gnss_summary = flow_tokens_pre, flow_summary_pre
            else:
                if gnss_seq is None:
                    raise ValueError("gnss_seq is required when gnss_input_dim > 0.")
                gnss_tokens, gnss_summary_base = self.gnss_encoder(gnss_seq)
                gnss_summary = (
                    self.gnss_readout(gnss_tokens, self.target_step)
                    if self.use_lead_query and self.gnss_readout is not None
                    else gnss_summary_base
                )

        bottleneck_tokens = self.bottleneck_tokens(feats.bottleneck)
        decoder_tokens = self.decoder_tokens(feats.decoder)
        bottleneck_bias = self._graph_attn_bias(graph_bias_map, bottleneck_tokens.shape[1])
        decoder_bias = self._graph_attn_bias(graph_bias_map, decoder_tokens.shape[1])

        bottleneck_tokens = self._fuse_spatial_tokens(
            bottleneck_tokens,
            farm_tokens,
            gnss_tokens,
            self.bottleneck_self,
            self.bottleneck_farm,
            self.bottleneck_gnss,
            bottleneck_bias,
        )
        decoder_tokens = self._fuse_spatial_tokens(
            decoder_tokens,
            farm_tokens,
            gnss_tokens,
            self.decoder_self,
            self.decoder_farm,
            self.decoder_gnss,
            decoder_bias,
        )

        spatial_summary = torch.cat(
            [self.spatial_summary(bottleneck_tokens), self.spatial_summary(decoder_tokens)],
            dim=-1,
        )
        # GNSS information is injected into the spatial tokens and only serves as
        # auxiliary context for the final farm-level prediction head.
        pred = self.head(spatial_summary, farm_summary, None)
        if return_aux:
            return pred, aux
        return pred


def build_farm_v2_power_model(
    arch: str,
    *,
    input_channels: int,
    output_channels: int,
    input_hw: tuple[int, int],
    farm_input_dim: int,
    gnss_input_dim: int,
    gnss_mode: str = "raw",
    graph_mode: str = "static",
    horizon: int = 16,
    gnss_station_input_dim: int = 0,
    gnss_station_static_dim: int = 0,
    hidden_dim: int,
    downscaling_factors: Sequence[int],
    layers: Sequence[int],
    heads: Sequence[int],
    head_dim: int,
    window_size: int,
    relative_pos_embedding: bool,
    fusion_hidden: int,
    farm_hidden: int,
    gnss_hidden: int,
    head_dropout: float,
    token_dim: int = 128,
    temporal_heads: int = 4,
    temporal_layers: int = 2,
    cross_attn_heads: int = 4,
    spatial_pool_hw: tuple[int, int] = (3, 3),
    graph_logit_bias: bool = True,
    farm_temporal_backbone: str = "transformer",
    temporal_readout: str = "summary",
    target_step: int = 1,
    gnss_station_weight_idx: int = -1,
    graph_edge_index: torch.Tensor | None = None,
    graph_edge_attr: torch.Tensor | None = None,
    graph_edge_attr_columns: Sequence[str] = (),
    grid_flat_idx: torch.Tensor | None = None,
    grid_count_flat: torch.Tensor | None = None,
    graph_coverage_threshold: float = 0.3,
) -> nn.Module:
    if arch == "rainformer_power":
        return RainformerPowerFarmV2Net(
            input_channels=input_channels,
            output_channels=output_channels,
            input_hw=input_hw,
            farm_input_dim=farm_input_dim,
            gnss_input_dim=gnss_input_dim,
            hidden_dim=hidden_dim,
            downscaling_factors=downscaling_factors,
            layers=layers,
            heads=heads,
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            fusion_hidden=fusion_hidden,
            farm_hidden=farm_hidden,
            gnss_hidden=gnss_hidden,
            head_dropout=head_dropout,
        )
    if arch == "windformer_power":
        return WindformerPowerFarmV2Net(
            input_channels=input_channels,
            output_channels=output_channels,
            input_hw=input_hw,
            farm_input_dim=farm_input_dim,
            gnss_input_dim=gnss_input_dim,
            gnss_mode=gnss_mode,
            graph_mode=graph_mode,
            horizon=horizon,
            gnss_station_input_dim=gnss_station_input_dim,
            gnss_station_static_dim=gnss_station_static_dim,
            hidden_dim=hidden_dim,
            downscaling_factors=downscaling_factors,
            layers=layers,
            heads=heads,
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            fusion_hidden=fusion_hidden,
            farm_hidden=farm_hidden,
            gnss_hidden=gnss_hidden,
            head_dropout=head_dropout,
            token_dim=token_dim,
            temporal_heads=temporal_heads,
            temporal_layers=temporal_layers,
            cross_attn_heads=cross_attn_heads,
            spatial_pool_hw=spatial_pool_hw,
            graph_logit_bias=graph_logit_bias,
            farm_temporal_backbone=farm_temporal_backbone,
            temporal_readout=temporal_readout,
            target_step=target_step,
            gnss_station_weight_idx=gnss_station_weight_idx,
            graph_edge_index=graph_edge_index,
            graph_edge_attr=graph_edge_attr,
            graph_edge_attr_columns=graph_edge_attr_columns,
            grid_flat_idx=grid_flat_idx,
            grid_count_flat=grid_count_flat,
            graph_coverage_threshold=graph_coverage_threshold,
        )
    raise ValueError(f"Unsupported farm_v2 power arch: {arch}")
