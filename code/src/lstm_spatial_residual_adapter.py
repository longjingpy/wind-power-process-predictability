"""
LSTM-first farm_v2 adapter with a lightweight spatial residual branch.

This branch keeps the temporal model as the main forecasting path and uses a
GNSS-flow-informed spatial encoder only as a residual correction path.

References:
- Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Computation 1997.
  https://doi.org/10.1162/neco.1997.9.8.1735
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
  https://arxiv.org/abs/1706.03762
- Tsai et al., "Multimodal Transformer for Unaligned Multimodal Language Sequences",
  ACL 2019. https://arxiv.org/abs/1906.00295
- Ying et al., "Do Transformers Really Perform Bad for Graph Representation?",
  NeurIPS 2021. https://arxiv.org/abs/2106.05234
- Arevalo et al., "Gated Multimodal Units for Information Fusion", ICLR Workshop
  2017. https://arxiv.org/abs/1702.01992
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .gnss_prior import build_gnss_prior_tensor_config, estimate_station_prior_tensor
from .gnn import GraphConv, LearnedAdjacency
from .paper_models import GNSSPriorRuntimeBundle, apply_gnss_prior_runtime_bundle
from .windformer_adapter import (
    AttentionSummaryPool,
    CrossAttentionBlock,
    DynamicGraphBiasBuilder,
    GNSSFlowEncoder,
    LearnedPositionalEncoding,
    SpatialTokenProjector,
    TokenSelfAttentionBlock,
)


def perturb_graph_direction(
    flow_uv: torch.Tensor,
    mode: str = "none",
    *,
    constant_direction_deg: float = 0.0,
    shuffle_seed: int = 20260720,
) -> torch.Tensor:
    """Apply an inference-only intervention to graph-path direction vectors."""
    mode = str(mode or "none").strip().lower()
    if mode == "none":
        return flow_uv
    if flow_uv.ndim != 3 or flow_uv.shape[-1] != 2:
        raise ValueError(f"Expected graph direction [B,H,2], got {tuple(flow_uv.shape)}")
    if mode == "origin_shuffled":
        if int(flow_uv.shape[0]) < 2:
            raise ValueError("origin_shuffled requires an inference batch with at least two origins.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(shuffle_seed))
        permutation = torch.randperm(int(flow_uv.shape[0]), generator=generator).to(flow_uv.device)
        return flow_uv.index_select(0, permutation)

    angle_deg = {"rotated_90": 90.0, "rotated_180": 180.0}.get(mode)
    if mode == "constant":
        angle_deg = float(constant_direction_deg)
    if angle_deg is None:
        raise ValueError(
            "graph direction perturbation must be one of "
            "{'none', 'origin_shuffled', 'rotated_90', 'rotated_180', 'constant'}."
        )
    angle = torch.as_tensor(angle_deg * torch.pi / 180.0, dtype=flow_uv.dtype, device=flow_uv.device)
    if mode == "constant":
        magnitude = torch.linalg.vector_norm(flow_uv, dim=-1)
        return torch.stack((magnitude * torch.cos(angle), magnitude * torch.sin(angle)), dim=-1)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    return torch.stack((u * cos_angle - v * sin_angle, u * sin_angle + v * cos_angle), dim=-1)


class GEGLUFeedForward(nn.Module):
    """
    GEGLU feed-forward block for stronger token mixing than a plain MLP.

    GEGLU follows gated Transformer feed-forward practice used in modern sequence
    models to improve expressiveness while keeping the residual path stable.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.in_proj = nn.Linear(int(dim), int(hidden_dim) * 2)
        self.out_proj = nn.Linear(int(hidden_dim), int(dim))
        self.dropout = nn.Dropout(float(max(dropout, 0.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        value, gate = self.in_proj(h).chunk(2, dim=-1)
        h = value * F.gelu(gate)
        h = self.dropout(h)
        h = self.out_proj(h)
        return self.dropout(h)


class ResidualCausalLSTMBlock(nn.Module):
    """
    Residual uni-directional LSTM block.

    The block keeps the LSTM causal and adds Transformer-style residual + FFN
    structure so the main temporal trunk remains modern without changing the
    fundamental recurrent inductive bias.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.lstm = nn.LSTM(
            input_size=int(dim),
            hidden_size=int(dim),
            num_layers=1,
            batch_first=True,
        )
        self.out_proj = nn.Linear(int(dim), int(dim))
        self.dropout = nn.Dropout(float(max(dropout, 0.0)))
        self.ffn = GEGLUFeedForward(int(dim), max(int(dim) * 4, 128), float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h, _ = self.lstm(h)
        x = x + self.dropout(self.out_proj(h))
        x = x + self.ffn(x)
        return x


class ResidualCausalLSTMEncoder(nn.Module):
    """Project sequence inputs to shared tokens and encode them with residual LSTM blocks."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        layers: int,
        dropout: float,
        summary_heads: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(model_dim))
        self.pos = LearnedPositionalEncoding(int(model_dim))
        self.blocks = nn.ModuleList(
            [ResidualCausalLSTMBlock(int(model_dim), float(dropout)) for _ in range(max(int(layers), 1))]
        )
        self.norm = nn.LayerNorm(int(model_dim))
        self.summary = AttentionSummaryPool(int(model_dim), max(int(summary_heads), 1))

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.input_proj(seq)
        tokens = self.pos(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        summary = self.summary(tokens)
        return tokens, summary


class MultiLeadQueryReadout(nn.Module):
    """Lead-conditioned readout for shared temporal tokens."""

    def __init__(self, dim: int, heads: int, max_target_step: int = 256) -> None:
        super().__init__()
        self.max_target_step = max(int(max_target_step), 1)
        self.query = nn.Parameter(torch.randn(1, 1, int(dim)) * 0.02)
        self.step_embed = nn.Embedding(self.max_target_step + 1, int(dim))
        self.attn = nn.MultiheadAttention(int(dim), max(int(heads), 1), batch_first=True)
        self.norm = nn.LayerNorm(int(dim))

    def forward(self, tokens: torch.Tensor, start_step: int, horizon: int) -> torch.Tensor:
        batch = tokens.shape[0]
        device = tokens.device
        step_idx = torch.arange(int(horizon), device=device, dtype=torch.long) + int(max(start_step, 1))
        step_idx = torch.clamp(step_idx, min=1, max=self.max_target_step)
        base_query = self.query.expand(batch, int(horizon), -1)
        step_query = self.step_embed(step_idx).unsqueeze(0).expand(batch, -1, -1)
        query = base_query + step_query
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        return self.norm(pooled)


class ConvNormGELU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, stride=int(stride), padding=1, bias=False),
            nn.BatchNorm2d(int(out_channels)),
            nn.GELU(),
            nn.Dropout2d(float(max(dropout, 0.0))),
            nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(out_channels)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass(frozen=True)
class SpatialResidualBackboneOutputs:
    bottleneck: torch.Tensor
    decoder: torch.Tensor


class SpatialResidual2p2Backbone(nn.Module):
    """
    Lightweight 2-down + 2-up spatial backbone with skip concatenation.

    The branch intentionally stays shallower than the Windformer U-shape so the
    temporal trunk remains dominant and the spatial path acts only as a residual
    correction path.
    """

    def __init__(self, input_channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        hidden = max(int(hidden_dim), 16)
        self.stem = ConvNormGELU(int(input_channels), hidden, stride=1, dropout=float(dropout))
        self.enc1 = ConvNormGELU(hidden, hidden, stride=2, dropout=float(dropout))
        self.enc2 = ConvNormGELU(hidden, hidden * 2, stride=2, dropout=float(dropout))
        self.up1 = nn.ConvTranspose2d(hidden * 2, hidden, kernel_size=2, stride=2)
        self.dec1 = ConvNormGELU(hidden * 2, hidden, stride=1, dropout=float(dropout))
        self.up2 = nn.ConvTranspose2d(hidden, hidden, kernel_size=2, stride=2)
        self.dec2 = ConvNormGELU(hidden * 2, hidden, stride=1, dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> SpatialResidualBackboneOutputs:
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        y1 = self.up1(x2)
        if y1.shape[-2:] != x1.shape[-2:]:
            y1 = F.interpolate(y1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))
        y2 = self.up2(y1)
        if y2.shape[-2:] != x0.shape[-2:]:
            y2 = F.interpolate(y2, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.dec2(torch.cat([y2, x0], dim=1))
        return SpatialResidualBackboneOutputs(bottleneck=x2, decoder=y2)


class LSTMSpatialResidualFarmV2Net(nn.Module):
    """
    LSTM-first farm_v2 model with GNSS-flow-informed spatial residual correction.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        input_hw: tuple[int, int],
        farm_input_dim: int,
        farm_feature_columns: Sequence[str] = (),
        gnss_input_dim: int = 0,
        gnss_mode: str = "raw",
        graph_mode: str = "static",
        horizon: int = 16,
        gnss_station_input_dim: int = 0,
        gnss_station_static_dim: int = 0,
        lstm_main_hidden_dim: int = 256,
        lstm_main_layers: int = 3,
        lstm_main_dropout: float = 0.1,
        lstm_cov_hidden_dim: int = 128,
        lstm_cov_layers: int = 2,
        lstm_cross_attn_heads: int = 4,
        lstm_readout_heads: int = 4,
        spatial_aux_hidden_dim: int = 64,
        spatial_aux_dropout: float = 0.1,
        graph_logit_bias: bool = True,
        target_step: int = 1,
        gnss_station_weight_idx: int = -1,
        gnss_station_feature_columns: Sequence[str] = (),
        graph_edge_index: torch.Tensor | None = None,
        graph_edge_attr: torch.Tensor | None = None,
        graph_edge_attr_columns: Sequence[str] = (),
        grid_flat_idx: torch.Tensor | None = None,
        grid_count_flat: torch.Tensor | None = None,
        graph_coverage_threshold: float = 0.3,
        gnss_prior_source: str = "strict_upwind",
        gnss_prior_top_k: int = 3,
        gnss_prior_confidence_min: float = 0.25,
        gnss_prior_confidence_power: float = 1.0,
        gnss_prior_confidence_gating: bool = True,
        gnss_prior_calibration_state: dict[str, object] | None = None,
        gnss_prior_checkpoint: str | None = None,
        gnss_prior_model_bundle: GNSSPriorRuntimeBundle | None = None,
        gnss_prior_perturbation: str = "none",
        gnss_prior_dummy_direction_deg: float = 0.0,
        gnss_graph_direction_perturbation: str = "none",
        gnss_graph_direction_constant_deg: float = 0.0,
        gnss_graph_direction_shuffle_seed: int = 20260720,
        gnss_flow_direction_only: bool = False,
        gnss_flow_active_steps: int = 0,
        residual_gate_init_bias: float = -2.0,
        branch_mode: str = "spatial_residual",
        turbine_node_input_dim: int = 0,
        enable_spatial_residual: bool = True,
    ) -> None:
        super().__init__()
        branch_mode = str(branch_mode or "spatial_residual")
        if branch_mode not in {"temporal_only", "gnss_gnn", "spatial_residual"}:
            raise ValueError("branch_mode must be one of {'temporal_only', 'gnss_gnn', 'spatial_residual'}.")
        if not bool(enable_spatial_residual) and branch_mode == "spatial_residual":
            branch_mode = "temporal_only"
        self.gnss_mode = str(gnss_mode)
        self.graph_mode = str(graph_mode)
        self.horizon = int(horizon)
        self.target_step = int(max(target_step, 1))
        self.graph_logit_bias = bool(graph_logit_bias)
        self.branch_mode = branch_mode
        self.enable_spatial_residual = bool(branch_mode == "spatial_residual")
        self.enable_gnss_gnn = bool(branch_mode == "gnss_gnn")
        self.model_dim = int(lstm_main_hidden_dim)
        self.point_channels = max(int(output_channels) // max(self.horizon, 1), 1)
        self.gnss_station_weight_idx = int(gnss_station_weight_idx)
        self.gnss_prior_source = str(gnss_prior_source or "strict_upwind")
        self.gnss_prior_top_k = max(int(gnss_prior_top_k), 1)
        self.gnss_prior_confidence_min = float(max(gnss_prior_confidence_min, 0.0))
        self.gnss_prior_confidence_power = float(max(gnss_prior_confidence_power, 1e-6))
        self.gnss_prior_confidence_gating = bool(gnss_prior_confidence_gating)
        self.gnss_prior_calibration_state = dict(gnss_prior_calibration_state or {}) or None
        self.gnss_prior_checkpoint = str(gnss_prior_checkpoint or "") or None
        self.gnss_prior_model_bundle = gnss_prior_model_bundle
        self.gnss_prior_perturbation = str(gnss_prior_perturbation or "none").strip().lower()
        self.gnss_prior_dummy_direction_deg = float(gnss_prior_dummy_direction_deg)
        self.gnss_graph_direction_perturbation = str(gnss_graph_direction_perturbation or "none").strip().lower()
        self.gnss_graph_direction_constant_deg = float(gnss_graph_direction_constant_deg)
        self.gnss_graph_direction_shuffle_seed = int(gnss_graph_direction_shuffle_seed)
        self.gnss_flow_direction_only = bool(gnss_flow_direction_only)
        self.gnss_flow_active_steps = int(max(gnss_flow_active_steps, 0))
        self.turbine_node_input_dim = int(max(turbine_node_input_dim, 0))
        self.gnss_prior_config = build_gnss_prior_tensor_config(tuple(gnss_station_feature_columns))
        self.farm_feature_columns = tuple(str(c) for c in farm_feature_columns)

        if self.gnss_mode not in {"raw", "flow"}:
            raise ValueError("gnss_mode must be 'raw' or 'flow'.")
        if self.graph_mode not in {"static", "flow_dynamic"}:
            raise ValueError("graph_mode must be 'static' or 'flow_dynamic'.")

        self.farm_encoder = ResidualCausalLSTMEncoder(
            input_dim=int(farm_input_dim),
            model_dim=self.model_dim,
            layers=int(lstm_main_layers),
            dropout=float(lstm_main_dropout),
            summary_heads=int(lstm_readout_heads),
        )
        self.farm_gnss_cross = (
            CrossAttentionBlock(self.model_dim, int(lstm_cross_attn_heads), float(lstm_main_dropout))
            if (self.gnss_mode == "raw" and int(gnss_input_dim) > 0)
            or (self.gnss_mode == "flow" and int(gnss_station_input_dim) > 0)
            else None
        )
        self.gnss_encoder: nn.Module | None
        if self.gnss_mode == "flow" and int(gnss_station_input_dim) > 0:
            flow_readout = "lead_query" if self.horizon == 1 else "summary"
            self.gnss_encoder = GNSSFlowEncoder(
                station_input_dim=int(gnss_station_input_dim),
                station_static_dim=int(gnss_station_static_dim),
                model_dim=self.model_dim,
                hidden_dim=max(int(lstm_cov_hidden_dim), self.model_dim),
                heads=int(lstm_readout_heads),
                layers=int(lstm_cov_layers),
                dropout=float(lstm_main_dropout),
                horizon=self.horizon,
                temporal_readout=flow_readout,
                target_step=self.target_step,
                station_weight_idx=self.gnss_station_weight_idx,
                direction_only=self.gnss_flow_direction_only,
            )
        elif self.gnss_mode == "raw" and int(gnss_input_dim) > 0:
            self.gnss_encoder = ResidualCausalLSTMEncoder(
                input_dim=int(gnss_input_dim),
                model_dim=self.model_dim,
                layers=int(lstm_cov_layers),
                dropout=float(lstm_main_dropout),
                summary_heads=int(lstm_readout_heads),
            )
        else:
            self.gnss_encoder = None

        self.temporal_summary = AttentionSummaryPool(self.model_dim, max(int(lstm_readout_heads), 1))
        self.readout = MultiLeadQueryReadout(self.model_dim, max(int(lstm_readout_heads), 1))
        self.base_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(float(max(lstm_main_dropout, 0.0))),
            nn.Linear(self.model_dim, self.point_channels),
        )

        self.spatial_backbone = SpatialResidual2p2Backbone(
            int(input_channels),
            int(spatial_aux_hidden_dim),
            float(spatial_aux_dropout),
        )
        self.bottleneck_tokens = SpatialTokenProjector(int(spatial_aux_hidden_dim) * 2, self.model_dim, (3, 3))
        self.decoder_tokens = SpatialTokenProjector(int(spatial_aux_hidden_dim), self.model_dim, (3, 3))
        self.bottleneck_self = TokenSelfAttentionBlock(self.model_dim, int(lstm_cross_attn_heads), float(spatial_aux_dropout))
        self.decoder_self = TokenSelfAttentionBlock(self.model_dim, int(lstm_cross_attn_heads), float(spatial_aux_dropout))
        self.bottleneck_farm = CrossAttentionBlock(self.model_dim, int(lstm_cross_attn_heads), float(spatial_aux_dropout))
        self.decoder_farm = CrossAttentionBlock(self.model_dim, int(lstm_cross_attn_heads), float(spatial_aux_dropout))
        self.bottleneck_gnss = (
            CrossAttentionBlock(self.model_dim, int(lstm_cross_attn_heads), float(spatial_aux_dropout))
            if self.gnss_encoder is not None
            else None
        )
        self.decoder_gnss = (
            CrossAttentionBlock(self.model_dim, int(lstm_cross_attn_heads), float(spatial_aux_dropout))
            if self.gnss_encoder is not None
            else None
        )
        self.spatial_summary = AttentionSummaryPool(self.model_dim, max(int(lstm_cross_attn_heads), 1))

        fusion_dim = self.model_dim + self.model_dim * 2 + (self.model_dim if self.gnss_encoder is not None else 0)
        fusion_hidden = max(self.model_dim, int(spatial_aux_hidden_dim) * 4, 128)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(float(max(spatial_aux_dropout, 0.0))),
            nn.Linear(fusion_hidden, self.point_channels),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(float(max(spatial_aux_dropout, 0.0))),
            nn.Linear(fusion_hidden, 1),
        )
        gate_last = self.gate_head[-1]
        if isinstance(gate_last, nn.Linear):
            nn.init.constant_(gate_last.bias, float(residual_gate_init_bias))

        self.gnn_node_proj: nn.Module | None = None
        self.gnn_flow_proj: nn.Module | None = None
        self.gnn_convs: nn.ModuleList | None = None
        self.gnn_learned_adj: LearnedAdjacency | None = None
        self.gnn_node_score: nn.Module | None = None
        self.gnn_delta_head: nn.Module | None = None
        self.gnn_gate_head: nn.Module | None = None
        if self.enable_gnss_gnn:
            if self.turbine_node_input_dim <= 0:
                raise ValueError("branch_mode='gnss_gnn' requires turbine_node_input_dim > 0.")
            if graph_edge_index is None or graph_edge_index.numel() == 0:
                raise ValueError("branch_mode='gnss_gnn' requires graph_edge_index with at least one edge.")
            n_nodes = int(graph_edge_index.max().item()) + 1
            adj_mask = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
            adj_mask[graph_edge_index[0].long(), graph_edge_index[1].long()] = 1.0
            adj_mask = torch.maximum(adj_mask, torch.eye(n_nodes, dtype=torch.float32))
            col_to_idx = {str(name): idx for idx, name in enumerate(graph_edge_attr_columns)}
            edge_count = int(graph_edge_index.shape[1])
            if graph_edge_attr is not None and edge_count > 0 and "distance_m" in col_to_idx:
                edge_distance = graph_edge_attr[:, col_to_idx["distance_m"]].detach().float().cpu()
                edge_base_weight = torch.exp(-edge_distance / 1500.0)
            else:
                edge_base_weight = torch.ones((edge_count,), dtype=torch.float32)
            if graph_edge_attr is not None and edge_count > 0 and "bearing_rad" in col_to_idx:
                edge_bearing = graph_edge_attr[:, col_to_idx["bearing_rad"]].detach().float().cpu()
            else:
                edge_bearing = torch.zeros((edge_count,), dtype=torch.float32)
            self.register_buffer("gnn_edge_src", graph_edge_index[0].detach().long(), persistent=False)
            self.register_buffer("gnn_edge_dst", graph_edge_index[1].detach().long(), persistent=False)
            self.register_buffer("gnn_edge_base_weight", edge_base_weight, persistent=False)
            self.register_buffer("gnn_edge_bearing", edge_bearing, persistent=False)
            self.gnn_node_proj = nn.Sequential(
                nn.LayerNorm(self.turbine_node_input_dim),
                nn.Linear(self.turbine_node_input_dim, self.model_dim),
            )
            self.gnn_flow_proj = nn.Linear(2, self.model_dim)
            self.gnn_convs = nn.ModuleList(
                [GraphConv(self.model_dim, self.model_dim, float(max(spatial_aux_dropout, 0.0))) for _ in range(2)]
            )
            self.gnn_learned_adj = LearnedAdjacency(n_nodes, adj_mask)
            self.gnn_node_score = nn.Linear(self.model_dim, 1)
            gnn_fusion_dim = self.model_dim + self.model_dim + (self.model_dim if self.gnss_encoder is not None else 0)
            self.gnn_delta_head = nn.Sequential(
                nn.LayerNorm(gnn_fusion_dim),
                nn.Linear(gnn_fusion_dim, fusion_hidden),
                nn.GELU(),
                nn.Dropout(float(max(spatial_aux_dropout, 0.0))),
                nn.Linear(fusion_hidden, self.point_channels),
            )
            self.gnn_gate_head = nn.Sequential(
                nn.LayerNorm(gnn_fusion_dim),
                nn.Linear(gnn_fusion_dim, fusion_hidden),
                nn.GELU(),
                nn.Dropout(float(max(spatial_aux_dropout, 0.0))),
                nn.Linear(fusion_hidden, 1),
            )
            gnn_gate_last = self.gnn_gate_head[-1]
            if isinstance(gnn_gate_last, nn.Linear):
                nn.init.constant_(gnn_gate_last.bias, float(residual_gate_init_bias))

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

    def set_spatial_branch_trainable(self, trainable: bool) -> None:
        modules: list[nn.Module] = [
            self.spatial_backbone,
            self.bottleneck_tokens,
            self.decoder_tokens,
            self.bottleneck_self,
            self.decoder_self,
            self.bottleneck_farm,
            self.decoder_farm,
            self.spatial_summary,
            self.delta_head,
            self.gate_head,
        ]
        if self.bottleneck_gnss is not None:
            modules.append(self.bottleneck_gnss)
        if self.decoder_gnss is not None:
            modules.append(self.decoder_gnss)
        if self.gnn_node_proj is not None:
            modules.append(self.gnn_node_proj)
        if self.gnn_flow_proj is not None:
            modules.append(self.gnn_flow_proj)
        if self.gnn_convs is not None:
            modules.extend(list(self.gnn_convs))
        if self.gnn_learned_adj is not None:
            modules.append(self.gnn_learned_adj)
        if self.gnn_node_score is not None:
            modules.append(self.gnn_node_score)
        if self.gnn_delta_head is not None:
            modules.append(self.gnn_delta_head)
        if self.gnn_gate_head is not None:
            modules.append(self.gnn_gate_head)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = bool(trainable)

    def set_temporal_trunk_trainable(self, trainable: bool) -> None:
        modules: list[nn.Module] = [
            self.farm_encoder,
            self.temporal_summary,
            self.readout,
            self.base_head,
        ]
        # When raw GNSS covariates are part of the temporal stage-1 model, they
        # should stay frozen together with the main temporal trunk during the
        # stage-2 spatial adaptation period. Flow-mode GNSS modules are new in
        # stage-2 and therefore remain trainable.
        if self.gnss_mode == "raw":
            if self.gnss_encoder is not None:
                modules.append(self.gnss_encoder)
            if self.farm_gnss_cross is not None:
                modules.append(self.farm_gnss_cross)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = bool(trainable)

    def _graph_attn_bias(self, graph_bias_map: torch.Tensor | None, token_count: int) -> torch.Tensor | None:
        if not self.graph_logit_bias or graph_bias_map is None or graph_bias_map.numel() == 0:
            return None
        pooled = F.adaptive_avg_pool2d(graph_bias_map, (3, 3))
        if pooled.shape[1] > 1:
            pooled = pooled.mean(dim=1, keepdim=True)
        token_bias = pooled.flatten(2).squeeze(1)
        if token_bias.shape[1] != token_count:
            token_bias = token_bias[:, :token_count]
        token_bias = torch.tanh(token_bias)
        return 0.5 * (token_bias.unsqueeze(2) + token_bias.unsqueeze(1))

    def _prior_confidence_scale(self, prior_confidence: torch.Tensor | None) -> torch.Tensor | None:
        if prior_confidence is None or prior_confidence.numel() == 0:
            return None
        if not self.gnss_prior_confidence_gating:
            return torch.ones_like(prior_confidence)
        scale = (prior_confidence - self.gnss_prior_confidence_min) / max(1.0 - self.gnss_prior_confidence_min, 1e-6)
        scale = torch.clamp(scale, min=0.0, max=1.0)
        return scale.pow(self.gnss_prior_confidence_power)

    def _build_prior_farm_context(self, farm_seq: torch.Tensor) -> torch.Tensor | None:
        bundle = self.gnss_prior_model_bundle
        if bundle is None:
            return None
        target_cols = tuple(getattr(bundle, "farm_context_columns", ()) or ())
        if not target_cols:
            return None
        batch, steps = int(farm_seq.shape[0]), int(farm_seq.shape[1])
        out = farm_seq.new_zeros((batch, steps, len(target_cols)))
        if not self.farm_feature_columns:
            return out
        source_index = {str(name): idx for idx, name in enumerate(self.farm_feature_columns)}
        for j, col in enumerate(target_cols):
            idx = source_index.get(str(col))
            if idx is None or idx >= farm_seq.shape[-1]:
                continue
            out[..., j] = farm_seq[..., int(idx)]
        return out

    def _lead_activity_mask(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        active_steps = int(self.gnss_flow_active_steps)
        if active_steps <= 0 or active_steps >= int(self.horizon):
            return None
        mask = torch.zeros((int(self.horizon),), dtype=dtype, device=device)
        mask[:active_steps] = 1.0
        return mask.view(1, int(self.horizon), 1)

    def _build_gnn_dynamic_adj(
        self,
        flow_uv_hat: torch.Tensor,
        prior_conf_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.gnn_learned_adj is None:
            raise RuntimeError("GNSS-GNN adjacency requested without learned adjacency module.")
        batch, horizon, _ = flow_uv_hat.shape
        learned_adj = self.gnn_learned_adj().to(device=flow_uv_hat.device, dtype=flow_uv_hat.dtype)
        learned_adj = learned_adj.unsqueeze(0).unsqueeze(0).expand(batch, horizon, -1, -1)
        if not hasattr(self, "gnn_edge_src") or self.gnn_edge_src.numel() == 0:
            return learned_adj
        flow_dir = torch.atan2(flow_uv_hat[..., 1], flow_uv_hat[..., 0])
        bearing = self.gnn_edge_bearing.to(device=flow_uv_hat.device, dtype=flow_uv_hat.dtype).view(1, 1, -1)
        base_weight = self.gnn_edge_base_weight.to(device=flow_uv_hat.device, dtype=flow_uv_hat.dtype).view(1, 1, -1)
        directional = ((torch.cos(flow_dir.unsqueeze(-1) - bearing) + 1.0) * 0.5).clamp(min=0.0).pow(1.5)
        edge_weight = base_weight * directional
        n_nodes = learned_adj.shape[-1]
        dyn_adj = torch.zeros((batch, horizon, n_nodes, n_nodes), dtype=flow_uv_hat.dtype, device=flow_uv_hat.device)
        dyn_adj[:, :, self.gnn_edge_src.long(), self.gnn_edge_dst.long()] = edge_weight
        dyn_adj = dyn_adj + torch.eye(n_nodes, dtype=flow_uv_hat.dtype, device=flow_uv_hat.device).view(1, 1, n_nodes, n_nodes)
        dyn_adj = dyn_adj / (dyn_adj.sum(dim=-1, keepdim=True) + 1e-6)
        if prior_conf_scale is not None:
            alpha = prior_conf_scale.unsqueeze(-1).unsqueeze(-1)
            return alpha * dyn_adj + (1.0 - alpha) * learned_adj
        return 0.7 * dyn_adj + 0.3 * learned_adj

    def _run_gnss_gnn_branch(
        self,
        turbine_last_node_features: torch.Tensor,
        flow_uv_for_graph: torch.Tensor,
        prior_conf_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.gnn_node_proj is None
            or self.gnn_flow_proj is None
            or self.gnn_convs is None
            or self.gnn_node_score is None
        ):
            raise RuntimeError("branch_mode='gnss_gnn' requires GNSS-GNN modules to be initialized.")
        node_base = self.gnn_node_proj(turbine_last_node_features)
        flow_embed = self.gnn_flow_proj(flow_uv_for_graph)
        adj = self._build_gnn_dynamic_adj(flow_uv_for_graph, prior_conf_scale=prior_conf_scale)
        summaries: list[torch.Tensor] = []
        for lead_idx in range(int(self.horizon)):
            h_graph = node_base + flow_embed[:, lead_idx, :].unsqueeze(1)
            lead_adj = adj[:, lead_idx, :, :]
            for conv in self.gnn_convs:
                h_graph = h_graph + torch.relu(conv(h_graph, lead_adj))
            scores = self.gnn_node_score(h_graph).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            summaries.append((h_graph * weights.unsqueeze(-1)).sum(dim=1))
        return torch.stack(summaries, dim=1), adj

    def _align_prior_horizon(self, prior_uv: torch.Tensor) -> torch.Tensor:
        if prior_uv.dim() != 3:
            raise ValueError(f"Expected prior_uv [B,H,2], got {tuple(prior_uv.shape)}")
        if int(prior_uv.shape[1]) == int(self.horizon):
            return prior_uv
        if int(prior_uv.shape[1]) > int(self.horizon):
            return prior_uv[:, : int(self.horizon), :]
        pad_steps = int(self.horizon) - int(prior_uv.shape[1])
        pad_value = prior_uv[:, -1:, :].expand(-1, pad_steps, -1)
        return torch.cat([prior_uv, pad_value], dim=1)

    def _perturb_prior_station_tensor(self, gnss_station_x: torch.Tensor) -> torch.Tensor:
        if self.gnss_prior_perturbation != "shuffled":
            return gnss_station_x
        if gnss_station_x.dim() != 4 or int(gnss_station_x.shape[2]) <= 1:
            return gnss_station_x
        station_perm = torch.arange(int(gnss_station_x.shape[2]) - 1, -1, -1, device=gnss_station_x.device)
        shuffled = gnss_station_x.clone()
        source = gnss_station_x.index_select(2, station_perm)
        candidate_indices = [
            self.gnss_prior_config.u_idx,
            self.gnss_prior_config.v_idx,
            self.gnss_prior_config.wind_speed_idx,
        ]
        for feat_idx in candidate_indices:
            if 0 <= int(feat_idx) < int(gnss_station_x.shape[-1]):
                shuffled[..., int(feat_idx)] = source[..., int(feat_idx)]
        return shuffled

    def _apply_final_prior_perturbation(self, prior_uv: torch.Tensor) -> torch.Tensor:
        mode = self.gnss_prior_perturbation
        if mode != "dummy_constant" or prior_uv.numel() == 0:
            return prior_uv
        speed = torch.sqrt(prior_uv[..., 0] ** 2 + prior_uv[..., 1] ** 2)
        angle = torch.as_tensor(
            self.gnss_prior_dummy_direction_deg * torch.pi / 180.0,
            dtype=prior_uv.dtype,
            device=prior_uv.device,
        )
        out = prior_uv.clone()
        out[..., 0] = speed * torch.cos(angle)
        out[..., 1] = speed * torch.sin(angle)
        return out

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
        turbine_last_node_features: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        gnss_tokens = None
        gnss_summary = None
        flow_uv_hat = None
        prior_conf_scale = None
        lead_active_mask = self._lead_activity_mask(device=farm_seq.device, dtype=farm_seq.dtype)

        if self.gnss_mode == "flow" and self.gnss_encoder is not None:
            if gnss_station_x is None or gnss_station_mask is None or gnss_station_static is None:
                raise ValueError("flow GNSS mode requires gnss_station_x/mask/static tensors.")
            if self.gnss_prior_source == "distilled_gnss":
                if self.gnss_prior_model_bundle is None:
                    raise ValueError("gnss_prior_source='distilled_gnss' requires a loaded GNSS prior runtime bundle.")
                conf_source = "calibrated" if self.gnss_prior_calibration_state is not None else "strict_upwind"
                _, prior_conf_seq = estimate_station_prior_tensor(
                    gnss_station_x,
                    gnss_station_mask,
                    self.gnss_prior_config,
                    source=conf_source,
                    top_k=self.gnss_prior_top_k,
                    calibration_state=self.gnss_prior_calibration_state if conf_source == "calibrated" else None,
                )
                prior_uv_last = apply_gnss_prior_runtime_bundle(
                    self.gnss_prior_model_bundle,
                    station_x=gnss_station_x,
                    station_mask=gnss_station_mask,
                    farm_context=self._build_prior_farm_context(farm_seq),
                    station_static=gnss_station_static,
                )
                prior_uv_last = self._align_prior_horizon(prior_uv_last)
            else:
                prior_station_x = self._perturb_prior_station_tensor(gnss_station_x)
                prior_uv_seq, prior_conf_seq = estimate_station_prior_tensor(
                    prior_station_x,
                    gnss_station_mask,
                    self.gnss_prior_config,
                    source=self.gnss_prior_source,
                    top_k=self.gnss_prior_top_k,
                    calibration_state=self.gnss_prior_calibration_state,
                )
                prior_uv_last = prior_uv_seq[:, -1:, :].expand(-1, self.horizon, -1)
            prior_uv_last = self._apply_final_prior_perturbation(prior_uv_last)
            prior_conf_window = prior_conf_seq.mean(dim=1)
            prior_conf_horizon = prior_conf_window.unsqueeze(1).expand(-1, self.horizon)
            if lead_active_mask is not None:
                prior_conf_horizon = prior_conf_horizon * lead_active_mask.squeeze(-1)
            prior_conf_scale = self._prior_confidence_scale(prior_conf_horizon)
            gnss_tokens, gnss_summary, flow_uv_hat = self.gnss_encoder(
                gnss_station_x,
                gnss_station_mask,
                gnss_station_static,
            )
            aux["flow_uv_hat"] = flow_uv_hat
            aux["gnss_prior_uv_hat"] = prior_uv_last
            aux["gnss_prior_confidence"] = prior_conf_horizon
            if self.dynamic_graph_builder is not None:
                if turbine_last_power is None:
                    raise ValueError("flow_dynamic graph mode requires turbine_last_power.")
                flow_uv_for_graph = flow_uv_hat
                if prior_conf_scale is not None:
                    flow_uv_for_graph = (
                        prior_conf_scale.unsqueeze(-1) * flow_uv_hat
                        + (1.0 - prior_conf_scale.unsqueeze(-1)) * prior_uv_last
                    )
                graph_bias_map = self.dynamic_graph_builder(
                    last_power=turbine_last_power,
                    flow_uv_hat=flow_uv_for_graph,
                    gnss_coverage=gnss_coverage,
                )
                if prior_conf_scale is not None:
                    graph_bias_map = graph_bias_map * prior_conf_scale.unsqueeze(-1).unsqueeze(-1)
                if lead_active_mask is not None:
                    graph_bias_map = graph_bias_map * lead_active_mask.unsqueeze(-1)
                aux["graph_bias_map"] = graph_bias_map
                if self.enable_spatial_residual:
                    x_grid = torch.cat([x_grid, graph_bias_map], dim=1)
        elif self.gnss_mode == "raw" and self.gnss_encoder is not None:
            if gnss_seq is None:
                raise ValueError("gnss_seq is required when gnss_mode='raw' and gnss_input_dim > 0.")
            gnss_tokens, gnss_summary = self.gnss_encoder(gnss_seq)

        farm_tokens, _ = self.farm_encoder(farm_seq)
        if gnss_tokens is not None and self.farm_gnss_cross is not None:
            farm_tokens = self.farm_gnss_cross(farm_tokens, gnss_tokens)
        lead_tokens = self.readout(farm_tokens, self.target_step, self.horizon)
        base_raw = self.base_head(lead_tokens)

        batch = farm_seq.shape[0]
        if self.branch_mode == "spatial_residual" and self.enable_spatial_residual:
            feats = self.spatial_backbone(x_grid)
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
            spatial_summary_expanded = spatial_summary.unsqueeze(1).expand(-1, self.horizon, -1)
            fusion_parts = [lead_tokens, spatial_summary_expanded]
            if gnss_summary is not None:
                fusion_parts.append(gnss_summary.unsqueeze(1).expand(-1, self.horizon, -1))
            fusion_context = torch.cat(fusion_parts, dim=-1)
            delta_raw = self.delta_head(fusion_context)
            residual_gate = torch.sigmoid(self.gate_head(fusion_context))
            if prior_conf_scale is not None:
                residual_gate = residual_gate * prior_conf_scale.unsqueeze(-1)
            if lead_active_mask is not None:
                residual_gate = residual_gate * lead_active_mask
            spatial_delta = residual_gate * delta_raw
        elif self.enable_gnss_gnn:
            if turbine_last_node_features is None:
                raise ValueError("branch_mode='gnss_gnn' requires turbine_last_node_features.")
            flow_uv_for_graph = aux.get("gnss_prior_uv_hat")
            if flow_uv_hat is not None:
                flow_uv_for_graph = flow_uv_hat
                prior_uv = aux.get("gnss_prior_uv_hat")
                if prior_conf_scale is not None and prior_uv is not None:
                    flow_uv_for_graph = (
                        prior_conf_scale.unsqueeze(-1) * flow_uv_hat
                        + (1.0 - prior_conf_scale.unsqueeze(-1)) * prior_uv
                    )
            if flow_uv_for_graph is None:
                raise ValueError("branch_mode='gnss_gnn' requires GNSS flow prior or encoder output.")
            flow_uv_for_graph = perturb_graph_direction(
                flow_uv_for_graph,
                self.gnss_graph_direction_perturbation,
                constant_direction_deg=self.gnss_graph_direction_constant_deg,
                shuffle_seed=self.gnss_graph_direction_shuffle_seed,
            )
            graph_summary, gnn_dynamic_adj = self._run_gnss_gnn_branch(
                turbine_last_node_features,
                flow_uv_for_graph,
                prior_conf_scale=prior_conf_scale,
            )
            aux["gnn_graph_direction"] = flow_uv_for_graph
            aux["gnn_dynamic_adj"] = gnn_dynamic_adj
            fusion_parts = [lead_tokens, graph_summary]
            if gnss_summary is not None:
                fusion_parts.append(gnss_summary.unsqueeze(1).expand(-1, self.horizon, -1))
            fusion_context = torch.cat(fusion_parts, dim=-1)
            delta_raw = self.gnn_delta_head(fusion_context)
            residual_gate = torch.sigmoid(self.gnn_gate_head(fusion_context))
            if prior_conf_scale is not None:
                residual_gate = residual_gate * prior_conf_scale.unsqueeze(-1)
            if lead_active_mask is not None:
                residual_gate = residual_gate * lead_active_mask
            spatial_delta = residual_gate * delta_raw
        else:
            residual_gate = torch.zeros((batch, self.horizon, 1), dtype=base_raw.dtype, device=base_raw.device)
            spatial_delta = torch.zeros_like(base_raw)

        final_raw = base_raw + spatial_delta
        aux["temporal_base_pred"] = base_raw.reshape(batch, -1)
        aux["spatial_delta_pred"] = spatial_delta.reshape(batch, -1)
        aux["residual_gate"] = residual_gate.squeeze(-1)
        if graph_bias_map is not None and "graph_bias_map" not in aux:
            aux["graph_bias_map"] = graph_bias_map
        if return_aux:
            return final_raw.reshape(batch, -1), aux
        return final_raw.reshape(batch, -1)


class LSTMRampExpertFarmV2Net(LSTMSpatialResidualFarmV2Net):
    """
    LSTM-first farm_v2 model with future ramp risk heads and soft-gated experts.

    The model keeps the same temporal trunk as the main predictor, then learns
    separate upward/downward correction magnitudes gated by lead-specific ramp
    risk probabilities. This follows the ramp-event forecasting intuition that
    future regime risk should be modeled explicitly instead of inferred only from
    a single point forecast.

    References:
    - Florita et al., "Probabilistic Forecasting of Wind Power Ramp Events Using
      Autoregressive Logit Models", European Journal of Operational Research, 2018.
      https://doi.org/10.1016/j.ejor.2017.10.001
    - Arevalo et al., "Gated Multimodal Units for Information Fusion", ICLR
      Workshop 2017. https://arxiv.org/abs/1702.01992
    """

    def __init__(
        self,
        *,
        ramp_expert_hidden_dim: int = 128,
        ramp_expert_gate_init_bias: float = -1.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        context_dim = self.model_dim * (2 + (1 if self.gnss_encoder is not None else 0))
        hidden_dim = max(int(ramp_expert_hidden_dim), self.model_dim, 128)
        head_dropout = float(max(kwargs.get("lstm_main_dropout", 0.0), kwargs.get("spatial_aux_dropout", 0.0)))
        self.ramp_up_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, self.point_channels),
        )
        self.ramp_down_head = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, self.point_channels),
        )
        self.ramp_up_gate = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.ramp_down_gate = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, 1),
        )
        for gate_module in (self.ramp_up_gate, self.ramp_down_gate):
            gate_last = gate_module[-1]
            if isinstance(gate_last, nn.Linear):
                nn.init.constant_(gate_last.bias, float(ramp_expert_gate_init_bias))

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
        turbine_last_node_features: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        gnss_tokens = None
        gnss_summary = None
        flow_uv_hat = None
        prior_conf_scale = None

        if self.gnss_mode == "flow" and self.gnss_encoder is not None:
            if gnss_station_x is None or gnss_station_mask is None or gnss_station_static is None:
                raise ValueError("flow GNSS mode requires gnss_station_x/mask/static tensors.")
            if self.gnss_prior_source == "distilled_gnss":
                if self.gnss_prior_model_bundle is None:
                    raise ValueError("gnss_prior_source='distilled_gnss' requires a loaded GNSS prior runtime bundle.")
                conf_source = "calibrated" if self.gnss_prior_calibration_state is not None else "strict_upwind"
                _, prior_conf_seq = estimate_station_prior_tensor(
                    gnss_station_x,
                    gnss_station_mask,
                    self.gnss_prior_config,
                    source=conf_source,
                    top_k=self.gnss_prior_top_k,
                    calibration_state=self.gnss_prior_calibration_state if conf_source == "calibrated" else None,
                )
                prior_uv_last = apply_gnss_prior_runtime_bundle(
                    self.gnss_prior_model_bundle,
                    station_x=gnss_station_x,
                    station_mask=gnss_station_mask,
                    farm_context=None,
                    station_static=gnss_station_static,
                )
                prior_uv_last = self._align_prior_horizon(prior_uv_last)
            else:
                prior_station_x = self._perturb_prior_station_tensor(gnss_station_x)
                prior_uv_seq, prior_conf_seq = estimate_station_prior_tensor(
                    prior_station_x,
                    gnss_station_mask,
                    self.gnss_prior_config,
                    source=self.gnss_prior_source,
                    top_k=self.gnss_prior_top_k,
                    calibration_state=self.gnss_prior_calibration_state,
                )
                prior_uv_last = prior_uv_seq[:, -1:, :].expand(-1, self.horizon, -1)
            prior_uv_last = self._apply_final_prior_perturbation(prior_uv_last)
            prior_conf_window = prior_conf_seq.mean(dim=1)
            prior_conf_horizon = prior_conf_window.unsqueeze(1).expand(-1, self.horizon)
            if lead_active_mask is not None:
                prior_conf_horizon = prior_conf_horizon * lead_active_mask.squeeze(-1)
            prior_conf_scale = self._prior_confidence_scale(prior_conf_horizon)
            gnss_tokens, gnss_summary, flow_uv_hat = self.gnss_encoder(
                gnss_station_x,
                gnss_station_mask,
                gnss_station_static,
            )
            aux["flow_uv_hat"] = flow_uv_hat
            aux["gnss_prior_uv_hat"] = prior_uv_last
            aux["gnss_prior_confidence"] = prior_conf_horizon
            if self.dynamic_graph_builder is not None:
                if turbine_last_power is None:
                    raise ValueError("flow_dynamic graph mode requires turbine_last_power.")
                flow_uv_for_graph = flow_uv_hat
                if prior_conf_scale is not None:
                    flow_uv_for_graph = (
                        prior_conf_scale.unsqueeze(-1) * flow_uv_hat
                        + (1.0 - prior_conf_scale.unsqueeze(-1)) * prior_uv_last
                    )
                graph_bias_map = self.dynamic_graph_builder(
                    last_power=turbine_last_power,
                    flow_uv_hat=flow_uv_for_graph,
                    gnss_coverage=gnss_coverage,
                )
                if prior_conf_scale is not None:
                    graph_bias_map = graph_bias_map * prior_conf_scale.unsqueeze(-1).unsqueeze(-1)
                if lead_active_mask is not None:
                    graph_bias_map = graph_bias_map * lead_active_mask.unsqueeze(-1)
                aux["graph_bias_map"] = graph_bias_map
                if self.enable_spatial_residual:
                    x_grid = torch.cat([x_grid, graph_bias_map], dim=1)
        elif self.gnss_mode == "raw" and self.gnss_encoder is not None:
            if gnss_seq is None:
                raise ValueError("gnss_seq is required when gnss_mode='raw' and gnss_input_dim > 0.")
            gnss_tokens, gnss_summary = self.gnss_encoder(gnss_seq)

        farm_tokens, farm_summary = self.farm_encoder(farm_seq)
        if gnss_tokens is not None and self.farm_gnss_cross is not None:
            farm_tokens = self.farm_gnss_cross(farm_tokens, gnss_tokens)
            farm_summary = self.temporal_summary(farm_tokens)
        lead_tokens = self.readout(farm_tokens, self.target_step, self.horizon)
        base_raw = self.base_head(lead_tokens)

        batch = farm_seq.shape[0]
        farm_summary_expanded = farm_summary.unsqueeze(1).expand(-1, self.horizon, -1)
        ramp_context_parts = [lead_tokens, farm_summary_expanded]
        if gnss_summary is not None:
            ramp_context_parts.append(gnss_summary.unsqueeze(1).expand(-1, self.horizon, -1))
        ramp_context = torch.cat(ramp_context_parts, dim=-1)
        ramp_up_logits = self.ramp_up_gate(ramp_context)
        ramp_down_logits = self.ramp_down_gate(ramp_context)
        ramp_up_prob = torch.sigmoid(ramp_up_logits)
        ramp_down_prob = torch.sigmoid(ramp_down_logits)
        # Constrain expert magnitudes to be non-negative so the up/down experts
        # remain directionally interpretable in the final residual correction.
        ramp_up_delta = F.softplus(self.ramp_up_head(ramp_context))
        ramp_down_delta = F.softplus(self.ramp_down_head(ramp_context))
        ramp_delta = (ramp_up_prob * ramp_up_delta) - (ramp_down_prob * ramp_down_delta)

        if self.branch_mode == "spatial_residual" and self.enable_spatial_residual:
            feats = self.spatial_backbone(x_grid)
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
            spatial_summary_expanded = spatial_summary.unsqueeze(1).expand(-1, self.horizon, -1)
            fusion_parts = [lead_tokens, spatial_summary_expanded]
            if gnss_summary is not None:
                fusion_parts.append(gnss_summary.unsqueeze(1).expand(-1, self.horizon, -1))
            fusion_context = torch.cat(fusion_parts, dim=-1)
            delta_raw = self.delta_head(fusion_context)
            residual_gate = torch.sigmoid(self.gate_head(fusion_context))
            if prior_conf_scale is not None:
                residual_gate = residual_gate * prior_conf_scale.unsqueeze(-1)
            if lead_active_mask is not None:
                residual_gate = residual_gate * lead_active_mask
            spatial_delta = residual_gate * delta_raw
        elif self.enable_gnss_gnn:
            if turbine_last_node_features is None:
                raise ValueError("branch_mode='gnss_gnn' requires turbine_last_node_features.")
            flow_uv_for_graph = aux.get("gnss_prior_uv_hat")
            if flow_uv_hat is not None:
                flow_uv_for_graph = flow_uv_hat
                prior_uv = aux.get("gnss_prior_uv_hat")
                if prior_conf_scale is not None and prior_uv is not None:
                    flow_uv_for_graph = (
                        prior_conf_scale.unsqueeze(-1) * flow_uv_hat
                        + (1.0 - prior_conf_scale.unsqueeze(-1)) * prior_uv
                    )
            if flow_uv_for_graph is None:
                raise ValueError("branch_mode='gnss_gnn' requires GNSS flow prior or encoder output.")
            graph_summary = self._run_gnss_gnn_branch(
                turbine_last_node_features,
                flow_uv_for_graph,
                prior_conf_scale=prior_conf_scale,
            )
            fusion_parts = [lead_tokens, graph_summary]
            if gnss_summary is not None:
                fusion_parts.append(gnss_summary.unsqueeze(1).expand(-1, self.horizon, -1))
            fusion_context = torch.cat(fusion_parts, dim=-1)
            delta_raw = self.gnn_delta_head(fusion_context)
            residual_gate = torch.sigmoid(self.gnn_gate_head(fusion_context))
            if prior_conf_scale is not None:
                residual_gate = residual_gate * prior_conf_scale.unsqueeze(-1)
            if lead_active_mask is not None:
                residual_gate = residual_gate * lead_active_mask
            spatial_delta = residual_gate * delta_raw
        else:
            residual_gate = torch.zeros((batch, self.horizon, 1), dtype=base_raw.dtype, device=base_raw.device)
            spatial_delta = torch.zeros_like(base_raw)

        final_raw = base_raw + ramp_delta + spatial_delta
        aux["temporal_base_pred"] = base_raw.reshape(batch, -1)
        aux["spatial_delta_pred"] = spatial_delta.reshape(batch, -1)
        aux["residual_gate"] = residual_gate.squeeze(-1)
        aux["ramp_expert_delta_pred"] = ramp_delta.reshape(batch, -1)
        aux["ramp_up_prob"] = ramp_up_prob.squeeze(-1)
        aux["ramp_down_prob"] = ramp_down_prob.squeeze(-1)
        aux["ramp_up_logits"] = ramp_up_logits.squeeze(-1)
        aux["ramp_down_logits"] = ramp_down_logits.squeeze(-1)
        if graph_bias_map is not None and "graph_bias_map" not in aux:
            aux["graph_bias_map"] = graph_bias_map
        if return_aux:
            return final_raw.reshape(batch, -1), aux
        return final_raw.reshape(batch, -1)


def build_lstm_spatial_residual_farm_v2_model(
    *,
    input_channels: int,
    output_channels: int,
    input_hw: tuple[int, int],
    farm_input_dim: int,
    farm_feature_columns: Sequence[str] = (),
    gnss_input_dim: int,
    gnss_mode: str = "raw",
    graph_mode: str = "static",
    horizon: int = 16,
    gnss_station_input_dim: int = 0,
    gnss_station_static_dim: int = 0,
    lstm_main_hidden_dim: int = 256,
    lstm_main_layers: int = 3,
    lstm_main_dropout: float = 0.1,
    lstm_cov_hidden_dim: int = 128,
    lstm_cov_layers: int = 2,
    lstm_cross_attn_heads: int = 4,
    lstm_readout_heads: int = 4,
    spatial_aux_hidden_dim: int = 64,
    spatial_aux_dropout: float = 0.1,
    graph_logit_bias: bool = True,
    target_step: int = 1,
    gnss_station_weight_idx: int = -1,
    gnss_station_feature_columns: Sequence[str] = (),
    graph_edge_index: torch.Tensor | None = None,
    graph_edge_attr: torch.Tensor | None = None,
    graph_edge_attr_columns: Sequence[str] = (),
    grid_flat_idx: torch.Tensor | None = None,
    grid_count_flat: torch.Tensor | None = None,
    graph_coverage_threshold: float = 0.3,
    gnss_prior_source: str = "strict_upwind",
    gnss_prior_top_k: int = 3,
    gnss_prior_confidence_min: float = 0.25,
    gnss_prior_confidence_power: float = 1.0,
    gnss_prior_confidence_gating: bool = True,
    gnss_prior_calibration_state: dict[str, object] | None = None,
    gnss_prior_checkpoint: str | None = None,
    gnss_prior_model_bundle: GNSSPriorRuntimeBundle | None = None,
    gnss_prior_perturbation: str = "none",
    gnss_prior_dummy_direction_deg: float = 0.0,
    gnss_graph_direction_perturbation: str = "none",
    gnss_graph_direction_constant_deg: float = 0.0,
    gnss_graph_direction_shuffle_seed: int = 20260720,
    gnss_flow_direction_only: bool = False,
    gnss_flow_active_steps: int = 0,
    residual_gate_init_bias: float = -2.0,
    branch_mode: str = "spatial_residual",
    turbine_node_input_dim: int = 0,
    enable_spatial_residual: bool = True,
) -> nn.Module:
    return LSTMSpatialResidualFarmV2Net(
        input_channels=int(input_channels),
        output_channels=int(output_channels),
        input_hw=input_hw,
        farm_input_dim=int(farm_input_dim),
        farm_feature_columns=tuple(farm_feature_columns),
        gnss_input_dim=int(gnss_input_dim),
        gnss_mode=str(gnss_mode),
        graph_mode=str(graph_mode),
        horizon=int(horizon),
        gnss_station_input_dim=int(gnss_station_input_dim),
        gnss_station_static_dim=int(gnss_station_static_dim),
        lstm_main_hidden_dim=int(lstm_main_hidden_dim),
        lstm_main_layers=int(lstm_main_layers),
        lstm_main_dropout=float(lstm_main_dropout),
        lstm_cov_hidden_dim=int(lstm_cov_hidden_dim),
        lstm_cov_layers=int(lstm_cov_layers),
        lstm_cross_attn_heads=int(lstm_cross_attn_heads),
        lstm_readout_heads=int(lstm_readout_heads),
        spatial_aux_hidden_dim=int(spatial_aux_hidden_dim),
        spatial_aux_dropout=float(spatial_aux_dropout),
        graph_logit_bias=bool(graph_logit_bias),
        target_step=int(target_step),
        gnss_station_weight_idx=int(gnss_station_weight_idx),
        gnss_station_feature_columns=tuple(gnss_station_feature_columns),
        graph_edge_index=graph_edge_index,
        graph_edge_attr=graph_edge_attr,
        graph_edge_attr_columns=graph_edge_attr_columns,
        grid_flat_idx=grid_flat_idx,
        grid_count_flat=grid_count_flat,
        graph_coverage_threshold=float(graph_coverage_threshold),
        gnss_prior_source=str(gnss_prior_source),
        gnss_prior_top_k=int(gnss_prior_top_k),
        gnss_prior_confidence_min=float(gnss_prior_confidence_min),
        gnss_prior_confidence_power=float(gnss_prior_confidence_power),
        gnss_prior_confidence_gating=bool(gnss_prior_confidence_gating),
        gnss_prior_calibration_state=gnss_prior_calibration_state,
        gnss_prior_checkpoint=gnss_prior_checkpoint,
        gnss_prior_model_bundle=gnss_prior_model_bundle,
        gnss_prior_perturbation=str(gnss_prior_perturbation),
        gnss_prior_dummy_direction_deg=float(gnss_prior_dummy_direction_deg),
        gnss_graph_direction_perturbation=str(gnss_graph_direction_perturbation),
        gnss_graph_direction_constant_deg=float(gnss_graph_direction_constant_deg),
        gnss_graph_direction_shuffle_seed=int(gnss_graph_direction_shuffle_seed),
        gnss_flow_direction_only=bool(gnss_flow_direction_only),
        gnss_flow_active_steps=int(gnss_flow_active_steps),
        residual_gate_init_bias=float(residual_gate_init_bias),
        branch_mode=str(branch_mode),
        turbine_node_input_dim=int(turbine_node_input_dim),
        enable_spatial_residual=bool(enable_spatial_residual),
    )


def build_lstm_ramp_expert_farm_v2_model(
    *,
    input_channels: int,
    output_channels: int,
    input_hw: tuple[int, int],
    farm_input_dim: int,
    farm_feature_columns: Sequence[str] = (),
    gnss_input_dim: int,
    gnss_mode: str = "raw",
    graph_mode: str = "static",
    horizon: int = 16,
    gnss_station_input_dim: int = 0,
    gnss_station_static_dim: int = 0,
    lstm_main_hidden_dim: int = 256,
    lstm_main_layers: int = 3,
    lstm_main_dropout: float = 0.1,
    lstm_cov_hidden_dim: int = 128,
    lstm_cov_layers: int = 2,
    lstm_cross_attn_heads: int = 4,
    lstm_readout_heads: int = 4,
    spatial_aux_hidden_dim: int = 64,
    spatial_aux_dropout: float = 0.1,
    graph_logit_bias: bool = True,
    target_step: int = 1,
    gnss_station_weight_idx: int = -1,
    gnss_station_feature_columns: Sequence[str] = (),
    graph_edge_index: torch.Tensor | None = None,
    graph_edge_attr: torch.Tensor | None = None,
    graph_edge_attr_columns: Sequence[str] = (),
    grid_flat_idx: torch.Tensor | None = None,
    grid_count_flat: torch.Tensor | None = None,
    graph_coverage_threshold: float = 0.3,
    gnss_prior_source: str = "strict_upwind",
    gnss_prior_top_k: int = 3,
    gnss_prior_confidence_min: float = 0.25,
    gnss_prior_confidence_power: float = 1.0,
    gnss_prior_confidence_gating: bool = True,
    gnss_prior_calibration_state: dict[str, object] | None = None,
    gnss_prior_checkpoint: str | None = None,
    gnss_prior_model_bundle: GNSSPriorRuntimeBundle | None = None,
    gnss_prior_perturbation: str = "none",
    gnss_prior_dummy_direction_deg: float = 0.0,
    gnss_graph_direction_perturbation: str = "none",
    gnss_graph_direction_constant_deg: float = 0.0,
    gnss_graph_direction_shuffle_seed: int = 20260720,
    gnss_flow_direction_only: bool = False,
    gnss_flow_active_steps: int = 0,
    residual_gate_init_bias: float = -2.0,
    branch_mode: str = "spatial_residual",
    turbine_node_input_dim: int = 0,
    enable_spatial_residual: bool = True,
    ramp_expert_hidden_dim: int = 128,
    ramp_expert_gate_init_bias: float = -1.5,
) -> nn.Module:
    return LSTMRampExpertFarmV2Net(
        input_channels=int(input_channels),
        output_channels=int(output_channels),
        input_hw=input_hw,
        farm_input_dim=int(farm_input_dim),
        farm_feature_columns=tuple(farm_feature_columns),
        gnss_input_dim=int(gnss_input_dim),
        gnss_mode=str(gnss_mode),
        graph_mode=str(graph_mode),
        horizon=int(horizon),
        gnss_station_input_dim=int(gnss_station_input_dim),
        gnss_station_static_dim=int(gnss_station_static_dim),
        lstm_main_hidden_dim=int(lstm_main_hidden_dim),
        lstm_main_layers=int(lstm_main_layers),
        lstm_main_dropout=float(lstm_main_dropout),
        lstm_cov_hidden_dim=int(lstm_cov_hidden_dim),
        lstm_cov_layers=int(lstm_cov_layers),
        lstm_cross_attn_heads=int(lstm_cross_attn_heads),
        lstm_readout_heads=int(lstm_readout_heads),
        spatial_aux_hidden_dim=int(spatial_aux_hidden_dim),
        spatial_aux_dropout=float(spatial_aux_dropout),
        graph_logit_bias=bool(graph_logit_bias),
        target_step=int(target_step),
        gnss_station_weight_idx=int(gnss_station_weight_idx),
        gnss_station_feature_columns=tuple(gnss_station_feature_columns),
        graph_edge_index=graph_edge_index,
        graph_edge_attr=graph_edge_attr,
        graph_edge_attr_columns=graph_edge_attr_columns,
        grid_flat_idx=grid_flat_idx,
        grid_count_flat=grid_count_flat,
        graph_coverage_threshold=float(graph_coverage_threshold),
        gnss_prior_source=str(gnss_prior_source),
        gnss_prior_top_k=int(gnss_prior_top_k),
        gnss_prior_confidence_min=float(gnss_prior_confidence_min),
        gnss_prior_confidence_power=float(gnss_prior_confidence_power),
        gnss_prior_confidence_gating=bool(gnss_prior_confidence_gating),
        gnss_prior_calibration_state=gnss_prior_calibration_state,
        gnss_prior_checkpoint=gnss_prior_checkpoint,
        gnss_prior_model_bundle=gnss_prior_model_bundle,
        gnss_prior_perturbation=str(gnss_prior_perturbation),
        gnss_prior_dummy_direction_deg=float(gnss_prior_dummy_direction_deg),
        gnss_graph_direction_perturbation=str(gnss_graph_direction_perturbation),
        gnss_graph_direction_constant_deg=float(gnss_graph_direction_constant_deg),
        gnss_graph_direction_shuffle_seed=int(gnss_graph_direction_shuffle_seed),
        gnss_flow_direction_only=bool(gnss_flow_direction_only),
        gnss_flow_active_steps=int(gnss_flow_active_steps),
        residual_gate_init_bias=float(residual_gate_init_bias),
        branch_mode=str(branch_mode),
        turbine_node_input_dim=int(turbine_node_input_dim),
        enable_spatial_residual=bool(enable_spatial_residual),
        ramp_expert_hidden_dim=int(ramp_expert_hidden_dim),
        ramp_expert_gate_init_bias=float(ramp_expert_gate_init_bias),
    )
