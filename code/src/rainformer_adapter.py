"""
Rainformer adapter for wind farm forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import torch
from torch import nn

_RAINFORMER_IMPORT_ERROR: Exception | None = None

try:
    from .rainformer import Net as RainformerNet
    from .rainformer import StageModule, StageModule_up, StageModule_up_final
except Exception as exc:  # pragma: no cover - import guard
    RainformerNet = None
    StageModule = None
    StageModule_up = None
    StageModule_up_final = None
    _RAINFORMER_IMPORT_ERROR = exc

PAPER_RF_GRID = (288, 288)
PAPER_RF_HIDDEN_DIM = 96
PAPER_RF_DOWNSCALE = (4, 2, 2, 2)
PAPER_RF_LAYERS = (2, 2, 2, 2)
PAPER_RF_HEADS = (3, 6, 12, 24)
PAPER_RF_HEAD_DIM = 32
PAPER_RF_WINDOW_SIZE = 9
PAPER_RF_RELATIVE_POS = True


@dataclass(frozen=True)
class TurbineGrid:
    grid_h: int
    grid_w: int
    rows: torch.Tensor
    cols: torch.Tensor
    flat_idx: torch.Tensor
    count_flat: torch.Tensor
    collision_cells: int


def _require_rainformer() -> None:
    if _RAINFORMER_IMPORT_ERROR is not None or RainformerNet is None:
        raise ImportError(
            "Rainformer modules unavailable. Ensure einops is installed and "
            "src/rainformer is importable."
        ) from _RAINFORMER_IMPORT_ERROR


def _validate_paper_params(
    input_hw: tuple[int, int],
    hidden_dim: int,
    downscaling_factors: Sequence[int],
    layers: Sequence[int],
    heads: Sequence[int],
    head_dim: int,
    window_size: int,
    relative_pos_embedding: bool,
) -> None:
    if tuple(input_hw) != PAPER_RF_GRID:
        raise ValueError("Rainformer paper architecture requires 288x288 input grid.")
    if int(hidden_dim) != PAPER_RF_HIDDEN_DIM:
        raise ValueError("Rainformer paper architecture requires hidden_dim=96.")
    if tuple(map(int, downscaling_factors)) != PAPER_RF_DOWNSCALE:
        raise ValueError("Rainformer paper architecture requires downscale=4,2,2,2.")
    if tuple(map(int, layers)) != PAPER_RF_LAYERS:
        raise ValueError("Rainformer paper architecture requires layers=2,2,2,2.")
    if tuple(map(int, heads)) != PAPER_RF_HEADS:
        raise ValueError("Rainformer paper architecture requires heads=3,6,12,24.")
    if int(head_dim) != PAPER_RF_HEAD_DIM:
        raise ValueError("Rainformer paper architecture requires head_dim=32.")
    if int(window_size) != PAPER_RF_WINDOW_SIZE:
        raise ValueError("Rainformer paper architecture requires window_size=9.")
    if bool(relative_pos_embedding) != PAPER_RF_RELATIVE_POS:
        raise ValueError("Rainformer paper architecture requires relative_pos_embedding=True.")


def build_turbine_grid(
    df: pd.DataFrame,
    turbine_ids: Sequence[str],
    grid_h: int,
    grid_w: int,
) -> TurbineGrid:
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("Missing lat/lon columns; cannot build Rainformer grid.")

    meta = df[["turbine_id", "lat", "lon"]].dropna().drop_duplicates()
    if meta.empty:
        raise ValueError("No turbine coordinates found for Rainformer grid.")

    lat_min = meta["lat"].min()
    lat_max = meta["lat"].max()
    lon_min = meta["lon"].min()
    lon_max = meta["lon"].max()
    lat_range = float(lat_max - lat_min) if lat_max != lat_min else 1.0
    lon_range = float(lon_max - lon_min) if lon_max != lon_min else 1.0

    coord_map: dict[str, tuple[int, int]] = {}
    for _, row in meta.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        lat_norm = (lat - lat_min) / lat_range
        lon_norm = (lon - lon_min) / lon_range
        r = int(round((grid_h - 1) * (1.0 - lat_norm)))
        c = int(round((grid_w - 1) * lon_norm))
        r = max(0, min(grid_h - 1, r))
        c = max(0, min(grid_w - 1, c))
        coord_map[str(row["turbine_id"])] = (r, c)

    rows = []
    cols = []
    for tid in turbine_ids:
        key = str(tid)
        if key not in coord_map:
            raise ValueError(f"Missing lat/lon for turbine_id={tid}")
        r, c = coord_map[key]
        rows.append(r)
        cols.append(c)

    rows_t = torch.tensor(rows, dtype=torch.long)
    cols_t = torch.tensor(cols, dtype=torch.long)
    flat_idx = rows_t * grid_w + cols_t
    count_flat = torch.bincount(flat_idx, minlength=grid_h * grid_w).float()
    collision_cells = int((count_flat > 1).sum().item())

    return TurbineGrid(
        grid_h=grid_h,
        grid_w=grid_w,
        rows=rows_t,
        cols=cols_t,
        flat_idx=flat_idx,
        count_flat=count_flat,
        collision_cells=collision_cells,
    )


def uv_seq_to_grid(
    uv_seq: torch.Tensor,
    flat_idx: torch.Tensor,
    count_flat: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """
    uv_seq shape: (B, T, N, 2) -> grid (B, 2*T, H, W)
    """
    batch, steps, _, _ = uv_seq.shape
    flat_size = grid_h * grid_w
    idx = flat_idx.unsqueeze(0).expand(batch, -1)
    grid_flat = torch.zeros((batch, 2 * steps, flat_size), device=uv_seq.device)

    for t in range(steps):
        grid_flat[:, 2 * t].scatter_add_(1, idx, uv_seq[:, t, :, 0])
        grid_flat[:, 2 * t + 1].scatter_add_(1, idx, uv_seq[:, t, :, 1])

    denom = count_flat.clamp(min=1.0).view(1, 1, flat_size)
    grid_flat = grid_flat / denom
    return grid_flat.view(batch, 2 * steps, grid_h, grid_w)


def seq_to_grid(
    x_seq: torch.Tensor,
    flat_idx: torch.Tensor,
    count_flat: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """
    x_seq shape: (B, T, N, F) -> grid (B, T*F, H, W)
    """
    batch, steps, n_nodes, feat = x_seq.shape
    flat_size = grid_h * grid_w
    x_flat = x_seq.permute(0, 1, 3, 2).reshape(batch, steps * feat, n_nodes)
    idx = flat_idx.view(1, 1, n_nodes).expand(batch, steps * feat, n_nodes)
    grid_flat = torch.zeros((batch, steps * feat, flat_size), device=x_seq.device)
    grid_flat.scatter_add_(2, idx, x_flat)
    denom = count_flat.clamp(min=1.0).view(1, 1, flat_size)
    grid_flat = grid_flat / denom
    return grid_flat.view(batch, steps * feat, grid_h, grid_w)


def target_to_grid(
    y_seq: torch.Tensor,
    flat_idx: torch.Tensor,
    count_flat: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """
    y_seq shape: (B, N, H) -> grid (B, H, grid_h, grid_w)
    """
    batch, n_nodes, horizon = y_seq.shape
    flat_size = grid_h * grid_w
    y_flat = y_seq.permute(0, 2, 1)
    idx = flat_idx.view(1, 1, n_nodes).expand(batch, horizon, n_nodes)
    grid_flat = torch.zeros((batch, horizon, flat_size), device=y_seq.device)
    grid_flat.scatter_add_(2, idx, y_flat)
    denom = count_flat.clamp(min=1.0).view(1, 1, flat_size)
    grid_flat = grid_flat / denom
    return grid_flat.view(batch, horizon, grid_h, grid_w)


def grid_to_uv_seq(
    grid: torch.Tensor,
    flat_idx: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """
    grid shape: (B, 2*horizon, H, W) -> uv (B, horizon, N, 2)
    """
    batch, _, grid_h, grid_w = grid.shape
    flat = grid.view(batch, 2 * horizon, grid_h * grid_w)
    idx = flat_idx.unsqueeze(0).expand(batch, -1)
    preds = torch.zeros((batch, horizon, idx.shape[1], 2), device=grid.device)
    for t in range(horizon):
        preds[:, t, :, 0] = flat[:, 2 * t].gather(1, idx)
        preds[:, t, :, 1] = flat[:, 2 * t + 1].gather(1, idx)
    return preds


def grid_to_seq(
    grid: torch.Tensor,
    flat_idx: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """
    grid shape: (B, H, grid_h, grid_w) -> seq (B, N, H)
    """
    batch, _, grid_h, grid_w = grid.shape
    flat = grid.view(batch, horizon, grid_h * grid_w)
    n_nodes = flat_idx.shape[0]
    idx = flat_idx.view(1, 1, n_nodes).expand(batch, horizon, n_nodes)
    seq = flat.gather(2, idx)
    return seq.permute(0, 2, 1)


def node_features_to_grid(
    node_features: torch.Tensor,
    flat_idx: torch.Tensor,
    count_flat: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """
    node_features: (N, F) or (B, N, F) -> grid (B, F, H, W)
    """
    if node_features.dim() == 2:
        node_features = node_features.unsqueeze(0)
    if node_features.dim() != 3:
        raise ValueError("node_features must have shape (N, F) or (B, N, F).")
    batch, n_nodes, feat = node_features.shape
    flat_size = grid_h * grid_w
    node_flat = node_features.permute(0, 2, 1)
    idx = flat_idx.view(1, 1, n_nodes).expand(batch, feat, n_nodes)
    grid_flat = torch.zeros((batch, feat, flat_size), device=node_features.device, dtype=node_features.dtype)
    grid_flat.scatter_add_(2, idx, node_flat)
    denom = count_flat.clamp(min=1.0).view(1, 1, flat_size).to(node_features.dtype)
    grid_flat = grid_flat / denom
    return grid_flat.view(batch, feat, grid_h, grid_w)


def graph_bias_from_edges(
    last_power: torch.Tensor,
    last_wind: torch.Tensor | None,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_attr_columns: Sequence[str],
    flat_idx: torch.Tensor,
    count_flat: torch.Tensor,
    grid_h: int,
    grid_w: int,
    sigma_m: float = 1500.0,
    z_scale: float = 200.0,
    roughness_alpha: float = 0.1,
) -> torch.Tensor:
    """
    Build a graph-informed spatial bias map from turbine power and static edge priors.

    Structural priors are inspired by Graphormer-style graph biases:
    Ying et al. (2021), "Do Transformers Really Perform Bad for Graph Representation?",
    NeurIPS. https://arxiv.org/abs/2106.05234
    """
    if last_power.dim() != 2:
        raise ValueError("last_power must have shape (B, N).")
    if edge_index.numel() == 0:
        bias = torch.zeros((last_power.shape[0], last_power.shape[1], 1), device=last_power.device, dtype=last_power.dtype)
        return node_features_to_grid(bias, flat_idx, count_flat, grid_h, grid_w)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    col_to_idx = {str(name): i for i, name in enumerate(edge_attr_columns)}
    weight = torch.ones((edge_attr.shape[0],), device=last_power.device, dtype=last_power.dtype)
    if "distance_m" in col_to_idx:
        dist = edge_attr[:, col_to_idx["distance_m"]].to(last_power.device, dtype=last_power.dtype)
        weight = weight * torch.exp(-dist / max(float(sigma_m), 1.0))
    if "elevation_diff_m" in col_to_idx:
        dz = edge_attr[:, col_to_idx["elevation_diff_m"]].to(last_power.device, dtype=last_power.dtype)
        weight = weight * torch.exp(-torch.abs(dz) / max(float(z_scale), 1.0))
    if "roughness_pair" in col_to_idx:
        rough = edge_attr[:, col_to_idx["roughness_pair"]].to(last_power.device, dtype=last_power.dtype)
        rough = torch.clamp(rough, min=1e-6) ** float(max(roughness_alpha, 0.0))
        weight = weight * rough
    if "wake_prior" in col_to_idx:
        wake = edge_attr[:, col_to_idx["wake_prior"]].to(last_power.device, dtype=last_power.dtype)
        weight = weight * torch.clamp(wake, min=0.0)

    dyn_weight = weight.view(1, -1)
    if last_wind is not None:
        wind_src = torch.clamp(last_wind[:, src], min=0.0)
        wind_scale = wind_src / wind_src.mean(dim=1, keepdim=True).clamp(min=1e-6)
        dyn_weight = dyn_weight * torch.clamp(wind_scale, min=0.5, max=2.0)
    power_src = torch.clamp(last_power[:, src], min=0.0)
    power_scale = power_src / power_src.mean(dim=1, keepdim=True).clamp(min=1e-6)
    dyn_weight = dyn_weight * torch.clamp(power_scale, min=0.5, max=2.5)

    messages = power_src * dyn_weight
    influence = torch.zeros_like(last_power)
    influence.scatter_add_(1, dst.view(1, -1).expand(last_power.shape[0], -1), messages)
    influence = influence.unsqueeze(-1)
    return node_features_to_grid(influence, flat_idx, count_flat, grid_h, grid_w)


def resolve_uv_indices(feature_columns: Sequence[str]) -> tuple[int, int]:
    cols = list(feature_columns)
    if "u100" in cols and "v100" in cols:
        return cols.index("u100"), cols.index("v100")
    if "u140" in cols and "v140" in cols:
        return cols.index("u140"), cols.index("v140")
    if "u10" in cols and "v10" in cols:
        return cols.index("u10"), cols.index("v10")
    raise ValueError("No uv feature columns found for Rainformer input.")


def _check_divisible(value: int, factor: int, name: str) -> None:
    if value % factor != 0:
        raise ValueError(f"{name}={value} must be divisible by {factor} for Rainformer.")


def _validate_stage_hw(h_w: tuple[int, int], window_size: int, stage: str) -> None:
    _check_divisible(h_w[0], window_size, f"{stage}_h")
    _check_divisible(h_w[1], window_size, f"{stage}_w")


class RainformerBackbone(nn.Module):
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
            raise ValueError("Rainformer expects 4-stage configurations.")

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4, x3)
        x6 = self.stage6(x5, x2)
        x7 = self.stage7(x6, x1)
        x8 = self.stage8(x7)
        return x8


class RainformerWindNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        input_hw: tuple[int, int],
        hidden_dim: int = 96,
        downscaling_factors: Sequence[int] = (4, 2, 2, 2),
        layers: Sequence[int] = (2, 2, 2, 2),
        heads: Sequence[int] = (3, 6, 12, 24),
        head_dim: int = 32,
        window_size: int = 9,
        relative_pos_embedding: bool = True,
        rf_version: str = "paper",
    ) -> None:
        super().__init__()
        self.rf_version = rf_version
        if rf_version == "paper":
            _require_rainformer()
            _validate_paper_params(
                input_hw=input_hw,
                hidden_dim=hidden_dim,
                downscaling_factors=downscaling_factors,
                layers=layers,
                heads=heads,
                head_dim=head_dim,
                window_size=window_size,
                relative_pos_embedding=relative_pos_embedding,
            )
            self.pre = (
                nn.Identity()
                if input_channels == output_channels
                else nn.Conv2d(input_channels, output_channels, kernel_size=1)
            )
            self.core = RainformerNet(
                input_channel=output_channels,
                hidden_dim=PAPER_RF_HIDDEN_DIM,
                downscaling_factors=PAPER_RF_DOWNSCALE,
                layers=PAPER_RF_LAYERS,
                heads=PAPER_RF_HEADS,
                head_dim=PAPER_RF_HEAD_DIM,
                window_size=PAPER_RF_WINDOW_SIZE,
                relative_pos_embedding=PAPER_RF_RELATIVE_POS,
            )
        elif rf_version == "legacy":
            self.backbone = RainformerBackbone(
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
            self.head = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        else:
            raise ValueError(f"Unknown rf_version={rf_version}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rf_version == "paper":
            return self.core(self.pre(x))
        y = self.backbone(x)
        return self.head(y)


class RainformerPowerNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        input_hw: tuple[int, int],
        hidden_dim: int = 32,
        downscaling_factors: Sequence[int] = (2, 2, 2, 2),
        layers: Sequence[int] = (2, 2, 2, 2),
        heads: Sequence[int] = (1, 2, 4, 8),
        head_dim: int = 16,
        window_size: int = 2,
        relative_pos_embedding: bool = True,
        head_layers: int = 2,
        context_dim: int = 0,
        context_hidden: int = 128,
        rf_version: str = "paper",
    ) -> None:
        super().__init__()
        self.rf_version = rf_version
        if rf_version == "paper":
            _require_rainformer()
            _validate_paper_params(
                input_hw=input_hw,
                hidden_dim=hidden_dim,
                downscaling_factors=downscaling_factors,
                layers=layers,
                heads=heads,
                head_dim=head_dim,
                window_size=window_size,
                relative_pos_embedding=relative_pos_embedding,
            )
            if int(context_dim) > 0:
                raise ValueError("Rainformer paper architecture does not use context features.")
            if int(head_layers) not in (1,):
                raise ValueError("Rainformer paper architecture does not use extra head layers.")
            self.pre = (
                nn.Identity()
                if input_channels == output_channels
                else nn.Conv2d(input_channels, output_channels, kernel_size=1)
            )
            self.core = RainformerNet(
                input_channel=output_channels,
                hidden_dim=PAPER_RF_HIDDEN_DIM,
                downscaling_factors=PAPER_RF_DOWNSCALE,
                layers=PAPER_RF_LAYERS,
                heads=PAPER_RF_HEADS,
                head_dim=PAPER_RF_HEAD_DIM,
                window_size=PAPER_RF_WINDOW_SIZE,
                relative_pos_embedding=PAPER_RF_RELATIVE_POS,
            )
            self.context_dim = 0
            self.context_mlp = None
        elif rf_version == "legacy":
            self.backbone = RainformerBackbone(
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
            self.context_dim = int(context_dim)
            if self.context_dim > 0:
                self.context_mlp = nn.Sequential(
                    nn.Linear(self.context_dim, context_hidden),
                    nn.ReLU(),
                    nn.Linear(context_hidden, input_channels * 2),
                )
            else:
                self.context_mlp = None

            head_layers = max(int(head_layers), 1)
            layers_list: list[nn.Module] = []
            in_ch = input_channels
            for _ in range(head_layers - 1):
                layers_list.append(nn.Conv2d(in_ch, in_ch, kernel_size=1))
                layers_list.append(nn.ReLU())
            layers_list.append(nn.Conv2d(in_ch, output_channels, kernel_size=1))
            self.head = nn.Sequential(*layers_list)
        else:
            raise ValueError(f"Unknown rf_version={rf_version}")

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.rf_version == "paper":
            if context is not None:
                raise ValueError("Rainformer paper architecture does not use context.")
            return self.core(self.pre(x))
        y = self.backbone(x)
        if self.context_mlp is not None:
            if context is None:
                raise ValueError("context is required when context_dim > 0.")
            gamma_beta = self.context_mlp(context)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
            y = y * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.head(y)


class RainformerPowerFarmNet(nn.Module):
    """Direct farm-level power head on top of Rainformer backbone."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        input_hw: tuple[int, int],
        hidden_dim: int = 32,
        downscaling_factors: Sequence[int] = (2, 2, 2, 2),
        layers: Sequence[int] = (2, 2, 2, 2),
        heads: Sequence[int] = (1, 2, 4, 8),
        head_dim: int = 16,
        window_size: int = 2,
        relative_pos_embedding: bool = True,
        farm_hidden: int = 128,
        context_dim: int = 0,
        context_hidden: int = 128,
        head_dropout: float = 0.1,
        head_downscale_layers: int = 4,
        legacy_mlp_head: bool = False,
        rf_version: str = "legacy",
    ) -> None:
        super().__init__()
        if rf_version != "legacy":
            raise ValueError("RainformerPowerFarmNet requires rf_version='legacy'.")
        self.backbone = RainformerBackbone(
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
        self.context_dim = int(context_dim)
        if self.context_dim > 0:
            # FiLM-style conditional modulation for multi-modal fusion.
            # Perez et al. (2018), "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI.
            # arXiv:1709.07871, https://arxiv.org/abs/1709.07871
            # Closely related affine conditioning in normalization layers:
            # Dumoulin et al. (2017), "A Learned Representation for Artistic Style", ICLR.
            # https://arxiv.org/abs/1610.07629
            self.context_mlp = nn.Sequential(
                nn.Linear(self.context_dim, context_hidden),
                nn.ReLU(),
                nn.Linear(context_hidden, input_channels * 2),
            )
        else:
            self.context_mlp = None

        self.legacy_mlp_head = bool(legacy_mlp_head)
        if self.legacy_mlp_head:
            # Backward-compatible farm head for older checkpoints with keys:
            # head.1.weight / head.3.weight
            hidden = int(farm_hidden) if int(farm_hidden) > 0 else max(int(input_channels) // 2, 8)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(input_channels), hidden),
                nn.GELU(),
                nn.Linear(hidden, int(output_channels)),
            )
            self.downscale_head = nn.Identity()
            self.out_conv_1x1 = nn.Identity()
            self.regressor = nn.Identity()
        else:
            # Extra downscales before scalar farm head:
            # 288/144 grids are over-parameterized for single-value regression.
            # We explicitly keep multiple stride-2 compressions here for stronger
            # spatial bottlenecking prior to farm-level projection.
            head_downscale_layers = max(int(head_downscale_layers), 1)
            head_channels = [int(input_channels)]
            for _ in range(head_downscale_layers):
                head_channels.append(max(head_channels[-1] // 2, 8))
            head_blocks: list[nn.Module] = []
            for in_ch, out_ch in zip(head_channels[:-1], head_channels[1:]):
                head_blocks.extend(
                    [
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.GELU(),
                        nn.Dropout2d(float(max(head_dropout, 0.0))),
                    ]
                )
            self.downscale_head = nn.Sequential(*head_blocks)
            self.out_conv_1x1 = nn.Conv2d(head_channels[-1], output_channels, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            if farm_hidden and farm_hidden > 0:
                self.regressor = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(output_channels, farm_hidden),
                    nn.GELU(),
                    nn.Dropout(float(max(head_dropout, 0.0))),
                    nn.Linear(farm_hidden, output_channels),
                )
            else:
                self.regressor = nn.Flatten()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.backbone(x)
        if self.context_mlp is not None:
            if context is None:
                raise ValueError("context is required when context_dim > 0.")
            gamma_beta = self.context_mlp(context)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
            # Feature-wise affine modulation: y = x * (1 + gamma) + beta (FiLM form).
            feat = feat * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        if self.legacy_mlp_head:
            feat = self.pool(feat)
            return self.head(feat)
        feat = self.downscale_head(feat)
        feat = self.out_conv_1x1(feat)
        feat = self.pool(feat)
        return self.regressor(feat)


class RainformerPowerFarmV2Net(nn.Module):
    """
    Farm-centric Rainformer head with turbine auxiliary modality, static-node channels,
    graph-bias channels, and GNSS late fusion.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        input_hw: tuple[int, int],
        farm_input_dim: int,
        gnss_input_dim: int = 0,
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
    ) -> None:
        super().__init__()
        self.backbone = RainformerBackbone(
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
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(input_channels, fusion_hidden, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.farm_encoder = nn.GRU(
            input_size=int(farm_input_dim),
            hidden_size=int(farm_hidden),
            num_layers=1,
            batch_first=True,
        )
        self.gnss_input_dim = int(gnss_input_dim)
        if self.gnss_input_dim > 0:
            self.gnss_encoder = nn.GRU(
                input_size=self.gnss_input_dim,
                hidden_size=int(gnss_hidden),
                num_layers=1,
                batch_first=True,
            )
            fusion_in = int(fusion_hidden + farm_hidden + gnss_hidden)
        else:
            self.gnss_encoder = None
            fusion_in = int(fusion_hidden + farm_hidden)
        self.head = nn.Sequential(
            nn.Linear(fusion_in, max(int(fusion_hidden), int(output_channels))),
            nn.GELU(),
            nn.Dropout(float(max(head_dropout, 0.0))),
            nn.Linear(max(int(fusion_hidden), int(output_channels)), int(output_channels)),
        )

    def forward(
        self,
        x_grid: torch.Tensor,
        farm_seq: torch.Tensor,
        gnss_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat = self.backbone(x_grid)
        spatial_emb = self.spatial_proj(feat)
        _, farm_hidden = self.farm_encoder(farm_seq)
        farm_emb = farm_hidden[-1]
        if self.gnss_encoder is not None:
            if gnss_seq is None:
                raise ValueError("gnss_seq is required when gnss_input_dim > 0.")
            _, gnss_hidden = self.gnss_encoder(gnss_seq)
            fusion = torch.cat([spatial_emb, farm_emb, gnss_hidden[-1]], dim=1)
        else:
            fusion = torch.cat([spatial_emb, farm_emb], dim=1)
        return self.head(fusion)
