"""
Transformer prior model for GNSS station sequences to farm-level ERA5 flow.

The model keeps the problem isolated from the power forecaster: it only learns
whether station-level GNSS/meteorological sequences can explain future farm
u/v flow targets. This makes the dynamic-graph feasibility check auditable.

References:
- Vaswani et al. (2017), "Attention Is All You Need", NeurIPS.
  https://doi.org/10.48550/arXiv.1706.03762
- Wen et al. (2023), "Transformers in Time Series: A Survey", IJCAI 2023.
  https://doi.org/10.24963/ijcai.2023/759
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn


@dataclass
class GNSSPriorTransformerConfig:
    seq_len: int
    station_count: int
    station_feature_dim: int
    farm_context_dim: int
    station_static_dim: int
    horizon: int
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    station_pooling: str = "flat"
    direction_bins: int = 12
    aux_direction_head: bool = False


GNSS_PRIOR_TEACHER_ONLY_PREFIXES = (
    "u100",
    "v100",
    "wind_speed_100",
    "wind_dir_rad",
    "met_quality_flag",
    "upwind_alignment",
    "upwind_mask",
    "upwind_weight",
)


def infer_teacher_only_station_features(feature_columns: list[str] | tuple[str, ...]) -> list[str]:
    cols = list(feature_columns)
    blocked: list[str] = []
    for col in cols:
        if any(str(col).startswith(prefix) for prefix in GNSS_PRIOR_TEACHER_ONLY_PREFIXES):
            blocked.append(str(col))
    return blocked


def build_station_feature_keep_mask(
    feature_columns: list[str] | tuple[str, ...],
    *,
    gnss_only_inference: bool,
    masked_feature_names: list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    cols = list(feature_columns)
    blocked = set(masked_feature_names or ())
    if not blocked and bool(gnss_only_inference):
        blocked = set(infer_teacher_only_station_features(cols))
    keep = [0.0 if col in blocked else 1.0 for col in cols]
    return torch.tensor(keep, dtype=torch.float32)


@dataclass(frozen=True)
class GNSSPriorRuntimeBundle:
    model: Any
    backend: str
    station_mean: torch.Tensor
    station_std: torch.Tensor
    farm_mean: torch.Tensor
    farm_std: torch.Tensor
    station_feature_keep_mask: torch.Tensor
    feature_columns: tuple[str, ...]
    farm_context_columns: tuple[str, ...]
    masked_feature_names: tuple[str, ...]
    teacher_anchor_columns: tuple[str, ...]
    gnss_only_inference: bool
    direction_only: bool
    focus_steps: int
    horizon: int


def _normalize_direction_vectors(x: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
    x = x / norm
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def load_gnss_prior_runtime_bundle(
    run_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    checkpoint_name: str = "checkpoint_best.pt",
) -> GNSSPriorRuntimeBundle:
    run_path = Path(run_dir)
    meta_path = run_path / "meta.json"
    ckpt_path = run_path / checkpoint_name
    joblib_path = run_path / "model.joblib"
    summary_path = run_path / "summary.json"
    if not ckpt_path.exists():
        ckpt_path = run_path / "checkpoint_last.pt"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    elif summary_path.exists():
        meta = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"Missing GNSS prior meta/summary under {run_path}")
    arch = str(meta.get("arch", ""))
    if arch == "gnss_prior_extratrees":
        if not joblib_path.exists():
            raise FileNotFoundError(f"Missing GNSS prior ExtraTrees joblib under {run_path}")
        payload = joblib.load(joblib_path)
        model = payload.get("model", payload) if isinstance(payload, dict) else payload
        return GNSSPriorRuntimeBundle(
            model=model,
            backend="extratrees",
            station_mean=torch.zeros(0, dtype=torch.float32, device=device),
            station_std=torch.zeros(0, dtype=torch.float32, device=device),
            farm_mean=torch.zeros(0, dtype=torch.float32, device=device),
            farm_std=torch.zeros(0, dtype=torch.float32, device=device),
            station_feature_keep_mask=torch.ones(0, dtype=torch.float32, device=device),
            feature_columns=tuple(meta.get("gnss_station_feature_columns", [])),
            farm_context_columns=tuple(meta.get("farm_context_columns", [])),
            masked_feature_names=tuple(),
            teacher_anchor_columns=tuple(),
            gnss_only_inference=bool(meta.get("gnss_only_inference", False)),
            direction_only=bool(meta.get("direction_only", True)),
            focus_steps=int(meta.get("focus_steps", meta.get("horizon", 1)) or meta.get("horizon", 1)),
            horizon=int(meta.get("horizon", meta.get("focus_steps", 1)) or meta.get("focus_steps", 1)),
        )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing GNSS prior checkpoint under {run_path}")
    cfg = GNSSPriorTransformerConfig(**dict(meta["model_cfg"]))
    model = GNSSPriorTransformerNet(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    feature_columns = tuple(meta.get("gnss_station_feature_columns", []))
    masked_feature_names = tuple(meta.get("masked_station_feature_names", []))
    keep_mask = build_station_feature_keep_mask(
        feature_columns,
        gnss_only_inference=bool(meta.get("gnss_only_inference", False)),
        masked_feature_names=list(masked_feature_names),
    )
    stats = dict(meta.get("stats", {}))
    bundle = GNSSPriorRuntimeBundle(
        model=model.to(device),
        backend="transformer",
        station_mean=torch.as_tensor(stats.get("station_mean", []), dtype=torch.float32, device=device),
        station_std=torch.as_tensor(stats.get("station_std", []), dtype=torch.float32, device=device),
        farm_mean=torch.as_tensor(stats.get("farm_mean", []), dtype=torch.float32, device=device),
        farm_std=torch.as_tensor(stats.get("farm_std", []), dtype=torch.float32, device=device),
        station_feature_keep_mask=keep_mask.to(device),
        feature_columns=feature_columns,
        farm_context_columns=tuple(meta.get("farm_context_columns", [])),
        masked_feature_names=masked_feature_names,
        teacher_anchor_columns=tuple(meta.get("teacher_anchor_columns", [])),
        gnss_only_inference=bool(meta.get("gnss_only_inference", False)),
        direction_only=bool(meta.get("direction_only", False)),
        focus_steps=int(meta.get("focus_steps", meta.get("model_cfg", {}).get("horizon", cfg.horizon)) or cfg.horizon),
        horizon=int(meta.get("model_cfg", {}).get("horizon", cfg.horizon)),
    )
    return bundle


def apply_gnss_prior_runtime_bundle(
    bundle: GNSSPriorRuntimeBundle,
    *,
    station_x: torch.Tensor,
    station_mask: torch.Tensor,
    farm_context: torch.Tensor | None = None,
    station_static: torch.Tensor | None = None,
) -> torch.Tensor:
    if str(getattr(bundle, "backend", "transformer")) == "extratrees":
        if farm_context is None:
            farm_context = torch.zeros(
                (station_x.shape[0], station_x.shape[1], len(bundle.farm_context_columns)),
                dtype=station_x.dtype,
                device=station_x.device,
            )
        if station_static is None:
            station_static = torch.zeros(
                (station_x.shape[0], station_x.shape[2], 0),
                dtype=station_x.dtype,
                device=station_x.device,
            )
        feat = torch.cat(
            [
                station_x.reshape(station_x.shape[0], -1),
                station_mask.reshape(station_mask.shape[0], -1),
                station_static.reshape(station_static.shape[0], -1),
                farm_context.reshape(farm_context.shape[0], -1),
            ],
            dim=1,
        )
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        pred_np = bundle.model.predict(feat.detach().cpu().numpy())
        pred = torch.from_numpy(np.asarray(pred_np, dtype=np.float32)).to(device=station_x.device, dtype=station_x.dtype)
        pred = pred.view(station_x.shape[0], int(bundle.focus_steps), 2)
        if bundle.direction_only:
            pred = _normalize_direction_vectors(pred)
        return pred
    keep_mask = bundle.station_feature_keep_mask.view(1, 1, 1, -1).to(dtype=station_x.dtype, device=station_x.device)
    station_mean = bundle.station_mean.view(1, 1, 1, -1).to(dtype=station_x.dtype, device=station_x.device)
    station_std = torch.where(
        bundle.station_std.to(dtype=station_x.dtype, device=station_x.device) < 1e-6,
        torch.ones_like(bundle.station_std.to(dtype=station_x.dtype, device=station_x.device)),
        bundle.station_std.to(dtype=station_x.dtype, device=station_x.device),
    ).view(1, 1, 1, -1)
    norm_station = ((station_x - station_mean) / station_std) * keep_mask
    if farm_context is None:
        farm_context = torch.zeros(
            (station_x.shape[0], station_x.shape[1], int(bundle.farm_mean.numel())),
            dtype=station_x.dtype,
            device=station_x.device,
        )
    farm_mean = bundle.farm_mean.view(1, 1, -1).to(dtype=farm_context.dtype, device=farm_context.device)
    farm_std = torch.where(
        bundle.farm_std.to(dtype=farm_context.dtype, device=farm_context.device) < 1e-6,
        torch.ones_like(bundle.farm_std.to(dtype=farm_context.dtype, device=farm_context.device)),
        bundle.farm_std.to(dtype=farm_context.dtype, device=farm_context.device),
    ).view(1, 1, -1)
    norm_farm = (farm_context - farm_mean) / farm_std
    with torch.no_grad():
        pred = bundle.model(norm_station, station_mask, norm_farm, station_static)
        if bundle.teacher_anchor_columns and farm_context is not None:
            try:
                u_idx = int(bundle.farm_context_columns.index(bundle.teacher_anchor_columns[0]))
                v_idx = int(bundle.farm_context_columns.index(bundle.teacher_anchor_columns[1]))
                anchor = farm_context[:, -1, [u_idx, v_idx]].to(dtype=pred.dtype, device=pred.device)
                anchor = torch.nan_to_num(anchor, nan=0.0, posinf=0.0, neginf=0.0)
                pred = pred + anchor.unsqueeze(1)
            except ValueError:
                pass
        if bundle.direction_only:
            pred = _normalize_direction_vectors(pred)
        return pred


class GNSSPriorTransformerNet(nn.Module):
    """
    Multi-step GNSS prior model with lead-query decoding.

    Input per time step preserves the full station tensor by flattening the
    station-feature grid instead of collapsing it into a simple weighted mean.
    """

    def __init__(self, cfg: GNSSPriorTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        station_step_dim = cfg.station_count * (cfg.station_feature_dim + 1)
        self.station_pooling = str(cfg.station_pooling or "flat")
        if self.station_pooling == "flat":
            self.input_proj = nn.Linear(station_step_dim + cfg.farm_context_dim, cfg.d_model)
            self.station_proj = None
            self.context_proj = None
        elif self.station_pooling == "encoded_mean":
            self.input_proj = None
            self.station_proj = nn.Linear(cfg.station_feature_dim + 1, cfg.d_model)
            self.context_proj = nn.Linear(cfg.d_model + cfg.farm_context_dim, cfg.d_model)
        else:
            raise ValueError(f"Unsupported station_pooling: {self.station_pooling}")
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.static_proj = (
            nn.Linear(cfg.station_count * cfg.station_static_dim, cfg.d_model)
            if cfg.station_static_dim > 0
            else None
        )
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
        self.lead_queries = nn.Parameter(torch.zeros(1, cfg.horizon, cfg.d_model))
        self.readout = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 2)
        self.aux_dir_head = (
            nn.Linear(cfg.d_model, max(int(cfg.direction_bins), 2))
            if bool(cfg.aux_direction_head)
            else None
        )

    def forward(
        self,
        station_x: torch.Tensor,
        station_mask: torch.Tensor,
        farm_context: torch.Tensor | None = None,
        station_static: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if station_x.dim() != 4:
            raise ValueError(f"GNSSPriorTransformerNet expects station_x [B,T,S,F], got {tuple(station_x.shape)}")
        if station_mask.dim() != 3:
            raise ValueError(f"GNSSPriorTransformerNet expects station_mask [B,T,S], got {tuple(station_mask.shape)}")
        batch, steps, station_count, feat_dim = station_x.shape
        if station_count != self.cfg.station_count or feat_dim != self.cfg.station_feature_dim:
            raise ValueError(
                "Station tensor shape does not match config: "
                f"got {(station_count, feat_dim)} expected {(self.cfg.station_count, self.cfg.station_feature_dim)}"
            )
        if farm_context is None:
            farm_context = torch.zeros(
                (batch, steps, self.cfg.farm_context_dim),
                dtype=station_x.dtype,
                device=station_x.device,
            )
        if self.station_pooling == "flat":
            station_tokens = torch.cat([station_x, station_mask.unsqueeze(-1)], dim=-1).reshape(batch, steps, -1)
            x = torch.cat([station_tokens, farm_context], dim=-1)
            h = self.input_proj(x)
        else:
            station_tokens = torch.cat([station_x, station_mask.unsqueeze(-1)], dim=-1)
            station_hidden = self.station_proj(station_tokens)
            mask = station_mask.unsqueeze(-1).to(dtype=station_hidden.dtype)
            pooled = (station_hidden * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
            x = torch.cat([pooled, farm_context], dim=-1)
            h = self.context_proj(x)
        h = h + self.pos_embed[:, :steps, :]
        if self.static_proj is not None and station_static is not None and station_static.numel() > 0:
            if station_static.dim() == 2:
                station_static = station_static.unsqueeze(0).expand(batch, -1, -1)
            static_flat = station_static.reshape(batch, -1)
            h = h + self.static_proj(static_flat).unsqueeze(1)
        h = self.encoder(h)
        queries = self.lead_queries.expand(batch, -1, -1)
        decoded, _ = self.readout(queries, h, h, need_weights=False)
        decoded = self.norm(decoded)
        pred = self.head(decoded)
        if bool(return_aux) and self.aux_dir_head is not None:
            return pred, self.aux_dir_head(decoded)
        return pred
