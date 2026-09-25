from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .farm_v2_dataset import FarmV2PowerDataset, FarmV2Stats
from .power_baselines import (
    apply_direct_multi_step_arx_baseline,
    fit_direct_multi_step_arx_coeffs,
    fit_recursive_ar1_coeffs,
)
from .rainformer_adapter import graph_bias_from_edges, node_features_to_grid, seq_to_grid


@dataclass(frozen=True)
class FarmV2PreparedBatch:
    farm_x_raw: torch.Tensor
    farm_x_norm: torch.Tensor
    turbine_x_norm: torch.Tensor
    gnss_seq: torch.Tensor
    gnss_station_x: torch.Tensor
    gnss_station_mask: torch.Tensor
    gnss_station_static: torch.Tensor
    gnss_coverage: torch.Tensor
    target: torch.Tensor
    target_flow_uv: torch.Tensor
    target_flow_mask: torch.Tensor
    window_flags: torch.Tensor
    origin_idx: torch.Tensor
    grid_x: torch.Tensor
    turbine_last_power: torch.Tensor
    graph_bias_map: torch.Tensor | None


@dataclass(frozen=True)
class FarmV2ModelTensors:
    flat_idx: torch.Tensor
    count_flat: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    static_node: torch.Tensor
    edge_attr_columns: tuple[str, ...]


def resolve_farm_v2_runtime_modes(
    dataset: FarmV2PowerDataset,
    requested_gnss_mode: str = "raw",
    requested_graph_mode: str = "static",
) -> tuple[str, str, str | None]:
    """
    Resolve the effective GNSS / graph runtime modes for farm_v2.

    When station-structured GNSS products are missing, flow mode cannot provide
    meaningful auxiliary context. In that case we fall back to the static-graph
    path so training/eval can still run on SCADA-only datasets.
    """

    gnss_mode = str(requested_gnss_mode or "raw")
    graph_mode = str(requested_graph_mode or "static")
    station_feature_ready = bool(getattr(dataset, "gnss_station_feature_columns", []))
    station_ids_ready = bool(getattr(dataset, "gnss_selected_station_ids", []))
    station_tensor = getattr(dataset, "gnss_station_values", None)
    station_count = int(station_tensor.shape[1]) if station_tensor is not None and station_tensor.ndim >= 2 else 0
    gnss_flow_ready = bool(station_feature_ready and station_ids_ready and station_count > 0)

    fallback_reason: str | None = None
    if gnss_mode == "flow" and not gnss_flow_ready:
        fallback_reason = (
            "farm_v2 dataset has no usable GNSS station products; "
            "falling back from gnss_mode=flow / graph_mode=flow_dynamic to raw/static."
        )
        gnss_mode = "raw"
        graph_mode = "static"
    elif graph_mode == "flow_dynamic" and gnss_mode != "flow":
        fallback_reason = "flow_dynamic graph requires gnss_mode=flow; falling back to graph_mode=static."
        graph_mode = "static"

    return gnss_mode, graph_mode, fallback_reason


def build_model_tensors(dataset: FarmV2PowerDataset, device: torch.device) -> FarmV2ModelTensors:
    flat_idx = torch.as_tensor(dataset.meta.get("grid_flat_idx", []), dtype=torch.long, device=device)
    count_flat = torch.as_tensor(dataset.meta.get("grid_count_flat", []), dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(dataset.edge_index, dtype=torch.long, device=device)
    edge_attr = torch.as_tensor(dataset.edge_attr, dtype=torch.float32, device=device)
    static_node = torch.as_tensor(dataset.static_node_values, dtype=torch.float32, device=device)
    return FarmV2ModelTensors(
        flat_idx=flat_idx,
        count_flat=count_flat,
        edge_index=edge_index,
        edge_attr=edge_attr,
        static_node=static_node,
        edge_attr_columns=tuple(dataset.edge_attr_columns),
    )


def _safe_tensor_std(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 1e-6, torch.ones_like(x), x)


def _normalize_static_node(static_node: torch.Tensor) -> torch.Tensor:
    if static_node.numel() == 0:
        return static_node
    mean = static_node.mean(dim=0, keepdim=True)
    std = _safe_tensor_std(static_node.std(dim=0, unbiased=False, keepdim=True))
    return (static_node - mean) / std


def prepare_batch(
    batch: dict[str, torch.Tensor],
    dataset: FarmV2PowerDataset,
    stats: FarmV2Stats,
    tensors: FarmV2ModelTensors,
    device: torch.device,
    grid_h: int,
    grid_w: int,
    static_bias: bool = True,
    graph_bias: bool = True,
    graph_mode: str = "static",
) -> FarmV2PreparedBatch:
    farm_x_raw = batch["farm_x"].to(device)
    turbine_x = batch["turbine_x"].to(device)
    target = batch["target"].to(device)
    gnss_x = batch["gnss_x"].to(device)
    gnss_mask = batch["gnss_mask"].to(device)
    gnss_station_x = batch["gnss_station_x"].to(device)
    gnss_station_mask = batch["gnss_station_mask"].to(device)
    gnss_station_static = batch["gnss_station_static"].to(device)
    gnss_coverage = batch["gnss_coverage"].to(device)
    target_flow_uv = batch["target_flow_uv"].to(device)
    target_flow_mask = batch["target_flow_mask"].to(device)
    window_flags = batch["window_flags"].to(device)

    farm_mean = torch.as_tensor(stats.farm_mean, dtype=farm_x_raw.dtype, device=device)
    farm_std = torch.as_tensor(stats.farm_std, dtype=farm_x_raw.dtype, device=device)
    turbine_mean = torch.as_tensor(stats.turbine_mean, dtype=turbine_x.dtype, device=device)
    turbine_std = torch.as_tensor(stats.turbine_std, dtype=turbine_x.dtype, device=device)

    farm_x_norm = (farm_x_raw - farm_mean) / farm_std
    turbine_x_norm = (turbine_x - turbine_mean) / turbine_std

    grid_x = seq_to_grid(turbine_x_norm, tensors.flat_idx, tensors.count_flat, grid_h, grid_w)

    if static_bias and tensors.static_node.numel() > 0:
        static_grid = node_features_to_grid(
            _normalize_static_node(tensors.static_node),
            tensors.flat_idx,
            tensors.count_flat,
            grid_h,
            grid_w,
        )
        static_grid = static_grid.expand(farm_x_raw.shape[0], -1, -1, -1)
        grid_x = torch.cat([grid_x, static_grid], dim=1)

    graph_bias_map = None
    if tensors.edge_index.numel() > 0:
        power_idx = dataset.turbine_feature_columns.index("power_5min_mean")
        last_power = turbine_x[:, -1, :, power_idx]
        last_wind = None
        if "wind_speed_1min_inst" in dataset.turbine_feature_columns:
            wind_idx = dataset.turbine_feature_columns.index("wind_speed_1min_inst")
            last_wind = turbine_x[:, -1, :, wind_idx]
        elif "wind_speed_5min_mean" in dataset.turbine_feature_columns:
            wind_idx = dataset.turbine_feature_columns.index("wind_speed_5min_mean")
            last_wind = turbine_x[:, -1, :, wind_idx]
        bias_grid = graph_bias_from_edges(
            last_power=last_power,
            last_wind=last_wind,
            edge_index=tensors.edge_index,
            edge_attr=tensors.edge_attr,
            edge_attr_columns=tensors.edge_attr_columns,
            flat_idx=tensors.flat_idx,
            count_flat=tensors.count_flat,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        if str(graph_mode) == "static":
            graph_bias_map = bias_grid
        if graph_bias and str(graph_mode) == "static":
            grid_x = torch.cat([grid_x, bias_grid], dim=1)
    else:
        power_idx = dataset.turbine_feature_columns.index("power_5min_mean")
        last_power = turbine_x[:, -1, :, power_idx]

    gnss_seq = torch.cat([gnss_x, gnss_mask], dim=-1) if gnss_x.shape[-1] > 0 else gnss_x
    return FarmV2PreparedBatch(
        farm_x_raw=farm_x_raw,
        farm_x_norm=farm_x_norm,
        turbine_x_norm=turbine_x_norm,
        gnss_seq=gnss_seq,
        gnss_station_x=gnss_station_x,
        gnss_station_mask=gnss_station_mask,
        gnss_station_static=gnss_station_static,
        gnss_coverage=gnss_coverage,
        target=target,
        target_flow_uv=target_flow_uv,
        target_flow_mask=target_flow_mask,
        window_flags=window_flags,
        origin_idx=batch["origin_idx"].to(device),
        grid_x=grid_x,
        turbine_last_power=last_power,
        graph_bias_map=graph_bias_map,
    )


def farm_v2_output_channels(horizon: int, probabilistic: bool, quantile_count: int) -> int:
    return int(horizon * quantile_count) if probabilistic else int(horizon)


def flow_direction_mae_stepwise(
    pred_uv: torch.Tensor,
    target_uv: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[list[float], list[int]]:
    if pred_uv.numel() == 0 or target_uv.numel() == 0:
        steps = int(pred_uv.shape[1]) if pred_uv.dim() >= 2 else 0
        return [float("nan")] * steps, [0] * steps
    steps = min(int(pred_uv.shape[1]), int(target_uv.shape[1]))
    pred_dir = torch.atan2(pred_uv[:, :steps, 1], pred_uv[:, :steps, 0])
    target_dir = torch.atan2(target_uv[:, :steps, 1], target_uv[:, :steps, 0])
    delta = torch.atan2(torch.sin(pred_dir - target_dir), torch.cos(pred_dir - target_dir)).abs()
    delta_deg = delta * (180.0 / np.pi)
    if mask is None:
        valid = torch.ones_like(delta_deg, dtype=torch.bool)
    else:
        valid = mask[:, :steps].to(dtype=torch.bool, device=delta_deg.device)
    mae: list[float] = []
    counts: list[int] = []
    for step in range(steps):
        step_valid = valid[:, step]
        counts.append(int(step_valid.sum().item()))
        if not bool(torch.any(step_valid).item()):
            mae.append(float("nan"))
            continue
        mae.append(float(delta_deg[:, step][step_valid].mean().item()))
    return mae, counts


def fit_farm_v2_residual_baseline(
    dataset: FarmV2PowerDataset,
    train_indices: Sequence[int],
    mode: str = "persistence",
) -> tuple[dict[str, Any] | list[float] | None, list[float] | None]:
    mode_key = str(mode).lower()
    if mode_key not in {"lag1_ar1", "arx_wind"}:
        return None, None
    if not train_indices or "power_5min_mean" not in dataset.farm_feature_columns:
        return None, None
    if mode_key == "arx_wind":
        # Use a causal direct multi-step ARX-style anchor with recent power/wind
        # history plus future calendar harmonics, following Nielsen et al.
        # (1996), "Wind Power Prediction Using ARX Models and Neural Networks"
        # and Chevillon (2007), "Direct Multi-step Estimation and Forecasting".
        farm_cols = list(dataset.farm_feature_columns)
        if "wind_speed_5min_mean" in farm_cols:
            wind_col = "wind_speed_5min_mean"
            wind_aux_col = "wind_speed_1min_inst" if "wind_speed_1min_inst" in farm_cols else None
        elif "wind_speed_1min_inst" in farm_cols:
            wind_col = "wind_speed_1min_inst"
            wind_aux_col = None
        else:
            return None, None
        power_col = "power_5min_mean"
        power_aux_col = "power_1min_inst" if "power_1min_inst" in farm_cols else None
        lag_count = int(max(dataset.t_in, 1))
        train_idx = np.asarray(train_indices, dtype=np.int64)

        def _history(value_col: str) -> np.ndarray:
            col_idx = farm_cols.index(value_col)
            starts = dataset.sample_input_start_idx[train_idx]
            ends = dataset.sample_input_end_idx[train_idx]
            out = np.full((train_idx.size, lag_count), np.nan, dtype=np.float64)
            for row_idx, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
                seq = dataset.farm_values[int(start) : int(end) + 1, col_idx].astype(np.float64, copy=False)
                if seq.size <= 0:
                    continue
                take = min(seq.size, lag_count)
                out[row_idx, -take:] = seq[-take:]
                if take < lag_count:
                    out[row_idx, : lag_count - take] = seq[0]
            return out

        power_history = _history(power_col)
        wind_history = _history(wind_col)
        power_aux_history = _history(power_aux_col) if power_aux_col is not None else None
        wind_aux_history = _history(wind_aux_col) if wind_aux_col is not None else None
        target_power = dataset.farm_targets[dataset.sample_origin_idx[train_idx], : int(dataset.horizon)]
        lead_minutes = np.arange(1, int(dataset.horizon) + 1, dtype=np.int64) * int(dataset.step_out_min)
        coef, bias = fit_direct_multi_step_arx_coeffs(
            power_history=power_history,
            wind_history=wind_history,
            power_aux_history=power_aux_history,
            wind_aux_history=wind_aux_history,
            origin_times=np.asarray([dataset.sample_times[int(i)] for i in train_idx], dtype="datetime64[ns]"),
            lead_minutes=lead_minutes,
            target_power=target_power,
            ridge=1e-3,
        )
        clip_max_kw = None
        power_capacity_mw = float(dataset.meta.get("power_capacity_mw", 0.0) or 0.0)
        if power_capacity_mw > 0:
            clip_max_kw = power_capacity_mw * 1000.0
        elif np.isfinite(target_power).any():
            clip_max_kw = float(np.nanmax(target_power))
        payload: dict[str, Any] = {
            "mode": "arx_wind",
            "lag_count": int(lag_count),
            "lead_minutes": lead_minutes.astype(np.int64, copy=False).tolist(),
            "power_col": power_col,
            "wind_col": wind_col,
            "power_aux_col": power_aux_col,
            "wind_aux_col": wind_aux_col,
            "coef": coef.astype(np.float32, copy=False).tolist(),
            "bias": bias.astype(np.float32, copy=False).tolist(),
            "clip_max_kw": None if clip_max_kw is None else float(clip_max_kw),
        }
        return payload, None
    power_idx = dataset.farm_feature_columns.index("power_5min_mean")
    train_origin = dataset.sample_origin_idx[np.asarray(train_indices, dtype=np.int64)]
    current_train = dataset.farm_values[train_origin, power_idx]
    next_train = dataset.farm_targets[train_origin, 0]
    a, b = fit_recursive_ar1_coeffs(current_train, next_train, int(dataset.horizon))
    return a.astype(np.float32, copy=False).tolist(), b.astype(np.float32, copy=False).tolist()


def farm_v2_power_baseline(
    farm_x_raw: torch.Tensor,
    farm_feature_columns: Sequence[str],
    horizon: int,
    mode: str = "persistence",
    baseline_a: dict[str, Any] | Sequence[float] | None = None,
    baseline_b: Sequence[float] | None = None,
    origin_times: np.ndarray | list | None = None,
) -> torch.Tensor:
    power_idx = list(farm_feature_columns).index("power_5min_mean")
    base = farm_x_raw[:, -1, power_idx : power_idx + 1]
    horizon = int(max(horizon, 1))
    mode_key = str(mode).lower()
    if mode_key == "arx_wind" and isinstance(baseline_a, dict):
        farm_cols = list(farm_feature_columns)
        power_col = str(baseline_a.get("power_col", "power_5min_mean"))
        wind_col = str(baseline_a.get("wind_col", "wind_speed_5min_mean"))
        if power_col not in farm_cols or wind_col not in farm_cols or origin_times is None:
            return base.expand(-1, horizon)
        power_hist = farm_x_raw[:, :, farm_cols.index(power_col)].detach().cpu().numpy().astype(np.float64, copy=False)
        wind_hist = farm_x_raw[:, :, farm_cols.index(wind_col)].detach().cpu().numpy().astype(np.float64, copy=False)
        power_aux_col = baseline_a.get("power_aux_col")
        wind_aux_col = baseline_a.get("wind_aux_col")
        power_aux_hist = None
        wind_aux_hist = None
        if isinstance(power_aux_col, str) and power_aux_col in farm_cols:
            power_aux_hist = (
                farm_x_raw[:, :, farm_cols.index(power_aux_col)]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
        if isinstance(wind_aux_col, str) and wind_aux_col in farm_cols:
            wind_aux_hist = (
                farm_x_raw[:, :, farm_cols.index(wind_aux_col)]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
        coef = np.asarray(baseline_a.get("coef", []), dtype=np.float64)
        bias = np.asarray(baseline_a.get("bias", []), dtype=np.float64)
        lead_minutes = np.asarray(baseline_a.get("lead_minutes", []), dtype=np.int64)
        if coef.ndim != 2 or coef.shape[0] <= 0 or bias.ndim != 1 or lead_minutes.ndim != 1:
            return base.expand(-1, horizon)
        usable_horizon = min(horizon, coef.shape[0], bias.shape[0], lead_minutes.shape[0])
        if usable_horizon <= 0:
            return base.expand(-1, horizon)
        pred = apply_direct_multi_step_arx_baseline(
            power_history=power_hist,
            wind_history=wind_hist,
            power_aux_history=power_aux_hist,
            wind_aux_history=wind_aux_hist,
            origin_times=np.asarray(origin_times, dtype="datetime64[ns]"),
            lead_minutes=lead_minutes[:usable_horizon],
            coef=coef[:usable_horizon],
            bias=bias[:usable_horizon],
            clip_max=baseline_a.get("clip_max_kw"),
        )
        pred_t = torch.as_tensor(pred, dtype=farm_x_raw.dtype, device=farm_x_raw.device)
        if usable_horizon < horizon:
            tail = pred_t[:, -1:].expand(-1, horizon - usable_horizon)
            pred_t = torch.cat([pred_t, tail], dim=1)
        return pred_t
    if mode_key != "lag1_ar1" or baseline_a is None or baseline_b is None:
        return base.expand(-1, horizon)
    a = torch.as_tensor(list(baseline_a), dtype=farm_x_raw.dtype, device=farm_x_raw.device).reshape(1, -1)
    b = torch.as_tensor(list(baseline_b), dtype=farm_x_raw.dtype, device=farm_x_raw.device).reshape(1, -1)
    if a.shape[1] < horizon or b.shape[1] < horizon:
        return base.expand(-1, horizon)
    return base * a[:, :horizon] + b[:, :horizon]


def decode_farm_v2_prediction(
    raw_pred: torch.Tensor,
    farm_x_raw: torch.Tensor,
    farm_feature_columns: Sequence[str],
    power_scale: float,
    short_delta_scale: float,
    output_activation: str,
    short_delta_steps: int = 0,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    if output_activation == "softplus":
        pred = torch.nn.functional.softplus(raw_pred) * float(power_scale)
    else:
        pred = raw_pred * float(power_scale)
    base = baseline
    if base is None:
        power_idx = list(farm_feature_columns).index("power_5min_mean")
        base = farm_x_raw[:, -1, power_idx : power_idx + 1].expand(-1, pred.shape[1])
    short_steps = min(int(max(short_delta_steps, 0)), pred.shape[1])
    if short_steps > 0:
        pred = pred.clone()
        pred[:, :short_steps] = raw_pred[:, :short_steps] * float(short_delta_scale) + base[:, :short_steps]
    return pred


def decode_farm_v2_quantiles(
    raw_pred: torch.Tensor,
    farm_x_raw: torch.Tensor,
    farm_feature_columns: Sequence[str],
    power_scale: float,
    short_delta_scale: float,
    output_activation: str,
    horizon: int,
    quantile_count: int,
    short_delta_steps: int = 0,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    raw_q = raw_pred.view(raw_pred.shape[0], horizon, quantile_count)
    if output_activation == "softplus":
        pred_q = torch.nn.functional.softplus(raw_q) * float(power_scale)
    else:
        pred_q = raw_q * float(power_scale)
    base = baseline
    if base is None:
        power_idx = list(farm_feature_columns).index("power_5min_mean")
        base = farm_x_raw[:, -1, power_idx : power_idx + 1].expand(-1, horizon)
    base = base.unsqueeze(-1)
    short_steps = min(int(max(short_delta_steps, 0)), pred_q.shape[1])
    if short_steps > 0:
        pred_q = pred_q.clone()
        pred_q[:, :short_steps, :] = raw_q[:, :short_steps, :] * float(short_delta_scale) + base
    return pred_q
