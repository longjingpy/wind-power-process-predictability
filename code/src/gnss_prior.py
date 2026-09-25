"""
GNSS wind-direction prior utilities for farm_v2 spatial models.

This module separates deterministic GNSS-flow priors from the main forecasting
model so the quality of the direction prior can be audited before it is allowed
to drive a dynamic graph.

References:
- Lee et al. (2025), "Short-term forecasting of 3D wind field based on deep
  learning and water vapor", ESS Open Archive.
  https://essopenarchive.org/users/927137/articles/1346713-short-term-forecasting-of-3d-wind-field-based-on-deep-learning-and-water-vapor
- Zhang et al. (2025), "Communications to Circulations: Real-Time 3D Wind Field
  Prediction Using 5G GNSS Signals and Deep Learning", arXiv.
  https://arxiv.org/abs/2509.16068
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class GNSSPriorTensorConfig:
    u_idx: int = -1
    v_idx: int = -1
    available_idx: int = -1
    upwind_weight_idx: int = -1
    wind_speed_idx: int = -1
    quality_idx: int = -1
    met_quality_idx: int = -1
    product_value_idx: int = -1
    product_value_diff1_idx: int = -1
    product_value_diff3_idx: int = -1
    pressure_idx: int = -1
    temperature_idx: int = -1
    dewpoint_idx: int = -1
    bearing_idx: int = -1
    distance_weight_idx: int = -1
    upwind_alignment_idx: int = -1


def build_gnss_prior_tensor_config(feature_columns: list[str] | tuple[str, ...]) -> GNSSPriorTensorConfig:
    cols = list(feature_columns)

    def _idx(name: str) -> int:
        return cols.index(name) if name in cols else -1

    return GNSSPriorTensorConfig(
        u_idx=_idx("u100") if "u100" in cols else _idx("gnss_proxy_u"),
        v_idx=_idx("v100") if "v100" in cols else _idx("gnss_proxy_v"),
        available_idx=_idx("available"),
        upwind_weight_idx=_idx("upwind_weight"),
        wind_speed_idx=_idx("wind_speed_100"),
        quality_idx=_idx("quality_flag"),
        met_quality_idx=_idx("met_quality_flag"),
        product_value_idx=_idx("product_value"),
        product_value_diff1_idx=_idx("product_value_diff1"),
        product_value_diff3_idx=_idx("product_value_diff3"),
        pressure_idx=_idx("sp_hpa") if "sp_hpa" in cols else _idx("sp_pa"),
        temperature_idx=_idx("t2m_k"),
        dewpoint_idx=_idx("d2m_k"),
        bearing_idx=_idx("station_to_farm_bearing_rad"),
        distance_weight_idx=_idx("distance_weight"),
        upwind_alignment_idx=_idx("upwind_alignment"),
    )


def _safe_angle_delta(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (pred - target)))


def circular_direction_error_deg(pred_dir_rad: np.ndarray, target_dir_rad: np.ndarray) -> np.ndarray:
    delta = _safe_angle_delta(pred_dir_rad, target_dir_rad)
    return np.abs(np.degrees(delta))


def _weighted_average_np(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denom = np.clip(weights.sum(axis=-1, keepdims=True), 1e-8, None)
    return (values * weights).sum(axis=-1, keepdims=False) / denom.squeeze(-1)


def _topk_weight_mask(weights: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or weights.shape[-1] == 0:
        return np.zeros_like(weights, dtype=bool)
    top_k = min(int(top_k), int(weights.shape[-1]))
    order = np.argsort(weights, axis=-1)
    keep = np.zeros_like(weights, dtype=bool)
    top_idx = order[..., -top_k:]
    np.put_along_axis(keep, top_idx, True, axis=-1)
    return keep


def _safe_standardize_np(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return out
    mean = float(np.nanmean(arr[mask]))
    std = float(np.nanstd(arr[mask]))
    if not np.isfinite(std) or std < 1e-6:
        return out
    out[mask] = np.clip((arr[mask] - mean) / std, -3.0, 3.0)
    return out


def _safe_standardize_torch(values: torch.Tensor) -> torch.Tensor:
    arr = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    valid = torch.isfinite(values).float()
    count = valid.sum(dim=-1, keepdim=True)
    mean = (arr * valid).sum(dim=-1, keepdim=True) / count.clamp(min=1.0)
    centered = (arr - mean) * valid
    var = (centered * centered).sum(dim=-1, keepdim=True) / count.clamp(min=1.0)
    std = torch.sqrt(var).clamp(min=1e-6)
    z = centered / std
    z = torch.where(count >= 2.0, z, torch.zeros_like(z))
    return torch.clamp(z, min=-3.0, max=3.0)


def _physics_guided_prior_numpy(
    group: pd.DataFrame,
    *,
    valid: np.ndarray,
    upwind_weight: np.ndarray,
    fallback_u: np.ndarray,
    fallback_v: np.ndarray,
) -> tuple[float, float, float]:
    def _col(name: str) -> np.ndarray:
        if name not in group.columns:
            return np.full(len(group), np.nan, dtype=float)
        return pd.to_numeric(group[name], errors="coerce").to_numpy(dtype=float)

    product = _col("product_value")
    product_diff1 = _col("product_value_diff1")
    product_diff3 = _col("product_value_diff3")
    pressure = _col("sp_hpa") if "sp_hpa" in group.columns else _col("sp_pa")
    temperature = _col("t2m_k")
    dewpoint = _col("d2m_k")
    bearing = _col("station_to_farm_bearing_rad")
    distance_weight = _col("distance_weight")
    upwind_alignment = _col("upwind_alignment")

    moisture = _safe_standardize_np(product)
    tendency = 0.7 * _safe_standardize_np(product_diff1) + 0.3 * _safe_standardize_np(product_diff3)
    dew = _safe_standardize_np(dewpoint)
    dryness = _safe_standardize_np(temperature - dewpoint)
    pressure_term = _safe_standardize_np(pressure)
    signal = moisture + 0.45 * tendency + 0.30 * dew - 0.25 * dryness - 0.15 * pressure_term
    signal = np.where(valid, signal, 0.0)

    if np.isfinite(distance_weight).any():
        dist = np.clip(np.nan_to_num(distance_weight, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    else:
        dist = np.ones_like(signal, dtype=float)
    if np.isfinite(upwind_alignment).any():
        align = np.clip(0.5 + 0.5 * np.nan_to_num(upwind_alignment, nan=0.0, posinf=1.0, neginf=-1.0), 0.0, 1.0)
    else:
        align = np.where(valid, 1.0, 0.0)
    uw = np.clip(np.nan_to_num(upwind_weight, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if np.any(uw > 0):
        uw = uw / np.clip(float(np.nanmax(uw)), 1e-6, None)
    geom = valid.astype(float) * (0.25 + 0.75 * dist) * (0.35 + 0.65 * align) * (0.5 + 0.5 * uw)

    if not np.isfinite(bearing).any():
        anchor_u = float(np.nanmean(fallback_u[valid])) if valid.any() else 0.0
        anchor_v = float(np.nanmean(fallback_v[valid])) if valid.any() else 0.0
        return anchor_u, anchor_v, 0.0

    grad_u = float(np.sum(signal * geom * np.cos(bearing)))
    grad_v = float(np.sum(signal * geom * np.sin(bearing)))
    grad_norm = float(np.sqrt(grad_u**2 + grad_v**2))
    grad_strength = float(
        np.clip(
            grad_norm / np.clip(np.sum(np.abs(signal) * geom), 1e-6, None),
            0.0,
            1.0,
        )
    )

    anchor_weight = np.where(uw > 0, uw, valid.astype(float))
    if float(anchor_weight.sum()) <= 0:
        anchor_weight = valid.astype(float)
    anchor_u = float(np.sum(fallback_u * anchor_weight) / np.clip(anchor_weight.sum(), 1e-6, None))
    anchor_v = float(np.sum(fallback_v * anchor_weight) / np.clip(anchor_weight.sum(), 1e-6, None))
    anchor_speed = float(np.sqrt(anchor_u**2 + anchor_v**2))
    anchor_norm = float(np.sqrt(anchor_u**2 + anchor_v**2))
    if grad_norm <= 1e-6:
        return anchor_u, anchor_v, 0.0

    grad_dir_u = grad_u / np.clip(grad_norm, 1e-6, None)
    grad_dir_v = grad_v / np.clip(grad_norm, 1e-6, None)
    if anchor_norm > 1e-6:
        anchor_dir_u = anchor_u / anchor_norm
        anchor_dir_v = anchor_v / anchor_norm
        alignment = float(np.clip(anchor_dir_u * grad_dir_u + anchor_dir_v * grad_dir_v, -1.0, 1.0))
    else:
        alignment = 0.0
    blend = float(np.clip(0.15 + 0.55 * grad_strength * (0.5 + 0.5 * max(alignment, 0.0)), 0.15, 0.70))
    target_speed = anchor_speed if anchor_speed > 1e-6 else 1.0
    prior_u = (1.0 - blend) * anchor_u + blend * grad_dir_u * target_speed
    prior_v = (1.0 - blend) * anchor_v + blend * grad_dir_v * target_speed
    return float(prior_u), float(prior_v), float(np.clip(0.5 + 0.5 * max(alignment, 0.0), 0.0, 1.0) * grad_strength)


def _physics_guided_prior_tensor(
    station_x: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    config: GNSSPriorTensorConfig,
    *,
    fallback_u: torch.Tensor,
    fallback_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _feat(idx: int) -> torch.Tensor:
        if idx < 0 or idx >= station_x.shape[-1]:
            return torch.zeros_like(fallback_u)
        return torch.nan_to_num(station_x[..., idx], nan=0.0, posinf=0.0, neginf=0.0)

    product = _feat(config.product_value_idx)
    product_diff1 = _feat(config.product_value_diff1_idx)
    product_diff3 = _feat(config.product_value_diff3_idx)
    pressure = _feat(config.pressure_idx)
    temperature = _feat(config.temperature_idx)
    dewpoint = _feat(config.dewpoint_idx)
    bearing = _feat(config.bearing_idx)
    distance_weight = torch.clamp(_feat(config.distance_weight_idx), min=0.0, max=1.0)
    if config.upwind_alignment_idx >= 0:
        upwind_alignment = torch.clamp(0.5 + 0.5 * _feat(config.upwind_alignment_idx), min=0.0, max=1.0)
    else:
        upwind_alignment = valid.float()

    moisture = _safe_standardize_torch(product)
    tendency = 0.7 * _safe_standardize_torch(product_diff1) + 0.3 * _safe_standardize_torch(product_diff3)
    dew = _safe_standardize_torch(dewpoint)
    dryness = _safe_standardize_torch(temperature - dewpoint)
    pressure_term = _safe_standardize_torch(pressure)
    signal = (moisture + 0.45 * tendency + 0.30 * dew - 0.25 * dryness - 0.15 * pressure_term) * valid.float()

    max_weight = weight.amax(dim=-1, keepdim=True).clamp(min=1e-6)
    weight_norm = torch.where(max_weight > 0, weight / max_weight, torch.zeros_like(weight))
    geom = valid.float() * (0.25 + 0.75 * distance_weight) * (0.35 + 0.65 * upwind_alignment) * (0.5 + 0.5 * weight_norm)
    grad_u = (signal * geom * torch.cos(bearing)).sum(dim=-1)
    grad_v = (signal * geom * torch.sin(bearing)).sum(dim=-1)
    grad_norm = torch.sqrt(grad_u * grad_u + grad_v * grad_v).clamp(min=1e-6)
    grad_strength = torch.clamp(
        grad_norm / ((signal.abs() * geom).sum(dim=-1).clamp(min=1e-6)),
        min=0.0,
        max=1.0,
    )

    anchor_weight = torch.where(weight.sum(dim=-1, keepdim=True) > 0, weight, valid.float())
    anchor_denom = anchor_weight.sum(dim=-1).clamp(min=1e-6)
    anchor_u = (fallback_u * anchor_weight).sum(dim=-1) / anchor_denom
    anchor_v = (fallback_v * anchor_weight).sum(dim=-1) / anchor_denom
    anchor_norm = torch.sqrt(anchor_u * anchor_u + anchor_v * anchor_v).clamp(min=1e-6)
    grad_dir_u = grad_u / grad_norm
    grad_dir_v = grad_v / grad_norm
    anchor_dir_u = anchor_u / anchor_norm
    anchor_dir_v = anchor_v / anchor_norm
    alignment = torch.clamp(anchor_dir_u * grad_dir_u + anchor_dir_v * grad_dir_v, min=-1.0, max=1.0)
    blend = torch.clamp(0.15 + 0.55 * grad_strength * (0.5 + 0.5 * torch.clamp(alignment, min=0.0)), min=0.15, max=0.70)
    target_speed = torch.where(anchor_norm > 1e-6, anchor_norm, torch.ones_like(anchor_norm))
    prior_u = (1.0 - blend) * anchor_u + blend * grad_dir_u * target_speed
    prior_v = (1.0 - blend) * anchor_v + blend * grad_dir_v * target_speed
    extra_conf = torch.clamp((0.5 + 0.5 * torch.clamp(alignment, min=0.0)) * grad_strength, min=0.0, max=1.0)
    return prior_u, prior_v, extra_conf


def compute_station_prior_numpy(
    station_df: pd.DataFrame,
    *,
    source: str = "strict_upwind",
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Aggregate station u/v into a farm-level deterministic prior per timestamp.

    The prior is intentionally simple and auditable. It is meant to answer
    whether GNSS-derived direction information is usable before it is injected
    into the dynamic graph.
    """

    work = station_df.copy()
    for col in (
        "u100",
        "v100",
        "available",
        "upwind_weight",
        "wind_speed_100",
        "quality_flag",
        "met_quality_flag",
        "product_value",
        "product_value_diff1",
        "product_value_diff3",
        "sp_pa",
        "sp_hpa",
        "t2m_k",
        "d2m_k",
        "station_to_farm_bearing_rad",
        "distance_weight",
        "upwind_alignment",
    ):
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["datetime", "u100", "v100"])
    if work.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "prior_u100",
                "prior_v100",
                "prior_dir_rad",
                "prior_speed_100",
                "prior_confidence",
                "station_count",
                "available_ratio",
                "upwind_positive_ratio",
                "upwind_weight_sum",
                "gnss_prior_source",
            ]
        )
    work["available"] = work["available"].fillna(0.0)
    work["upwind_weight"] = work["upwind_weight"].fillna(0.0)
    rows: list[dict[str, Any]] = []
    for dt, group in work.groupby("datetime", sort=True):
        valid = group["available"].to_numpy(dtype=float) > 0.5
        if not valid.any():
            continue
        u = group["u100"].to_numpy(dtype=float)
        v = group["v100"].to_numpy(dtype=float)
        upwind_weight = np.clip(group["upwind_weight"].to_numpy(dtype=float), 0.0, None)
        source_name = str(source or "strict_upwind")
        if source_name == "mean":
            weights = valid.astype(float)
        elif source_name == "topk_weighted":
            keep = _topk_weight_mask(upwind_weight.reshape(1, -1), int(top_k)).reshape(-1)
            weights = np.where(keep, upwind_weight, 0.0)
            if float(weights.sum()) <= 0:
                weights = valid.astype(float)
        else:
            weights = upwind_weight.copy()
            if float(weights.sum()) <= 0:
                weights = valid.astype(float)
        weights = weights * valid.astype(float)
        if float(weights.sum()) <= 0:
            continue
        anchor_u = float(np.sum(u * weights) / np.clip(weights.sum(), 1e-8, None))
        anchor_v = float(np.sum(v * weights) / np.clip(weights.sum(), 1e-8, None))
        physics_extra_conf = 1.0
        if source_name == "physics_guided":
            prior_u, prior_v, physics_extra_conf = _physics_guided_prior_numpy(
                group,
                valid=valid,
                upwind_weight=weights,
                fallback_u=u,
                fallback_v=v,
            )
        else:
            prior_u, prior_v = anchor_u, anchor_v
        speed = float(np.sqrt(prior_u**2 + prior_v**2))
        quality_vals = []
        for q_col in ("quality_flag", "met_quality_flag"):
            vals = pd.to_numeric(group[q_col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                quality_vals.append(float(np.nanmean(np.clip(vals, 0.0, 1.0))))
        quality_score = float(np.nanmean(quality_vals)) if quality_vals else 1.0
        available_ratio = float(valid.mean())
        upwind_positive_ratio = float((upwind_weight[valid] > 0).mean()) if valid.any() else 0.0
        weight_sum = float(weights.sum())
        weight_strength = float(np.clip(weight_sum / max(valid.sum(), 1), 0.0, 1.0))
        confidence = float(
            np.clip(
                available_ratio
                * (0.5 + 0.5 * upwind_positive_ratio)
                * (0.5 + 0.5 * quality_score)
                * (0.5 + 0.5 * weight_strength),
                0.0,
                1.0,
            )
        )
        if source_name == "physics_guided":
            confidence = float(np.clip(confidence * (0.6 + 0.4 * physics_extra_conf), 0.0, 1.0))
        rows.append(
            {
                "datetime": dt,
                "prior_u100": prior_u,
                "prior_v100": prior_v,
                "prior_dir_rad": float(np.arctan2(prior_v, prior_u)),
                "prior_speed_100": speed,
                "prior_confidence": confidence,
                "station_count": int(valid.sum()),
                "available_ratio": available_ratio,
                "upwind_positive_ratio": upwind_positive_ratio,
                "upwind_weight_sum": weight_sum,
                "gnss_prior_source": source_name,
            }
        )
    return pd.DataFrame(rows)


def evaluate_prior_against_farm_flow(
    farm_df: pd.DataFrame,
    prior_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = farm_df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    merged = work.merge(prior_df, on="datetime", how="inner")
    merged = merged.dropna(subset=["era5_flow_u100", "era5_flow_v100", "prior_u100", "prior_v100"])
    if merged.empty:
        return merged, {
            "rows": 0,
            "dir_mae_deg": float("nan"),
            "dir_median_deg": float("nan"),
            "dir_p90_deg": float("nan"),
            "u_corr": float("nan"),
            "v_corr": float("nan"),
            "speed_corr": float("nan"),
            "prior_confidence_mean": float("nan"),
        }
    merged["farm_dir_rad"] = np.arctan2(merged["era5_flow_v100"], merged["era5_flow_u100"])
    merged["prior_dir_mae_deg"] = circular_direction_error_deg(
        merged["prior_dir_rad"].to_numpy(dtype=float),
        merged["farm_dir_rad"].to_numpy(dtype=float),
    )
    merged["farm_speed_100"] = np.sqrt(merged["era5_flow_u100"] ** 2 + merged["era5_flow_v100"] ** 2)
    summary = {
        "rows": int(len(merged)),
        "dir_mae_deg": float(merged["prior_dir_mae_deg"].mean()),
        "dir_median_deg": float(merged["prior_dir_mae_deg"].median()),
        "dir_p90_deg": float(merged["prior_dir_mae_deg"].quantile(0.9)),
        "u_corr": float(merged["era5_flow_u100"].corr(merged["prior_u100"])),
        "v_corr": float(merged["era5_flow_v100"].corr(merged["prior_v100"])),
        "speed_corr": float(merged["farm_speed_100"].corr(merged["prior_speed_100"])),
        "prior_confidence_mean": float(merged["prior_confidence"].mean()),
    }
    return merged, summary


def fit_calibrated_prior(
    prior_frame: pd.DataFrame,
    farm_frame: pd.DataFrame,
    *,
    train_end_month: str,
    ridge: float = 1e-3,
) -> pd.DataFrame:
    """
    Fit a light linear calibrator from GNSS prior + recent farm context to site u/v.

    A small ridge regularizer is used to stabilize the regression when a month
    contains sparse GNSS support. The goal is not model complexity, but a
    transparent bias correction before the dynamic graph consumes the prior.
    """

    prior = prior_frame.copy()
    farm = farm_frame.copy()
    prior["datetime"] = pd.to_datetime(prior["datetime"], errors="coerce")
    farm["datetime"] = pd.to_datetime(farm["datetime"], errors="coerce")
    keep_cols = [
        "datetime",
        "era5_flow_u100",
        "era5_flow_v100",
        "power_5min_mean",
        "wind_speed_5min_mean",
        "farm_power_diff1",
        "farm_wind_diff1",
    ]
    merged = prior.merge(farm[keep_cols], on="datetime", how="inner")
    merged = merged.dropna(
        subset=[
            "prior_u100",
            "prior_v100",
            "prior_confidence",
            "era5_flow_u100",
            "era5_flow_v100",
        ]
    )
    if merged.empty:
        return pd.DataFrame()
    merged["month"] = merged["datetime"].dt.to_period("M").astype(str)
    x_cols = [
        "prior_u100",
        "prior_v100",
        "prior_speed_100",
        "prior_confidence",
        "station_count",
        "available_ratio",
        "upwind_positive_ratio",
        "upwind_weight_sum",
        "power_5min_mean",
        "wind_speed_5min_mean",
        "farm_power_diff1",
        "farm_wind_diff1",
    ]
    for col in x_cols:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    train_mask = merged["month"] <= str(train_end_month)
    if not train_mask.any():
        return pd.DataFrame()
    x_train = merged.loc[train_mask, x_cols].to_numpy(dtype=float)
    x_train = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    xtx = x_train.T @ x_train + float(ridge) * np.eye(x_train.shape[1])
    xty_u = x_train.T @ merged.loc[train_mask, "era5_flow_u100"].to_numpy(dtype=float)
    xty_v = x_train.T @ merged.loc[train_mask, "era5_flow_v100"].to_numpy(dtype=float)
    beta_u = np.linalg.solve(xtx, xty_u)
    beta_v = np.linalg.solve(xtx, xty_v)
    x_all = merged[x_cols].to_numpy(dtype=float)
    x_all = np.concatenate([np.ones((x_all.shape[0], 1)), x_all], axis=1)
    merged["prior_u100"] = x_all @ beta_u
    merged["prior_v100"] = x_all @ beta_v
    merged["prior_speed_100"] = np.sqrt(merged["prior_u100"] ** 2 + merged["prior_v100"] ** 2)
    merged["prior_dir_rad"] = np.arctan2(merged["prior_v100"], merged["prior_u100"])
    merged["gnss_prior_source"] = "calibrated"
    return merged[
        [
            "datetime",
            "prior_u100",
            "prior_v100",
            "prior_dir_rad",
            "prior_speed_100",
            "prior_confidence",
            "station_count",
            "available_ratio",
            "upwind_positive_ratio",
            "upwind_weight_sum",
            "gnss_prior_source",
        ]
    ].copy()


TENSOR_CALIBRATION_X_COLS = [
    "prior_u100",
    "prior_v100",
    "prior_speed_100",
    "prior_confidence",
    "station_count",
    "available_ratio",
    "upwind_positive_ratio",
    "upwind_weight_sum",
]


def fit_tensor_calibrated_prior_state(
    station_df: pd.DataFrame,
    farm_df: pd.DataFrame,
    *,
    train_datetimes: pd.Series | list[pd.Timestamp] | np.ndarray,
    base_source: str = "strict_upwind",
    top_k: int = 3,
    ridge: float = 1e-3,
) -> dict[str, Any]:
    """
    Fit a lightweight GNSS-only linear calibration state for tensor-side priors.

    The tensor model cannot safely access the full pandas calibration frame at
    inference time, so we fit a small linear map offline and store the learned
    coefficients in model metadata for reproducible deployment.
    """

    base_source = str(base_source or "strict_upwind")
    prior = compute_station_prior_numpy(
        station_df,
        source=base_source,
        top_k=int(top_k),
    )
    if prior.empty and base_source != "mean":
        prior = compute_station_prior_numpy(
            station_df,
            source="mean",
            top_k=int(top_k),
        )
        base_source = "mean"
    if prior.empty:
        raise ValueError("Cannot fit tensor calibrated prior: empty deterministic base prior.")
    farm = farm_df.copy()
    farm["datetime"] = pd.to_datetime(farm["datetime"], errors="coerce")
    keep_cols = ["datetime", "era5_flow_u100", "era5_flow_v100"]
    merged = prior.merge(farm[keep_cols], on="datetime", how="inner")
    merged = merged.dropna(subset=["prior_u100", "prior_v100", "era5_flow_u100", "era5_flow_v100"]).copy()
    if merged.empty:
        raise ValueError("Cannot fit tensor calibrated prior: no overlapping farm flow rows.")
    train_times = pd.to_datetime(pd.Series(train_datetimes), errors="coerce").dropna().dt.floor("min")
    if train_times.empty:
        raise ValueError("Cannot fit tensor calibrated prior: empty training datetimes.")
    merged["datetime"] = pd.to_datetime(merged["datetime"], errors="coerce").dt.floor("min")
    train_time_ns = train_times.astype("int64").to_numpy()
    train_mask = merged["datetime"].astype("int64").isin(train_time_ns)
    fit_scope = "train_window"
    if not train_mask.any() and base_source != "mean":
        prior = compute_station_prior_numpy(
            station_df,
            source="mean",
            top_k=int(top_k),
        )
        merged = prior.merge(farm[keep_cols], on="datetime", how="inner")
        merged = merged.dropna(subset=["prior_u100", "prior_v100", "era5_flow_u100", "era5_flow_v100"]).copy()
        merged["datetime"] = pd.to_datetime(merged["datetime"], errors="coerce").dt.floor("min")
        train_mask = merged["datetime"].astype("int64").isin(train_time_ns)
        base_source = "mean"
    if not train_mask.any():
        # Small smoke windows can legitimately land before the first valid
        # GNSS-derived u/v rows. In that case fall back to the full overlap
        # pool so the calibrated prior remains usable for engineering probes.
        train_mask = np.ones(len(merged), dtype=bool)
        fit_scope = "all_overlap_fallback"
    x_train = merged.loc[train_mask, TENSOR_CALIBRATION_X_COLS].to_numpy(dtype=float)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_train = np.concatenate([np.ones((x_train.shape[0], 1), dtype=float), x_train], axis=1)
    xtx = x_train.T @ x_train + float(ridge) * np.eye(x_train.shape[1], dtype=float)
    xty_u = x_train.T @ merged.loc[train_mask, "era5_flow_u100"].to_numpy(dtype=float)
    xty_v = x_train.T @ merged.loc[train_mask, "era5_flow_v100"].to_numpy(dtype=float)
    beta_u = np.linalg.solve(xtx, xty_u)
    beta_v = np.linalg.solve(xtx, xty_v)
    return {
        "kind": "gnss_only_linear",
        "base_source": str(base_source),
        "top_k": int(top_k),
        "ridge": float(ridge),
        "fit_scope": str(fit_scope),
        "x_cols": list(TENSOR_CALIBRATION_X_COLS),
        "beta_u": beta_u.tolist(),
        "beta_v": beta_v.tolist(),
    }


def apply_tensor_calibrated_prior_tensor(
    prior_uv: torch.Tensor,
    prior_confidence: torch.Tensor,
    station_count: torch.Tensor,
    available_ratio: torch.Tensor,
    upwind_positive_ratio: torch.Tensor,
    upwind_weight_sum: torch.Tensor,
    calibration_state: dict[str, Any],
) -> torch.Tensor:
    """
    Apply an offline-fitted GNSS-only linear calibration in tensor space.
    """

    x_cols = list(calibration_state.get("x_cols") or TENSOR_CALIBRATION_X_COLS)
    feature_map = {
        "prior_u100": prior_uv[..., 0],
        "prior_v100": prior_uv[..., 1],
        "prior_speed_100": torch.sqrt(prior_uv[..., 0] ** 2 + prior_uv[..., 1] ** 2),
        "prior_confidence": prior_confidence,
        "station_count": station_count,
        "available_ratio": available_ratio,
        "upwind_positive_ratio": upwind_positive_ratio,
        "upwind_weight_sum": upwind_weight_sum,
    }
    x = torch.stack([feature_map[name] for name in x_cols], dim=-1)
    ones = torch.ones_like(x[..., :1])
    x = torch.cat([ones, x], dim=-1)
    beta_u = torch.as_tensor(calibration_state["beta_u"], dtype=prior_uv.dtype, device=prior_uv.device)
    beta_v = torch.as_tensor(calibration_state["beta_v"], dtype=prior_uv.dtype, device=prior_uv.device)
    pred_u = torch.matmul(x, beta_u)
    pred_v = torch.matmul(x, beta_v)
    return torch.stack([pred_u, pred_v], dim=-1)


def _confidence_from_components(
    valid: torch.Tensor,
    weight: torch.Tensor,
    quality: torch.Tensor | None,
) -> torch.Tensor:
    coverage = valid.float().mean(dim=-1)
    positive_ratio = ((weight > 0).float() * valid.float()).sum(dim=-1) / valid.float().sum(dim=-1).clamp(min=1.0)
    weight_strength = weight.sum(dim=-1) / valid.float().sum(dim=-1).clamp(min=1.0)
    weight_strength = torch.clamp(weight_strength, min=0.0, max=1.0)
    if quality is None:
        quality_score = torch.ones_like(coverage)
    else:
        quality_score = (quality * valid.float()).sum(dim=-1) / valid.float().sum(dim=-1).clamp(min=1.0)
        quality_score = torch.clamp(quality_score, min=0.0, max=1.0)
    confidence = coverage * (0.5 + 0.5 * positive_ratio) * (0.5 + 0.5 * quality_score) * (0.5 + 0.5 * weight_strength)
    return torch.clamp(confidence, min=0.0, max=1.0)


def estimate_station_prior_tensor(
    station_x: torch.Tensor,
    station_mask: torch.Tensor,
    config: GNSSPriorTensorConfig,
    *,
    source: str = "strict_upwind",
    top_k: int = 3,
    calibration_state: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate a deterministic site-level wind prior from station tensors.

    Returns:
      prior_uv_seq: [B, T, 2]
      prior_conf_seq: [B, T]
    """

    if config.u_idx < 0 or config.v_idx < 0 or station_x.numel() == 0:
        batch = station_x.shape[0]
        steps = station_x.shape[1] if station_x.dim() >= 2 else 1
        prior_uv = torch.zeros((batch, steps, 2), dtype=station_x.dtype, device=station_x.device)
        prior_conf = torch.zeros((batch, steps), dtype=station_x.dtype, device=station_x.device)
        return prior_uv, prior_conf
    valid = station_mask.to(dtype=torch.bool)
    if 0 <= config.available_idx < station_x.shape[-1]:
        available = station_x[..., config.available_idx] > 0.5
        valid = valid & available
    u = station_x[..., config.u_idx]
    v = station_x[..., config.v_idx]
    finite_uv = torch.isfinite(u) & torch.isfinite(v)
    valid = valid & finite_uv
    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if 0 <= config.upwind_weight_idx < station_x.shape[-1]:
        upwind_weight = torch.nan_to_num(
            torch.clamp(station_x[..., config.upwind_weight_idx], min=0.0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    else:
        upwind_weight = torch.zeros_like(u)
    quality_terms = []
    if 0 <= config.quality_idx < station_x.shape[-1]:
        quality_terms.append(
            torch.nan_to_num(
                torch.clamp(station_x[..., config.quality_idx], min=0.0, max=1.0),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
        )
    if 0 <= config.met_quality_idx < station_x.shape[-1]:
        quality_terms.append(
            torch.nan_to_num(
                torch.clamp(station_x[..., config.met_quality_idx], min=0.0, max=1.0),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
        )
    quality = torch.stack(quality_terms, dim=0).mean(dim=0) if quality_terms else None
    source = str(source or "strict_upwind")
    base_source = str(source)
    if source == "calibrated":
        if not calibration_state:
            raise ValueError("gnss_prior_source='calibrated' requires a tensor calibration state.")
        base_source = str(calibration_state.get("base_source", "strict_upwind") or "strict_upwind")

    if base_source == "mean":
        weight = valid.float()
    elif base_source == "topk_weighted":
        weight = upwind_weight * valid.float()
        if top_k > 0 and weight.shape[-1] > top_k:
            keep_values, keep_idx = torch.topk(weight, k=int(top_k), dim=-1)
            keep_mask = torch.zeros_like(weight, dtype=torch.bool)
            keep_mask.scatter_(-1, keep_idx, keep_values > 0)
            weight = torch.where(keep_mask, weight, torch.zeros_like(weight))
            zero_weight = weight.sum(dim=-1, keepdim=True) <= 0
            weight = torch.where(zero_weight, valid.float(), weight)
        else:
            zero_weight = weight.sum(dim=-1, keepdim=True) <= 0
            weight = torch.where(zero_weight, valid.float(), weight)
    else:
        weight = upwind_weight * valid.float()
        zero_weight = weight.sum(dim=-1, keepdim=True) <= 0
        weight = torch.where(zero_weight, valid.float(), weight)
    denom = weight.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    anchor_u = (u * weight).sum(dim=-1, keepdim=True) / denom
    anchor_v = (v * weight).sum(dim=-1, keepdim=True) / denom
    if base_source == "physics_guided":
        physics_u, physics_v, physics_extra_conf = _physics_guided_prior_tensor(
            station_x,
            valid,
            weight,
            config,
            fallback_u=u,
            fallback_v=v,
        )
        prior_u = physics_u.unsqueeze(-1)
        prior_v = physics_v.unsqueeze(-1)
    else:
        prior_u = anchor_u
        prior_v = anchor_v
    prior_uv = torch.cat([prior_u, prior_v], dim=-1)
    prior_conf = _confidence_from_components(valid, weight, quality)
    if base_source == "physics_guided":
        prior_conf = torch.clamp(prior_conf * (0.6 + 0.4 * physics_extra_conf), min=0.0, max=1.0)
    if source == "calibrated":
        valid_float = valid.float()
        station_count = valid_float.sum(dim=-1)
        available_ratio = valid_float.mean(dim=-1)
        upwind_positive_ratio = ((upwind_weight > 0).float() * valid_float).sum(dim=-1) / valid_float.sum(dim=-1).clamp(min=1.0)
        upwind_weight_sum = weight.sum(dim=-1)
        prior_uv = apply_tensor_calibrated_prior_tensor(
            prior_uv,
            prior_conf,
            station_count,
            available_ratio,
            upwind_positive_ratio,
            upwind_weight_sum,
            calibration_state,
        )
    prior_uv = torch.nan_to_num(prior_uv, nan=0.0, posinf=0.0, neginf=0.0)
    prior_conf = torch.nan_to_num(prior_conf, nan=0.0, posinf=0.0, neginf=0.0)
    return prior_uv, prior_conf
