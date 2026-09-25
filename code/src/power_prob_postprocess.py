from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def quantile_map(
    pred_vals: np.ndarray,
    pred_ref: np.ndarray,
    target_ref: np.ndarray,
    n_quantiles: int,
) -> np.ndarray:
    q = np.linspace(0.0, 1.0, max(20, int(n_quantiles)))
    pred_q = np.quantile(pred_ref, q)
    target_q = np.quantile(target_ref, q)
    pred_q = np.maximum.accumulate(pred_q)
    return np.interp(pred_vals, pred_q, target_q, left=target_q[0], right=target_q[-1])


def apply_power_tail_cdf_match_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tail_min_kw: float,
    n_quantiles: int,
    alpha: float,
    min_samples: int,
    cap_kw: float | None,
) -> tuple[np.ndarray, dict[str, object]]:
    info: dict[str, object] = {
        "enabled": tail_min_kw > 0,
        "tail_min_kw": float(tail_min_kw),
        "n_quantiles": int(n_quantiles),
        "alpha": float(alpha),
        "min_samples": int(min_samples),
        "applied": False,
        "reason": "disabled",
    }
    if tail_min_kw <= 0:
        return y_pred, info

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    true_flat = y_true_arr.reshape(-1)
    pred_flat = y_pred_arr.reshape(-1)
    valid = np.isfinite(true_flat) & np.isfinite(pred_flat)
    tail_mask = valid & (pred_flat >= float(tail_min_kw))
    tail_rows = int(np.count_nonzero(tail_mask))
    info["tail_rows"] = tail_rows
    if tail_rows < max(10, int(min_samples)):
        info["reason"] = "insufficient_tail_samples"
        return y_pred, info

    pred_tail_ref = pred_flat[tail_mask]
    true_tail_ref = true_flat[tail_mask]
    if np.nanmax(pred_tail_ref) <= np.nanmin(pred_tail_ref):
        info["reason"] = "degenerate_tail_pred"
        return y_pred, info

    mapped_tail = quantile_map(
        pred_tail_ref,
        pred_tail_ref,
        true_tail_ref,
        max(20, int(n_quantiles)),
    )
    blend = float(np.clip(alpha, 0.0, 1.0))
    pred_new_flat = pred_flat.copy()
    pred_new_flat[tail_mask] = (1.0 - blend) * pred_tail_ref + blend * mapped_tail
    pred_new_flat = np.clip(pred_new_flat, 0.0, float(cap_kw) if cap_kw and cap_kw > 0 else np.inf)

    mae_before_all = float(np.mean(np.abs(true_flat[valid] - pred_flat[valid]))) if np.any(valid) else float("nan")
    mae_after_all = (
        float(np.mean(np.abs(true_flat[valid] - pred_new_flat[valid]))) if np.any(valid) else float("nan")
    )
    mae_before_tail = float(np.mean(np.abs(true_tail_ref - pred_tail_ref)))
    mae_after_tail = float(np.mean(np.abs(true_tail_ref - pred_new_flat[tail_mask])))
    info.update(
        {
            "applied": True,
            "reason": "applied",
            "mae_before_all": mae_before_all,
            "mae_after_all": mae_after_all,
            "mae_before_tail": mae_before_tail,
            "mae_after_tail": mae_after_tail,
        }
    )
    return pred_new_flat.reshape(y_pred_arr.shape), info


def apply_power_tail_cdf_match_df(
    df_pred: pd.DataFrame,
    *,
    tail_min_kw: float,
    n_quantiles: int,
    alpha: float,
    min_samples: int,
    cap_kw: float | None,
    pred_col: str = "pred_power",
    target_col: str = "target_power",
) -> tuple[pd.DataFrame, dict[str, object]]:
    info: dict[str, object] = {
        "enabled": tail_min_kw > 0,
        "tail_min_kw": float(tail_min_kw),
        "n_quantiles": int(n_quantiles),
        "alpha": float(alpha),
        "min_samples": int(min_samples),
        "applied": False,
        "reason": "disabled",
    }
    if tail_min_kw <= 0 or df_pred.empty:
        return df_pred, info
    if pred_col not in df_pred.columns or target_col not in df_pred.columns:
        info["reason"] = "missing_columns"
        return df_pred, info

    pred_new, info = apply_power_tail_cdf_match_arrays(
        df_pred[target_col].to_numpy(dtype=float, copy=False),
        df_pred[pred_col].to_numpy(dtype=float, copy=False),
        tail_min_kw=tail_min_kw,
        n_quantiles=n_quantiles,
        alpha=alpha,
        min_samples=min_samples,
        cap_kw=cap_kw,
    )
    out = df_pred.copy()
    out[pred_col] = pred_new.reshape(-1)
    return out, info


def add_probabilistic_quantiles_df(
    df_pred: pd.DataFrame,
    quantiles: list[float],
    bins: int,
    min_count: int,
    cap_kw: float | None,
    *,
    pred_col: str = "pred_power",
    target_col: str = "target_power",
    output_prefix: str = "pred_p",
) -> tuple[pd.DataFrame, dict[str, object]]:
    info: dict[str, object] = {
        "quantiles": quantiles,
        "bins": int(bins),
        "min_count": int(min_count),
        "applied": False,
    }
    if df_pred.empty or not quantiles or pred_col not in df_pred.columns or target_col not in df_pred.columns:
        return df_pred, info

    pred = df_pred[pred_col].to_numpy(dtype=float, copy=False)
    target = df_pred[target_col].to_numpy(dtype=float, copy=False)
    resid = target - pred
    valid = np.isfinite(pred) & np.isfinite(resid)
    if not np.any(valid):
        return df_pred, info

    quantiles = sorted({float(q) for q in quantiles if 0.0 < q < 1.0})
    if not quantiles:
        return df_pred, info

    if bins < 1:
        bins = 1
    if cap_kw is not None and np.isfinite(cap_kw) and cap_kw > 0:
        edges = np.linspace(0.0, float(cap_kw), bins + 1)
    else:
        edges = np.quantile(pred[valid], np.linspace(0.0, 1.0, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_idx = np.clip(np.digitize(pred, edges) - 1, 0, bins - 1)
    global_q = np.quantile(resid[valid], quantiles)
    bin_q = np.zeros((bins, len(quantiles)), dtype=float)
    bin_counts = np.zeros(bins, dtype=int)
    for bucket in range(bins):
        mask = (bin_idx == bucket) & valid
        bin_counts[bucket] = int(mask.sum())
        if bin_counts[bucket] >= min_count:
            bin_q[bucket] = np.quantile(resid[mask], quantiles)
        else:
            bin_q[bucket] = global_q

    out = df_pred.copy()
    # Residual-quantile calibration for predictive intervals.
    # Gneiting et al. (2005), "Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics",
    # Monthly Weather Review, DOI: 10.1175/MWR2904.1
    for qi, q in enumerate(quantiles):
        label = int(round(q * 100))
        out[f"{output_prefix}{label:02d}"] = pred + bin_q[bin_idx, qi]
    info["bin_counts"] = bin_counts.tolist()
    info["applied"] = True
    return out, info


def compute_probabilistic_metrics_from_frame(
    df_pred: pd.DataFrame,
    *,
    target_col: str = "target_power",
    prefix: str = "pred_p",
    target_coverage: float = 0.8,
) -> dict[str, Any] | None:
    if df_pred.empty or target_col not in df_pred.columns:
        return None
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{2}})$")
    q_cols: list[tuple[float, str]] = []
    for col in df_pred.columns:
        match = pattern.match(col)
        if match:
            q_cols.append((int(match.group(1)) / 100.0, col))
    q_cols = sorted(q_cols, key=lambda item: item[0])
    if len(q_cols) < 2:
        return None

    quantiles = [q for q, _ in q_cols]
    cols = [c for _, c in q_cols]
    pred_q = df_pred[cols].to_numpy(dtype=float, copy=False)
    y = df_pred[target_col].to_numpy(dtype=float, copy=False)
    valid = np.isfinite(y)
    for idx in range(pred_q.shape[1]):
        valid &= np.isfinite(pred_q[:, idx])
    if not np.any(valid):
        return None
    pred_q = pred_q[valid]
    y = y[valid]

    median_idx = min(range(len(quantiles)), key=lambda idx: abs(quantiles[idx] - 0.5))
    wis_terms: list[np.ndarray] = [np.abs(pred_q[:, median_idx] - y)]
    interval_pairs: list[tuple[float, int, int]] = []
    q_map = {q: i for i, q in enumerate(quantiles)}
    for q in quantiles:
        if q >= 0.5:
            continue
        q_hi = 1.0 - q
        if q_hi in q_map:
            alpha = float(np.clip(1.0 - (q_hi - q), 1e-6, 0.999999))
            interval_pairs.append((alpha, q_map[q], q_map[q_hi]))

    target_cov = float(np.clip(target_coverage, 0.05, 0.99))
    target_alpha = 1.0 - target_cov
    selected_pair = min(interval_pairs, key=lambda item: abs(item[0] - target_alpha)) if interval_pairs else None
    coverage_error = float("nan")
    interval_width = float("nan")
    for alpha, idx_lo, idx_hi in interval_pairs:
        lower = pred_q[:, idx_lo]
        upper = pred_q[:, idx_hi]
        interval = upper - lower
        under = np.maximum(lower - y, 0.0)
        over = np.maximum(y - upper, 0.0)
        score = interval + (2.0 / alpha) * under + (2.0 / alpha) * over
        wis_terms.append((alpha / 2.0) * score)
        if selected_pair is not None and idx_lo == selected_pair[1] and idx_hi == selected_pair[2]:
            inside = ((y >= lower) & (y <= upper)).astype(float)
            coverage_error = float(abs(np.mean(inside) - (1.0 - alpha)))
            interval_width = float(np.mean(interval))

    wis = float(np.mean([np.mean(term) for term in wis_terms]))
    point_mae = float(np.mean(np.abs(pred_q[:, median_idx] - y)))
    return {
        "available_quantiles": quantiles,
        "sample_count": int(y.size),
        "wis": wis,
        "coverage_error": coverage_error,
        "interval_width": interval_width,
        "median_mae": point_mae,
        "target_coverage": float(target_cov),
        "prefix": prefix,
    }


def compute_grouped_probabilistic_metrics_from_frame(
    df_pred: pd.DataFrame,
    *,
    group_columns: dict[str, str],
    target_col: str = "target_power",
    prefix: str = "pred_p",
    target_coverage: float = 0.8,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int | float]]]:
    metrics: dict[str, dict[str, Any]] = {}
    counts: dict[str, dict[str, int | float]] = {}
    for group_name, col in group_columns.items():
        if col not in df_pred.columns:
            continue
        mask = df_pred[col].fillna(0).astype(bool)
        subset = df_pred.loc[mask].copy()
        counts[group_name] = {
            "row_count": int(len(subset)),
            "origin_count": int(subset["origin_time"].nunique()) if "origin_time" in subset.columns else int(len(subset)),
            "ratio": float(mask.mean()) if len(mask) else 0.0,
        }
        if subset.empty:
            continue
        subset_metrics = compute_probabilistic_metrics_from_frame(
            subset,
            target_col=target_col,
            prefix=prefix,
            target_coverage=target_coverage,
        )
        if subset_metrics is not None:
            metrics[group_name] = subset_metrics
    return metrics, counts


def apply_power_postprocess_df(
    df_pred: pd.DataFrame,
    *,
    enable_tail_cdf: bool,
    tail_min_kw: float,
    tail_quantiles: int,
    tail_alpha: float,
    tail_min_samples: int,
    enable_residual_quantiles: bool,
    residual_quantiles: list[float],
    residual_bins: int,
    residual_min_count: int,
    cap_kw: float | None,
    pred_col: str = "pred_power",
    target_col: str = "target_power",
    output_prefix: str = "pred_p",
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = df_pred.copy()
    info: dict[str, object] = {
        "postprocess_enabled": bool(enable_tail_cdf or enable_residual_quantiles),
        "tail_cdf_applied": False,
        "residual_quantile_calibration_applied": False,
        "postprocess_order": [],
        "tail_cdf_info": None,
        "residual_quantile_info": None,
    }
    if enable_tail_cdf:
        out, tail_info = apply_power_tail_cdf_match_df(
            out,
            tail_min_kw=tail_min_kw,
            n_quantiles=tail_quantiles,
            alpha=tail_alpha,
            min_samples=tail_min_samples,
            cap_kw=cap_kw,
            pred_col=pred_col,
            target_col=target_col,
        )
        info["tail_cdf_info"] = tail_info
        info["tail_cdf_applied"] = bool(tail_info.get("applied"))
        info["postprocess_order"].append("tail_cdf")
    if enable_residual_quantiles:
        out, quant_info = add_probabilistic_quantiles_df(
            out,
            residual_quantiles,
            residual_bins,
            residual_min_count,
            cap_kw,
            pred_col=pred_col,
            target_col=target_col,
            output_prefix=output_prefix,
        )
        info["residual_quantile_info"] = quant_info
        info["residual_quantile_calibration_applied"] = bool(quant_info.get("applied"))
        info["postprocess_order"].append("residual_quantile_calibration")
    return out, info
