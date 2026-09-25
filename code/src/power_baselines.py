"""
Simple statistical power baselines shared by eval/report.

References:
- Yule (1927), "On a method of investigating periodicities in disturbed series",
  Philosophical Transactions of the Royal Society A.
  https://doi.org/10.1098/rsta.1927.0007
- Nielsen et al. (1996), "Wind Power Prediction Using ARX Models and Neural Networks",
  International Conference: Modelling, Identification and Control.
  https://enfor.dk/pub/bib/paprep/public/arx-nn-iasted-1996.pdf
- Chevillon (2007), "Direct Multi-step Estimation and Forecasting",
  Journal of Economic Surveys.
  https://doi.org/10.1111/j.1467-6419.2007.00518.x
- Lei et al. (2013), "A review on the forecasting of wind speed and generated power",
  Renewable and Sustainable Energy Reviews.
  https://doi.org/10.1016/j.rser.2013.08.062
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_recursive_ar1_coeffs(
    current_power: np.ndarray,
    next_power: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(current_power, dtype=float).reshape(-1)
    y = np.asarray(next_power, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    a1 = 1.0
    b1 = 0.0
    if x.size >= 2:
        mean_x = float(x.mean())
        mean_y = float(y.mean())
        var_x = float(np.mean(np.square(x)) - mean_x**2)
        if var_x >= 1e-6:
            cov_xy = float(np.mean(x * y) - mean_x * mean_y)
            a1 = cov_xy / var_x
            b1 = mean_y - a1 * mean_x

    a = np.ones(int(horizon), dtype=np.float64)
    b = np.zeros(int(horizon), dtype=np.float64)
    for k in range(int(horizon)):
        h = k + 1
        ah = float(a1**h)
        a[k] = ah
        if abs(1.0 - a1) < 1e-8:
            b[k] = float(b1 * h)
        else:
            b[k] = float(b1 * (1.0 - ah) / (1.0 - a1))
    return a, b


def apply_recursive_linear_baseline(
    current_power: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    base = np.asarray(current_power, dtype=float).reshape(-1, 1)
    a_arr = np.asarray(a, dtype=float).reshape(1, -1)
    b_arr = np.asarray(b, dtype=float).reshape(1, -1)
    return base * a_arr + b_arr


def build_lagged_history_matrix(
    values: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    value_col: int,
    lag_count: int,
) -> np.ndarray:
    """
    Extract fixed-length lag histories from a time-indexed feature matrix.

    The returned design matrix is ordered from oldest -> newest lag so direct
    multi-step regressors can operate on the full recent power trajectory rather
    than only the last single lag.
    """

    arr = np.asarray(values, dtype=float)
    starts = np.asarray(start_indices, dtype=np.int64).reshape(-1)
    ends = np.asarray(end_indices, dtype=np.int64).reshape(-1)
    lag_count = int(max(lag_count, 1))
    history = np.full((starts.size, lag_count), np.nan, dtype=np.float64)
    for row_idx, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        seq = arr[int(start) : int(end) + 1, int(value_col)].reshape(-1)
        if seq.size <= 0:
            continue
        take = min(seq.size, lag_count)
        history[row_idx, -take:] = seq[-take:]
        if take < lag_count:
            # Repeat the earliest available value when the window is shorter
            # than the requested lag depth; this keeps the baseline deterministic
            # without leaking future values.
            history[row_idx, : lag_count - take] = seq[0]
    return history


def fit_direct_multi_lag_linear_coeffs(
    history_power: np.ndarray,
    target_power: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit direct multi-step linear regressors on recent power history.

    Direct multi-horizon regression is typically stronger than a recursive
    one-step AR(1) extrapolation for multi-step forecasting because each lead is
    estimated against its own target horizon.
    Chevillon (2007), Journal of Economic Surveys.
    """

    x = np.asarray(history_power, dtype=float)
    y = np.asarray(target_power, dtype=float)
    if x.ndim != 2:
        raise ValueError("history_power must be 2D [samples, lags].")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError("target_power must be 1D or 2D [samples, horizon].")
    if x.shape[0] != y.shape[0]:
        raise ValueError("history_power and target_power must have the same sample count.")

    sample_count, lag_count = x.shape
    horizon = y.shape[1]
    coef = np.zeros((horizon, lag_count), dtype=np.float64)
    bias = np.zeros(horizon, dtype=np.float64)

    fallback_idx = lag_count - 1
    for step in range(horizon):
        step_target = y[:, step]
        mask = np.all(np.isfinite(x), axis=1) & np.isfinite(step_target)
        if int(mask.sum()) < max(3, lag_count + 1):
            coef[step, fallback_idx] = 1.0
            bias[step] = 0.0
            continue

        x_fit = x[mask]
        y_fit = step_target[mask]
        design = np.concatenate([x_fit, np.ones((x_fit.shape[0], 1), dtype=np.float64)], axis=1)
        xtx = design.T @ design
        reg = np.eye(xtx.shape[0], dtype=np.float64) * float(max(ridge, 0.0))
        reg[-1, -1] = 0.0  # do not penalize the intercept
        xty = design.T @ y_fit
        try:
            beta = np.linalg.solve(xtx + reg, xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xtx + reg) @ xty
        coef[step] = beta[:-1]
        bias[step] = float(beta[-1])

    return coef, bias


def apply_direct_multi_lag_linear_baseline(
    history_power: np.ndarray,
    coef: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    x = np.asarray(history_power, dtype=float)
    coef_arr = np.asarray(coef, dtype=float)
    bias_arr = np.asarray(bias, dtype=float).reshape(1, -1)
    if x.ndim != 2:
        raise ValueError("history_power must be 2D [samples, lags].")
    if coef_arr.ndim != 2:
        raise ValueError("coef must be 2D [horizon, lags].")
    if x.shape[1] != coef_arr.shape[1]:
        raise ValueError("history_power lag dimension must match coef.")
    return x @ coef_arr.T + bias_arr


def build_future_calendar_harmonics(
    origin_times: np.ndarray | list,
    lead_minutes: int,
) -> np.ndarray:
    """
    Build deterministic future-time harmonics for direct multi-step baselines.

    Nielsen et al. (1996) showed that short-term wind-power ARX models benefit
    from exogenous wind information together with a diurnal profile. We encode
    the future timestamp using hour-of-day and day-of-week harmonics so the
    baseline remains causal at inference time.
    """

    future = pd.to_datetime(origin_times) + pd.to_timedelta(int(lead_minutes), unit="m")
    hour_float = future.hour.to_numpy(dtype=np.float64) + future.minute.to_numpy(dtype=np.float64) / 60.0
    dow_float = future.dayofweek.to_numpy(dtype=np.float64) + hour_float / 24.0
    hour_rad = 2.0 * np.pi * hour_float / 24.0
    dow_rad = 2.0 * np.pi * dow_float / 7.0
    return np.column_stack(
        [
            np.sin(hour_rad),
            np.cos(hour_rad),
            np.sin(dow_rad),
            np.cos(dow_rad),
        ]
    ).astype(np.float64, copy=False)


def build_direct_arx_design_matrix(
    *,
    power_history: np.ndarray,
    wind_history: np.ndarray,
    origin_times: np.ndarray | list,
    lead_minutes: int,
    power_aux_history: np.ndarray | None = None,
    wind_aux_history: np.ndarray | None = None,
) -> np.ndarray:
    """
    Assemble a direct multi-step ARX-style design matrix.

    The design follows the wind-power forecasting literature by combining
    endogenous power lags with exogenous wind-speed lags and deterministic
    calendar terms. We fit each horizon directly rather than recursively.
    Nielsen et al. (1996), MIC.
    Chevillon (2007), Journal of Economic Surveys.
    """

    blocks: list[np.ndarray] = []
    power_hist = np.asarray(power_history, dtype=np.float64)
    wind_hist = np.asarray(wind_history, dtype=np.float64)
    if power_hist.ndim != 2 or wind_hist.ndim != 2:
        raise ValueError("power_history and wind_history must be 2D [samples, lags].")
    blocks.append(np.sqrt(np.clip(power_hist, a_min=0.0, a_max=None)))
    if power_aux_history is not None:
        aux = np.asarray(power_aux_history, dtype=np.float64)
        if aux.ndim != 2:
            raise ValueError("power_aux_history must be 2D [samples, lags].")
        blocks.append(np.sqrt(np.clip(aux, a_min=0.0, a_max=None)))
    blocks.append(np.sqrt(np.clip(wind_hist, a_min=0.0, a_max=None)))
    blocks.append(np.clip(wind_hist, a_min=0.0, a_max=None))
    if wind_aux_history is not None:
        aux = np.asarray(wind_aux_history, dtype=np.float64)
        if aux.ndim != 2:
            raise ValueError("wind_aux_history must be 2D [samples, lags].")
        blocks.append(np.sqrt(np.clip(aux, a_min=0.0, a_max=None)))
        blocks.append(np.clip(aux, a_min=0.0, a_max=None))
    blocks.append(build_future_calendar_harmonics(origin_times, int(lead_minutes)))
    return np.concatenate(blocks, axis=1)


def fit_direct_multi_step_arx_coeffs(
    *,
    power_history: np.ndarray,
    wind_history: np.ndarray,
    origin_times: np.ndarray | list,
    lead_minutes: np.ndarray,
    target_power: np.ndarray,
    power_aux_history: np.ndarray | None = None,
    wind_aux_history: np.ndarray | None = None,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a direct multi-step ARX-style wind-power baseline.

    Each forecast horizon is estimated separately using recent power history,
    recent wind-speed history, and deterministic future calendar terms.
    This matches the ARX + exogenous wind-speed philosophy widely used in
    short-term wind-power forecasting while keeping the implementation causal
    and lightweight.
    Nielsen et al. (1996), MIC.
    Lei et al. (2013), RSER.
    """

    lead_arr = np.asarray(lead_minutes, dtype=np.int64).reshape(-1)
    target = np.asarray(target_power, dtype=np.float64)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    if target.ndim != 2:
        raise ValueError("target_power must be 1D or 2D [samples, horizon].")
    horizon = target.shape[1]
    if lead_arr.size != horizon:
        raise ValueError("lead_minutes must have one value per horizon step.")

    coef: np.ndarray | None = None
    bias = np.zeros(horizon, dtype=np.float64)
    for step in range(horizon):
        design = build_direct_arx_design_matrix(
            power_history=power_history,
            wind_history=wind_history,
            power_aux_history=power_aux_history,
            wind_aux_history=wind_aux_history,
            origin_times=origin_times,
            lead_minutes=int(lead_arr[step]),
        )
        target_step = np.sqrt(np.clip(target[:, step], a_min=0.0, a_max=None))
        mask = np.all(np.isfinite(design), axis=1) & np.isfinite(target_step)
        if coef is None:
            coef = np.zeros((horizon, design.shape[1]), dtype=np.float64)
        if int(mask.sum()) < max(8, design.shape[1] + 1):
            continue
        x_fit = design[mask]
        y_fit = target_step[mask]
        design_fit = np.concatenate([x_fit, np.ones((x_fit.shape[0], 1), dtype=np.float64)], axis=1)
        xtx = design_fit.T @ design_fit
        reg = np.eye(xtx.shape[0], dtype=np.float64) * float(max(ridge, 0.0))
        reg[-1, -1] = 0.0
        xty = design_fit.T @ y_fit
        try:
            beta = np.linalg.solve(xtx + reg, xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xtx + reg) @ xty
        coef[step] = beta[:-1]
        bias[step] = float(beta[-1])
    if coef is None:
        coef = np.zeros((horizon, 0), dtype=np.float64)
    return coef, bias


def apply_direct_multi_step_arx_baseline(
    *,
    power_history: np.ndarray,
    wind_history: np.ndarray,
    origin_times: np.ndarray | list,
    lead_minutes: np.ndarray,
    coef: np.ndarray,
    bias: np.ndarray,
    power_aux_history: np.ndarray | None = None,
    wind_aux_history: np.ndarray | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    lead_arr = np.asarray(lead_minutes, dtype=np.int64).reshape(-1)
    coef_arr = np.asarray(coef, dtype=np.float64)
    bias_arr = np.asarray(bias, dtype=np.float64).reshape(-1)
    if coef_arr.ndim != 2:
        raise ValueError("coef must be 2D [horizon, features].")
    if coef_arr.shape[0] != lead_arr.size or bias_arr.size != lead_arr.size:
        raise ValueError("coef, bias, and lead_minutes must share the same horizon.")
    pred = np.zeros((np.asarray(power_history).shape[0], lead_arr.size), dtype=np.float64)
    for step in range(lead_arr.size):
        design = build_direct_arx_design_matrix(
            power_history=power_history,
            wind_history=wind_history,
            power_aux_history=power_aux_history,
            wind_aux_history=wind_aux_history,
            origin_times=origin_times,
            lead_minutes=int(lead_arr[step]),
        )
        step_pred = np.clip(design @ coef_arr[step] + bias_arr[step], a_min=0.0, a_max=None) ** 2
        if clip_max is not None and np.isfinite(float(clip_max)):
            step_pred = np.clip(step_pred, a_min=0.0, a_max=float(clip_max))
        pred[:, step] = step_pred
    return pred
