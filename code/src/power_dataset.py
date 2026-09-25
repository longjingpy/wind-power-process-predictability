"""
Dataset utilities for Rainformer power forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_POWER_FEATURE_COLUMNS = [
    "power",
    "farm_power_sum",
    "wind_speed_inst",
    "wind_speed",
    "wind_speed_100",
    "wind_dir_sin",
    "wind_dir_cos",
    "u100",
    "v100",
    "t2m",
    "sp",
    "q",
    "rho",
    "z0",
    "l_stability_norm",
    "is_stable",
]

SCADA_ONLY_POWER_FEATURE_COLUMNS = [
    "power",
    "farm_power_sum",
    "wind_speed_inst",
    "wind_speed",
    "power_diff1",
    "wind_speed_inst_diff1",
    "wind_inst_minus_avg",
    "power_over_wind_inst",
    "hour_sin",
    "hour_cos",
]


@dataclass
class PowerStats:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    power_scale: float


class WindowedPowerDataset(Dataset):
    """Windowed dataset for power forecasting."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Iterable[str],
        t_in: int,
        horizon: int,
        step_out: int,
        power_col: str = "power",
        target_mode: str = "point",
        target_window: int | None = None,
        max_samples: int = 0,
        power_scale_pctl: float = 99.5,
        enforce_contiguous: bool = True,
        expected_freq_min: int | None = None,
        require_complete_input: bool = True,
        require_complete_target: bool = True,
    ) -> None:
        self.feature_columns = [c for c in feature_columns if c in df.columns]
        self.feature_columns = [c for c in self.feature_columns if not df[c].isna().all()]
        if not self.feature_columns:
            raise ValueError("No feature columns found for power dataset.")
        if power_col not in df.columns:
            raise ValueError(f"Missing power column '{power_col}' for power dataset.")

        self.turbine_ids = sorted(df["turbine_id"].astype(str).unique().tolist())

        times = pd.Index(pd.to_datetime(df["datetime"]).sort_values().unique())

        t_count = len(times)
        n_turbines = len(self.turbine_ids)
        f_count = len(self.feature_columns)

        features = np.full((t_count, n_turbines, f_count), np.nan, dtype=np.float32)
        power = np.full((t_count, n_turbines), np.nan, dtype=np.float32)

        time_codes = pd.Categorical(pd.to_datetime(df["datetime"]), categories=times, ordered=True).codes
        turbine_codes = pd.Categorical(df["turbine_id"].astype(str), categories=self.turbine_ids, ordered=True).codes
        valid_rows = (time_codes >= 0) & (turbine_codes >= 0)
        if not np.all(valid_rows):
            time_codes = time_codes[valid_rows]
            turbine_codes = turbine_codes[valid_rows]
            df = df.loc[valid_rows].copy()

        feature_values = df[self.feature_columns].to_numpy(dtype=np.float32, copy=False)
        power_values = pd.to_numeric(df[power_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        features[time_codes, turbine_codes, :] = feature_values
        power[time_codes, turbine_codes] = power_values

        nan_mask = np.isnan(features)
        feature_mean = np.nanmean(features.reshape(-1, f_count), axis=0)
        feature_std = np.nanstd(features.reshape(-1, f_count), axis=0)
        feature_mean = np.nan_to_num(feature_mean, nan=0.0)
        feature_std = np.nan_to_num(feature_std, nan=1.0)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

        if nan_mask.any():
            features[nan_mask] = np.take(feature_mean, np.where(nan_mask)[2])

        self.features = torch.from_numpy(features)
        self.power = torch.from_numpy(power)
        self.times = times
        self.t_in = t_in
        self.horizon = horizon
        self.step_out = step_out
        self.target_mode = target_mode
        self.target_window = int(target_window) if target_window is not None else step_out
        self.enforce_contiguous = bool(enforce_contiguous)
        self.require_complete_input = bool(require_complete_input)
        self.require_complete_target = bool(require_complete_target)
        if self.target_window < 1:
            raise ValueError("target_window must be >= 1.")
        if self.target_mode not in {"point", "avg", "delta"}:
            raise ValueError("target_mode must be one of: point, avg, delta.")
        if self.target_mode == "avg" and self.target_window > self.step_out:
            raise ValueError("target_window must be <= step_out when target_mode='avg'.")
        self.nan_mask = nan_mask

        times_dt = pd.to_datetime(times)
        times_ns = times_dt.view("int64")
        diffs_ns = np.diff(times_ns)
        positive_diffs = diffs_ns[diffs_ns > 0]
        inferred_freq_ns = int(np.median(positive_diffs)) if positive_diffs.size > 0 else 0
        expected_freq_ns = int(expected_freq_min) * 60 * 1_000_000_000 if expected_freq_min else 0
        base_freq_ns = expected_freq_ns if expected_freq_ns > 0 else inferred_freq_ns
        self.expected_freq_min = int(expected_freq_min) if expected_freq_min else None
        self.inferred_freq_min = int(round(inferred_freq_ns / (60 * 1_000_000_000))) if inferred_freq_ns > 0 else None
        self.base_freq_min = int(round(base_freq_ns / (60 * 1_000_000_000))) if base_freq_ns > 0 else None
        bad_gap = np.ones(max(0, t_count - 1), dtype=np.bool_)
        if base_freq_ns > 0 and bad_gap.size > 0:
            bad_gap = (diffs_ns <= 0) | (diffs_ns != base_freq_ns)
        gap_prefix = np.zeros(t_count, dtype=np.int64)
        if bad_gap.size > 0:
            gap_prefix[1:] = np.cumsum(bad_gap.astype(np.int64))

        max_t = t_count - step_out * horizon
        candidate_t = np.arange(t_in - 1, max_t, dtype=np.int64)
        total_candidates = int(candidate_t.size)
        dropped_non_contiguous = 0
        dropped_incomplete_input = 0
        dropped_incomplete_target = 0

        if total_candidates > 0:
            start_idx = candidate_t - t_in + 1
            end_idx = candidate_t + step_out * horizon
            keep_mask = np.ones(total_candidates, dtype=bool)

            if self.enforce_contiguous and base_freq_ns > 0:
                # Keep fixed-cadence windows to preserve forecast horizon semantics and avoid
                # temporal leakage from irregular jumps.
                # Reference: Bergmeir et al., "A note on the validity of cross-validation for
                # evaluating autoregressive time series prediction", CSDA 2018.
                # https://doi.org/10.1016/j.csda.2017.11.003
                contiguous_mask = (gap_prefix[end_idx] - gap_prefix[start_idx]) == 0
                dropped_non_contiguous = int((~contiguous_mask).sum())
                keep_mask &= contiguous_mask

            power_complete = ~np.isnan(power).any(axis=1)
            feature_complete = ~np.isnan(features).any(axis=(1, 2))

            if self.require_complete_input:
                input_complete = power_complete & feature_complete
                input_bad_prefix = np.zeros(t_count + 1, dtype=np.int64)
                input_bad_prefix[1:] = np.cumsum((~input_complete).astype(np.int64))
                input_mask = (input_bad_prefix[candidate_t + 1] - input_bad_prefix[start_idx]) == 0
                dropped_incomplete_input = int((keep_mask & (~input_mask)).sum())
                keep_mask &= input_mask

            if self.require_complete_target:
                if target_mode == "avg":
                    target_mask = np.ones(total_candidates, dtype=bool)
                    power_bad_prefix = np.zeros(t_count + 1, dtype=np.int64)
                    power_bad_prefix[1:] = np.cumsum((~power_complete).astype(np.int64))
                    for i in range(horizon):
                        end = candidate_t + (i + 1) * step_out + 1
                        start = end - self.target_window
                        step_mask = (power_bad_prefix[end] - power_bad_prefix[start]) == 0
                        target_mask &= step_mask
                else:
                    target_steps = candidate_t[:, None] + step_out * np.arange(1, horizon + 1, dtype=np.int64)
                    target_mask = power_complete[target_steps].all(axis=1)
                    if target_mode == "delta":
                        target_mask &= power_complete[candidate_t]
                dropped_incomplete_target = int((keep_mask & (~target_mask)).sum())
                keep_mask &= target_mask

            indices = candidate_t[keep_mask].tolist()
        else:
            indices = []

        if max_samples and max_samples > 0 and len(indices) > max_samples:
            rng = np.random.default_rng(7)
            indices = rng.choice(indices, size=max_samples, replace=False).tolist()

        self.indices = sorted(indices)
        self.sample_times = [times[idx] for idx in self.indices]
        self.total_candidates = total_candidates
        self.dropped_non_contiguous = dropped_non_contiguous
        self.dropped_incomplete_input = dropped_incomplete_input
        self.dropped_incomplete_target = dropped_incomplete_target

        power_scale = float(np.nanpercentile(power, power_scale_pctl))
        if not np.isfinite(power_scale) or power_scale <= 0:
            power_scale = float(np.nanmax(power))
        if not np.isfinite(power_scale) or power_scale <= 0:
            power_scale = 1.0
        self.stats = PowerStats(
            feature_mean=feature_mean.astype(np.float32),
            feature_std=feature_std.astype(np.float32),
            power_scale=power_scale,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t_idx = self.indices[idx]
        x_seq = self.features[t_idx - self.t_in + 1 : t_idx + 1]
        if self.target_mode == "avg":
            y_list = []
            for i in range(self.horizon):
                end = t_idx + (i + 1) * self.step_out + 1
                start = end - self.target_window
                window = self.power[start:end]
                y_list.append(torch.nanmean(window, dim=0))
            y_seq = torch.stack(y_list, dim=1)
        else:
            y_steps = [t_idx + (i + 1) * self.step_out for i in range(self.horizon)]
            y_seq = self.power[y_steps].transpose(0, 1)
            if self.target_mode == "delta":
                base = self.power[t_idx].unsqueeze(1)
                y_seq = y_seq - base
        return x_seq, y_seq

    def fill_missing(self, feature_mean: np.ndarray) -> None:
        """Fill original missing feature values with the provided mean."""
        if not hasattr(self, "nan_mask") or self.nan_mask is None:
            return
        if not self.nan_mask.any():
            return
        features = self.features.numpy()
        features[self.nan_mask] = np.take(feature_mean, np.where(self.nan_mask)[2])
