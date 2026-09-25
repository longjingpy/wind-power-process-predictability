from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_FARM_V2_FARM_FEATURE_COLUMNS = [
    "power_5min_mean",
    "power_1min_inst",
    "farm_power_diff1",
    "farm_power_diff3",
    "farm_power_roll_mean_3",
    "farm_power_roll_std_3",
    "farm_power_ratio_inst_mean",
    "wind_speed_5min_mean",
    "wind_speed_1min_inst",
    "farm_wind_diff1",
    "farm_wind_roll_mean_3",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

# Fast preset keeps the strongest short-horizon SCADA signals only: recent power,
# recent wind, first differences, and diurnal phase. This follows wind-power
# forecasting reviews that consistently rank recent power/wind history among the
# most informative predictors for ultra-short-term horizons.
# Lei et al. (2013), "A review on the forecasting of wind speed and generated
# power", Renewable and Sustainable Energy Reviews, DOI: 10.1016/j.rser.2013.08.062
FAST_FARM_V2_FARM_FEATURE_COLUMNS = [
    "power_5min_mean",
    "power_1min_inst",
    "farm_power_diff1",
    "wind_speed_5min_mean",
    "wind_speed_1min_inst",
    "farm_wind_diff1",
    "hour_sin",
    "hour_cos",
]

# Ramp-aware fast preset adds only causal ramp-state summaries from the input
# window: current ramp magnitude plus recent rolling mean/max of ramp magnitude.
# This keeps feature count low while making regime intensity explicit for
# long-lead forecasting where over-smoothing is common.
RAMP_FAST_FARM_V2_FARM_FEATURE_COLUMNS = [
    "power_5min_mean",
    "power_1min_inst",
    "farm_power_diff1",
    "farm_power_ramp_abs_1",
    "farm_power_ramp_abs_roll_mean_6",
    "farm_power_ramp_abs_roll_max_6",
    "wind_speed_5min_mean",
    "wind_speed_1min_inst",
    "farm_wind_diff1",
    "farm_wind_ramp_abs_roll_mean_6",
    "farm_wind_ramp_abs_roll_max_6",
    "hour_sin",
    "hour_cos",
]

DEFAULT_FARM_V2_TURBINE_FEATURE_COLUMNS = [
    "power_5min_mean",
    "power_1min_inst",
    "wind_speed_5min_mean",
    "wind_speed_1min_inst",
    "power_diff1",
    "power_diff3",
    "power_roll_mean_3",
    "power_roll_std_3",
    "wind_speed_diff1",
    "wind_speed_diff3",
    "wind_inst_minus_avg",
    "power_over_wind_inst",
    "power_minus_farm_mean",
]

# Fast preset for turbine auxiliaries keeps local power/wind level, first-order
# changes, and two normalized interaction terms that remain useful for turbine
# heterogeneity without carrying the full feature bank.
# Lei et al. (2013), "A review on the forecasting of wind speed and generated
# power", Renewable and Sustainable Energy Reviews, DOI: 10.1016/j.rser.2013.08.062
FAST_FARM_V2_TURBINE_FEATURE_COLUMNS = [
    "power_5min_mean",
    "power_1min_inst",
    "wind_speed_5min_mean",
    "wind_speed_1min_inst",
    "power_diff1",
    "wind_speed_diff1",
    "power_over_wind_inst",
    "power_minus_farm_mean",
]

RAMP_FAST_FARM_V2_TURBINE_FEATURE_COLUMNS = [
    "power_5min_mean",
    "power_1min_inst",
    "wind_speed_5min_mean",
    "wind_speed_1min_inst",
    "power_diff1",
    "power_ramp_abs_1",
    "power_ramp_abs_roll_mean_6",
    "wind_speed_diff1",
    "wind_speed_ramp_abs_1",
    "wind_speed_ramp_abs_roll_mean_6",
    "power_over_wind_inst",
    "power_minus_farm_mean",
]

DEFAULT_FARM_V2_STATIC_NODE_COLUMNS = [
    "lat",
    "lon",
    "elevation_m",
    "slope_deg",
    "landcover",
    "z0",
]

WINDOW_FLAG_COLUMNS = [
    "window_has_high_ramp",
    "window_has_long_zero",
    "window_has_gap_fill",
    "window_has_zero_power_high_wind",
]


@dataclass(frozen=True)
class FarmV2Bundle:
    root: Path
    meta: dict[str, Any]
    farm_features: pd.DataFrame
    turbine_features: pd.DataFrame
    sample_index: pd.DataFrame
    graph_static: pd.DataFrame
    gnss_station_features: pd.DataFrame
    gnss_station_meta: pd.DataFrame


@dataclass
class FarmV2Stats:
    farm_mean: np.ndarray
    farm_std: np.ndarray
    turbine_mean: np.ndarray
    turbine_std: np.ndarray
    power_scale: float
    short_delta_scale: float


def _root_from_path(path: Path) -> Path:
    # Treat suffixless paths as dataset roots even when the path has not been
    # materialized yet on the current machine. This keeps relative dataset
    # directories stable across Windows/Linux sync workflows.
    if path.exists():
        return path if path.is_dir() else path.parent
    if path.suffix:
        return path.parent
    return path


def load_farm_v2_meta(path: Path) -> dict[str, Any]:
    root = _root_from_path(path)
    meta_path = root / "dataset_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing farm_v2 dataset meta: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("dataset_version", "")) != "farm_v2":
        raise ValueError(f"Dataset at {root} is not farm_v2.")
    return meta


def is_farm_v2_dataset(path: Path) -> bool:
    try:
        meta = load_farm_v2_meta(path)
    except Exception:
        return False
    return str(meta.get("dataset_version", "")) == "farm_v2"


def load_farm_v2_bundle(path: Path) -> FarmV2Bundle:
    root = _root_from_path(path)
    meta = load_farm_v2_meta(root)
    farm_features = pd.read_parquet(root / "farm_features.parquet")
    turbine_features = pd.read_parquet(root / "turbine_features.parquet")
    sample_index = pd.read_parquet(root / "sample_index.parquet")
    graph_static = pd.read_parquet(root / "graph_static.parquet")
    gnss_station_features_path = root / "gnss_station_features.parquet"
    gnss_station_meta_path = root / "gnss_station_meta.parquet"
    gnss_station_features = pd.read_parquet(gnss_station_features_path) if gnss_station_features_path.exists() else pd.DataFrame()
    gnss_station_meta = pd.read_parquet(gnss_station_meta_path) if gnss_station_meta_path.exists() else pd.DataFrame()
    farm_features["datetime"] = pd.to_datetime(farm_features["datetime"])
    turbine_features["datetime"] = pd.to_datetime(turbine_features["datetime"])
    if not gnss_station_features.empty and "datetime" in gnss_station_features.columns:
        gnss_station_features["datetime"] = pd.to_datetime(gnss_station_features["datetime"])
    if "origin_time" in sample_index.columns:
        sample_index["origin_time"] = pd.to_datetime(sample_index["origin_time"])
    return FarmV2Bundle(
        root=root,
        meta=meta,
        farm_features=farm_features,
        turbine_features=turbine_features,
        sample_index=sample_index,
        graph_static=graph_static,
        gnss_station_features=gnss_station_features,
        gnss_station_meta=gnss_station_meta,
    )


class FarmV2PowerDataset(Dataset):
    def __init__(
        self,
        path: Path,
        farm_feature_columns: Iterable[str] | None = None,
        turbine_feature_columns: Iterable[str] | None = None,
        gnss_feature_columns: Iterable[str] | None = None,
        horizon_override: int | None = None,
        target_step_override: int | None = None,
        max_samples: int = 0,
        use_default_eligible: bool = True,
    ) -> None:
        bundle = load_farm_v2_bundle(path)
        self.bundle = bundle
        self.meta = dict(bundle.meta)
        self.root = bundle.root

        farm_df = bundle.farm_features.sort_values("datetime").reset_index(drop=True).copy()
        turbine_df = bundle.turbine_features.sort_values(["datetime", "turbine_id"]).reset_index(drop=True).copy()
        sample_df = bundle.sample_index.sort_values("origin_idx").reset_index(drop=True).copy()

        available_target_columns = [
            c for c in bundle.meta.get("target_columns", []) if c in farm_df.columns
        ]
        if not available_target_columns:
            available_target_columns = [c for c in farm_df.columns if c.startswith("target_power_step_")]
        if not available_target_columns:
            raise ValueError("farm_v2 dataset is missing target_power_step_* columns.")
        available_horizon = len(available_target_columns)
        requested_horizon = int(horizon_override) if horizon_override is not None else int(bundle.meta.get("horizon", available_horizon) or available_horizon)
        target_step = int(target_step_override) if target_step_override is not None else int(bundle.meta.get("target_step", 1) or 1)
        if requested_horizon <= 0:
            raise ValueError("farm_v2 horizon must be positive.")
        if target_step <= 0:
            raise ValueError("farm_v2 target_step must be positive.")
        if requested_horizon > available_horizon:
            raise ValueError(
                f"Requested farm_v2 horizon={requested_horizon} exceeds available target columns={available_horizon}."
            )
        target_end = target_step - 1 + requested_horizon
        if target_end > available_horizon:
            raise ValueError(
                f"Requested farm_v2 target_step={target_step} horizon={requested_horizon} exceeds available target columns={available_horizon}."
            )
        self.target_columns = available_target_columns[target_step - 1 : target_end]
        available_target_flow_u_columns = [
            c for c in bundle.meta.get("target_flow_u_columns", []) if c in farm_df.columns
        ]
        if not available_target_flow_u_columns:
            available_target_flow_u_columns = [c for c in farm_df.columns if c.startswith("target_flow_u_step_")]
        self.target_flow_u_columns = available_target_flow_u_columns[target_step - 1 : target_end]
        available_target_flow_v_columns = [
            c for c in bundle.meta.get("target_flow_v_columns", []) if c in farm_df.columns
        ]
        if not available_target_flow_v_columns:
            available_target_flow_v_columns = [c for c in farm_df.columns if c.startswith("target_flow_v_step_")]
        self.target_flow_v_columns = available_target_flow_v_columns[target_step - 1 : target_end]
        self.meta["horizon"] = requested_horizon
        self.meta["target_step"] = target_step
        self.meta["target_columns"] = list(self.target_columns)
        if self.target_flow_u_columns:
            self.meta["target_flow_u_columns"] = list(self.target_flow_u_columns)
        if self.target_flow_v_columns:
            self.meta["target_flow_v_columns"] = list(self.target_flow_v_columns)

        farm_cols = list(farm_feature_columns or bundle.meta.get("farm_feature_columns", DEFAULT_FARM_V2_FARM_FEATURE_COLUMNS))
        self.farm_feature_columns = [c for c in farm_cols if c in farm_df.columns]
        if not self.farm_feature_columns:
            raise ValueError("No farm feature columns found for farm_v2 dataset.")

        turbine_cols = list(
            turbine_feature_columns or bundle.meta.get("turbine_feature_columns", DEFAULT_FARM_V2_TURBINE_FEATURE_COLUMNS)
        )
        self.turbine_feature_columns = [c for c in turbine_cols if c in turbine_df.columns]
        if not self.turbine_feature_columns:
            raise ValueError("No turbine feature columns found for farm_v2 dataset.")

        gnss_cols = list(gnss_feature_columns or bundle.meta.get("gnss_feature_columns", []))
        self.gnss_feature_columns = [c for c in gnss_cols if c in farm_df.columns]
        self.gnss_missing_columns = [f"{c}_missing" for c in self.gnss_feature_columns if f"{c}_missing" in farm_df.columns]
        station_feature_cols = list(bundle.meta.get("gnss_station_feature_columns", []))
        if not station_feature_cols and not bundle.gnss_station_features.empty:
            station_feature_cols = [
                c
                for c in bundle.gnss_station_features.columns
                if c not in ("datetime", "station_id")
            ]
        self.gnss_station_feature_columns = [
            c for c in station_feature_cols if c in bundle.gnss_station_features.columns
        ]
        station_static_cols = list(bundle.meta.get("gnss_station_static_columns", []))
        if not station_static_cols and not bundle.gnss_station_meta.empty:
            station_static_cols = [
                c
                for c in ("distance_m", "bearing_rad", "elevation_diff_m")
                if c in bundle.gnss_station_meta.columns
            ]
        self.gnss_station_static_columns = [
            c for c in station_static_cols if c in bundle.gnss_station_meta.columns
        ]
        self.static_node_columns = [
            c for c in bundle.meta.get("static_node_columns", DEFAULT_FARM_V2_STATIC_NODE_COLUMNS)
            if c in turbine_df.columns
        ]
        self.window_flag_columns = [c for c in WINDOW_FLAG_COLUMNS if c in sample_df.columns]

        if use_default_eligible and "is_train_eligible_default" in sample_df.columns:
            sample_df = sample_df[sample_df["is_train_eligible_default"].astype(bool)].copy()
        elif "is_target_complete" in sample_df.columns:
            mask = sample_df["is_contiguous"].astype(bool) & sample_df["is_input_complete"].astype(bool) & sample_df["is_target_complete"].astype(bool)
            sample_df = sample_df[mask].copy()

        if sample_df.empty:
            raise ValueError("farm_v2 dataset has no eligible samples.")

        if max_samples and len(sample_df) > max_samples:
            sample_df = sample_df.iloc[: int(max_samples)].copy()

        self.sample_df = sample_df.reset_index(drop=True)
        self.sample_times = pd.to_datetime(self.sample_df["origin_time"]).tolist()
        self.sample_origin_idx = self.sample_df["origin_idx"].astype(int).to_numpy()
        self.sample_input_start_idx = self.sample_df["input_start_idx"].astype(np.int64).to_numpy(copy=True)
        self.sample_input_end_idx = self.sample_df["input_end_idx"].astype(np.int64).to_numpy(copy=True)
        self.window_flag_values = (
            self.sample_df[self.window_flag_columns].to_numpy(dtype=np.float32, copy=True)
            if self.window_flag_columns
            else np.zeros((len(self.sample_df), 0), dtype=np.float32)
        )
        self.indices = list(range(len(self.sample_df)))
        self.t_in = int(bundle.meta.get("t_in", 0) or 0)
        self.horizon = requested_horizon
        self.target_step = target_step
        self.step_out_min = int(bundle.meta.get("step_out_min", 15) or 15)
        self.freq_min = int(bundle.meta.get("freq_min", 5) or 5)
        self.target_window_min = int(bundle.meta.get("target_window_min", 15) or 15)

        self.times = pd.to_datetime(farm_df["datetime"]).tolist()
        self.turbine_ids = [str(x) for x in bundle.meta.get("turbine_ids", [])]
        if not self.turbine_ids:
            self.turbine_ids = sorted(turbine_df["turbine_id"].astype(str).unique().tolist())

        farm_numeric = farm_df[self.farm_feature_columns].apply(pd.to_numeric, errors="coerce")
        self.farm_values = farm_numeric.to_numpy(dtype=np.float32, copy=True)
        self.farm_targets = farm_df[self.target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
        if self.target_flow_u_columns and self.target_flow_v_columns:
            flow_u = farm_df[self.target_flow_u_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
            flow_v = farm_df[self.target_flow_v_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
            flow_steps = min(flow_u.shape[1], flow_v.shape[1], len(self.target_columns) or flow_u.shape[1])
            flow_u = flow_u[:, :flow_steps]
            flow_v = flow_v[:, :flow_steps]
            self.flow_target_mask = np.isfinite(flow_u) & np.isfinite(flow_v)
            flow_u = np.nan_to_num(flow_u, nan=0.0, posinf=0.0, neginf=0.0)
            flow_v = np.nan_to_num(flow_v, nan=0.0, posinf=0.0, neginf=0.0)
            self.flow_target_values = np.stack([flow_u, flow_v], axis=-1).astype(np.float32, copy=False)
            self.has_flow_targets = bool(flow_steps > 0)
        else:
            self.flow_target_values = np.zeros((len(farm_df), max(len(self.target_columns), 0), 2), dtype=np.float32)
            self.flow_target_mask = np.zeros((len(farm_df), max(len(self.target_columns), 0)), dtype=bool)
            self.has_flow_targets = False
        self.gnss_values = (
            farm_df[self.gnss_feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
            if self.gnss_feature_columns
            else np.zeros((len(farm_df), 0), dtype=np.float32)
        )
        self.gnss_mask = (
            1.0
            - farm_df[self.gnss_missing_columns].apply(pd.to_numeric, errors="coerce").fillna(1.0).to_numpy(dtype=np.float32, copy=True)
            if self.gnss_missing_columns
            else np.ones((len(farm_df), len(self.gnss_feature_columns)), dtype=np.float32)
        )
        self.gnss_selected_station_ids = [str(x) for x in bundle.meta.get("gnss_selected_stations", [])]
        if not self.gnss_selected_station_ids and not bundle.gnss_station_meta.empty and "station_id" in bundle.gnss_station_meta.columns:
            self.gnss_selected_station_ids = bundle.gnss_station_meta["station_id"].astype(str).drop_duplicates().tolist()
        n_station_times = len(farm_df)
        n_stations = len(self.gnss_selected_station_ids)
        n_station_features = len(self.gnss_station_feature_columns)
        station_values = np.zeros((n_station_times, n_stations, n_station_features), dtype=np.float32)
        station_mask = np.zeros((n_station_times, n_stations), dtype=np.float32)
        station_static = np.zeros((n_stations, len(self.gnss_station_static_columns)), dtype=np.float32)
        if n_stations > 0 and not bundle.gnss_station_features.empty:
            station_df = bundle.gnss_station_features.copy()
            time_codes = pd.Categorical(
                pd.to_datetime(station_df["datetime"]),
                categories=pd.to_datetime(farm_df["datetime"]),
                ordered=True,
            ).codes
            station_codes = pd.Categorical(
                station_df["station_id"].astype(str),
                categories=self.gnss_selected_station_ids,
                ordered=True,
            ).codes
            valid_rows = (time_codes >= 0) & (station_codes >= 0)
            if np.any(valid_rows):
                station_numeric = (
                    station_df.loc[valid_rows, self.gnss_station_feature_columns]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                )
                station_values[time_codes[valid_rows], station_codes[valid_rows], :] = station_numeric.to_numpy(
                    dtype=np.float32,
                    copy=True,
                )
                if "available" in station_df.columns:
                    avail = pd.to_numeric(station_df.loc[valid_rows, "available"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)
                    station_mask[time_codes[valid_rows], station_codes[valid_rows]] = (avail > 0).astype(np.float32)
                else:
                    station_mask[time_codes[valid_rows], station_codes[valid_rows]] = 1.0
        if n_stations > 0 and not bundle.gnss_station_meta.empty and self.gnss_station_static_columns:
            station_meta_df = bundle.gnss_station_meta.copy()
            station_meta_df["station_id"] = station_meta_df["station_id"].astype(str)
            station_meta_df = station_meta_df.set_index("station_id").reindex(self.gnss_selected_station_ids)
            station_static = (
                station_meta_df[self.gnss_station_static_columns]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32, copy=True)
            )
        self.gnss_station_values = station_values
        self.gnss_station_mask = station_mask
        self.gnss_station_static_values = station_static
        self.gnss_station_coverage = (
            station_mask.mean(axis=1).astype(np.float32, copy=False)
            if n_stations > 0
            else np.zeros((n_station_times,), dtype=np.float32)
        )

        n_times = len(farm_df)
        n_turbines = len(self.turbine_ids)
        n_turbine_features = len(self.turbine_feature_columns)
        turb_values = np.full((n_times, n_turbines, n_turbine_features), np.nan, dtype=np.float32)
        time_codes = pd.Categorical(pd.to_datetime(turbine_df["datetime"]), categories=pd.to_datetime(farm_df["datetime"]), ordered=True).codes
        turbine_codes = pd.Categorical(turbine_df["turbine_id"].astype(str), categories=self.turbine_ids, ordered=True).codes
        valid_rows = (time_codes >= 0) & (turbine_codes >= 0)
        turbine_numeric = turbine_df.loc[valid_rows, self.turbine_feature_columns].apply(pd.to_numeric, errors="coerce")
        turb_values[time_codes[valid_rows], turbine_codes[valid_rows], :] = turbine_numeric.to_numpy(dtype=np.float32, copy=True)
        self.turbine_values = turb_values

        static_df = (
            turbine_df[["turbine_id"] + self.static_node_columns]
            .drop_duplicates(subset=["turbine_id"], keep="first")
            .copy()
        )
        static_df["turbine_id"] = static_df["turbine_id"].astype(str)
        static_df = static_df.set_index("turbine_id").reindex(self.turbine_ids)
        self.static_node_values = static_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)

        graph_df = bundle.graph_static.copy()
        if "record_type" in graph_df.columns:
            graph_df = graph_df[graph_df["record_type"] == "edge"].copy()
        graph_df["src"] = graph_df["src"].astype(str)
        graph_df["dst"] = graph_df["dst"].astype(str)
        edge_cols = [
            c for c in ("distance_m", "bearing_rad", "elevation_diff_m", "roughness_pair", "wake_prior")
            if c in graph_df.columns
        ]
        src_codes = pd.Categorical(graph_df["src"], categories=self.turbine_ids, ordered=True).codes
        dst_codes = pd.Categorical(graph_df["dst"], categories=self.turbine_ids, ordered=True).codes
        valid_edge = (src_codes >= 0) & (dst_codes >= 0)
        self.edge_index = np.stack([src_codes[valid_edge], dst_codes[valid_edge]], axis=0).astype(np.int64, copy=False)
        self.edge_attr = (
            graph_df.loc[valid_edge, edge_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)
            if edge_cols
            else np.zeros((int(valid_edge.sum()), 0), dtype=np.float32)
        )
        self.edge_attr_columns = edge_cols

    def __len__(self) -> int:
        return len(self.sample_df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        idx = int(idx)
        start = int(self.sample_input_start_idx[idx])
        end = int(self.sample_input_end_idx[idx])
        origin_idx = int(self.sample_origin_idx[idx])
        farm_x = torch.from_numpy(self.farm_values[start : end + 1])
        turbine_x = torch.from_numpy(self.turbine_values[start : end + 1])
        gnss_x = torch.from_numpy(self.gnss_values[start : end + 1])
        gnss_mask = torch.from_numpy(self.gnss_mask[start : end + 1])
        gnss_station_x = torch.from_numpy(self.gnss_station_values[start : end + 1])
        gnss_station_mask = torch.from_numpy(self.gnss_station_mask[start : end + 1])
        gnss_station_static = torch.from_numpy(self.gnss_station_static_values)
        gnss_coverage = torch.from_numpy(self.gnss_station_coverage[start : end + 1])
        target = torch.from_numpy(self.farm_targets[origin_idx])
        target_flow_uv = torch.from_numpy(self.flow_target_values[origin_idx])
        target_flow_mask = torch.from_numpy(self.flow_target_mask[origin_idx].astype(np.float32, copy=False))
        window_flags = torch.from_numpy(self.window_flag_values[idx])
        return {
            "farm_x": farm_x,
            "turbine_x": turbine_x,
            "gnss_x": gnss_x,
            "gnss_mask": gnss_mask,
            "gnss_station_x": gnss_station_x,
            "gnss_station_mask": gnss_station_mask,
            "gnss_station_static": gnss_station_static,
            "gnss_coverage": gnss_coverage,
            "target": target,
            "target_flow_uv": target_flow_uv,
            "target_flow_mask": target_flow_mask,
            "window_flags": window_flags,
            "origin_idx": torch.tensor(origin_idx, dtype=torch.long),
        }

    def compute_stats(self, train_indices: Iterable[int], power_scale_pctl: float, short_delta_steps: int = 0) -> FarmV2Stats:
        idx = np.asarray(list(train_indices), dtype=np.int64)
        if idx.size == 0:
            raise ValueError("compute_stats requires non-empty train_indices.")
        starts = self.sample_input_start_idx[idx]
        ends = self.sample_input_end_idx[idx]
        max_window = int(np.max(ends - starts)) + 1
        offsets = np.arange(max_window, dtype=np.int64)
        window_idx = starts[:, None] + offsets[None, :]
        valid_mask = window_idx <= ends[:, None]
        input_idx = np.unique(window_idx[valid_mask])
        farm_train = self.farm_values[input_idx]
        turbine_train = self.turbine_values[input_idx]

        farm_mean = np.nanmean(farm_train, axis=0)
        farm_std = np.nanstd(farm_train, axis=0)
        farm_mean = np.nan_to_num(farm_mean, nan=0.0)
        farm_std = np.nan_to_num(farm_std, nan=1.0)
        farm_std = np.where(farm_std < 1e-6, 1.0, farm_std)

        turbine_mean = np.nanmean(turbine_train.reshape(-1, turbine_train.shape[-1]), axis=0)
        turbine_std = np.nanstd(turbine_train.reshape(-1, turbine_train.shape[-1]), axis=0)
        turbine_mean = np.nan_to_num(turbine_mean, nan=0.0)
        turbine_std = np.nan_to_num(turbine_std, nan=1.0)
        turbine_std = np.where(turbine_std < 1e-6, 1.0, turbine_std)

        target_vals = self.farm_targets[self.sample_origin_idx[idx]]
        power_scale = float(np.nanpercentile(target_vals, power_scale_pctl))
        if not np.isfinite(power_scale) or power_scale <= 0:
            power_scale = float(np.nanmax(target_vals))
        if not np.isfinite(power_scale) or power_scale <= 0:
            power_scale = 1.0

        short_delta_scale = power_scale
        power_col = "power_5min_mean"
        if short_delta_steps > 0 and power_col in self.farm_feature_columns:
            p_idx = self.farm_feature_columns.index(power_col)
            base = self.farm_values[self.sample_origin_idx[idx], p_idx : p_idx + 1]
            delta = target_vals[:, : min(int(short_delta_steps), target_vals.shape[1])] - base
            short_delta_scale = float(np.nanpercentile(np.abs(delta), power_scale_pctl))
            if not np.isfinite(short_delta_scale) or short_delta_scale <= 0:
                short_delta_scale = power_scale

        return FarmV2Stats(
            farm_mean=farm_mean.astype(np.float32),
            farm_std=farm_std.astype(np.float32),
            turbine_mean=turbine_mean.astype(np.float32),
            turbine_std=turbine_std.astype(np.float32),
            power_scale=float(power_scale),
            short_delta_scale=float(short_delta_scale),
        )
