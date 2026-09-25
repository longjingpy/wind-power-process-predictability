"""
Purpose: Merge SCADA and ERA5 features, apply stability filtering, and save features.
Input: data/processed/scada_5min.parquet and data/processed/era5_turbines/*.parquet.
Output: data/processed/features/features_YYYYMM.parquet and graph_static.parquet.
Prereq: pandas, numpy installed.
Next: Train model and evaluate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.farm_v2_dataset import (
    DEFAULT_FARM_V2_FARM_FEATURE_COLUMNS,
    DEFAULT_FARM_V2_STATIC_NODE_COLUMNS,
    DEFAULT_FARM_V2_TURBINE_FEATURE_COLUMNS,
)
from src.gnss_flow import (
    StationProductConfig,
    add_upwind_features,
    build_station_feature_table,
    load_table_or_dir,
    normalize_station_products,
    resample_station_products,
    select_nearest_stations,
    select_strict_upwind_stations,
    add_ztd_gradient_direction_proxy,
)

_PROGRESS_LAST_LEN = 0


def _emit_progress_line(msg: str, *, final: bool = False) -> None:
    global _PROGRESS_LAST_LEN
    if not sys.stdout:
        return
    if sys.stdout.isatty():
        pad = ""
        if len(msg) < _PROGRESS_LAST_LEN:
            pad = " " * (_PROGRESS_LAST_LEN - len(msg))
        sys.stdout.write("\r" + msg + pad)
        if final:
            sys.stdout.write("\n")
            _PROGRESS_LAST_LEN = 0
        else:
            _PROGRESS_LAST_LEN = len(msg)
        sys.stdout.flush()
    else:
        print(msg, flush=True)


def _haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371000.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = phi2 - phi1
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _bearing(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dlambda = np.deg2rad(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return np.arctan2(y, x)


LAT_KEYS = ("lat", "latitude", "纬度")
LON_KEYS = ("lon", "longitude", "经度")
ELEV_KEYS = ("elevation_m", "elevation", "elev", "海拔", "海拔高度")
ID_KEYS = ("turbine_id", "编号", "机组", "name", "machine_num", "machine", "pow_fan_id", "fan_id")
RAW_ID_KEYS = ("名称", "机组", "turbine", "id", "machine_num", "machine", "pow_fan_id", "fan_id")
RAW_TIME_KEYS = ("时间", "timestamp", "datetime", "time")
RAW_WIND_KEYS = ("风速", "windspeed", "wind_speed", "speed")
RAW_POWER_KEYS = ("有功", "功率", "power", "active")
RAW_EXCLUDE_FILES = {"TurbinsToPower.csv"}


def _pick_column(header: Iterable[str], keys: Tuple[str, ...]) -> int | None:
    for idx, col in enumerate(header):
        col_norm = col.strip().lower()
        if col_norm in keys or any(key in col_norm for key in keys):
            return idx
    return None


def _parse_coord(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    for ch in ("°", "o", "'", "\"", "′", "″", "’", "”"):
        value = value.replace(ch, " ")
    parts = [p for p in value.replace(",", " ").split() if p]
    try:
        if len(parts) == 1:
            return float(parts[0])
        deg = float(parts[0])
        minutes = float(parts[1]) if len(parts) > 1 else 0.0
        seconds = float(parts[2]) if len(parts) > 2 else 0.0
    except ValueError:
        return None
    sign = -1.0 if deg < 0 else 1.0
    return sign * (abs(deg) + minutes / 60.0 + seconds / 3600.0)


def _load_coords(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        id_idx = _pick_column(header, ID_KEYS)
        if id_idx is None:
            id_idx = 1
        lat_idx = _pick_column(header, LAT_KEYS)
        lon_idx = _pick_column(header, LON_KEYS)
        elev_idx = _pick_column(header, ELEV_KEYS)
        if lat_idx is None or lon_idx is None:
            raise ValueError("Missing lat/lon columns in coords file.")
        rows = []
        for row in reader:
            if lat_idx >= len(row) or lon_idx >= len(row):
                continue
            lat = _parse_coord(row[lat_idx])
            lon = _parse_coord(row[lon_idx])
            if lat is None or lon is None:
                continue
            turbine_id = row[id_idx].strip() if id_idx < len(row) else ""
            elev = np.nan
            if elev_idx is not None and elev_idx < len(row):
                elev_raw = row[elev_idx].strip().lower().replace("m", "").strip()
                try:
                    elev = float(elev_raw)
                except ValueError:
                    elev = np.nan
            rows.append({"turbine_id": turbine_id, "lat": lat, "lon": lon, "elevation_m": elev})
    return pd.DataFrame(rows)


def _load_era5_static_priors(era5_path: Path) -> pd.DataFrame:
    """
    Load turbine-level static priors from ERA5 interpolation output.

    farm_v2 keeps SCADA as the main dynamic source, but static priors such as
    elevation and surface roughness z0 are valid auxiliary descriptors even in
    `--scada-only` mode. We aggregate them per turbine and merge as node attrs.
    """
    if not era5_path.exists():
        return pd.DataFrame(columns=["turbine_id"])
    if era5_path.is_dir():
        files = sorted(era5_path.glob("*.parquet"))
    else:
        files = [era5_path]
    frames: list[pd.DataFrame] = []
    keep_cols = ["turbine_id", "lat", "lon", "elevation_m", "z0"]
    for path in files:
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        cols = [c for c in keep_cols if c in frame.columns]
        if "turbine_id" not in cols:
            continue
        frames.append(frame[cols].copy())
    if not frames:
        return pd.DataFrame(columns=["turbine_id"])
    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        return pd.DataFrame(columns=["turbine_id"])
    for col in ("lat", "lon", "elevation_m", "z0"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    agg_map = {"lat": "median", "lon": "median", "elevation_m": "median", "z0": "median"}
    agg_map = {k: v for k, v in agg_map.items() if k in merged.columns}
    if not agg_map:
        return merged[["turbine_id"]].drop_duplicates().reset_index(drop=True)
    return merged.groupby("turbine_id", as_index=False).agg(agg_map)


def _build_static_graph(turbines: pd.DataFrame, k: int) -> pd.DataFrame:
    coords = turbines[["lat", "lon"]].to_numpy()
    n = coords.shape[0]
    lat = coords[:, 0][:, None]
    lon = coords[:, 1][:, None]
    dist = _haversine(lat, lon, lat.T, lon.T)
    bearing = _bearing(lat, lon, lat.T, lon.T)

    edges = []
    for i in range(n):
        order = np.argsort(dist[i])
        for j in order[1 : k + 1]:
            edges.append(
                {
                    "src": turbines.iloc[i]["turbine_id"],
                    "dst": turbines.iloc[j]["turbine_id"],
                    "distance_m": float(dist[i, j]),
                    "bearing_rad": float(bearing[i, j]),
                }
            )
    return pd.DataFrame(edges)


def _load_era5(era5_path: Path) -> pd.DataFrame:
    if era5_path.is_dir():
        files = sorted(era5_path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No ERA5 parquet files in {era5_path}")
        frames = [pd.read_parquet(path) for path in files]
        return pd.concat(frames, ignore_index=True)
    return pd.read_parquet(era5_path)


def _load_scada(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def _resample_scada(df: pd.DataFrame, freq: str, label: str, closed: str) -> pd.DataFrame:
    records = []
    for turbine_id, group in df.groupby("turbine_id"):
        group = group.sort_values("datetime").set_index("datetime")
        agg = group.resample(freq, label=label, closed=closed).mean(numeric_only=True)
        agg["turbine_id"] = turbine_id
        records.append(agg.reset_index())
    return pd.concat(records, ignore_index=True)


def _resample_era5(df: pd.DataFrame, freq: str, label: str, closed: str) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    records = []
    for turbine_id, group in df.groupby("turbine_id"):
        group = group.sort_values("datetime").set_index("datetime")
        # interpolate numeric columns only to avoid object-dtype warnings
        numeric = group.select_dtypes(include=[np.number]).infer_objects(copy=False)
        interp = numeric.resample(freq, label=label, closed=closed).interpolate(method="time")
        interp["turbine_id"] = turbine_id
        records.append(interp.reset_index())
    return pd.concat(records, ignore_index=True)


def _add_targets(df: pd.DataFrame, freq_minutes: int, horizon_minutes: int, source_col: str) -> pd.DataFrame:
    steps = int(horizon_minutes / freq_minutes)
    df = df.sort_values(["turbine_id", "datetime"])
    target_time = df["datetime"] + pd.to_timedelta(horizon_minutes, unit="m")
    future_value = df.groupby("turbine_id")[source_col].shift(-steps)
    future_time = pd.to_datetime(df.groupby("turbine_id")["datetime"].shift(-steps))
    df["target_wind_speed"] = future_value.where(future_time.eq(target_time))
    df["target_time"] = target_time
    return df


def _add_scada_derived_features(df: pd.DataFrame, freq_minutes: int) -> pd.DataFrame:
    """Build lag/ramp/context features from SCADA only (no future leakage)."""
    if df.empty or "turbine_id" not in df.columns or "datetime" not in df.columns:
        return df
    if freq_minutes <= 0:
        return df

    df = df.sort_values(["turbine_id", "datetime"]).copy()
    g = df.groupby("turbine_id", sort=False)
    dt = pd.to_datetime(df["datetime"])
    lag1_dt = pd.to_datetime(g["datetime"].shift(1))
    lag3_dt = pd.to_datetime(g["datetime"].shift(3))
    lag1_gap = (dt - lag1_dt).dt.total_seconds().div(60.0)
    lag3_gap = (dt - lag3_dt).dt.total_seconds().div(60.0)
    valid_lag1 = lag1_gap.eq(float(freq_minutes))
    valid_lag3 = lag3_gap.eq(float(freq_minutes * 3))

    base_cols = [c for c in ("power", "wind_speed_inst", "wind_speed") if c in df.columns]
    for col in base_cols:
        series = g[col]
        lag1 = series.shift(1).where(valid_lag1)
        lag3 = series.shift(3).where(valid_lag3)
        df[f"{col}_lag1"] = lag1
        df[f"{col}_lag3"] = lag3
        df[f"{col}_diff1"] = df[col] - lag1
        df[f"{col}_diff3"] = df[col] - lag3
        df[f"{col}_roll_mean_3"] = (
            series.rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"{col}_roll_std_3"] = (
            series.rolling(window=3, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
        )

    eps = 1e-3
    if "wind_speed_inst" in df.columns and "wind_speed" in df.columns:
        df["wind_inst_minus_avg"] = df["wind_speed_inst"] - df["wind_speed"]
        df["wind_inst_ratio_avg"] = df["wind_speed_inst"] / (df["wind_speed"] + 0.1 + eps)
    if "power" in df.columns and "wind_speed_inst" in df.columns:
        df["power_over_wind_inst"] = df["power"] / (df["wind_speed_inst"] + 0.5 + eps)
    if "power" in df.columns and "wind_speed" in df.columns:
        df["power_over_wind_avg"] = df["power"] / (df["wind_speed"] + 0.5 + eps)

    by_time = df.groupby("datetime", sort=False)
    if "power" in df.columns:
        df["farm_power_sum"] = by_time["power"].transform("sum")
        df["farm_power_mean"] = by_time["power"].transform("mean")
        df["power_minus_farm_mean"] = df["power"] - df["farm_power_mean"]
    if "wind_speed_inst" in df.columns:
        df["farm_wind_inst_mean"] = by_time["wind_speed_inst"].transform("mean")
        df["wind_inst_minus_farm_mean"] = df["wind_speed_inst"] - df["farm_wind_inst_mean"]
    if "wind_speed" in df.columns:
        df["farm_wind_mean"] = by_time["wind_speed"].transform("mean")
        df["wind_minus_farm_mean"] = df["wind_speed"] - df["farm_wind_mean"]

    hour = dt.dt.hour + dt.dt.minute / 60.0
    dow = dt.dt.dayofweek + hour / 24.0
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _merge_gnss_daily_features(df: pd.DataFrame, gnss_daily_path: Path) -> pd.DataFrame:
    """
    Merge daily GNSS features onto timestamp rows.

    Causal convention: features from day D are only visible from D+1 onward
    (avoid same-day lookahead leakage during offline build).
    """
    if df.empty:
        return df
    if not gnss_daily_path or not gnss_daily_path.exists():
        print(f"Warning: GNSS daily file not found, skip merge: {gnss_daily_path}")
        return df

    suffix = gnss_daily_path.suffix.lower()
    if suffix == ".parquet":
        daily = pd.read_parquet(gnss_daily_path)
    elif suffix in (".csv", ".txt"):
        daily = pd.read_csv(gnss_daily_path)
    else:
        print(f"Warning: unsupported GNSS daily format {gnss_daily_path}, skip merge.")
        return df

    if daily.empty:
        print(f"Warning: GNSS daily file is empty: {gnss_daily_path}")
        return df
    if "date" not in daily.columns:
        print(f"Warning: GNSS daily file missing required 'date' column: {gnss_daily_path}")
        return df

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.floor("D")
    daily = daily.sort_values("date")
    daily["merge_date"] = daily["date"] + pd.to_timedelta(1, unit="D")

    feature_cols = [c for c in daily.columns if c not in ("date", "merge_date")]
    if not feature_cols:
        print(f"Warning: GNSS daily file has no feature columns: {gnss_daily_path}")
        return df

    out = df.copy()
    out["_merge_date"] = pd.to_datetime(out["datetime"]).dt.floor("D")
    out = out.merge(daily[["merge_date"] + feature_cols], left_on="_merge_date", right_on="merge_date", how="left")
    out = out.drop(columns=["_merge_date", "merge_date"])

    for col in feature_cols:
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            miss_col = f"{col}_missing"
            if miss_col not in out.columns:
                out[miss_col] = out[col].isna().astype(float)

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0.0)

    avail_cols = [c for c in feature_cols if c.endswith("_available")]
    if avail_cols:
        coverage = float((out[avail_cols].sum(axis=1) > 0).mean())
        print(
            f"Merged GNSS daily features: cols={len(feature_cols)} "
            f"coverage={coverage:.2%} source={gnss_daily_path}"
        )
    else:
        print(f"Merged GNSS daily features: cols={len(feature_cols)} source={gnss_daily_path}")
    return out


def _load_gnss_table(path: Path) -> pd.DataFrame | None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix == ".gz":
        name = path.name.lower()
        if name.endswith(".csv.gz") or name.endswith(".txt.gz"):
            return pd.read_csv(path, compression="gzip")
    return None


def _merge_gnss_raw_features(
    df: pd.DataFrame,
    gnss_raw_path: Path,
    freq: str,
    label: str,
    closed: str,
    time_col: str = "datetime",
    prefix: str = "gnss",
    station_col: str = "",
    station_value: str = "",
    agg: str = "last",
) -> pd.DataFrame:
    """
    Merge high-frequency GNSS time-series features onto SCADA timeline.

    Causal convention: resample with right-closed/right-labeled bins and use
    "last" value in each bin (no future leakage).
    """
    if df.empty:
        return df
    if not gnss_raw_path or not gnss_raw_path.exists():
        print(f"Warning: GNSS raw file/dir not found, skip merge: {gnss_raw_path}")
        return df

    frames: list[pd.DataFrame] = []
    if gnss_raw_path.is_dir():
        files = sorted([p for p in gnss_raw_path.iterdir() if p.is_file()])
        for p in files:
            t = _load_gnss_table(p)
            if t is not None and not t.empty:
                frames.append(t)
    else:
        t = _load_gnss_table(gnss_raw_path)
        if t is None:
            print(f"Warning: unsupported GNSS raw format {gnss_raw_path}, skip merge.")
            return df
        if not t.empty:
            frames.append(t)

    if not frames:
        print(f"Warning: GNSS raw source is empty: {gnss_raw_path}")
        return df

    raw = pd.concat(frames, ignore_index=True)
    cand_time_cols = [time_col, "datetime", "timestamp", "time", "dt"]
    chosen_time_col = next((c for c in cand_time_cols if c in raw.columns), "")
    if not chosen_time_col:
        print(
            f"Warning: GNSS raw missing time column (candidates={cand_time_cols}), "
            f"skip merge: {gnss_raw_path}"
        )
        return df
    raw = raw.copy()
    raw[chosen_time_col] = pd.to_datetime(raw[chosen_time_col], errors="coerce")
    raw = raw.dropna(subset=[chosen_time_col])
    if raw.empty:
        print(f"Warning: GNSS raw has no valid timestamps: {gnss_raw_path}")
        return df

    station_col = station_col.strip()
    station_value = station_value.strip()
    if station_col and station_value and station_col in raw.columns:
        raw = raw[raw[station_col].astype(str) == station_value].copy()
        if raw.empty:
            print(
                f"Warning: GNSS raw station filter removed all rows: "
                f"{station_col}={station_value}"
            )
            return df

    exclude = {chosen_time_col, station_col, "date"}
    feature_cols = [
        c for c in raw.columns if c not in exclude and pd.api.types.is_numeric_dtype(raw[c])
    ]
    if not feature_cols:
        print(f"Warning: GNSS raw has no numeric feature columns: {gnss_raw_path}")
        return df

    series = raw[[chosen_time_col] + feature_cols].sort_values(chosen_time_col).set_index(chosen_time_col)
    if agg == "mean":
        agg_df = series.resample(freq, label=label, closed=closed).mean()
    else:
        agg_df = series.resample(freq, label=label, closed=closed).last()
    agg_df = agg_df.reset_index().rename(columns={chosen_time_col: "datetime"})

    prefix = prefix.strip()
    rename_map: dict[str, str] = {}
    for c in feature_cols:
        if not prefix:
            rename_map[c] = c
            continue
        low = c.lower()
        if low.startswith(f"{prefix.lower()}_") or low.startswith("gnss_"):
            rename_map[c] = c
        else:
            rename_map[c] = f"{prefix}_{c}"
    agg_df = agg_df.rename(columns=rename_map)
    merged_feature_cols = [rename_map[c] for c in feature_cols]

    avail_col = "gnss_raw_available" if not prefix else f"{prefix}_raw_available"
    agg_df[avail_col] = agg_df[merged_feature_cols].notna().any(axis=1).astype(float)

    out = df.copy()
    out = out.merge(agg_df, on="datetime", how="left")
    for col in merged_feature_cols:
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            miss_col = f"{col}_missing"
            if miss_col not in out.columns:
                out[miss_col] = out[col].isna().astype(float)
    for col in merged_feature_cols + [avail_col]:
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0.0)

    coverage = float((out[avail_col] > 0).mean()) if avail_col in out.columns else float("nan")
    print(
        f"Merged GNSS raw features: cols={len(merged_feature_cols)} "
        f"coverage={coverage:.2%} agg={agg} freq={freq} source={gnss_raw_path}"
    )
    return out


def _choose_gnss_raw_station_value_col(
    raw: pd.DataFrame,
    *,
    time_col: str,
    station_col: str,
    requested_value_col: str,
) -> str:
    requested = str(requested_value_col or "").strip()
    if requested and requested in raw.columns:
        return requested
    exclude = {time_col, station_col, "date"}
    numeric_cols = [c for c in raw.columns if c not in exclude and pd.api.types.is_numeric_dtype(raw[c])]
    if not numeric_cols:
        return ""
    preferred = [
        c
        for c in numeric_cols
        if "quality" not in c.lower()
        and "available" not in c.lower()
        and not c.lower().endswith("_missing")
    ]
    return preferred[0] if preferred else numeric_cols[0]


def _choose_gnss_raw_station_quality_col(raw: pd.DataFrame, requested_quality_col: str) -> str:
    requested = str(requested_quality_col or "").strip()
    if requested and requested in raw.columns:
        return requested
    candidates = [
        "quality_flag",
        "available",
        "raw_available",
        "gnss_raw_available",
    ]
    for col in raw.columns:
        low = str(col).lower()
        if "available" in low or "quality" in low:
            candidates.append(col)
    for col in candidates:
        if col in raw.columns:
            return str(col)
    return ""


def _load_station_products_from_gnss_raw(
    *,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, object]] | None:
    if not args.gnss_raw:
        return None
    raw_path = Path(args.gnss_raw)
    if not raw_path.exists():
        return None
    raw = load_table_or_dir(raw_path)
    if raw.empty:
        print(f"Warning: GNSS raw source is empty for station-flow fallback: {raw_path}")
        return None
    time_col = str(args.gnss_time_col or "datetime").strip()
    if time_col not in raw.columns:
        fallback_time = next((c for c in ("datetime", "timestamp", "time", "dt") if c in raw.columns), "")
        if not fallback_time:
            print(f"Warning: GNSS raw missing time column for station-flow fallback: {raw_path}")
            return None
        time_col = fallback_time
    station_col = str(args.gnss_station_col or "").strip()
    if not station_col or station_col not in raw.columns:
        fallback_station = next((c for c in ("station_id", "station", "site", "id") if c in raw.columns), "")
        if not fallback_station:
            print(
                "Warning: GNSS raw missing station column for station-flow fallback; "
                "provide --gnss-station-col or a raw table with station_id/station."
            )
            return None
        station_col = fallback_station
    value_col = _choose_gnss_raw_station_value_col(
        raw,
        time_col=time_col,
        station_col=station_col,
        requested_value_col=str(args.gnss_station_value_col),
    )
    if not value_col:
        print(f"Warning: GNSS raw has no numeric station value column for station-flow fallback: {raw_path}")
        return None
    quality_col = _choose_gnss_raw_station_quality_col(raw, str(args.gnss_station_quality_col))
    product_df = raw.copy()
    if not quality_col:
        quality_col = "quality_flag"
        product_df[quality_col] = 1.0
    meta_df = None
    if args.gnss_station_meta:
        meta_path = Path(args.gnss_station_meta)
        if meta_path.exists():
            meta_df = load_table_or_dir(meta_path)
    config = StationProductConfig(
        time_col=time_col,
        station_col=station_col,
        value_col=value_col,
        quality_col=quality_col,
        lat_col=args.gnss_station_lat_col,
        lon_col=args.gnss_station_lon_col,
        elev_col=args.gnss_station_elev_col,
    )
    station_df = normalize_station_products(product_df, config=config, meta_df=meta_df)
    if station_df.empty:
        print(f"Warning: GNSS raw station-flow fallback normalized to empty table: {raw_path}")
        return None
    station_df = resample_station_products(
        station_df,
        freq=args.freq,
        label=args.resample_label,
        closed=args.resample_closed,
    )
    print(
        "Using GNSS raw station-flow fallback: "
        f"value_col={value_col} station_col={station_col} source={raw_path}"
    )
    payload = {
        "gnss_processing_route": "raw_station_direct",
        "gnss_product_source": "gnss_raw_direct",
        "gnss_primary_variable": str(value_col),
        "ppp_success_stations": [],
    }
    return station_df, meta_df, payload


def _add_gnss_temporal_features(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    freq_minutes: int,
) -> pd.DataFrame:
    """
    Add causal temporal derivatives/statistics for GNSS series.

    Current processed GNSS raw source only exposes satellite count. To avoid
    handing a single scalar channel to the GNSS encoder, we build a small causal
    feature family: diff1 / diff3 / roll_mean_3 / roll_std_3, plus missing flags.
    """
    if df.empty or not feature_cols:
        return df
    out = df.sort_values("datetime").copy()
    dt = pd.to_datetime(out["datetime"])
    prev_ok = dt.diff().dt.total_seconds().div(60.0).eq(float(freq_minutes))
    prev3_ok = dt.diff(3).dt.total_seconds().div(60.0).eq(float(freq_minutes * 3))
    roll_ok = prev_ok & prev_ok.shift(1, fill_value=False)
    for col in feature_cols:
        base = pd.to_numeric(out[col], errors="coerce")
        miss_col = f"{col}_missing"
        miss = pd.to_numeric(out[miss_col], errors="coerce").fillna(1.0) if miss_col in out.columns else base.isna().astype(float)
        miss_bool = miss > 0.5

        lag1 = base.shift(1)
        lag3 = base.shift(3)
        valid_diff1 = prev_ok & (~miss_bool) & (~miss_bool.shift(1, fill_value=True))
        valid_diff3 = prev3_ok & (~miss_bool) & (~miss_bool.shift(3, fill_value=True))
        valid_roll = roll_ok & (~miss_bool) & (~miss_bool.shift(1, fill_value=True)) & (~miss_bool.shift(2, fill_value=True))

        diff1_col = f"{col}_diff1"
        diff3_col = f"{col}_diff3"
        roll_mean_col = f"{col}_roll_mean_3"
        roll_std_col = f"{col}_roll_std_3"

        out[diff1_col] = np.where(valid_diff1, (base - lag1).to_numpy(dtype=float), 0.0)
        out[diff3_col] = np.where(valid_diff3, (base - lag3).to_numpy(dtype=float), 0.0)

        rolling = base.rolling(window=3, min_periods=3)
        roll_mean = rolling.mean()
        roll_std = rolling.std(ddof=0)
        out[roll_mean_col] = np.where(valid_roll, roll_mean.to_numpy(dtype=float), 0.0)
        out[roll_std_col] = np.where(valid_roll, roll_std.fillna(0.0).to_numpy(dtype=float), 0.0)

        out[f"{diff1_col}_missing"] = (~valid_diff1).astype(float)
        out[f"{diff3_col}_missing"] = (~valid_diff3).astype(float)
        out[f"{roll_mean_col}_missing"] = (~valid_roll).astype(float)
        out[f"{roll_std_col}_missing"] = (~valid_roll).astype(float)
    return out


def _infer_freq_min(times: pd.Series) -> int | None:
    times = pd.to_datetime(times).sort_values().unique()
    if len(times) < 2:
        return None
    deltas = np.diff(times).astype("timedelta64[m]").astype(int)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return None
    return int(np.median(deltas))


def _max_gap_min(times: pd.Series) -> float:
    values = pd.to_datetime(times).sort_values().unique()
    if len(values) < 2:
        return 0.0
    deltas = np.diff(values).astype("timedelta64[m]").astype(int)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return 0.0
    return float(np.max(deltas))


def _log_stage_stats(name: str, df: pd.DataFrame, expected_freq_min: int) -> None:
    if df.empty:
        print(f"[stage:{name}] rows=0 timestamps=0 turbines=0 max_gap_min=nan")
        return
    ts_count = int(df["datetime"].nunique()) if "datetime" in df.columns else 0
    tb_count = int(df["turbine_id"].nunique()) if "turbine_id" in df.columns else 0
    dt_min = str(pd.to_datetime(df["datetime"]).min()) if "datetime" in df.columns else "na"
    dt_max = str(pd.to_datetime(df["datetime"]).max()) if "datetime" in df.columns else "na"
    gap_max = _max_gap_min(df["datetime"]) if "datetime" in df.columns else float("nan")
    if expected_freq_min > 0 and np.isfinite(gap_max):
        gap_warn_threshold = expected_freq_min * 3
        level = "warning" if gap_max > gap_warn_threshold else "ok"
        print(
            f"[stage:{name}] rows={len(df)} timestamps={ts_count} turbines={tb_count} "
            f"dt=[{dt_min} ~ {dt_max}] max_gap_min={gap_max:.1f} ({level}, threshold={gap_warn_threshold})"
        )
    else:
        print(
            f"[stage:{name}] rows={len(df)} timestamps={ts_count} turbines={tb_count} "
            f"dt=[{dt_min} ~ {dt_max}] max_gap_min={gap_max:.1f}"
        )


def _instant_from_minute(df: pd.DataFrame, freq: str, label: str, closed: str, how: str) -> pd.DataFrame:
    records = []
    for turbine_id, group in df.groupby("turbine_id"):
        group = group.sort_values("datetime").set_index("datetime")
        resampled = group.resample(freq, label=label, closed=closed)
        if how == "last":
            inst = resampled.last()
        else:
            inst = resampled.first()
        inst["turbine_id"] = turbine_id
        records.append(inst.reset_index())
    return pd.concat(records, ignore_index=True)


def _drop_unstable(df: pd.DataFrame, l_threshold: float) -> pd.DataFrame:
    if "l_stability" not in df.columns:
        return df
    mask = df["l_stability"].isna() | (df["l_stability"].abs() >= l_threshold)
    return df.loc[mask].copy()


def _filter_zero_power_with_wind(df: pd.DataFrame, wind_min: float, power_max_kw: float) -> pd.DataFrame:
    if "power" not in df.columns:
        return df
    # Non-positive threshold disables this filter.
    if wind_min <= 0 or power_max_kw <= 0:
        return df
    wind_cols = [c for c in ("wind_speed_inst", "wind_speed") if c in df.columns]
    if not wind_cols:
        return df
    if "datetime" not in df.columns or "turbine_id" not in df.columns:
        return df
    wind = df[wind_cols].max(axis=1)
    # Keep per-turbine anomalies, but drop timestamps where the entire farm is in
    # low-power-with-wind state (likely curtailment/outage). This preserves temporal
    # continuity for sequence learning while excluding non-operational periods.
    # IEC 61400-12-1:2017 discusses excluding such non-normal operating points.
    # URL: https://webstore.iec.ch/publication/26603
    mask = (df["power"] < power_max_kw) & wind.notna() & (wind > wind_min)
    flagged = (
        df.loc[mask, ["datetime", "turbine_id"]]
        .groupby("datetime")["turbine_id"]
        .nunique()
    )
    total = df.groupby("datetime")["turbine_id"].nunique()
    drop_times = flagged[flagged >= total.reindex(flagged.index).fillna(0)].index
    removed_times = int(len(drop_times))
    if removed_times > 0:
        remove_mask = df["datetime"].isin(drop_times)
        removed = int(remove_mask.sum())
        print(
            "Low-power-with-wind filter removed",
            f"{removed} rows",
            f"across {removed_times} timestamps",
            "(all turbines flagged at timestamp)",
            f"(wind_min={wind_min}, power_max_kw={power_max_kw}).",
        )
        return df.loc[~remove_mask].copy()
    return df


def _filter_min_turbines(df: pd.DataFrame, min_turbines: int) -> pd.DataFrame:
    if min_turbines <= 0:
        return df
    # Data availability screening is standard in power performance measurement.
    # IEC 61400-12-1:2017, "Power performance measurements of electricity producing wind turbines"
    # URL: https://webstore.iec.ch/publication/26603
    counts = df.groupby("datetime")["turbine_id"].nunique()
    keep_times = counts[counts >= min_turbines].index
    removed = int(len(counts) - len(keep_times))
    if removed > 0:
        print(f"Min-turbines filter removed {removed} timestamps (<{min_turbines} turbines).")
    return df[df["datetime"].isin(keep_times)].copy()


def _filter_power_curve_bin_outliers(
    df: pd.DataFrame,
    *,
    wind_col: str,
    power_col: str,
    wind_min: float,
    bin_width: float,
    min_bin_samples: int,
    low_k: float,
    high_k: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    info: dict[str, object] = {
        "enabled": True,
        "wind_col": wind_col,
        "power_col": power_col,
        "wind_min": float(wind_min),
        "bin_width": float(bin_width),
        "min_bin_samples": int(min_bin_samples),
        "low_k": float(low_k),
        "high_k": float(high_k),
        "rows_before": int(len(df)),
        "rows_dropped": 0,
        "rows_after": int(len(df)),
        "reason": "disabled_or_missing",
    }
    if df.empty:
        return df, info
    if wind_col not in df.columns or power_col not in df.columns:
        return df, info
    if bin_width <= 0 or min_bin_samples <= 1 or low_k <= 0 or high_k <= 0:
        return df, info
    if "turbine_id" not in df.columns:
        return df, info

    wind = pd.to_numeric(df[wind_col], errors="coerce")
    power = pd.to_numeric(df[power_col], errors="coerce")
    valid = np.isfinite(wind) & np.isfinite(power) & (wind >= float(wind_min))
    if not np.any(valid):
        info["reason"] = "no_valid_rows"
        return df, info

    temp = pd.DataFrame(
        {
            "turbine_id": df.loc[valid, "turbine_id"].astype(str).to_numpy(),
            "wind": wind[valid].to_numpy(),
            "power": power[valid].to_numpy(),
        },
        index=df.index[valid],
    )
    temp["_bin"] = np.floor(temp["wind"] / float(bin_width)).astype(np.int32)

    grp = temp.groupby(["turbine_id", "_bin"], sort=False)["power"]
    temp["_count"] = grp.transform("size")
    temp["_median"] = grp.transform("median")
    # Robust bin-wise dispersion for SCADA cleaning:
    # Rousseeuw & Croux (1993), Journal of the American Statistical Association.
    # DOI: 10.1080/01621459.1993.10476408
    temp["_mad"] = grp.transform(lambda s: float(np.median(np.abs(s.to_numpy() - np.median(s.to_numpy())))))
    temp["_q25"] = grp.transform(lambda s: float(np.quantile(s.to_numpy(), 0.25)))
    temp["_q75"] = grp.transform(lambda s: float(np.quantile(s.to_numpy(), 0.75)))

    sigma_mad = 1.4826 * temp["_mad"].to_numpy()
    sigma_iqr = (temp["_q75"].to_numpy() - temp["_q25"].to_numpy()) / 1.349
    sigma = np.where(np.isfinite(sigma_mad) & (sigma_mad > 0), sigma_mad, sigma_iqr)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)

    center = temp["_median"].to_numpy()
    low_bound = center - float(low_k) * sigma
    high_bound = center + float(high_k) * sigma
    bin_ok = temp["_count"].to_numpy() >= int(min_bin_samples)
    outlier = bin_ok & (
        (temp["power"].to_numpy() < low_bound) | (temp["power"].to_numpy() > high_bound)
    )

    drop_idx = temp.index[outlier]
    dropped = int(len(drop_idx))
    info["rows_dropped"] = dropped
    if dropped <= 0:
        info["reason"] = "applied_no_rows"
        return df, info

    out = df.drop(index=drop_idx).copy()
    info["rows_after"] = int(len(out))
    info["reason"] = "applied"
    # Wind-power bin filtering follows SCADA quality-control practice:
    # IEC 61400-12-1 (power-curve bins, non-normal operation exclusion) and
    # Wang et al. (2023), Energies 16(1):203. DOI: 10.3390/en16010203
    dropped_ts = int(df.loc[drop_idx, "datetime"].nunique()) if "datetime" in df.columns else 0
    print(
        "Power-curve bin outlier filter removed",
        f"{dropped} rows",
        f"across {dropped_ts} timestamps",
        f"(wind>={wind_min}, bin={bin_width}m/s, min_bin={min_bin_samples}, k=({low_k},{high_k})).",
    )
    return out, info


def _build_neighbor_map(turbines: pd.DataFrame, k: int) -> dict[str, list[str]]:
    if k <= 0 or turbines.empty:
        return {}
    coords = turbines[["lat", "lon"]].to_numpy()
    ids = turbines["turbine_id"].astype(str).to_numpy()
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    neighbors: dict[str, list[str]] = {}
    for i, tid in enumerate(ids):
        order = np.argsort(dist[i])
        order = [idx for idx in order if idx != i]
        take = order[: min(k, len(order))]
        neighbors[str(tid)] = [str(ids[j]) for j in take]
    return neighbors


def _impute_missing_turbines(
    df: pd.DataFrame,
    *,
    expected_turbines: list[str],
    min_turbines: int,
    max_missing: int,
    neighbor_k: int,
    impute_cols: list[str],
) -> pd.DataFrame:
    if df.empty or not expected_turbines or min_turbines <= 0:
        return df
    if "lat" not in df.columns or "lon" not in df.columns:
        print("Warning: skip missing-turbine imputation (missing lat/lon).")
        return df
    if not impute_cols:
        return df
    expected = set(expected_turbines)
    if len(expected) <= min_turbines:
        return df
    counts = df.groupby("datetime")["turbine_id"].nunique()
    target_times = counts[counts == min_turbines].index
    if len(target_times) == 0:
        return df

    turbines = df[["turbine_id", "lat", "lon"]].drop_duplicates()
    neighbors = _build_neighbor_map(turbines, neighbor_k)
    coord_map = turbines.set_index("turbine_id")[["lat", "lon"]].to_dict("index")

    rows = []
    imputed_times = 0
    for dt, group in df[df["datetime"].isin(target_times)].groupby("datetime"):
        present = set(group["turbine_id"].astype(str).tolist())
        missing = list(expected - present)
        if len(missing) == 0 or len(missing) > max_missing:
            continue
        for mid in missing:
            neigh = [n for n in neighbors.get(mid, []) if n in present]
            if not neigh:
                continue
            neigh_rows = group[group["turbine_id"].astype(str).isin(neigh)]
            # Neighbor mean imputation as a simple spatial smoother (uniform-weight kNN),
            # a limiting case of inverse-distance interpolation.
            # Shepard, D. (1968) "A two-dimensional interpolation function for irregularly spaced data",
            # Proceedings of the 1968 ACM National Conference. DOI: 10.1145/800186.810616
            values = neigh_rows[impute_cols].mean(numeric_only=True)
            row = {"datetime": dt, "turbine_id": mid, "imputed_missing": 1}
            for col in impute_cols:
                row[col] = values.get(col, np.nan)
            if mid in coord_map:
                row.update(coord_map[mid])
            rows.append(row)
        if rows:
            imputed_times += 1

    if not rows:
        return df
    add_df = pd.DataFrame(rows)
    if "imputed_missing" not in df.columns:
        df = df.copy()
        df["imputed_missing"] = 0
    for col in df.columns:
        if col not in add_df.columns:
            add_df[col] = np.nan
    add_df = add_df[df.columns]
    merged = pd.concat([df, add_df], ignore_index=True)
    print(
        "Imputed missing turbines:",
        f"times={imputed_times}",
        f"rows={len(add_df)}",
        f"k={neighbor_k}",
        f"cols={impute_cols}",
    )
    return merged


def _filter_high_ramp(
    df: pd.DataFrame,
    capacity_mw: float,
    ramp_frac: float,
    freq_min: int,
) -> pd.DataFrame:
    if capacity_mw <= 0 or ramp_frac <= 0 or "power" not in df.columns:
        return df
    cap_kw = capacity_mw * 1000.0
    threshold_kw = ramp_frac * cap_kw
    farm = (
        df.groupby("datetime")["power"]
        .sum(min_count=1)
        .sort_index()
    )
    dt_min = farm.index.to_series().diff().dt.total_seconds().div(60)
    ramp = farm.diff().abs()
    # Ramp-event thresholds are often defined as fraction of rated capacity.
    # Cheneka et al., WES 2020, DOI: 10.5194/wes-5-1731-2020
    # Pichault et al., WES 2021, DOI: 10.5194/wes-6-131-2021
    mask = (ramp > threshold_kw) & (dt_min == freq_min)
    drop_times = farm.index[mask.fillna(False)]
    if len(drop_times) > 0:
        print(
            f"Ramp filter removed {len(drop_times)} timestamps "
            f"(>{ramp_frac:.0%} of capacity, cap={capacity_mw:.2f} MW)."
        )
    return df[~df["datetime"].isin(drop_times)].copy()


def _nan_run_mask(is_na: np.ndarray, limit: int, require_bracket: bool = True) -> np.ndarray:
    mask = np.zeros_like(is_na, dtype=bool)
    if limit <= 0 or len(is_na) == 0:
        return mask
    n = len(is_na)
    start = None
    for i, flag in enumerate(is_na):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i
            if end - start <= limit:
                if not require_bracket or (start > 0 and end < n):
                    mask[start:end] = True
            start = None
    if start is not None:
        end = n
        if end - start <= limit:
            if not require_bracket or (start > 0 and end < n):
                mask[start:end] = True
    return mask


def _kalman_smooth_series(
    y: np.ndarray,
    q_level_frac: float,
    q_trend_frac: float,
    r_frac: float,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y
    obs_mask = np.isfinite(y)
    if obs_mask.sum() == 0:
        return y
    if obs_mask.sum() == 1:
        const = y[obs_mask][0]
        return np.where(obs_mask, y, const)

    var_y = np.nanvar(y)
    if not np.isfinite(var_y) or var_y <= 0:
        const = y[obs_mask][0]
        return np.where(obs_mask, y, const)

    q_level = max(float(q_level_frac) * var_y, 1e-8)
    q_trend = max(float(q_trend_frac) * var_y, 1e-8)
    r_meas = max(float(r_frac) * var_y, 1e-8)

    # Local linear trend model with Kalman filter + RTS smoother.
    # Kalman (1960), J. Basic Eng., DOI: 10.1115/1.3662552
    # Rauch, Tung, Striebel (1965), AIAA Journal, DOI: 10.2514/3.3166
    f_mat = np.array([[1.0, 1.0], [0.0, 1.0]])
    h_mat = np.array([[1.0, 0.0]])
    q_mat = np.array([[q_level, 0.0], [0.0, q_trend]])
    r_mat = np.array([[r_meas]])

    first_idx = int(np.argmax(obs_mask))
    x_prev = np.array([y[first_idx], 0.0])
    p_prev = np.eye(2) * (var_y * 100.0 + 1.0)

    x_pred = np.zeros((n, 2))
    p_pred = np.zeros((n, 2, 2))
    x_filt = np.zeros((n, 2))
    p_filt = np.zeros((n, 2, 2))
    ident = np.eye(2)

    for t in range(n):
        x_p = f_mat @ x_prev
        p_p = f_mat @ p_prev @ f_mat.T + q_mat
        x_pred[t] = x_p
        p_pred[t] = p_p
        if obs_mask[t]:
            s_mat = h_mat @ p_p @ h_mat.T + r_mat
            k_gain = (p_p @ h_mat.T) / s_mat
            innovation = y[t] - (h_mat @ x_p)[0]
            x_f = x_p + k_gain[:, 0] * innovation
            p_f = (ident - k_gain @ h_mat) @ p_p
        else:
            x_f = x_p
            p_f = p_p
        x_filt[t] = x_f
        p_filt[t] = p_f
        x_prev = x_f
        p_prev = p_f

    x_smooth = np.zeros_like(x_filt)
    p_smooth = np.zeros_like(p_filt)
    x_smooth[-1] = x_filt[-1]
    p_smooth[-1] = p_filt[-1]
    for t in range(n - 2, -1, -1):
        p_p = p_pred[t + 1]
        p_inv = np.linalg.pinv(p_p)
        c_mat = p_filt[t] @ f_mat.T @ p_inv
        x_smooth[t] = x_filt[t] + c_mat @ (x_smooth[t + 1] - x_pred[t + 1])
        p_smooth[t] = p_filt[t] + c_mat @ (p_smooth[t + 1] - p_p) @ c_mat.T

    return x_smooth[:, 0]


def _fill_short_gaps(
    df: pd.DataFrame,
    freq: str,
    limit: int,
    method: str,
    kalman_q_level: float,
    kalman_q_trend: float,
    kalman_r: float,
) -> pd.DataFrame:
    if limit <= 0:
        return df
    if "turbine_id" not in df.columns or "datetime" not in df.columns:
        return df
    times = pd.DatetimeIndex(sorted(df["datetime"].unique()))
    times = times.rename("datetime")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    static_cols: list[str] = []
    if num_cols:
        per_turb = df.groupby("turbine_id")[num_cols].nunique(dropna=True)
        static_cols = per_turb.columns[(per_turb.max(axis=0) <= 1).values].tolist()
    interp_cols = [c for c in num_cols if c not in static_cols]
    static_map = df.groupby("turbine_id")[static_cols].first() if static_cols else None
    filled = []
    total = df["turbine_id"].nunique()
    last = time.perf_counter()
    for idx, (tid, group) in enumerate(df.groupby("turbine_id"), 1):
        g = group.set_index("datetime").sort_index()
        g = g.reindex(times)
        g.index.name = "datetime"
        if interp_cols:
            if method == "linear":
                # Short-gap linear interpolation (equivalent to mean of neighbors for 1-step gaps).
                # Hyndman & Athanasopoulos, "Forecasting: Principles and Practice", 3rd ed., OTexts, 2021.
                # URL: https://otexts.com/fpp3/
                g[interp_cols] = g[interp_cols].interpolate(
                    method="time",
                    limit=limit,
                    limit_direction="both",
                    limit_area="inside",
                )
            elif method == "ffill":
                # Causal upsampling: never use a later native observation to fill an earlier model-grid input.
                g[interp_cols] = g[interp_cols].ffill(limit=limit)
            elif method == "kalman":
                for col in interp_cols:
                    series = g[col].astype(float).to_numpy()
                    if not np.isnan(series).any():
                        continue
                    smooth = _kalman_smooth_series(series, kalman_q_level, kalman_q_trend, kalman_r)
                    fill_mask = _nan_run_mask(np.isnan(series), limit, require_bracket=True)
                    if fill_mask.any():
                        series[fill_mask] = smooth[fill_mask]
                    g[col] = series
            else:
                raise ValueError(f"Unknown fill method: {method}")
        if static_cols and static_map is not None:
            for col in static_cols:
                g[col] = static_map.loc[tid, col]
        g["turbine_id"] = tid
        filled.append(g.reset_index())
        now = time.perf_counter()
        if now - last >= 2.0:
            _emit_progress_line(f"Gap fill: turbine {idx}/{total}")
            last = now
    _emit_progress_line("Gap fill: done", final=True)
    result = pd.concat(filled, ignore_index=True)
    filled_rows = int(result[num_cols].isna().sum().sum())
    if filled_rows > 0:
        print(f"Gap fill left {filled_rows} remaining NaNs (limit={limit}).")
    return result


def _append_missing_indicators(df: pd.DataFrame, skip_cols: set[str] | None = None) -> tuple[pd.DataFrame, int]:
    """
    Add per-column missing indicators (<col>_missing) before downstream fills.

    This keeps data-quality semantics explicit for model input and diagnostics.
    """
    if df.empty:
        return df, 0
    out = df.copy()
    skip = set(skip_cols or set())
    created = 0
    for col in out.columns:
        if col in skip or col.endswith("_missing"):
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        miss_col = f"{col}_missing"
        if miss_col in out.columns:
            continue
        miss = out[col].isna()
        if miss.any():
            out[miss_col] = miss.astype(float)
            created += 1
    return out, created


def _pick_named_column(columns: Iterable[str], keys: tuple[str, ...]) -> str | None:
    for col in columns:
        col_norm = str(col).strip().lower()
        if col_norm in keys or any(key in col_norm for key in keys):
            return str(col)
    return None


def _load_raw_scada_minutes(
    input_dir: Path,
    pattern: str,
    encoding: str,
    wind_min: float,
    wind_max: float,
    power_min: float,
    power_max: float,
    recursive: bool = False,
    id_col_override: str = "",
    time_col_override: str = "",
    wind_col_override: str = "",
    power_col_override: str = "",
    start: str = "",
    end: str = "",
    turbine_id_prefix: str = "",
) -> pd.DataFrame:
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    files = [p for p in sorted(iterator) if p.name not in RAW_EXCLUDE_FILES]
    if not files:
        raise FileNotFoundError(f"No raw SCADA CSV files found in {input_dir} with pattern {pattern}")
    frames: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path, encoding=encoding, low_memory=False)
        if df.empty:
            continue
        id_col = id_col_override or _pick_named_column(df.columns, RAW_ID_KEYS)
        time_col = time_col_override or _pick_named_column(df.columns, RAW_TIME_KEYS)
        wind_col = wind_col_override or _pick_named_column(df.columns, RAW_WIND_KEYS)
        power_col = power_col_override or _pick_named_column(df.columns, RAW_POWER_KEYS)
        if not id_col or not time_col or not wind_col or not power_col:
            print(f"Skip raw SCADA file (missing required columns): {path.name}")
            continue
        missing_overrides = [col for col in (id_col, time_col, wind_col, power_col) if col not in df.columns]
        if missing_overrides:
            raise ValueError(f"Raw SCADA column override(s) not found in {path}: {missing_overrides}")
        cur = df[[id_col, time_col, wind_col, power_col]].copy()
        cur.columns = ["turbine_id", "datetime", "wind_speed_1min", "power_1min_inst"]
        cur["datetime"] = pd.to_datetime(cur["datetime"], errors="coerce")
        cur["wind_speed_1min"] = pd.to_numeric(cur["wind_speed_1min"], errors="coerce")
        cur["power_1min_inst"] = pd.to_numeric(cur["power_1min_inst"], errors="coerce")
        cur["wind_speed_1min"] = cur["wind_speed_1min"].clip(lower=0)
        cur["power_1min_inst"] = cur["power_1min_inst"].clip(lower=0)
        cur = cur.dropna(subset=["turbine_id", "datetime"])
        if turbine_id_prefix:
            numeric_ids = pd.to_numeric(cur["turbine_id"], errors="coerce")
            timestamp_has_encoded_ids = cur.groupby("datetime")["turbine_id"].transform(
                lambda values: pd.to_numeric(values, errors="coerce").max() > 117
            )
            raw_text = cur["turbine_id"].astype(str)
            suffix = pd.to_numeric(raw_text.str.removeprefix(str(turbine_id_prefix)), errors="coerce")
            remap = timestamp_has_encoded_ids & raw_text.str.startswith(str(turbine_id_prefix)) & suffix.between(1, 117)
            cur.loc[remap, "turbine_id"] = suffix.loc[remap]
        cur = cur[cur["wind_speed_1min"].between(wind_min, wind_max)]
        cur = cur[cur["power_1min_inst"] >= power_min]
        if power_max > 0:
            cur = cur[cur["power_1min_inst"] <= power_max]
        frames.append(cur)
    if not frames:
        raise ValueError("No valid raw minute-level SCADA files after parsing.")
    out = pd.concat(frames, ignore_index=True)
    if start:
        out = out[out["datetime"] >= pd.Timestamp(start)].copy()
    if end:
        end_time = pd.Timestamp(end)
        if len(str(end)) == 10:
            end_time = end_time + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        out = out[out["datetime"] <= end_time].copy()
    if out.empty:
        raise ValueError("No raw SCADA rows remain after the requested time window.")
    out["turbine_id"] = out["turbine_id"].astype(str)
    out = out.drop_duplicates(subset=["turbine_id", "datetime"], keep="first")
    out = out.sort_values(["turbine_id", "datetime"]).reset_index(drop=True)
    return out


def _resample_raw_scada_for_farm_v2(
    minute_df: pd.DataFrame,
    freq: str,
    label: str,
    closed: str,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for turbine_id, group in minute_df.groupby("turbine_id", sort=False):
        group = group.sort_values("datetime").set_index("datetime")
        mean_df = group.resample(freq, label=label, closed=closed).mean(numeric_only=True)
        inst_df = group.resample(freq, label=label, closed=closed).last()
        cur = pd.DataFrame(
            {
                "datetime": mean_df.index,
                "turbine_id": turbine_id,
                "wind_speed_5min_mean": mean_df["wind_speed_1min"].to_numpy(),
                "power_5min_mean": mean_df["power_1min_inst"].to_numpy(),
                "wind_speed_1min_inst": inst_df["wind_speed_1min"].to_numpy(),
                "power_1min_inst": inst_df["power_1min_inst"].to_numpy(),
            }
        )
        records.append(cur)
    out = pd.concat(records, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.sort_values(["datetime", "turbine_id"]).reset_index(drop=True)


def _load_processed_scada_for_farm_v2(args: argparse.Namespace) -> pd.DataFrame:
    scada = _load_scada(args.scada)
    scada_minute = _load_scada(args.scada_minute)
    for col in ("wind_speed", "power"):
        if col in scada.columns:
            scada[col] = pd.to_numeric(scada[col], errors="coerce").clip(lower=0)
        if col in scada_minute.columns:
            scada_minute[col] = pd.to_numeric(scada_minute[col], errors="coerce").clip(lower=0)
    if args.scada_offset_min:
        offset = pd.to_timedelta(args.scada_offset_min, unit="m")
        scada["datetime"] = pd.to_datetime(scada["datetime"]) + offset
        scada_minute["datetime"] = pd.to_datetime(scada_minute["datetime"]) + offset
    if args.freq != "5min":
        scada = _resample_scada(scada, args.freq, args.resample_label, args.resample_closed)
    inst = _instant_from_minute(
        scada_minute,
        args.freq,
        args.resample_label,
        args.resample_closed,
        args.instant_which,
    )
    inst = inst[["datetime", "turbine_id", "wind_speed", "power"]].rename(
        columns={
            "wind_speed": "wind_speed_1min_inst",
            "power": "power_1min_inst",
        }
    )
    merged = scada.rename(
        columns={
            "wind_speed": "wind_speed_5min_mean",
            "power": "power_5min_mean",
        }
    ).merge(inst, on=["datetime", "turbine_id"], how="left")
    merged["datetime"] = pd.to_datetime(merged["datetime"])
    return merged.sort_values(["datetime", "turbine_id"]).reset_index(drop=True)


def _tag_zero_power_high_wind_farm(
    df: pd.DataFrame,
    expected_turbines: list[str],
    wind_min: float,
    power_max_kw: float,
) -> pd.DataFrame:
    out = df.copy()
    out["flag_zero_power_high_wind_turbine"] = 0.0
    out["flag_zero_power_high_wind_farm"] = 0.0
    if wind_min <= 0 or power_max_kw <= 0 or out.empty:
        return out
    wind = out[["wind_speed_1min_inst", "wind_speed_5min_mean"]].max(axis=1)
    flagged = (out["power_5min_mean"] < power_max_kw) & wind.notna() & (wind > wind_min)
    out.loc[flagged, "flag_zero_power_high_wind_turbine"] = 1.0
    counts = out.groupby("datetime")["flag_zero_power_high_wind_turbine"].sum()
    full_times = counts[counts >= float(len(expected_turbines))].index
    out.loc[out["datetime"].isin(full_times), "flag_zero_power_high_wind_farm"] = 1.0
    return out


def _apply_power_curve_outlier_mask_v2(
    df: pd.DataFrame,
    *,
    wind_min: float,
    bin_width: float,
    min_bin_samples: int,
    low_k: float,
    high_k: float,
) -> pd.DataFrame:
    if df.empty:
        return df
    compat = df.copy()
    compat["power"] = compat["power_5min_mean"]
    compat["wind_speed_inst"] = compat["wind_speed_1min_inst"]
    compat["wind_speed"] = compat["wind_speed_5min_mean"]
    compat["_row_id"] = np.arange(len(compat))
    filtered, _ = _filter_power_curve_bin_outliers(
        compat,
        wind_col="wind_speed_inst",
        power_col="power",
        wind_min=wind_min,
        bin_width=bin_width,
        min_bin_samples=min_bin_samples,
        low_k=low_k,
        high_k=high_k,
    )
    kept_ids = set(filtered["_row_id"].astype(int).tolist())
    out = df.copy()
    out["flag_power_curve_outlier"] = 0.0
    dropped_mask = ~pd.Series(np.arange(len(out))).isin(list(kept_ids)).to_numpy()
    if dropped_mask.any():
        out.loc[dropped_mask, "flag_power_curve_outlier"] = 1.0
        for col in ("power_5min_mean", "power_1min_inst", "wind_speed_5min_mean", "wind_speed_1min_inst"):
            if col in out.columns:
                out.loc[dropped_mask, col] = np.nan
    return out


def _add_fill_flags(before_fill: pd.DataFrame, after_fill: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    base_cols = ["datetime", "turbine_id"] + [c for c in cols if c in before_fill.columns and c in after_fill.columns]
    prev = before_fill[base_cols].copy()
    cur = after_fill.copy()
    merged = cur.merge(prev, on=["datetime", "turbine_id"], how="left", suffixes=("", "_before"))
    flag_cols: list[str] = []
    for col in cols:
        before_col = f"{col}_before"
        if before_col not in merged.columns or col not in merged.columns:
            continue
        flag_col = f"{col}_filled"
        merged[flag_col] = merged[before_col].isna() & merged[col].notna()
        merged[flag_col] = merged[flag_col].astype(float)
        flag_cols.append(flag_col)
        merged = merged.drop(columns=[before_col])
    if "imputed_missing" in merged.columns:
        fill_any = merged["imputed_missing"].fillna(0).astype(float)
    else:
        fill_any = pd.Series(0.0, index=merged.index)
    for col in flag_cols:
        fill_any = np.maximum(fill_any, merged[col].fillna(0).astype(float))
    merged["flag_gap_filled_any"] = fill_any.astype(float)
    if "power_5min_mean_filled" in merged.columns or "power_1min_inst_filled" in merged.columns:
        merged["flag_gap_filled_power"] = np.maximum(
            merged.get("power_5min_mean_filled", 0.0),
            merged.get("power_1min_inst_filled", 0.0),
        ).astype(float)
    if "wind_speed_5min_mean_filled" in merged.columns or "wind_speed_1min_inst_filled" in merged.columns:
        merged["flag_gap_filled_wind"] = np.maximum(
            merged.get("wind_speed_5min_mean_filled", 0.0),
            merged.get("wind_speed_1min_inst_filled", 0.0),
        ).astype(float)
    return merged


def _add_turbine_derived_features_v2(df: pd.DataFrame, freq_minutes: int) -> pd.DataFrame:
    out = df.sort_values(["turbine_id", "datetime"]).copy()
    g = out.groupby("turbine_id", sort=False)
    dt = pd.to_datetime(out["datetime"])
    lag1_dt = pd.to_datetime(g["datetime"].shift(1))
    lag3_dt = pd.to_datetime(g["datetime"].shift(3))
    valid_lag1 = (dt - lag1_dt).dt.total_seconds().div(60.0).eq(float(freq_minutes))
    valid_lag3 = (dt - lag3_dt).dt.total_seconds().div(60.0).eq(float(freq_minutes * 3))
    for src in ("power_5min_mean", "wind_speed_5min_mean"):
        series = g[src]
        lag1 = series.shift(1).where(valid_lag1)
        lag3 = series.shift(3).where(valid_lag3)
        stem = "power" if src.startswith("power") else "wind_speed"
        out[f"{stem}_diff1"] = out[src] - lag1
        out[f"{stem}_diff3"] = out[src] - lag3
        out[f"{stem}_roll_mean_3"] = series.rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        out[f"{stem}_roll_std_3"] = (
            series.rolling(window=3, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
        )
        # Past-only ramp-state features summarize recent magnitude regimes without
        # leaking target-window information. Recent ramp intensity is repeatedly
        # reported as a key short-term wind-power predictor in review/ramp work:
        # Lei et al. (2013), RSER, DOI: 10.1016/j.rser.2013.08.062
        # Zhang et al. (2025), ESWA, DOI: 10.1016/j.eswa.2025.128200
        ramp_abs = out[f"{stem}_diff1"].abs().fillna(0.0)
        out[f"{stem}_ramp_abs_1"] = ramp_abs
        out[f"{stem}_ramp_abs_roll_mean_6"] = ramp_abs.rolling(window=6, min_periods=1).mean().fillna(0.0)
        out[f"{stem}_ramp_abs_roll_max_6"] = ramp_abs.rolling(window=6, min_periods=1).max().fillna(0.0)
    out["wind_inst_minus_avg"] = out["wind_speed_1min_inst"] - out["wind_speed_5min_mean"]
    out["power_over_wind_inst"] = out["power_5min_mean"] / (out["wind_speed_1min_inst"] + 0.5)
    by_time = out.groupby("datetime", sort=False)
    out["farm_power_mean"] = by_time["power_5min_mean"].transform("mean")
    out["power_minus_farm_mean"] = out["power_5min_mean"] - out["farm_power_mean"]
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def _build_farm_features_v2(df: pd.DataFrame, freq_minutes: int) -> pd.DataFrame:
    by_time = df.groupby("datetime", sort=False)
    farm = by_time.agg(
        power_5min_mean=("power_5min_mean", "sum"),
        power_1min_inst=("power_1min_inst", "sum"),
        wind_speed_5min_mean=("wind_speed_5min_mean", "mean"),
        wind_speed_1min_inst=("wind_speed_1min_inst", "mean"),
        farm_complete_turbines=("power_5min_mean", lambda s: int(np.isfinite(pd.to_numeric(s, errors="coerce")).sum())),
        flag_zero_power_high_wind_farm=("flag_zero_power_high_wind_farm", "max"),
        flag_gap_filled_any=("flag_gap_filled_any", "max"),
        flag_gap_filled_power=("flag_gap_filled_power", "max"),
        flag_gap_filled_wind=("flag_gap_filled_wind", "max"),
    ).reset_index()
    farm = farm.sort_values("datetime").reset_index(drop=True)
    farm["farm_power_diff1"] = farm["power_5min_mean"].diff(1)
    farm["farm_power_diff3"] = farm["power_5min_mean"].diff(3)
    farm["farm_power_roll_mean_3"] = farm["power_5min_mean"].rolling(window=3, min_periods=1).mean()
    farm["farm_power_roll_std_3"] = farm["power_5min_mean"].rolling(window=3, min_periods=2).std().fillna(0.0)
    farm["farm_power_ramp_abs_1"] = farm["farm_power_diff1"].abs().fillna(0.0)
    farm["farm_power_ramp_abs_roll_mean_6"] = (
        farm["farm_power_ramp_abs_1"].rolling(window=6, min_periods=1).mean().fillna(0.0)
    )
    farm["farm_power_ramp_abs_roll_max_6"] = (
        farm["farm_power_ramp_abs_1"].rolling(window=6, min_periods=1).max().fillna(0.0)
    )
    farm["farm_power_ratio_inst_mean"] = farm["power_1min_inst"] / (farm["power_5min_mean"] + 1.0)
    farm["farm_wind_diff1"] = farm["wind_speed_5min_mean"].diff(1)
    farm["farm_wind_ramp_abs_1"] = farm["farm_wind_diff1"].abs().fillna(0.0)
    farm["farm_wind_ramp_abs_roll_mean_6"] = (
        farm["farm_wind_ramp_abs_1"].rolling(window=6, min_periods=1).mean().fillna(0.0)
    )
    farm["farm_wind_ramp_abs_roll_max_6"] = (
        farm["farm_wind_ramp_abs_1"].rolling(window=6, min_periods=1).max().fillna(0.0)
    )
    farm["farm_wind_roll_mean_3"] = farm["wind_speed_5min_mean"].rolling(window=3, min_periods=1).mean()
    hour = farm["datetime"].dt.hour + farm["datetime"].dt.minute / 60.0
    dow = farm["datetime"].dt.dayofweek + hour / 24.0
    farm["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    farm["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    farm["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    farm["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    farm.replace([np.inf, -np.inf], np.nan, inplace=True)
    return farm


def _tag_ramp_and_zero_runs(
    farm: pd.DataFrame,
    *,
    farm_capacity_mw: float,
    high_ramp_frac: float,
    long_zero_minutes: int,
    freq_minutes: int,
) -> pd.DataFrame:
    out = farm.copy()
    cap_kw = float(max(farm_capacity_mw, 0.0)) * 1000.0
    ramp_thr = cap_kw * float(max(high_ramp_frac, 0.0))
    if ramp_thr > 0:
        out["flag_high_ramp_farm"] = (out["farm_power_diff1"].abs() >= ramp_thr).astype(float)
    else:
        q_thr = out["farm_power_diff1"].abs().quantile(0.95)
        out["flag_high_ramp_farm"] = (out["farm_power_diff1"].abs() >= q_thr).astype(float)
    zero_mask = (out["power_5min_mean"] <= 1e-3) & (out["farm_complete_turbines"] > 0)
    run_ids = zero_mask.ne(zero_mask.shift(fill_value=False)).cumsum()
    run_len = zero_mask.groupby(run_ids).transform("sum")
    min_steps = max(int(round(long_zero_minutes / max(freq_minutes, 1))), 1)
    out["flag_long_zero_run_farm"] = (zero_mask & (run_len >= min_steps)).astype(float)
    return out


def _attach_targets_v2(
    farm: pd.DataFrame,
    *,
    horizon: int,
    step_out_steps: int,
    target_window_steps: int,
    freq_minutes: int,
) -> pd.DataFrame:
    out = farm.copy()
    times = pd.to_datetime(out["datetime"]).reset_index(drop=True)
    power = pd.to_numeric(out["power_5min_mean"], errors="coerce").to_numpy(dtype=float, copy=True)
    diffs_ns = np.diff(times.view("int64"))
    base_ns = int(freq_minutes) * 60 * 1_000_000_000
    gap_prefix = np.zeros(len(times), dtype=np.int64)
    if len(times) > 1:
        bad_gap = (diffs_ns <= 0) | (diffs_ns != base_ns)
        gap_prefix[1:] = np.cumsum(bad_gap.astype(np.int64))
    for step in range(1, horizon + 1):
        values = np.full(len(out), np.nan, dtype=np.float32)
        end_pos = np.arange(len(out), dtype=np.int64) + step * step_out_steps
        start_pos = end_pos - target_window_steps + 1
        valid = (start_pos >= 0) & (end_pos < len(out))
        valid_idx = np.where(valid)[0]
        for idx in valid_idx:
            s = int(start_pos[idx])
            e = int(end_pos[idx])
            if (gap_prefix[e] - gap_prefix[idx]) != 0:
                continue
            window = power[s : e + 1]
            if not np.isfinite(window).all():
                continue
            values[idx] = float(np.mean(window))
        out[f"target_power_step_{step:02d}"] = values
    return out


def _attach_vector_targets_v2(
    frame: pd.DataFrame,
    *,
    u_col: str,
    v_col: str,
    horizon: int,
    step_out_steps: int,
    target_window_steps: int,
    freq_minutes: int,
) -> pd.DataFrame:
    out = frame.copy()
    if u_col not in out.columns or v_col not in out.columns:
        return out
    times = pd.to_datetime(out["datetime"]).reset_index(drop=True)
    u_vals = pd.to_numeric(out[u_col], errors="coerce").to_numpy(dtype=float, copy=True)
    v_vals = pd.to_numeric(out[v_col], errors="coerce").to_numpy(dtype=float, copy=True)
    diffs_ns = np.diff(times.view("int64"))
    base_ns = int(freq_minutes) * 60 * 1_000_000_000
    gap_prefix = np.zeros(len(times), dtype=np.int64)
    if len(times) > 1:
        bad_gap = (diffs_ns <= 0) | (diffs_ns != base_ns)
        gap_prefix[1:] = np.cumsum(bad_gap.astype(np.int64))
    for step in range(1, horizon + 1):
        u_target = np.full(len(out), np.nan, dtype=np.float32)
        v_target = np.full(len(out), np.nan, dtype=np.float32)
        end_pos = np.arange(len(out), dtype=np.int64) + step * step_out_steps
        start_pos = end_pos - target_window_steps + 1
        valid = (start_pos >= 0) & (end_pos < len(out))
        for idx in np.where(valid)[0]:
            s = int(start_pos[idx])
            e = int(end_pos[idx])
            if (gap_prefix[e] - gap_prefix[idx]) != 0:
                continue
            u_window = u_vals[s : e + 1]
            v_window = v_vals[s : e + 1]
            if not np.isfinite(u_window).all() or not np.isfinite(v_window).all():
                continue
            u_target[idx] = float(np.mean(u_window))
            v_target[idx] = float(np.mean(v_window))
        out[f"target_flow_u_step_{step:02d}"] = u_target
        out[f"target_flow_v_step_{step:02d}"] = v_target
    return out


def _load_source_counts_from_manifest(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    if not Path(path).exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty or "source" not in frame.columns or "status" not in frame.columns:
        return {}
    ok = frame[frame["status"].astype(str).isin(["downloaded", "exists"])].copy()
    if ok.empty:
        return {}
    counts = ok["source"].astype(str).value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _load_ppp_success_stations(
    station_df: pd.DataFrame,
    *,
    summary_path: Path | None,
    value_col: str,
) -> list[str]:
    if summary_path is not None and Path(summary_path).exists():
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        stations = payload.get("successful_stations", [])
        if isinstance(stations, list):
            return sorted({str(item) for item in stations if str(item).strip()})
    if value_col in station_df.columns:
        value = pd.to_numeric(station_df[value_col], errors="coerce")
    else:
        value = pd.Series(np.nan, index=station_df.index, dtype=float)
    if "available" in station_df.columns:
        available_raw = pd.to_numeric(station_df["available"], errors="coerce")
    else:
        available_raw = pd.Series(0.0, index=station_df.index, dtype=float)
    available = available_raw.fillna(0.0) > 0.0
    success = station_df.loc[available & value.notna(), "station_id"].astype(str).dropna().unique().tolist()
    return sorted(success)


def _build_gnss_station_products_v2(
    *,
    args: argparse.Namespace,
    farm: pd.DataFrame,
    turbines: pd.DataFrame,
    expected_freq_min: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    station_policy = str(getattr(args, "gnss_station_policy", "nearest") or "nearest")
    empty_meta: dict[str, object] = {
        "gnss_station_selected_ids": [],
        "gnss_station_count": 0,
        "gnss_station_coverage_ratio": 0.0,
        "gnss_station_policy": station_policy,
        "gnss_station_feature_columns": [],
        "gnss_station_static_columns": [],
        "gnss_station_upwind_weight_mean": 0.0,
        "gnss_station_upwind_nonzero_ratio": 0.0,
    }
    station_df = pd.DataFrame()
    meta_df = None
    route_override: dict[str, object] = {}
    if args.gnss_station_data:
        station_path = Path(args.gnss_station_data)
        if not station_path.exists():
            print(f"Warning: GNSS station source not found, skip station build: {station_path}")
            return pd.DataFrame(), pd.DataFrame(), empty_meta

        product_df = load_table_or_dir(station_path)
        if args.gnss_station_meta:
            meta_path = Path(args.gnss_station_meta)
            if meta_path.exists():
                meta_df = load_table_or_dir(meta_path)
            else:
                print(f"Warning: GNSS station meta not found, skip meta merge: {meta_path}")
        config = StationProductConfig(
            time_col=args.gnss_station_time_col,
            station_col=args.gnss_station_id_col,
            value_col=args.gnss_station_value_col,
            quality_col=args.gnss_station_quality_col,
            lat_col=args.gnss_station_lat_col,
            lon_col=args.gnss_station_lon_col,
            elev_col=args.gnss_station_elev_col,
        )
        station_df = normalize_station_products(product_df, config=config, meta_df=meta_df)
        if station_df.empty:
            print("Warning: GNSS station source is empty after normalization.")
            return pd.DataFrame(), pd.DataFrame(), empty_meta
        station_df = resample_station_products(
            station_df,
            freq=args.freq,
            label=args.resample_label,
            closed=args.resample_closed,
        )
        ppp_success_stations = _load_ppp_success_stations(
            station_df,
            summary_path=getattr(args, "gnss_ppp_summary", None),
            value_col=str(args.gnss_station_value_col),
        )
    else:
        fallback = _load_station_products_from_gnss_raw(args=args)
        if fallback is None:
            return pd.DataFrame(), pd.DataFrame(), empty_meta
        station_df, meta_df, route_override = fallback
        ppp_success_stations = list(route_override.get("ppp_success_stations", []))
    station_meta = (
        station_df[["station_id", "lat", "lon", "elevation_m"]]
        .dropna(subset=["lat", "lon"])
        .drop_duplicates(subset=["station_id"], keep="first")
        .reset_index(drop=True)
    )
    if station_meta.empty:
        print("Warning: GNSS station source has no valid lat/lon; skip station build.")
        return pd.DataFrame(), pd.DataFrame(), empty_meta

    if station_policy == "ztd_gradient":
        station_df = add_ztd_gradient_direction_proxy(station_df)

    farm_lat = float(pd.to_numeric(turbines["lat"], errors="coerce").median())
    farm_lon = float(pd.to_numeric(turbines["lon"], errors="coerce").median())
    if "elevation_m" in turbines.columns:
        elev_series = pd.to_numeric(turbines["elevation_m"], errors="coerce")
    else:
        elev_series = pd.Series([0.0], dtype=float)
    farm_elevation_m = float(elev_series.median())
    aligned_times = pd.to_datetime(farm["datetime"]).sort_values().unique()
    coverage_threshold_used = float(args.gnss_station_coverage_threshold)
    if station_policy in {"strict_upwind", "ztd_gradient"}:
        has_direction = (
            "wind_dir_rad" in station_df.columns
            or {"u100", "v100"}.issubset(station_df.columns)
            or {"gnss_proxy_u", "gnss_proxy_v"}.issubset(station_df.columns)
        )
        if not has_direction:
            raise ValueError(
                "GNSS directional station policy requires wind_dir_rad, u100/v100, or gnss_proxy_u/v."
            )
        thresholds = [float(args.gnss_station_coverage_threshold)]
        if thresholds[0] > 0.05:
            thresholds.append(0.05)
        selection = None
        selected_meta = pd.DataFrame()
        last_error: Exception | None = None
        for coverage_threshold_used in dict.fromkeys(thresholds):
            try:
                selected_meta, selection = select_strict_upwind_stations(
                    station_meta,
                    farm_lat=farm_lat,
                    farm_lon=farm_lon,
                    farm_elevation_m=farm_elevation_m,
                    aligned_times=aligned_times,
                    station_df=station_df,
                    coverage_threshold=float(coverage_threshold_used),
                    target_count=int(args.gnss_station_target_count_full),
                    min_count=int(args.gnss_station_min_count_smoke),
                )
                break
            except ValueError as exc:
                last_error = exc
        if selection is None:
            if last_error is not None:
                raise last_error
            raise ValueError("strict_upwind station selection failed without an explicit error.")
    else:
        selected_meta, selection = select_nearest_stations(
            station_meta,
            farm_lat=farm_lat,
            farm_lon=farm_lon,
            farm_elevation_m=farm_elevation_m,
            aligned_times=aligned_times,
            station_df=station_df,
            coverage_threshold=float(args.gnss_station_coverage_threshold),
            start_k=int(args.gnss_station_start_k),
            expansion_steps=(int(args.gnss_station_expand_k1), int(args.gnss_station_expand_k2)),
        )
    if selected_meta.empty:
        print("Warning: no GNSS stations selected for farm_v2.")
        return pd.DataFrame(), pd.DataFrame(), empty_meta

    station_features = build_station_feature_table(
        station_df,
        selected_station_ids=selection.selected_station_ids,
        freq_minutes=expected_freq_min,
    )
    full_index = pd.MultiIndex.from_product(
        [pd.Index(aligned_times, name="datetime"), pd.Index(selection.selected_station_ids, name="station_id")],
        names=["datetime", "station_id"],
    )
    station_features = (
        full_index.to_frame(index=False)
        .merge(station_features, on=["datetime", "station_id"], how="left")
        .sort_values(["datetime", "station_id"])
        .reset_index(drop=True)
    )
    if station_policy in {"strict_upwind", "ztd_gradient"}:
        selected_meta = selected_meta.copy()
        station_features = add_upwind_features(
            station_features,
            station_meta=selected_meta,
            coverage_threshold=float(coverage_threshold_used),
        )
    dynamic_cols = [c for c in station_features.columns if c not in ("datetime", "station_id")]
    for col in dynamic_cols:
        if col not in station_features.columns:
            station_features[col] = 0.0
        station_features[col] = pd.to_numeric(station_features[col], errors="coerce").fillna(0.0)
    coverage_by_station = (
        station_features.groupby("station_id", as_index=False)["available"].mean().rename(columns={"available": "station_coverage_ratio"})
    )
    selected_meta = selected_meta.merge(
        coverage_by_station,
        on="station_id",
        how="left",
        suffixes=("", "_observed"),
    )
    coverage_col = "station_coverage_ratio"
    observed_col = "station_coverage_ratio_observed"
    if observed_col in selected_meta.columns:
        base = (
            pd.to_numeric(selected_meta[coverage_col], errors="coerce")
            if coverage_col in selected_meta.columns
            else pd.Series(np.nan, index=selected_meta.index, dtype=float)
        )
        observed = pd.to_numeric(selected_meta[observed_col], errors="coerce")
        selected_meta[coverage_col] = observed.combine_first(base).fillna(0.0)
        selected_meta = selected_meta.drop(columns=[observed_col])
    elif coverage_col in selected_meta.columns:
        selected_meta[coverage_col] = pd.to_numeric(selected_meta[coverage_col], errors="coerce").fillna(0.0)
    else:
        selected_meta[coverage_col] = 0.0
    if station_policy in {"strict_upwind", "ztd_gradient"}:
        static_cols = [
            "distance_m",
            "farm_to_station_bearing_rad",
            "station_to_farm_bearing_rad",
            "elevation_diff_m",
            "station_coverage_ratio",
            "distance_weight",
            "upwind_alignment_mean",
            "selection_score",
        ]
    else:
        static_cols = [
            "distance_m",
            "farm_to_station_bearing_rad",
            "station_to_farm_bearing_rad",
            "elevation_diff_m",
            "station_coverage_ratio",
        ]
    station_meta_payload = {
        "gnss_station_selected_ids": list(selection.selected_station_ids),
        "gnss_station_count": int(selection.selected_station_count),
        "gnss_station_coverage_ratio": float(selection.coverage_ratio),
        "gnss_station_policy": station_policy,
        "gnss_station_feature_columns": dynamic_cols,
        "gnss_station_static_columns": static_cols,
        "gnss_station_selection_start_k": int(selection.start_k),
        "gnss_station_selection_final_k": int(selection.final_k),
        "gnss_processing_route": str(route_override.get("gnss_processing_route", args.gnss_processing_route)),
        "gnss_product_source": str(route_override.get("gnss_product_source", args.gnss_product_source)),
        "gnss_primary_variable": str(route_override.get("gnss_primary_variable", args.gnss_station_value_col)),
        "gnss_station_solver_min_count_smoke": int(args.gnss_station_min_count_smoke),
        "gnss_station_solver_target_full": int(args.gnss_station_target_count_full),
        "gnss_station_coverage_threshold_used": float(coverage_threshold_used),
        "gnss_station_upwind_weight_mean": float(selection.upwind_weight_mean),
        "gnss_station_upwind_nonzero_ratio": float(selection.upwind_nonzero_ratio),
        "ppp_success_stations": ppp_success_stations,
    }
    return station_features, selected_meta, station_meta_payload


def _build_sample_index_v2(
    farm: pd.DataFrame,
    turbine: pd.DataFrame,
    *,
    t_in: int,
    horizon: int,
    step_out_steps: int,
    target_window_steps: int,
    freq_minutes: int,
    expected_turbines: int,
    farm_feature_columns: list[str],
    turbine_feature_columns: list[str],
) -> pd.DataFrame:
    times = pd.to_datetime(farm["datetime"]).reset_index(drop=True)
    n_times = len(times)
    base_ns = int(freq_minutes) * 60 * 1_000_000_000
    diffs_ns = np.diff(times.view("int64"))
    gap_prefix = np.zeros(n_times, dtype=np.int64)
    if n_times > 1:
        bad_gap = (diffs_ns <= 0) | (diffs_ns != base_ns)
        gap_prefix[1:] = np.cumsum(bad_gap.astype(np.int64))

    farm_feature_cols = [c for c in farm_feature_columns if c in farm.columns]
    target_cols = [f"target_power_step_{step:02d}" for step in range(1, horizon + 1) if f"target_power_step_{step:02d}" in farm.columns]
    farm_complete = farm[farm_feature_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy()
    target_complete = farm[target_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy()

    turb_cols = ["datetime", "turbine_id"] + [c for c in turbine_feature_columns if c in turbine.columns]
    turb = turbine[turb_cols].copy()
    turb["turbine_id"] = turb["turbine_id"].astype(str)
    turb_feature_cols = [c for c in turbine_feature_columns if c in turb.columns]
    turb_numeric = turb[turb_feature_cols].apply(pd.to_numeric, errors="coerce")
    turb["complete_row"] = turb_numeric.notna().all(axis=1)
    grouped_complete = turb.groupby("datetime")["complete_row"]
    row_ok = grouped_complete.all() & (grouped_complete.size() >= expected_turbines)
    time_to_ok = row_ok.reindex(times, fill_value=False).to_numpy()

    rows: list[dict[str, object]] = []
    max_origin = n_times - step_out_steps * horizon
    for origin in range(t_in - 1, max_origin):
        start = origin - t_in + 1
        target_end_idx = origin + step_out_steps * horizon
        is_contiguous = bool((gap_prefix[target_end_idx] - gap_prefix[start]) == 0)
        input_complete = bool(farm_complete[start : origin + 1].all() and time_to_ok[start : origin + 1].all())
        target_ok = bool(target_complete[origin])
        window_slice = slice(start, target_end_idx + 1)
        rows.append(
            {
                "origin_idx": int(origin),
                "origin_time": times.iloc[origin],
                "input_start_idx": int(start),
                "input_end_idx": int(origin),
                "target_end_idx": int(target_end_idx),
                "is_contiguous": float(is_contiguous),
                "is_input_complete": float(input_complete),
                "is_target_complete": float(target_ok),
                "window_has_high_ramp": float(farm.iloc[window_slice]["flag_high_ramp_farm"].fillna(0).max() > 0),
                "window_has_long_zero": float(farm.iloc[window_slice]["flag_long_zero_run_farm"].fillna(0).max() > 0),
                "window_has_gap_fill": float(farm.iloc[window_slice]["flag_gap_filled_any"].fillna(0).max() > 0),
                "window_has_zero_power_high_wind": float(farm.iloc[window_slice]["flag_zero_power_high_wind_farm"].fillna(0).max() > 0),
            }
        )
    sample_index = pd.DataFrame(rows)
    sample_index["is_train_eligible_default"] = (
        sample_index["is_contiguous"].astype(bool)
        & sample_index["is_input_complete"].astype(bool)
        & sample_index["is_target_complete"].astype(bool)
        & ~sample_index["window_has_zero_power_high_wind"].astype(bool)
    ).astype(float)
    return sample_index


def _build_static_graph_v2(turbines: pd.DataFrame, k: int) -> pd.DataFrame:
    base = _build_static_graph(turbines[["turbine_id", "lat", "lon"]], k)
    meta = turbines.set_index("turbine_id")
    out = base.copy()
    out["record_type"] = "edge"
    out["src_lat"] = out["src"].map(meta["lat"])
    out["src_lon"] = out["src"].map(meta["lon"])
    out["dst_lat"] = out["dst"].map(meta["lat"])
    out["dst_lon"] = out["dst"].map(meta["lon"])
    zero_series = pd.Series(np.zeros(len(out), dtype=np.float64), index=out.index)
    src_elev = out["src"].map(meta["elevation_m"]) if "elevation_m" in meta.columns else zero_series
    dst_elev = out["dst"].map(meta["elevation_m"]) if "elevation_m" in meta.columns else zero_series
    src_z0 = out["src"].map(meta["z0"]) if "z0" in meta.columns else zero_series
    dst_z0 = out["dst"].map(meta["z0"]) if "z0" in meta.columns else zero_series
    out["elevation_diff_m"] = pd.to_numeric(dst_elev, errors="coerce").fillna(0.0) - pd.to_numeric(src_elev, errors="coerce").fillna(0.0)
    out["roughness_pair"] = np.sqrt(
        np.clip(pd.to_numeric(src_z0, errors="coerce").fillna(0.0).to_numpy(), 1e-6, None)
        * np.clip(pd.to_numeric(dst_z0, errors="coerce").fillna(0.0).to_numpy(), 1e-6, None)
    )
    out["wake_prior"] = np.exp(-pd.to_numeric(out["distance_m"], errors="coerce").fillna(0.0) / 1500.0)
    for col in DEFAULT_FARM_V2_STATIC_NODE_COLUMNS:
        if col in meta.columns:
            out[f"src_{col}"] = out["src"].map(meta[col])
            out[f"dst_{col}"] = out["dst"].map(meta[col])
    return out


def _build_farm_v2(args: argparse.Namespace) -> None:
    if str(getattr(args, "farm_v2_scada_source", "raw")) == "processed":
        print(f"farm_v2 SCADA source: processed ({args.scada}, {args.scada_minute})")
        merged = _load_processed_scada_for_farm_v2(args)
    else:
        print(f"farm_v2 SCADA source: raw ({args.raw_scada_dir})")
        raw_minute = _load_raw_scada_minutes(
            args.raw_scada_dir,
            args.raw_scada_pattern,
            args.raw_scada_encoding,
            args.raw_wind_min,
            args.raw_wind_max,
            args.raw_power_min,
            args.raw_power_max,
            recursive=bool(args.raw_scada_recursive),
            id_col_override=str(args.raw_id_col),
            time_col_override=str(args.raw_time_col),
            wind_col_override=str(args.raw_wind_col),
            power_col_override=str(args.raw_power_col),
            start=str(args.raw_start),
            end=str(args.raw_end),
            turbine_id_prefix=str(args.raw_turbine_id_prefix),
        )
        if args.scada_offset_min:
            raw_minute["datetime"] = raw_minute["datetime"] + pd.to_timedelta(args.scada_offset_min, unit="m")
        merged = _resample_raw_scada_for_farm_v2(raw_minute, args.freq, args.resample_label, args.resample_closed)
    expected_freq_min = args.expected_freq_min if args.expected_freq_min > 0 else int(pd.to_timedelta(args.freq).total_seconds() / 60.0)
    era5_dynamic = pd.DataFrame()

    if args.topo.exists():
        topo = pd.read_parquet(args.topo)
        merged = merged.merge(topo, on="turbine_id", how="left", suffixes=("", "_topo"))
        if "elevation_m_topo" in merged.columns:
            if "elevation_m" in merged.columns:
                merged["elevation_m"] = merged["elevation_m"].combine_first(merged["elevation_m_topo"])
            else:
                merged["elevation_m"] = merged["elevation_m_topo"]
            merged = merged.drop(columns=["elevation_m_topo"])
    elif args.coords.exists():
        coords = _load_coords(args.coords)
        merged = merged.merge(coords, on="turbine_id", how="left")

    if bool(getattr(args, "drop_missing_coords", False)):
        before = len(merged)
        merged = merged.dropna(subset=["lat", "lon"]).copy()
        dropped = before - len(merged)
        if dropped > 0:
            print(f"Dropped {dropped} SCADA rows without turbine coordinates.")

    if args.era5.exists():
        era5_static = _load_era5_static_priors(args.era5)
        if not era5_static.empty:
            merged = merged.merge(era5_static, on="turbine_id", how="left", suffixes=("", "_era5_static"))
            for base_col in ("lat", "lon", "elevation_m", "z0"):
                era_col = f"{base_col}_era5_static"
                if era_col not in merged.columns:
                    continue
                if base_col in merged.columns:
                    merged[base_col] = pd.to_numeric(merged[base_col], errors="coerce").combine_first(
                        pd.to_numeric(merged[era_col], errors="coerce")
                    )
                else:
                    merged[base_col] = pd.to_numeric(merged[era_col], errors="coerce")
                merged = merged.drop(columns=[era_col])

    if args.era5.exists():
        era5_dynamic = _load_era5(args.era5)
        era5_dynamic = _resample_era5(era5_dynamic, args.freq, args.resample_label, args.resample_closed)
        if args.era5_offset_hours:
            era5_dynamic["datetime"] = era5_dynamic["datetime"] + pd.to_timedelta(args.era5_offset_hours, unit="h")
        if args.era5_offset_min:
            era5_dynamic["datetime"] = era5_dynamic["datetime"] + pd.to_timedelta(args.era5_offset_min, unit="m")
        if not args.scada_only:
            merged = merged.merge(era5_dynamic, on=["turbine_id", "datetime"], how="left", suffixes=("", "_era5"))
            for base_col in ("lat", "lon", "elevation_m", "z0"):
                era_col = f"{base_col}_era5"
                if era_col not in merged.columns:
                    continue
                if base_col in merged.columns:
                    merged[base_col] = pd.to_numeric(merged[base_col], errors="coerce").combine_first(
                        pd.to_numeric(merged[era_col], errors="coerce")
                    )
                else:
                    merged[base_col] = pd.to_numeric(merged[era_col], errors="coerce")
                merged = merged.drop(columns=[era_col])

    if "landcover" in merged.columns:
        merged["landcover"] = merged["landcover"].fillna(-1)
    if "slope_deg" in merged.columns:
        merged["slope_deg"] = merged["slope_deg"].fillna(0.0)
    if "z0" in merged.columns:
        merged["z0"] = pd.to_numeric(merged["z0"], errors="coerce").fillna(1e-4)

    all_turbines = sorted(merged["turbine_id"].astype(str).unique().tolist())
    merged = _tag_zero_power_high_wind_farm(
        merged,
        expected_turbines=all_turbines,
        wind_min=args.zero_power_wind_min,
        power_max_kw=args.zero_power_max_kw,
    )
    merged = _apply_power_curve_outlier_mask_v2(
        merged,
        wind_min=args.pc_wind_min,
        bin_width=args.pc_bin_width,
        min_bin_samples=args.pc_min_bin_samples,
        low_k=args.pc_mad_low_k,
        high_k=args.pc_mad_high_k,
    )
    if args.impute_missing_turbines:
        impute_cols = [c for c in ("power_5min_mean", "power_1min_inst", "wind_speed_5min_mean", "wind_speed_1min_inst") if c in merged.columns]
        merged = _impute_missing_turbines(
            merged.rename(
                columns={
                    "power_5min_mean": "power",
                    "power_1min_inst": "power_inst_tmp",
                    "wind_speed_5min_mean": "wind_speed",
                    "wind_speed_1min_inst": "wind_speed_inst",
                }
            ),
            expected_turbines=all_turbines,
            min_turbines=args.min_turbines,
            max_missing=args.impute_max_missing,
            neighbor_k=args.impute_neighbor_k,
            impute_cols=["power", "wind_speed_inst", "wind_speed"],
        ).rename(
            columns={
                "power": "power_5min_mean",
                "power_inst_tmp": "power_1min_inst",
                "wind_speed": "wind_speed_5min_mean",
                "wind_speed_inst": "wind_speed_1min_inst",
            }
        )

    merged["power"] = merged["power_5min_mean"]
    merged["wind_speed"] = merged["wind_speed_5min_mean"]
    merged["wind_speed_inst"] = merged["wind_speed_1min_inst"]
    if args.add_missing_indicators:
        merged, created_missing_cols = _append_missing_indicators(
            merged,
            skip_cols={"datetime", "turbine_id", "imputed_missing", "flag_zero_power_high_wind_turbine", "flag_zero_power_high_wind_farm", "flag_power_curve_outlier"},
        )
        if created_missing_cols > 0:
            print(f"Added {created_missing_cols} missing-indicator columns before farm_v2 gap fill.")
    before_fill = merged.copy()
    if args.fill_missing:
        merged = _fill_short_gaps(
            merged,
            args.freq,
            args.fill_limit,
            args.fill_method,
            args.kalman_q_level,
            args.kalman_q_trend,
            args.kalman_r,
        )
        merged = _add_fill_flags(
            before_fill,
            merged,
            [
                "power_5min_mean",
                "power_1min_inst",
                "wind_speed_5min_mean",
                "wind_speed_1min_inst",
                "power",
                "wind_speed",
                "wind_speed_inst",
            ],
        )
    else:
        merged["flag_gap_filled_any"] = merged.get("imputed_missing", 0.0)
        merged["flag_gap_filled_power"] = merged.get("imputed_missing", 0.0)
        merged["flag_gap_filled_wind"] = merged.get("imputed_missing", 0.0)

    drop_cols = [c for c in ("power", "wind_speed", "wind_speed_inst") if c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    merged = _add_turbine_derived_features_v2(merged, expected_freq_min)

    farm = _build_farm_features_v2(merged, expected_freq_min)
    farm = _tag_ramp_and_zero_runs(
        farm,
        farm_capacity_mw=args.farm_capacity_mw,
        high_ramp_frac=args.high_ramp_threshold_frac,
        long_zero_minutes=args.long_zero_minutes,
        freq_minutes=expected_freq_min,
    )
    if not era5_dynamic.empty and {"datetime", "u100", "v100"}.issubset(era5_dynamic.columns):
        flow_teacher = (
            era5_dynamic.groupby("datetime", as_index=False)
            .agg(era5_flow_u100=("u100", "mean"), era5_flow_v100=("v100", "mean"))
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        farm = farm.merge(flow_teacher, on="datetime", how="left")
    if args.gnss_raw:
        farm = _merge_gnss_raw_features(
            farm,
            args.gnss_raw,
            freq=args.freq,
            label=args.resample_label,
            closed=args.resample_closed,
            time_col=args.gnss_time_col,
            prefix=args.gnss_prefix,
            station_col=args.gnss_station_col,
            station_value=args.gnss_station_value,
            agg="last",
        )
        gnss_base_cols = [
            c
            for c in farm.columns
            if c.startswith(f"{args.gnss_prefix}_") and not c.endswith("_missing") and not c.endswith("_available")
        ]
        farm = _add_gnss_temporal_features(
            farm,
            feature_cols=gnss_base_cols,
            freq_minutes=expected_freq_min,
        )
    elif args.gnss_daily:
        farm = _merge_gnss_daily_features(farm, args.gnss_daily)

    step_out_steps = int(args.step_out_min / expected_freq_min)
    target_window_steps = int(args.target_window_min / expected_freq_min)
    farm = _attach_targets_v2(
        farm,
        horizon=args.horizon,
        step_out_steps=step_out_steps,
        target_window_steps=target_window_steps,
        freq_minutes=expected_freq_min,
    )
    farm = _attach_vector_targets_v2(
        farm,
        u_col="era5_flow_u100",
        v_col="era5_flow_v100",
        horizon=args.horizon,
        step_out_steps=step_out_steps,
        target_window_steps=target_window_steps,
        freq_minutes=expected_freq_min,
    )

    gnss_feature_columns = [c for c in farm.columns if c.startswith(f"{args.gnss_prefix}_") and not c.endswith("_missing") and not c.endswith("_available")]
    static_cols = [c for c in DEFAULT_FARM_V2_STATIC_NODE_COLUMNS if c in merged.columns]
    farm_feature_columns = [c for c in DEFAULT_FARM_V2_FARM_FEATURE_COLUMNS if c in farm.columns]
    turbine_feature_columns = [c for c in DEFAULT_FARM_V2_TURBINE_FEATURE_COLUMNS if c in merged.columns]
    sample_index = _build_sample_index_v2(
        farm,
        merged,
        t_in=args.t_in,
        horizon=args.horizon,
        step_out_steps=step_out_steps,
        target_window_steps=target_window_steps,
        freq_minutes=expected_freq_min,
        expected_turbines=len(all_turbines),
        farm_feature_columns=farm_feature_columns,
        turbine_feature_columns=turbine_feature_columns,
    )

    if "lat" not in merged.columns or "lon" not in merged.columns:
        raise ValueError("farm_v2 requires lat/lon in turbine features. Provide --topo or --coords.")
    turbines = merged[
        ["turbine_id"] + [c for c in ("lat", "lon", "elevation_m", "slope_deg", "landcover", "z0") if c in merged.columns]
    ].drop_duplicates(subset=["turbine_id"]).reset_index(drop=True)
    graph = _build_static_graph_v2(turbines, args.knn)
    gnss_station_features, gnss_station_meta, gnss_station_meta_payload = _build_gnss_station_products_v2(
        args=args,
        farm=farm,
        turbines=turbines,
        expected_freq_min=expected_freq_min,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    farm.to_parquet(args.out_dir / "farm_features.parquet", index=False)
    merged.to_parquet(args.out_dir / "turbine_features.parquet", index=False)
    sample_index.to_parquet(args.out_dir / "sample_index.parquet", index=False)
    graph.to_parquet(args.out_dir / "graph_static.parquet", index=False)
    if not gnss_station_features.empty:
        gnss_station_features.to_parquet(args.out_dir / "gnss_station_features.parquet", index=False)
    if not gnss_station_meta.empty:
        gnss_station_meta.to_parquet(args.out_dir / "gnss_station_meta.parquet", index=False)
    meta = {
        "dataset_version": "farm_v2",
        "freq_min": expected_freq_min,
        "t_in": int(args.t_in),
        "horizon": int(args.horizon),
        "step_out_min": int(args.step_out_min),
        "target_window_min": int(args.target_window_min),
        "farm_feature_columns": farm_feature_columns,
        "turbine_feature_columns": turbine_feature_columns,
        "gnss_feature_columns": gnss_feature_columns,
        "gnss_station_feature_columns": list(gnss_station_meta_payload.get("gnss_station_feature_columns", [])),
        "gnss_station_static_columns": list(gnss_station_meta_payload.get("gnss_station_static_columns", [])),
        "gnss_selected_stations": list(gnss_station_meta_payload.get("gnss_station_selected_ids", [])),
        "gnss_station_policy": str(gnss_station_meta_payload.get("gnss_station_policy", "nearest")),
        "gnss_station_count": int(gnss_station_meta_payload.get("gnss_station_count", 0)),
        "gnss_flow_ready": bool(
            not gnss_station_features.empty
            and bool(gnss_station_meta_payload.get("gnss_station_selected_ids", []))
            and bool(gnss_station_meta_payload.get("gnss_station_feature_columns", []))
        ),
        "gnss_station_coverage_ratio": float(gnss_station_meta_payload.get("gnss_station_coverage_ratio", 0.0)),
        "gnss_station_coverage_threshold_used": float(
            gnss_station_meta_payload.get("gnss_station_coverage_threshold_used", args.gnss_station_coverage_threshold)
        ),
        "gnss_station_upwind_weight_mean": float(gnss_station_meta_payload.get("gnss_station_upwind_weight_mean", 0.0)),
        "gnss_station_upwind_nonzero_ratio": float(gnss_station_meta_payload.get("gnss_station_upwind_nonzero_ratio", 0.0)),
        "gnss_processing_route": gnss_station_meta_payload.get("gnss_processing_route"),
        "gnss_product_source": gnss_station_meta_payload.get("gnss_product_source"),
        "gnss_primary_variable": gnss_station_meta_payload.get("gnss_primary_variable"),
        "gnss_station_solver_min_count_smoke": int(gnss_station_meta_payload.get("gnss_station_solver_min_count_smoke", 0)),
        "gnss_station_solver_target_full": int(gnss_station_meta_payload.get("gnss_station_solver_target_full", 0)),
        "static_node_columns": static_cols,
        "target_columns": [f"target_power_step_{step:02d}" for step in range(1, args.horizon + 1)],
        "target_flow_u_columns": [f"target_flow_u_step_{step:02d}" for step in range(1, args.horizon + 1)],
        "target_flow_v_columns": [f"target_flow_v_step_{step:02d}" for step in range(1, args.horizon + 1)],
        "gnss_flow_label_source": "gnss_ztd_gradient_proxy" if args.gnss_station_policy == "ztd_gradient" else "era5_u100_v100",
        "turbine_ids": all_turbines,
        "graph_path": str(args.out_dir / "graph_static.parquet"),
        "gnss_station_features_path": str(args.out_dir / "gnss_station_features.parquet") if not gnss_station_features.empty else None,
        "gnss_station_meta_path": str(args.out_dir / "gnss_station_meta.parquet") if not gnss_station_meta.empty else None,
        "zero_power_wind_min": float(args.zero_power_wind_min),
        "zero_power_max_kw": float(args.zero_power_max_kw),
        "high_ramp_threshold_frac": float(args.high_ramp_threshold_frac),
        "long_zero_minutes": int(args.long_zero_minutes),
        "raw_scada_dir": str(args.raw_scada_dir),
        "raw_scada_pattern": str(args.raw_scada_pattern),
        "raw_scada_recursive": bool(args.raw_scada_recursive),
        "raw_turbine_id_prefix": str(args.raw_turbine_id_prefix),
        "farm_name": str(args.farm_name) if args.farm_name else None,
        "transfer_role": str(args.transfer_role) if args.transfer_role else None,
        "native_freq_min": int(args.native_freq_min) if args.native_freq_min > 0 else expected_freq_min,
        "fill_method": str(args.fill_method),
        "fill_limit": int(args.fill_limit),
        "farm_capacity_mw": float(args.farm_capacity_mw),
        "gnss_raw": str(args.gnss_raw) if args.gnss_raw else None,
        "gnss_daily": str(args.gnss_daily) if args.gnss_daily else None,
        "gnss_station_data": str(args.gnss_station_data) if args.gnss_station_data else None,
        "gnss_station_meta_input": str(args.gnss_station_meta) if args.gnss_station_meta else None,
        "gnss_download_manifest": str(args.gnss_download_manifest) if args.gnss_download_manifest else None,
        "download_source_counts": _load_source_counts_from_manifest(args.gnss_download_manifest),
        "ppp_success_stations": list(gnss_station_meta_payload.get("ppp_success_stations", [])),
        "troposphere_fallback_used": bool(args.gnss_troposphere_fallback_used),
        "included_regions": [item.strip() for item in str(args.gnss_included_regions).split(",") if item.strip()],
        "era5_completion_start": str(args.era5_completion_start) if args.era5_completion_start else None,
        "era5_completion_end": str(args.era5_completion_end) if args.era5_completion_end else None,
        "scada_only": bool(args.scada_only),
    }
    (args.out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {args.out_dir / 'farm_features.parquet'}")
    print(f"Saved: {args.out_dir / 'turbine_features.parquet'}")
    print(f"Saved: {args.out_dir / 'sample_index.parquet'}")
    print(f"Saved: {args.out_dir / 'graph_static.parquet'}")
    if not gnss_station_features.empty:
        print(f"Saved: {args.out_dir / 'gnss_station_features.parquet'}")
    if not gnss_station_meta.empty:
        print(f"Saved: {args.out_dir / 'gnss_station_meta.parquet'}")
    print(f"Saved: {args.out_dir / 'dataset_meta.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged dataset features.")
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="legacy",
        choices=["legacy", "farm_v2"],
        help="Dataset contract to build.",
    )
    parser.add_argument("--scada", type=Path, default=Path("data/processed/scada_5min.parquet"))
    parser.add_argument("--scada-minute", type=Path, default=Path("data/processed/scada_minute.parquet"))
    parser.add_argument("--raw-scada-dir", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--farm-v2-scada-source",
        type=str,
        default="raw",
        choices=["raw", "processed"],
        help="farm_v2 SCADA input source: raw minute CSVs or processed scada/scada-minute parquet files.",
    )
    parser.add_argument("--raw-scada-pattern", type=str, default="*_minute.csv")
    parser.add_argument("--raw-scada-encoding", type=str, default="gb18030")
    parser.add_argument("--raw-scada-recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--raw-id-col", type=str, default="")
    parser.add_argument("--raw-time-col", type=str, default="")
    parser.add_argument("--raw-wind-col", type=str, default="")
    parser.add_argument("--raw-power-col", type=str, default="")
    parser.add_argument("--raw-start", type=str, default="", help="Optional inclusive raw SCADA start timestamp.")
    parser.add_argument("--raw-end", type=str, default="", help="Optional inclusive raw SCADA end timestamp or date.")
    parser.add_argument("--raw-turbine-id-prefix", type=str, default="", help="Strip this prefix from encoded turbine IDs when a timestamp contains the encoded namespace.")
    parser.add_argument("--raw-wind-min", type=float, default=0.0)
    parser.add_argument("--raw-wind-max", type=float, default=60.0)
    parser.add_argument("--raw-power-min", type=float, default=0.0)
    parser.add_argument("--raw-power-max", type=float, default=0.0)
    parser.add_argument("--drop-missing-coords", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--farm-name", type=str, default="")
    parser.add_argument("--transfer-role", type=str, default="")
    parser.add_argument("--native-freq-min", type=int, default=0)
    parser.add_argument("--era5", type=Path, default=Path("data/processed/era5_turbines"))
    parser.add_argument("--topo", type=Path, default=Path("data/topo_extract.parquet"))
    parser.add_argument("--coords", type=Path, default=Path("data/real/whereTurbins.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/features"))
    parser.add_argument("--graph-out", type=Path, default=Path("data/processed/graph_static.parquet"))
    parser.add_argument("--scada-only", action="store_true", help="Build features without ERA5 merge.")
    parser.add_argument("--freq", type=str, default="5min")
    parser.add_argument(
        "--expected-freq-min",
        type=int,
        default=0,
        help="Expected data frequency in minutes (0 means infer from --freq).",
    )
    parser.add_argument("--strict-freq", action="store_true", help="Error if data frequency mismatches expected.")
    parser.add_argument("--horizon-min", type=int, default=240)
    parser.add_argument("--t-in", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--step-out-min", type=int, default=15)
    parser.add_argument("--target-window-min", type=int, default=15)
    parser.add_argument("--l-threshold", type=float, default=5.0)
    parser.add_argument("--keep-unstable", action="store_true")
    parser.add_argument("--era5-offset-hours", type=float, default=8.0)
    parser.add_argument("--era5-offset-min", type=float, default=0.0)
    parser.add_argument("--scada-offset-min", type=float, default=0.0)
    parser.add_argument("--resample-label", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--resample-closed", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--instant-which", type=str, default="last", choices=["first", "last"])
    parser.add_argument("--l-clip", type=float, default=500.0)
    parser.add_argument("--l-min-abs", type=float, default=1.0)
    parser.add_argument("--l-bin-th1", type=float, default=50.0)
    parser.add_argument("--l-bin-th2", type=float, default=100.0)
    parser.add_argument("--l-bin-th3", type=float, default=200.0)
    parser.add_argument("--knn", type=int, default=6)
    parser.add_argument("--farm-capacity-mw", type=float, default=87.45)
    parser.add_argument(
        "--ramp-threshold-frac",
        type=float,
        default=0.0,
        help="Deprecated: ramp filtering is disabled and this option is ignored.",
    )
    parser.add_argument("--min-turbines", type=int, default=33)
    parser.add_argument(
        "--impute-missing-turbines",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Impute missing turbine rows when only one turbine is missing.",
    )
    parser.add_argument("--impute-neighbor-k", type=int, default=4)
    parser.add_argument("--impute-max-missing", type=int, default=1)
    parser.add_argument("--zero-power-wind-min", type=float, default=0.0)
    parser.add_argument(
        "--zero-power-max-kw",
        type=float,
        default=0.0,
        help="Drop rows when wind_speed_inst/wind_speed > zero-power-wind-min and power < this threshold (kW). <=0 disables.",
    )
    parser.add_argument(
        "--power-curve-bin-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable robust wind-power bin outlier filtering.",
    )
    parser.add_argument(
        "--pc-wind-col",
        type=str,
        default="wind_speed_inst",
        help="Wind column used by power-curve bin outlier filter.",
    )
    parser.add_argument(
        "--pc-power-col",
        type=str,
        default="power",
        help="Power column used by power-curve bin outlier filter.",
    )
    parser.add_argument(
        "--pc-wind-min",
        type=float,
        default=3.0,
        help="Minimum wind speed (m/s) to apply power-curve bin filtering.",
    )
    parser.add_argument(
        "--pc-bin-width",
        type=float,
        default=0.5,
        help="Wind-speed bin width (m/s) for power-curve outlier filtering.",
    )
    parser.add_argument(
        "--pc-min-bin-samples",
        type=int,
        default=40,
        help="Minimum samples per (turbine, wind-bin) to apply outlier bounds.",
    )
    parser.add_argument(
        "--pc-mad-low-k",
        type=float,
        default=4.0,
        help="Lower robust-sigma multiplier for bin outlier filter.",
    )
    parser.add_argument(
        "--pc-mad-high-k",
        type=float,
        default=6.0,
        help="Upper robust-sigma multiplier for bin outlier filter.",
    )
    parser.add_argument("--fill-method", type=str, default="kalman", choices=["linear", "kalman", "ffill"])
    parser.add_argument("--fill-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fill-limit", type=int, default=1)
    parser.add_argument("--kalman-q-level", type=float, default=0.01)
    parser.add_argument("--kalman-q-trend", type=float, default=0.001)
    parser.add_argument("--kalman-r", type=float, default=0.1)
    parser.add_argument(
        "--add-missing-indicators",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add per-feature missing indicators (<col>_missing) before fill/merge corrections.",
    )
    parser.add_argument(
        "--gnss-daily",
        type=Path,
        default=None,
        help="Optional daily GNSS feature table (parquet/csv) with a 'date' column.",
    )
    parser.add_argument(
        "--gnss-raw",
        type=Path,
        default=None,
        help="Optional high-frequency GNSS time-series (file/dir: parquet/csv/txt).",
    )
    parser.add_argument(
        "--gnss-time-col",
        type=str,
        default="datetime",
        help="Timestamp column name in GNSS raw table.",
    )
    parser.add_argument(
        "--gnss-prefix",
        type=str,
        default="gnss",
        help="Prefix for merged GNSS raw feature columns.",
    )
    parser.add_argument(
        "--gnss-station-col",
        type=str,
        default="",
        help="Optional station-id column in GNSS raw table.",
    )
    parser.add_argument(
        "--gnss-station-value",
        type=str,
        default="",
        help="Optional station-id value to keep (used with --gnss-station-col).",
    )
    parser.add_argument(
        "--gnss-agg",
        type=str,
        default="last",
        choices=["last", "mean"],
        help="Resample aggregation for GNSS raw series.",
    )
    parser.add_argument(
        "--gnss-station-data",
        type=Path,
        default=None,
        help="farm_v2: optional long-table GNSS station product source (file or dir).",
    )
    parser.add_argument(
        "--gnss-station-meta",
        type=Path,
        default=None,
        help="farm_v2: optional station metadata table for GNSS station products.",
    )
    parser.add_argument("--gnss-station-time-col", type=str, default="datetime")
    parser.add_argument("--gnss-station-id-col", type=str, default="station_id")
    parser.add_argument("--gnss-station-value-col", type=str, default="pwv")
    parser.add_argument("--gnss-station-quality-col", type=str, default="quality_flag")
    parser.add_argument("--gnss-station-lat-col", type=str, default="lat")
    parser.add_argument("--gnss-station-lon-col", type=str, default="lon")
    parser.add_argument("--gnss-station-elev-col", type=str, default="elevation_m")
    parser.add_argument("--gnss-station-start-k", type=int, default=10)
    parser.add_argument("--gnss-station-expand-k1", type=int, default=15)
    parser.add_argument("--gnss-station-expand-k2", type=int, default=20)
    parser.add_argument("--gnss-station-coverage-threshold", type=float, default=0.8)
    parser.add_argument("--gnss-processing-route", type=str, default="ppp_ztd_pwv")
    parser.add_argument("--gnss-product-source", type=str, default="whu_igs_public")
    parser.add_argument(
        "--gnss-download-manifest",
        type=Path,
        default=None,
        help="Optional GNSS download manifest csv for source-count metadata.",
    )
    parser.add_argument(
        "--gnss-ppp-summary",
        type=Path,
        default=None,
        help="Optional PPP summary json with successful_stations.",
    )
    parser.add_argument(
        "--gnss-included-regions",
        type=str,
        default="",
        help="Optional comma-separated region codes, e.g. CHN,TWN,HKG,MAC.",
    )
    parser.add_argument(
        "--era5-completion-start",
        type=str,
        default="",
        help="Optional completeness reference start date for prototype metadata.",
    )
    parser.add_argument(
        "--era5-completion-end",
        type=str,
        default="",
        help="Optional completeness reference end date for prototype metadata.",
    )
    parser.add_argument(
        "--gnss-troposphere-fallback-used",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set when GNSS station products mix in public troposphere-product fallback data.",
    )
    parser.add_argument("--gnss-station-min-count-smoke", type=int, default=4)
    parser.add_argument("--gnss-station-target-count-full", type=int, default=10)
    parser.add_argument(
        "--gnss-station-policy",
        type=str,
        default="nearest",
        choices=["nearest", "strict_upwind", "ztd_gradient"],
        help="farm_v2 GNSS station selection policy: nearest, external-wind strict upwind, or GNSS-only ZTD-gradient direction.",
    )
    parser.add_argument(
        "--high-ramp-threshold-frac",
        type=float,
        default=0.1,
        help="farm_v2: high-ramp label threshold as fraction of farm capacity.",
    )
    parser.add_argument(
        "--long-zero-minutes",
        type=int,
        default=60,
        help="farm_v2: minimum zero-power duration for long-zero label.",
    )
    args = parser.parse_args()

    if args.dataset_version == "farm_v2":
        if args.out_dir == Path("data/processed/features"):
            args.out_dir = Path("data/processed/farm_v2")
        _build_farm_v2(args)
        return

    scada = _load_scada(args.scada)
    scada_minute = _load_scada(args.scada_minute)
    for col in ("wind_speed", "power"):
        if col in scada.columns:
            scada[col] = scada[col].clip(lower=0)
        if col in scada_minute.columns:
            scada_minute[col] = scada_minute[col].clip(lower=0)
    if args.scada_offset_min:
        offset = pd.to_timedelta(args.scada_offset_min, unit="m")
        scada["datetime"] = scada["datetime"] + offset
        scada_minute["datetime"] = scada_minute["datetime"] + offset
        print(f"Shifted SCADA datetime by {args.scada_offset_min} minutes.")
    if args.freq != "5min":
        scada = _resample_scada(scada, args.freq, args.resample_label, args.resample_closed)
    scada_inst = _instant_from_minute(
        scada_minute,
        args.freq,
        args.resample_label,
        args.resample_closed,
        args.instant_which,
    )[
        ["datetime", "turbine_id", "wind_speed"]
    ].rename(columns={"wind_speed": "wind_speed_inst"})
    merged = scada.merge(scada_inst, on=["turbine_id", "datetime"], how="left")

    if not args.scada_only:
        era5 = _load_era5(args.era5)
        era5 = _resample_era5(era5, args.freq, args.resample_label, args.resample_closed)
        if args.era5_offset_hours:
            era5["datetime"] = era5["datetime"] + pd.to_timedelta(args.era5_offset_hours, unit="h")
            print(f"Shifted ERA5 datetime by {args.era5_offset_hours} hours.")
        if args.era5_offset_min:
            era5["datetime"] = era5["datetime"] + pd.to_timedelta(args.era5_offset_min, unit="m")
            print(f"Shifted ERA5 datetime by {args.era5_offset_min} minutes.")
        merged = merged.merge(era5, on=["turbine_id", "datetime"], how="inner")

    expected_freq_min = args.expected_freq_min
    if expected_freq_min <= 0:
        expected_freq_min = int(pd.to_timedelta(args.freq).total_seconds() / 60.0)
    data_freq_min = _infer_freq_min(merged["datetime"])
    if data_freq_min and expected_freq_min and data_freq_min != expected_freq_min:
        msg = (
            f"Data frequency {data_freq_min} min mismatches expected {expected_freq_min} min "
            f"(--freq={args.freq})."
        )
        if args.strict_freq:
            raise ValueError(msg)
        print(f"Warning: {msg}")
    _log_stage_stats("merged_raw", merged, expected_freq_min)

    if args.topo.exists():
        topo = pd.read_parquet(args.topo)
        merged = merged.merge(topo, on="turbine_id", how="left", suffixes=("", "_topo"))
        if "elevation_m_topo" in merged.columns:
            merged["elevation_m"] = merged["elevation_m"].combine_first(merged["elevation_m_topo"])
            merged = merged.drop(columns=["elevation_m_topo"])
    elif args.coords.exists():
        coords = _load_coords(args.coords)
        merged = merged.merge(coords, on="turbine_id", how="left")

    all_turbines = merged["turbine_id"].astype(str).unique().tolist()

    merged = _filter_zero_power_with_wind(merged, args.zero_power_wind_min, args.zero_power_max_kw)
    _log_stage_stats("after_zero_power_with_wind", merged, expected_freq_min)
    if args.impute_missing_turbines:
        impute_cols = [c for c in ("power", "wind_speed_inst", "wind_speed") if c in merged.columns]
        merged = _impute_missing_turbines(
            merged,
            expected_turbines=all_turbines,
            min_turbines=args.min_turbines,
            max_missing=args.impute_max_missing,
            neighbor_k=args.impute_neighbor_k,
            impute_cols=impute_cols,
        )
        _log_stage_stats("after_impute_missing_turbines", merged, expected_freq_min)
    power_curve_outlier_info: dict[str, object] | None = None
    if args.power_curve_bin_filter:
        merged_before_pc = merged
        merged, power_curve_outlier_info = _filter_power_curve_bin_outliers(
            merged,
            wind_col=args.pc_wind_col,
            power_col=args.pc_power_col,
            wind_min=args.pc_wind_min,
            bin_width=args.pc_bin_width,
            min_bin_samples=args.pc_min_bin_samples,
            low_k=args.pc_mad_low_k,
            high_k=args.pc_mad_high_k,
        )
        if merged.empty:
            print("Warning: power-curve bin filter removed all rows; using pre-filter data.")
            merged = merged_before_pc
            if power_curve_outlier_info is not None:
                power_curve_outlier_info["reason"] = "removed_all_reverted"
                power_curve_outlier_info["rows_after"] = int(len(merged))
        _log_stage_stats("after_power_curve_bin_filter", merged, expected_freq_min)
    if args.ramp_threshold_frac > 0:
        print(
            "Warning: --ramp-threshold-frac is deprecated and ignored; "
            "ramp filtering is disabled."
        )

    if "l_stability" in merged.columns:
        l_vals = merged["l_stability"].copy()
        if args.l_clip and args.l_clip > 0:
            l_vals = l_vals.clip(lower=-args.l_clip, upper=args.l_clip)
        if args.l_min_abs and args.l_min_abs > 0:
            l_vals = l_vals.mask(l_vals.abs() < args.l_min_abs)
        merged["l_stability"] = l_vals

        abs_l = merged["l_stability"].abs()
        norm_ref = args.l_bin_th3 if args.l_bin_th3 > 0 else 200.0
        l_norm = merged["l_stability"] / norm_ref
        l_norm = l_norm.clip(lower=-1.0, upper=1.0)
        merged["l_stability_norm"] = l_norm
        abs_l_norm = l_norm.abs()
        stable_sign = merged["l_stability"] > 0
        merged["is_stable"] = np.where(
            merged["l_stability"].isna(),
            np.nan,
            np.where(abs_l_norm >= 1.0, 0.5, np.where(stable_sign, 1.0, 0.0)),
        )
        merged["is_stable"] = merged["is_stable"].fillna(0.5)
        th1_norm = args.l_bin_th1 / norm_ref
        th2_norm = args.l_bin_th2 / norm_ref
        merged["l_bin"] = np.select(
            [abs_l_norm < th1_norm, abs_l_norm < th2_norm, abs_l_norm < 1.0],
            [0, 1, 2],
            default=3,
        )
        total = len(merged)
        l_nan = int(merged["l_stability"].isna().sum())
        l_b1 = int((abs_l < args.l_bin_th1).sum())
        l_b2 = int(((abs_l >= args.l_bin_th1) & (abs_l < args.l_bin_th2)).sum())
        l_b3 = int(((abs_l >= args.l_bin_th2) & (abs_l < args.l_bin_th3)).sum())
        l_b4 = int((abs_l >= args.l_bin_th3).sum())
        print(
            "L filter stats:",
            f"total={total}",
            f"nan={l_nan}",
            f"|L|<{args.l_bin_th1}={l_b1}",
            f"[{args.l_bin_th1},{args.l_bin_th2})={l_b2}",
            f"[{args.l_bin_th2},{args.l_bin_th3})={l_b3}",
            f"|L|>={args.l_bin_th3}={l_b4}",
        )
    if "landcover" in merged.columns:
        merged["landcover"] = merged["landcover"].fillna(-1)
    if "slope_deg" in merged.columns:
        merged["slope_deg"] = merged["slope_deg"].fillna(0.0)
    if not args.keep_unstable:
        before_drop = len(merged)
        merged = _drop_unstable(merged, args.l_threshold)
        print(f"L filter removed {before_drop - len(merged)} rows.")
        _log_stage_stats("after_l_filter", merged, expected_freq_min)

    merged = _filter_min_turbines(merged, args.min_turbines)
    _log_stage_stats("after_min_turbines_post_anomaly", merged, expected_freq_min)

    if args.add_missing_indicators:
        merged, created_missing_cols = _append_missing_indicators(
            merged,
            skip_cols={"datetime", "turbine_id", "imputed_missing"},
        )
        if created_missing_cols > 0:
            print(f"Added {created_missing_cols} missing-indicator columns before gap fill.")

    if args.fill_missing:
        merged = _fill_short_gaps(
            merged,
            args.freq,
            args.fill_limit,
            args.fill_method,
            args.kalman_q_level,
            args.kalman_q_trend,
            args.kalman_r,
        )
        _log_stage_stats("after_gap_fill", merged, expected_freq_min)

    if "u100" in merged.columns and "v100" in merged.columns:
        merged["wind_dir_100"] = np.arctan2(merged["v100"], merged["u100"])
        merged["wind_speed_100"] = np.sqrt(merged["u100"] ** 2 + merged["v100"] ** 2)
        merged["wind_dir_sin"] = np.sin(merged["wind_dir_100"])
        merged["wind_dir_cos"] = np.cos(merged["wind_dir_100"])
    elif "u140" in merged.columns and "v140" in merged.columns:
        merged["wind_dir_140"] = np.arctan2(merged["v140"], merged["u140"])
        merged["wind_speed_140"] = np.sqrt(merged["u140"] ** 2 + merged["v140"] ** 2)
        merged["wind_dir_sin"] = np.sin(merged["wind_dir_140"])
        merged["wind_dir_cos"] = np.cos(merged["wind_dir_140"])
    if "u10" in merged.columns and "v10" in merged.columns:
        merged["wind_dir_10"] = np.arctan2(merged["v10"], merged["u10"])
        merged["wind_speed_10"] = np.sqrt(merged["u10"] ** 2 + merged["v10"] ** 2)

    ref_col = None
    if "wind_speed_100" in merged.columns:
        ref_col = "wind_speed_100"
    elif "wind_speed_140" in merged.columns:
        ref_col = "wind_speed_140"
    if "wind_speed_inst" in merged.columns and ref_col is not None:
        agg = (
            merged.groupby("datetime", as_index=False)[["wind_speed_inst", ref_col]]
            .mean(numeric_only=True)
            .sort_values("datetime")
        )
        if len(agg) > 10:
            lags = {}
            for lag in range(-2, 3):
                ref = agg[ref_col].shift(lag)
                valid = agg["wind_speed_inst"].notna() & ref.notna()
                if valid.sum() < 10:
                    continue
                lags[lag] = agg.loc[valid, "wind_speed_inst"].corr(ref[valid])
            if lags:
                best = max(lags, key=lags.get)
                print(f"Alignment check (wind_speed_inst vs {ref_col}):")
                for lag, corr in lags.items():
                    print(f"  lag {lag:+d} -> corr {corr:.4f}")
                print(f"  best lag: {best:+d} (suggest shift {(-best)} steps of {args.freq})")

    freq_minutes = int(pd.to_timedelta(args.freq).total_seconds() / 60.0)
    cols_before = set(merged.columns)
    merged = _add_scada_derived_features(merged, freq_minutes)
    added_cols = sorted(set(merged.columns) - cols_before)
    if added_cols:
        print(f"Added {len(added_cols)} SCADA-derived features.")
    if args.gnss_raw:
        merged = _merge_gnss_raw_features(
            merged,
            args.gnss_raw,
            freq=args.freq,
            label=args.resample_label,
            closed=args.resample_closed,
            time_col=args.gnss_time_col,
            prefix=args.gnss_prefix,
            station_col=args.gnss_station_col,
            station_value=args.gnss_station_value,
            agg=args.gnss_agg,
        )
    elif args.gnss_daily:
        merged = _merge_gnss_daily_features(merged, args.gnss_daily)
    merged = _add_targets(merged, freq_minutes, args.horizon_min, source_col="wind_speed_inst")
    merged = merged.dropna(subset=["target_wind_speed"])
    _log_stage_stats("final_target_ready", merged, expected_freq_min)

    if "lat" not in merged.columns or "lon" not in merged.columns:
        raise ValueError("Missing lat/lon in features. Provide --topo or --coords for turbine locations.")
    turbines = merged[["turbine_id", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    graph = _build_static_graph(turbines, args.knn)
    args.graph_out.parent.mkdir(parents=True, exist_ok=True)
    graph.to_parquet(args.graph_out, index=False)

    merged["month"] = merged["datetime"].dt.strftime("%Y%m")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for month, frame in merged.groupby("month"):
        out_path = args.out_dir / f"features_{month}.parquet"
        frame.drop(columns=["month"]).to_parquet(out_path, index=False)
        print(f"Saved: {out_path}")

    print(f"Saved: {args.graph_out}")


if __name__ == "__main__":
    main()
