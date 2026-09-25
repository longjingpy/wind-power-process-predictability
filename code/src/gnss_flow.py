"""
GNSS flow preprocessing utilities for farm_v2.

This module standardizes station-product tables into a long-table contract that
the farm_v2 pipeline can consume for GNSS-flow modeling.

References:
- Zhang et al. (2025), "Communications to Circulations: Real-Time 3D Wind Field
  Prediction Using 5G GNSS Signals and Deep Learning", arXiv.
  https://arxiv.org/abs/2509.16068
- Lee et al. (2025), "Short-term forecasting of 3D wind field based on deep
  learning and water vapor", ESS Open Archive.
  https://essopenarchive.org/users/927137/articles/1346713-short-term-forecasting-of-3d-wind-field-based-on-deep-learning-and-water-vapor
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CORE_STATION_COLUMNS = {
    "datetime",
    "station_id",
    "quality_flag",
    "available",
    "lat",
    "lon",
    "elevation_m",
}
STRICT_UPWIND_WIND_COLUMNS = ("wind_dir_rad", "u100", "v100", "gnss_proxy_u", "gnss_proxy_v")


def _haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius_m = 6371000.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = phi2 - phi1
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_m * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _bearing(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dlambda = np.deg2rad(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return np.arctan2(y, x)


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix == ".gz" and path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    raise ValueError(f"Unsupported GNSS station data format: {path}")


def load_table_or_dir(path: Path) -> pd.DataFrame:
    if path.is_dir():
        frames: list[pd.DataFrame] = []
        for item in sorted(p for p in path.iterdir() if p.is_file()):
            try:
                frame = _load_table(item)
            except Exception:
                continue
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    return _load_table(path)


@dataclass(frozen=True)
class StationProductConfig:
    time_col: str = "datetime"
    station_col: str = "station_id"
    value_col: str = "product_value"
    quality_col: str = "quality_flag"
    lat_col: str = "lat"
    lon_col: str = "lon"
    elev_col: str = "elevation_m"


@dataclass(frozen=True)
class StationSelectionSummary:
    selected_station_ids: tuple[str, ...]
    selected_station_count: int
    coverage_ratio: float
    start_k: int
    final_k: int
    policy: str = "nearest"
    upwind_weight_mean: float = 0.0
    upwind_nonzero_ratio: float = 0.0


def _annotate_station_geo(
    station_meta: pd.DataFrame,
    *,
    farm_lat: float,
    farm_lon: float,
    farm_elevation_m: float,
) -> pd.DataFrame:
    meta = station_meta.copy()
    meta["distance_m"] = _haversine(
        np.full(len(meta), float(farm_lat)),
        np.full(len(meta), float(farm_lon)),
        meta["lat"].to_numpy(dtype=float),
        meta["lon"].to_numpy(dtype=float),
    )
    meta["farm_to_station_bearing_rad"] = _bearing(
        np.full(len(meta), float(farm_lat)),
        np.full(len(meta), float(farm_lon)),
        meta["lat"].to_numpy(dtype=float),
        meta["lon"].to_numpy(dtype=float),
    )
    meta["station_to_farm_bearing_rad"] = _bearing(
        meta["lat"].to_numpy(dtype=float),
        meta["lon"].to_numpy(dtype=float),
        np.full(len(meta), float(farm_lat)),
        np.full(len(meta), float(farm_lon)),
    )
    # Keep the legacy field name for backward compatibility with older artifacts.
    meta["bearing_rad"] = meta["farm_to_station_bearing_rad"]
    meta["elevation_diff_m"] = pd.to_numeric(meta["elevation_m"], errors="coerce").fillna(0.0) - float(farm_elevation_m)
    return meta


def _aligned_station_presence(
    station_df: pd.DataFrame,
    *,
    aligned_index: pd.Index,
    station_ids: list[str],
) -> pd.DataFrame:
    if not station_ids:
        return pd.DataFrame(index=aligned_index)
    window = station_df[station_df["station_id"].astype(str).isin(station_ids)].copy()
    if window.empty:
        return pd.DataFrame(index=aligned_index, columns=station_ids).fillna(0.0)
    pivot = (
        window.assign(is_present=(pd.to_numeric(window["available"], errors="coerce").fillna(0.0) > 0).astype(float))
        .pivot_table(index="datetime", columns="station_id", values="is_present", aggfunc="max")
        .reindex(aligned_index)
        .reindex(columns=station_ids)
        .fillna(0.0)
    )
    return pivot


def resolve_station_flow_direction_rad(df: pd.DataFrame) -> pd.Series:
    if "wind_dir_rad" in df.columns:
        wind_dir = pd.to_numeric(df["wind_dir_rad"], errors="coerce")
        if wind_dir.notna().any():
            return wind_dir
    if {"u100", "v100"}.issubset(df.columns):
        u100 = pd.to_numeric(df["u100"], errors="coerce")
        v100 = pd.to_numeric(df["v100"], errors="coerce")
        if u100.notna().any() and v100.notna().any():
            arr = np.full(len(df), np.nan, dtype=float)
            valid = u100.notna() & v100.notna()
            arr[valid.to_numpy()] = np.arctan2(
                v100[valid].to_numpy(dtype=float, copy=True),
                u100[valid].to_numpy(dtype=float, copy=True),
            )
            return pd.Series(arr, index=df.index, dtype=float)
    if {"gnss_proxy_u", "gnss_proxy_v"}.issubset(df.columns):
        proxy_u = pd.to_numeric(df["gnss_proxy_u"], errors="coerce")
        proxy_v = pd.to_numeric(df["gnss_proxy_v"], errors="coerce")
        valid = proxy_u.notna() & proxy_v.notna() & ((proxy_u**2 + proxy_v**2) > 0)
        arr = np.full(len(df), np.nan, dtype=float)
        arr[valid.to_numpy()] = np.arctan2(
            proxy_v[valid].to_numpy(dtype=float, copy=True),
            proxy_u[valid].to_numpy(dtype=float, copy=True),
        )
        return pd.Series(arr, index=df.index, dtype=float)
    return pd.Series(np.full(len(df), np.nan, dtype=float), index=df.index, dtype=float)


def add_ztd_gradient_direction_proxy(station_df: pd.DataFrame) -> pd.DataFrame:
    """Add a GNSS-only horizontal ZTD-gradient direction proxy to station data."""
    required = {"datetime", "station_id", "lat", "lon", "product_value"}
    missing = sorted(required - set(station_df.columns))
    if missing:
        raise ValueError(f"ZTD-gradient proxy requires station columns: {missing}")

    work = station_df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work["product_value"] = pd.to_numeric(work["product_value"], errors="coerce")
    lat = pd.to_numeric(work["lat"], errors="coerce")
    lon = pd.to_numeric(work["lon"], errors="coerce")
    lat0 = float(lat.median())
    lon0 = float(lon.median())
    earth_radius_m = 6371000.0
    work["_proxy_x_m"] = earth_radius_m * np.cos(np.deg2rad(lat0)) * np.deg2rad(lon - lon0)
    work["_proxy_y_m"] = earth_radius_m * np.deg2rad(lat - lat0)
    work["gnss_proxy_u"] = 0.0
    work["gnss_proxy_v"] = 0.0
    work["gnss_proxy_available"] = 0.0

    for _, index in work.groupby("datetime", sort=False).groups.items():
        rows = work.loc[index]
        valid = rows[["_proxy_x_m", "_proxy_y_m", "product_value"]].notna().all(axis=1)
        if "available" in rows.columns:
            valid &= pd.to_numeric(rows["available"], errors="coerce").fillna(0.0).gt(0.5)
        if int(valid.sum()) < 3:
            continue
        design = np.column_stack(
            [
                np.ones(int(valid.sum()), dtype=float),
                rows.loc[valid, "_proxy_x_m"].to_numpy(dtype=float),
                rows.loc[valid, "_proxy_y_m"].to_numpy(dtype=float),
            ]
        )
        gradient = np.linalg.lstsq(
            design,
            rows.loc[valid, "product_value"].to_numpy(dtype=float),
            rcond=None,
        )[0][1:]
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm <= 1e-12:
            continue
        work.loc[index, "gnss_proxy_u"] = float(gradient[0] / norm)
        work.loc[index, "gnss_proxy_v"] = float(gradient[1] / norm)
        work.loc[index, "gnss_proxy_available"] = 1.0

    return work.drop(columns=["_proxy_x_m", "_proxy_y_m"])


def _dynamic_signal_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in CORE_STATION_COLUMNS:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any() or pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if "product_value" in cols:
        cols = ["product_value"] + [c for c in cols if c != "product_value"]
    return cols


def normalize_station_products(
    product_df: pd.DataFrame,
    *,
    config: StationProductConfig,
    meta_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if product_df.empty:
        return pd.DataFrame(
            columns=["datetime", "station_id", "product_value", "quality_flag", "lat", "lon", "elevation_m"]
        )
    work = product_df.copy()
    rename_map = {
        config.time_col: "datetime",
        config.station_col: "station_id",
        config.value_col: "product_value",
        config.quality_col: "quality_flag",
        config.lat_col: "lat",
        config.lon_col: "lon",
        config.elev_col: "elevation_m",
    }
    rename_map = {src: dst for src, dst in rename_map.items() if src in work.columns}
    work = work.rename(columns=rename_map)
    required = {"datetime", "station_id", "product_value"}
    missing = [name for name in required if name not in work.columns]
    if missing:
        raise ValueError(f"GNSS station data missing required columns: {missing}")
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work = work.dropna(subset=["datetime"])
    work["station_id"] = work["station_id"].astype(str).str.strip()
    work["product_value"] = pd.to_numeric(work["product_value"], errors="coerce")
    if "quality_flag" not in work.columns:
        work["quality_flag"] = 1.0
    work["quality_flag"] = pd.to_numeric(work["quality_flag"], errors="coerce").fillna(0.0)

    if meta_df is not None and not meta_df.empty:
        meta = meta_df.copy()
        meta = meta.rename(
            columns={
                config.station_col: "station_id",
                config.lat_col: "lat",
                config.lon_col: "lon",
                config.elev_col: "elevation_m",
            }
        )
        keep_cols = [c for c in ("station_id", "lat", "lon", "elevation_m") if c in meta.columns]
        if "station_id" in keep_cols:
            meta["station_id"] = meta["station_id"].astype(str).str.strip()
            meta = meta[keep_cols].drop_duplicates(subset=["station_id"], keep="first")
            work = work.merge(meta, on="station_id", how="left", suffixes=("", "_meta"))
    for col in ("lat", "lon", "elevation_m"):
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in work.columns:
        if col in CORE_STATION_COLUMNS:
            continue
        numeric = pd.to_numeric(work[col], errors="coerce")
        if numeric.notna().any() or pd.api.types.is_numeric_dtype(work[col]):
            work[col] = numeric
    work["available"] = (work["product_value"].notna() & (work["quality_flag"] > 0)).astype(float)
    return work.sort_values(["station_id", "datetime"]).reset_index(drop=True)


def resample_station_products(
    df: pd.DataFrame,
    *,
    freq: str,
    label: str,
    closed: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    frames: list[pd.DataFrame] = []
    signal_cols = _dynamic_signal_columns(df)
    for station_id, group in df.groupby("station_id", sort=True):
        group = group.sort_values("datetime").set_index("datetime")
        signal_frames = [group[[col]].resample(freq, label=label, closed=closed).last() for col in signal_cols]
        quality = group[["quality_flag"]].resample(freq, label=label, closed=closed).max()
        available = group[["available"]].resample(freq, label=label, closed=closed).max()
        for static_col in ("lat", "lon", "elevation_m"):
            if static_col in group.columns:
                quality[static_col] = pd.to_numeric(group[static_col], errors="coerce").median()
        out = pd.concat(signal_frames + [quality, available], axis=1).reset_index()
        out["station_id"] = station_id
        frames.append(out)
    merged = pd.concat(frames, ignore_index=True)
    merged["available"] = merged["available"].fillna(0.0).astype(float)
    return merged.sort_values(["station_id", "datetime"]).reset_index(drop=True)


def select_nearest_stations(
    station_meta: pd.DataFrame,
    *,
    farm_lat: float,
    farm_lon: float,
    farm_elevation_m: float,
    aligned_times: Iterable[pd.Timestamp],
    station_df: pd.DataFrame,
    coverage_threshold: float = 0.8,
    start_k: int = 10,
    expansion_steps: tuple[int, ...] = (15, 20),
) -> tuple[pd.DataFrame, StationSelectionSummary]:
    if station_meta.empty:
        return station_meta.copy(), StationSelectionSummary(tuple(), 0, 0.0, start_k, 0)
    meta = _annotate_station_geo(
        station_meta,
        farm_lat=farm_lat,
        farm_lon=farm_lon,
        farm_elevation_m=farm_elevation_m,
    )
    meta = meta.sort_values(["distance_m", "station_id"]).reset_index(drop=True)

    aligned_index = pd.Index(pd.to_datetime(list(aligned_times)).sort_values().unique())
    candidate_ks = [int(start_k)] + [int(k) for k in expansion_steps if int(k) > int(start_k)]
    final_k = min(candidate_ks[0], len(meta))
    final_cov = 0.0
    selected = meta.iloc[:final_k].copy()
    for k in candidate_ks:
        take_k = min(int(k), len(meta))
        trial = meta.iloc[:take_k].copy()
        trial_ids = trial["station_id"].astype(str).tolist()
        if not trial_ids:
            continue
        window = station_df[station_df["station_id"].astype(str).isin(trial_ids)].copy()
        if window.empty:
            final_k = take_k
            final_cov = 0.0
            selected = trial
            continue
        pivot = (
            window.assign(is_present=(window["available"] > 0).astype(float))
            .pivot_table(index="datetime", columns="station_id", values="is_present", aggfunc="max")
            .reindex(aligned_index)
            .fillna(0.0)
        )
        coverage = float(pivot.to_numpy(dtype=float).mean()) if not pivot.empty else 0.0
        final_k = take_k
        final_cov = coverage
        selected = trial
        if coverage >= float(coverage_threshold):
            break
    return selected, StationSelectionSummary(
        selected_station_ids=tuple(selected["station_id"].astype(str).tolist()),
        selected_station_count=int(len(selected)),
        coverage_ratio=float(final_cov),
        start_k=int(start_k),
        final_k=int(final_k),
        policy="nearest",
    )


def select_strict_upwind_stations(
    station_meta: pd.DataFrame,
    *,
    farm_lat: float,
    farm_lon: float,
    farm_elevation_m: float,
    aligned_times: Iterable[pd.Timestamp],
    station_df: pd.DataFrame,
    coverage_threshold: float = 0.8,
    target_count: int = 10,
    min_count: int = 4,
    distance_sigma_m: float = 350000.0,
) -> tuple[pd.DataFrame, StationSelectionSummary]:
    if station_meta.empty:
        return station_meta.copy(), StationSelectionSummary(tuple(), 0, 0.0, target_count, 0, policy="strict_upwind")
    meta = _annotate_station_geo(
        station_meta,
        farm_lat=farm_lat,
        farm_lon=farm_lon,
        farm_elevation_m=farm_elevation_m,
    )
    aligned_index = pd.Index(pd.to_datetime(list(aligned_times)).sort_values().unique())
    if aligned_index.empty:
        return meta.iloc[0:0].copy(), StationSelectionSummary(tuple(), 0, 0.0, target_count, 0, policy="strict_upwind")
    station_ids = meta["station_id"].astype(str).tolist()
    presence = _aligned_station_presence(
        station_df,
        aligned_index=aligned_index,
        station_ids=station_ids,
    )
    coverage_ratio = presence.mean(axis=0).reindex(station_ids).fillna(0.0)

    flow_df = station_df[station_df["station_id"].astype(str).isin(station_ids)].copy()
    flow_df["station_id"] = flow_df["station_id"].astype(str)
    flow_df["datetime"] = pd.to_datetime(flow_df["datetime"], errors="coerce")
    flow_df = flow_df.dropna(subset=["datetime"])
    flow_df["flow_dir_rad"] = resolve_station_flow_direction_rad(flow_df)
    if not flow_df["flow_dir_rad"].notna().any():
        raise ValueError(
            "strict_upwind GNSS station selection requires station direction fields in gnss station products "
            f"(expected one of {STRICT_UPWIND_WIND_COLUMNS})."
        )
    flow_df = flow_df.merge(
        meta[["station_id", "station_to_farm_bearing_rad", "distance_m"]],
        on="station_id",
        how="left",
    )
    flow_df["upwind_alignment"] = np.clip(
        np.cos(
            pd.to_numeric(flow_df["station_to_farm_bearing_rad"], errors="coerce").to_numpy(dtype=float, copy=True)
            - pd.to_numeric(flow_df["flow_dir_rad"], errors="coerce").to_numpy(dtype=float, copy=True)
        ),
        0.0,
        None,
    ) ** 2
    alignment = (
        flow_df.pivot_table(index="datetime", columns="station_id", values="upwind_alignment", aggfunc="mean")
        .reindex(aligned_index)
        .reindex(columns=station_ids)
        .fillna(0.0)
    )
    align_mean = alignment.mean(axis=0).reindex(station_ids).fillna(0.0)

    meta["station_coverage_ratio"] = meta["station_id"].astype(str).map(coverage_ratio).fillna(0.0).astype(float)
    meta["distance_weight"] = np.exp(-meta["distance_m"].to_numpy(dtype=float, copy=True) / float(max(distance_sigma_m, 1.0)))
    meta["upwind_alignment_mean"] = meta["station_id"].astype(str).map(align_mean).fillna(0.0).astype(float)
    meta["coverage_pass"] = meta["station_coverage_ratio"].ge(float(coverage_threshold)).astype(float)
    meta["selection_score"] = (
        meta["coverage_pass"].to_numpy(dtype=float, copy=True)
        * meta["station_coverage_ratio"].to_numpy(dtype=float, copy=True)
        * meta["upwind_alignment_mean"].to_numpy(dtype=float, copy=True)
        * meta["distance_weight"].to_numpy(dtype=float, copy=True)
    )
    ranked = meta.sort_values(
        ["selection_score", "station_coverage_ratio", "upwind_alignment_mean", "distance_m", "station_id"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    selected = ranked[ranked["selection_score"] > 0].head(max(int(target_count), 1)).copy()
    if len(selected) < int(min_count):
        raise ValueError(
            f"strict_upwind selected only {len(selected)} station(s), below min_count={int(min_count)}. "
            "Regenerate station ERA5 wind fields or relax coverage filtering."
        )
    selected_ids = selected["station_id"].astype(str).tolist()
    selected_presence = presence.reindex(columns=selected_ids).fillna(0.0)
    selected_alignment = alignment.reindex(columns=selected_ids).fillna(0.0)
    return selected.reset_index(drop=True), StationSelectionSummary(
        selected_station_ids=tuple(selected_ids),
        selected_station_count=int(len(selected_ids)),
        coverage_ratio=float(selected_presence.to_numpy(dtype=float).mean()) if not selected_presence.empty else 0.0,
        start_k=int(target_count),
        final_k=int(len(selected_ids)),
        policy="strict_upwind",
        upwind_weight_mean=float(selected["selection_score"].mean()) if not selected.empty else 0.0,
        upwind_nonzero_ratio=float((selected_alignment.to_numpy(dtype=float) > 0.0).mean()) if not selected_alignment.empty else 0.0,
    )


def add_upwind_features(
    station_features: pd.DataFrame,
    *,
    station_meta: pd.DataFrame,
    coverage_threshold: float,
) -> pd.DataFrame:
    if station_features.empty:
        return station_features
    work = station_features.copy()
    meta_cols = [
        "station_id",
        "station_to_farm_bearing_rad",
        "station_coverage_ratio",
        "distance_weight",
    ]
    missing_meta = [col for col in meta_cols if col not in station_meta.columns]
    if missing_meta:
        raise ValueError(f"strict_upwind station meta missing required columns: {missing_meta}")
    work = work.merge(station_meta[meta_cols], on="station_id", how="left")
    flow_dir = resolve_station_flow_direction_rad(work)
    available = pd.to_numeric(work.get("available", 0.0), errors="coerce").fillna(0.0)
    alignment = np.clip(
        np.cos(
            pd.to_numeric(work["station_to_farm_bearing_rad"], errors="coerce").to_numpy(dtype=float, copy=True)
            - flow_dir.to_numpy(dtype=float, copy=True)
        ),
        0.0,
        None,
    ) ** 2
    coverage_gate = pd.to_numeric(work["station_coverage_ratio"], errors="coerce").fillna(0.0).ge(float(coverage_threshold))
    weight = (
        available.to_numpy(dtype=float, copy=True)
        * coverage_gate.to_numpy(dtype=float, copy=True)
        * pd.to_numeric(work["station_coverage_ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=True)
        * pd.to_numeric(work["distance_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=True)
        * np.nan_to_num(alignment, nan=0.0, posinf=0.0, neginf=0.0)
    )
    work["upwind_alignment"] = np.nan_to_num(alignment, nan=0.0, posinf=0.0, neginf=0.0)
    work["upwind_mask"] = (weight > 0.0).astype(float)
    work["upwind_weight"] = weight.astype(float)
    return work


def build_station_feature_table(
    station_df: pd.DataFrame,
    *,
    selected_station_ids: Iterable[str],
    freq_minutes: int,
) -> pd.DataFrame:
    selected_ids = {str(x) for x in selected_station_ids}
    if not selected_ids:
        return pd.DataFrame(columns=["datetime", "station_id", "product_value", "quality_flag", "available"])
    work = station_df[station_df["station_id"].astype(str).isin(selected_ids)].copy()
    work = work.sort_values(["station_id", "datetime"]).reset_index(drop=True)
    signal_cols = _dynamic_signal_columns(work)
    derived_cols: list[str] = []
    for station_id, idx in work.groupby("station_id").groups.items():
        rows = work.loc[idx].copy()
        dt = pd.to_datetime(rows["datetime"])
        prev_ok = dt.diff().dt.total_seconds().div(60.0).eq(float(freq_minutes))
        prev3_ok = dt.diff(3).dt.total_seconds().div(60.0).eq(float(freq_minutes * 3))
        roll_ok = prev_ok & prev_ok.shift(1, fill_value=False)
        available = pd.to_numeric(rows["available"], errors="coerce").fillna(0.0) > 0.5
        valid_diff1 = prev_ok & available & available.shift(1, fill_value=False)
        valid_diff3 = prev3_ok & available & available.shift(3, fill_value=False)
        valid_roll = roll_ok & available & available.shift(1, fill_value=False) & available.shift(2, fill_value=False)
        for col in signal_cols:
            value = pd.to_numeric(rows[col], errors="coerce")
            lag1 = value.shift(1)
            lag3 = value.shift(3)
            diff1_col = f"{col}_diff1"
            diff3_col = f"{col}_diff3"
            roll_mean_col = f"{col}_roll_mean_3"
            roll_std_col = f"{col}_roll_std_3"
            work.loc[idx, diff1_col] = np.where(valid_diff1, (value - lag1).to_numpy(dtype=float), 0.0)
            work.loc[idx, diff3_col] = np.where(valid_diff3, (value - lag3).to_numpy(dtype=float), 0.0)
            roll_mean = value.rolling(window=3, min_periods=1).mean()
            roll_std = value.rolling(window=3, min_periods=2).std().fillna(0.0)
            work.loc[idx, roll_mean_col] = np.where(valid_roll, roll_mean.to_numpy(dtype=float), 0.0)
            work.loc[idx, roll_std_col] = np.where(valid_roll, roll_std.to_numpy(dtype=float), 0.0)
            derived_cols.extend([diff1_col, diff3_col, roll_mean_col, roll_std_col])
    out_cols = ["datetime", "station_id"] + signal_cols + ["quality_flag", "available"]
    for col in dict.fromkeys(derived_cols):
        out_cols.append(col)
    for col in out_cols[2:]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    return work[out_cols].copy()
