"""
Utilities for the strict GNSS meteorology route used by farm_v2 GNSS-flow.

References:
- Takasu, T. and Yasuda, A. (2009), "Development of the low-cost RTK-GPS
  receiver with an open source program package RTKLIB", International
  Symposium on GPS/GNSS. https://gpspp.sakura.ne.jp/paper2005/isgps2009_rtklib.pdf
- Saastamoinen, J. (1972), "Atmospheric Correction for the Troposphere and
  Stratosphere in Radio Ranging of Satellites", The Use of Artificial
  Satellites for Geodesy.
- Bevis, M. et al. (1994), "GPS meteorology: Mapping zenith wet delays onto
  precipitable water", Journal of Applied Meteorology, 33(3), 379-386.
  https://doi.org/10.1175/1520-0450(1994)033<0379:GMMZWD>2.0.CO;2
"""

from __future__ import annotations

import csv
import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


BKG_ROOT = "https://igs.bkg.bund.de"
BKG_STATION_API = f"{BKG_ROOT}/api/collections/stations/items"
BKG_RINEX_API = f"{BKG_ROOT}/api/files/rinex"

LAT_KEYS = ("lat", "latitude", "纬度")
LON_KEYS = ("lon", "longitude", "经度")
ELEV_KEYS = ("elevation", "height", "海拔", "高程")
ID_KEYS = ("turbine_id", "编号", "机组", "name")

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(frozen=True)
class FarmCenter:
    lat: float
    lon: float
    elevation_m: float
    turbine_count: int


def _pick_column(header: Iterable[str], keys: tuple[str, ...]) -> int | None:
    for idx, col in enumerate(header):
        col_norm = str(col).strip().lower()
        if col_norm in keys or any(key in col_norm for key in keys):
            return idx
    return None


def parse_dms_coord(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    for ch in ("°", "º", "'", '"', "′", "″", "’", "”"):
        text = text.replace(ch, " ")
    parts = [p for p in text.replace(",", " ").split() if p]
    try:
        if len(parts) == 1:
            return float(parts[0])
        deg = float(parts[0])
        minute = float(parts[1]) if len(parts) > 1 else 0.0
        second = float(parts[2]) if len(parts) > 2 else 0.0
    except ValueError:
        return None
    sign = -1.0 if deg < 0 else 1.0
    return sign * (abs(deg) + minute / 60.0 + second / 3600.0)


def _parse_elevation(value: object) -> float | None:
    text = str(value).replace("m", "").replace("M", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else None


def load_farm_center(coords_path: Path, encoding: str = "gb18030") -> FarmCenter:
    with coords_path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        lat_idx = _pick_column(header, LAT_KEYS)
        lon_idx = _pick_column(header, LON_KEYS)
        elev_idx = _pick_column(header, ELEV_KEYS)
        if lat_idx is None or lon_idx is None:
            lat_idx = 3
            lon_idx = 2
        lats: list[float] = []
        lons: list[float] = []
        elevs: list[float] = []
        for row in reader:
            if lat_idx >= len(row) or lon_idx >= len(row):
                continue
            lat = parse_dms_coord(row[lat_idx])
            lon = parse_dms_coord(row[lon_idx])
            if lat is None or lon is None:
                continue
            lats.append(lat)
            lons.append(lon)
            if elev_idx is not None and elev_idx < len(row):
                elev = _parse_elevation(row[elev_idx])
                if elev is not None:
                    elevs.append(elev)
    if not lats or not lons:
        raise ValueError(f"No valid farm coordinates found in {coords_path}")
    return FarmCenter(
        lat=float(np.mean(lats)),
        lon=float(np.mean(lons)),
        elevation_m=float(np.mean(elevs)) if elevs else 0.0,
        turbine_count=int(len(lats)),
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * math.atan2(math.sqrt(a), math.sqrt(max(1e-15, 1.0 - a)))


def utc_days(start: pd.Timestamp | datetime | str, end: pd.Timestamp | datetime | str) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start).tz_localize(None).normalize()
    end_ts = pd.Timestamp(end).tz_localize(None).normalize()
    return list(pd.date_range(start_ts, end_ts, freq="D"))


def yyyydoy(ts: pd.Timestamp | datetime | date) -> tuple[int, int]:
    stamp = pd.Timestamp(ts).tz_localize(None)
    return int(stamp.year), int(stamp.day_of_year)


def gps_week(ts: pd.Timestamp | datetime | date) -> int:
    stamp = pd.Timestamp(ts).tz_localize(None)
    gps_epoch = pd.Timestamp("1980-01-06")
    return int((stamp.normalize() - gps_epoch).days // 7)


def gps_week_start(week: int) -> pd.Timestamp:
    gps_epoch = pd.Timestamp("1980-01-06")
    return gps_epoch + pd.to_timedelta(int(week) * 7, unit="D")


def gps_week_tow_to_utc(week: int, tow_seconds: float) -> pd.Timestamp:
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    stamp = gps_epoch + timedelta(weeks=int(week), seconds=float(tow_seconds))
    return pd.Timestamp(stamp).tz_convert(None)


def fetch_bkg_station_catalog(session: requests.Session | None = None, timeout: float = 120.0) -> pd.DataFrame:
    http = session or requests.Session()
    response = http.get(BKG_STATION_API, headers={"Accept": "application/geo+json"}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    rows: list[dict[str, object]] = []
    for feature in payload.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates", [])
        if not isinstance(coords, list) or len(coords) < 2:
            continue
        props = feature.get("properties", {})
        projects = props.get("projects", []) or []
        rows.append(
            {
                "station_id": str(feature.get("id", "")).strip(),
                "lat": float(coords[1]),
                "lon": float(coords[0]),
                "country": props.get("country"),
                "agency": props.get("agency"),
                "projects": ",".join(str(item.get("name")) for item in projects if isinstance(item, dict)),
                "links": feature.get("properties", {}).get("links", []),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("BKG station catalog is empty.")
    return frame.sort_values("station_id").reset_index(drop=True)


def search_bkg_rinex(
    stations: Iterable[str],
    *,
    filetype: str,
    fileperiod: str,
    rinexversion: int,
    datasource: str,
    start: pd.Timestamp | datetime | str,
    end: pd.Timestamp | datetime | str,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> pd.DataFrame:
    params: list[tuple[str, str]] = []
    for station_id in stations:
        params.append(("stations", str(station_id)))
    params.extend(
        [
            ("filetype", str(filetype)),
            ("fileperiod", str(fileperiod)),
            ("rinexversion", str(rinexversion)),
            ("datasource", str(datasource)),
            ("starttime", pd.Timestamp(start).tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S")),
            ("endtime", pd.Timestamp(end).tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S")),
        ]
    )
    http = session or requests.Session()
    response = http.get(BKG_RINEX_API, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    frame = pd.DataFrame(data)
    if frame.empty:
        return frame
    for col in ("timeFilename", "timeFirstData", "timeLastData", "timeReceived", "timeModified"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True).dt.tz_convert(None)
    frame["station_id"] = frame["path"].astype(str).str.extract(r"/([A-Z0-9]{9})_")[0].fillna("")
    frame["download_url"] = frame["path"].astype(str).map(lambda x: f"{BKG_ROOT}{x}" if x.startswith("/") else x)
    frame["date"] = frame["timeFilename"].dt.normalize() if "timeFilename" in frame.columns else pd.NaT
    return frame


def parse_root_ftp_listing(html: str) -> list[str]:
    return re.findall(r'href="([^"]+)"', html)


def fetch_bkg_listing(path: str, session: requests.Session | None = None, timeout: float = 120.0) -> list[str]:
    http = session or requests.Session()
    response = http.get(f"{BKG_ROOT}{path}", timeout=timeout)
    response.raise_for_status()
    return parse_root_ftp_listing(response.text)


def precise_product_urls_for_window(
    start: pd.Timestamp | datetime | str,
    end: pd.Timestamp | datetime | str,
    *,
    session: requests.Session | None = None,
    orbit_interval: str = "15M",
    clock_interval: str = "30S",
    product_level: str = "FIN",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    http = session or requests.Session()
    product_rows: list[dict[str, object]] = []
    erp_rows: list[dict[str, object]] = []
    seen_weeks: set[int] = set()
    for day in utc_days(start, end):
        week = gps_week(day)
        if week not in seen_weeks:
            names = fetch_bkg_listing(f"/root_ftp/IGS/products/{week}/", session=http)
            seen_weeks.add(week)
            week_start = gps_week_start(week)
            year_i, doy_i = yyyydoy(week_start)
            erp_name = f"IGS0OPS{product_level}_{year_i}{doy_i:03d}0000_07D_01D_ERP.ERP.gz"
            if erp_name in names:
                erp_rows.append(
                    {
                        "gps_week": week,
                        "erp_name": erp_name,
                        "erp_url": f"{BKG_ROOT}/root_ftp/IGS/products/{week}/{erp_name}",
                    }
                )
            listing_cache = names
        else:
            listing_cache = fetch_bkg_listing(f"/root_ftp/IGS/products/{week}/", session=http)
        year_i, doy_i = yyyydoy(day)
        orbit_name = f"IGS0OPS{product_level}_{year_i}{doy_i:03d}0000_01D_{orbit_interval}_ORB.SP3.gz"
        clock_name = f"IGS0OPS{product_level}_{year_i}{doy_i:03d}0000_01D_{clock_interval}_CLK.CLK.gz"
        if orbit_name in listing_cache and clock_name in listing_cache:
            product_rows.append(
                {
                    "date": day.normalize(),
                    "gps_week": week,
                    "orbit_name": orbit_name,
                    "orbit_url": f"{BKG_ROOT}/root_ftp/IGS/products/{week}/{orbit_name}",
                    "clock_name": clock_name,
                    "clock_url": f"{BKG_ROOT}/root_ftp/IGS/products/{week}/{clock_name}",
                }
            )
    return pd.DataFrame(product_rows), pd.DataFrame(erp_rows).drop_duplicates(subset=["gps_week"], keep="first")


def download_file(
    url: str,
    out_path: Path,
    session: requests.Session | None = None,
    timeout: float = 120.0,
    retries: int = 5,
    backoff_seconds: float = 2.0,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    http = session or requests.Session()
    last_error: Exception | None = None
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    for attempt in range(int(retries)):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            curl_bin = shutil.which("curl.exe") or shutil.which("curl")
            if curl_bin:
                subprocess.run(
                    [
                        curl_bin,
                        "--fail",
                        "--location",
                        "--retry",
                        str(int(retries)),
                        "--retry-all-errors",
                        "--connect-timeout",
                        "30",
                        "--max-time",
                        str(int(timeout)),
                        "--output",
                        str(tmp_path),
                        url,
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            else:
                with http.get(
                    url,
                    stream=True,
                    timeout=timeout,
                    headers={"User-Agent": "WindPowerForcast/gnss-tropo"},
                ) as response:
                    response.raise_for_status()
                    with tmp_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            if chunk:
                                handle.write(chunk)
            tmp_path.replace(out_path)
            return out_path
        except Exception as exc:  # pragma: no cover - network variability
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt + 1 >= int(retries):
                break
            time.sleep(float(backoff_seconds) * float(attempt + 1))
    if last_error is not None:
        raise last_error
    return out_path


def parse_rinex_approx_xyz(path: Path) -> tuple[float, float, float] | None:
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            if "APPROX POSITION XYZ" in line:
                parts = line[:60].split()
                if len(parts) >= 3:
                    return float(parts[0]), float(parts[1]), float(parts[2])
            if "END OF HEADER" in line:
                break
    return None


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / max(1e-12, math.cos(lat)) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    h = p / max(1e-12, math.cos(lat)) - n
    return math.degrees(lat), math.degrees(lon), h
