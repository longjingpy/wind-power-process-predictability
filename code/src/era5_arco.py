"""
Utilities for downloading local ERA5 subsets from the public ARCO ERA5 Zarr.

References:
- Hersbach, H. et al. (2020), "The ERA5 global reanalysis", Quarterly Journal
  of the Royal Meteorological Society, 146(730), 1999-2049.
  https://doi.org/10.1002/qj.3803
- Google Cloud, "ERA5 weather reanalysis dataset" public dataset docs.
  https://cloud.google.com/storage/docs/public-datasets/era5
- Google Research, "ARCO-ERA5" repository.
  https://github.com/google-research/arco-era5
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr


ARCO_ERA5_ROOT = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def _sanitize_attrs(attrs: dict[str, object]) -> dict[str, object]:
    return {str(key): value for key, value in attrs.items() if not str(key).startswith("_")}


@dataclass
class ArcoEra5Store:
    root: str = ARCO_ERA5_ROOT
    token: str = "anon"

    def __post_init__(self) -> None:
        import gcsfs
        import zarr

        self._fs = gcsfs.GCSFileSystem(token=self.token)
        # Avoid noisy cross-event-loop cleanup errors from gcsfs on Windows
        # process shutdown after successful anonymous reads.
        self._fs.close_session = lambda *args, **kwargs: None  # type: ignore[method-assign]
        self._group = zarr.open_group(self._fs.get_mapper(self.root), mode="r")

    @cached_property
    def latitude(self) -> np.ndarray:
        return np.asarray(self._group["latitude"][:], dtype=np.float32)

    @cached_property
    def longitude(self) -> np.ndarray:
        return np.asarray(self._group["longitude"][:], dtype=np.float32)

    @cached_property
    def level(self) -> np.ndarray:
        return np.asarray(self._group["level"][:], dtype=np.int64)

    @cached_property
    def time_hours(self) -> np.ndarray:
        return np.asarray(self._group["time"][:], dtype=np.int64)

    @cached_property
    def time_origin(self) -> pd.Timestamp:
        units = str(self._group["time"].attrs.get("units", "hours since 1900-01-01 00:00:00"))
        if "since" not in units:
            raise ValueError(f"Unexpected ARCO time units: {units}")
        return pd.Timestamp(units.split("since", 1)[1].strip())

    def _time_slice(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[slice, pd.DatetimeIndex]:
        start_ts = pd.Timestamp(start).tz_localize(None)
        end_ts = pd.Timestamp(end).tz_localize(None)
        start_hour = int((start_ts - self.time_origin) / pd.Timedelta(hours=1))
        end_hour = int((end_ts - self.time_origin) / pd.Timedelta(hours=1))
        left = int(np.searchsorted(self.time_hours, start_hour, side="left"))
        right = int(np.searchsorted(self.time_hours, end_hour, side="right"))
        if right <= left:
            raise ValueError(f"No ARCO ERA5 samples found in {start_ts}..{end_ts}")
        offsets = self.time_hours[left:right]
        times = self.time_origin + pd.to_timedelta(offsets, unit="h")
        return slice(left, right), pd.DatetimeIndex(times)

    def _lat_slice(self, north: float, south: float) -> tuple[slice, np.ndarray]:
        lats = self.latitude
        mask = (lats <= float(north) + 1e-6) & (lats >= float(south) - 1e-6)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            raise ValueError(f"No latitude cells found in {south}..{north}")
        lat_slice = slice(int(idx[0]), int(idx[-1]) + 1)
        return lat_slice, lats[lat_slice]

    def _lon_slice(self, west: float, east: float) -> tuple[slice, np.ndarray]:
        lons = self.longitude
        west_360 = float(west) % 360.0
        east_360 = float(east) % 360.0
        if west_360 > east_360:
            raise ValueError("ARCO helper does not support longitude wraparound selections.")
        mask = (lons >= west_360 - 1e-6) & (lons <= east_360 + 1e-6)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            raise ValueError(f"No longitude cells found in {west}..{east}")
        lon_slice = slice(int(idx[0]), int(idx[-1]) + 1)
        return lon_slice, lons[lon_slice]

    def _level_index(self, levels_hpa: list[int] | None) -> tuple[slice | list[int] | None, np.ndarray | None]:
        if levels_hpa is None:
            return None, None
        level_map = {int(value): idx for idx, value in enumerate(self.level.tolist())}
        indices = [level_map[int(level)] for level in levels_hpa if int(level) in level_map]
        if len(indices) != len(levels_hpa):
            missing = [int(level) for level in levels_hpa if int(level) not in level_map]
            raise ValueError(f"Missing ARCO pressure levels: {missing}")
        return indices, np.asarray(levels_hpa, dtype=np.int64)

    def subset_dataset(
        self,
        *,
        variables: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        area: list[float],
        levels_hpa: list[int] | None = None,
    ) -> xr.Dataset:
        north, west, south, east = [float(v) for v in area]
        time_slice, times = self._time_slice(start, end)
        lat_slice, lat_vals = self._lat_slice(north, south)
        lon_slice, lon_vals = self._lon_slice(west, east)
        level_index, level_vals = self._level_index(levels_hpa)

        coords: dict[str, tuple[tuple[str], np.ndarray | pd.DatetimeIndex]] = {
            "time": (("time",), times),
            "latitude": (("latitude",), lat_vals),
            "longitude": (("longitude",), lon_vals),
        }
        if level_vals is not None:
            coords["level"] = (("level",), level_vals)

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]] = {}
        for name in variables:
            if name not in self._group:
                raise KeyError(f"ARCO ERA5 variable not found: {name}")
            arr = self._group[name]
            attrs = _sanitize_attrs(dict(arr.attrs))
            if level_index is None:
                data = np.asarray(arr[time_slice, lat_slice, lon_slice])
                data_vars[name] = (("time", "latitude", "longitude"), data, attrs)
            else:
                data = np.asarray(arr.oindex[time_slice, level_index, lat_slice, lon_slice])
                data_vars[name] = (("time", "level", "latitude", "longitude"), data, attrs)

        return xr.Dataset(
            data_vars={name: (dims, values, attrs) for name, (dims, values, attrs) in data_vars.items()},
            coords=coords,
        )


@lru_cache(maxsize=4)
def _cached_store(token: str = "anon") -> ArcoEra5Store:
    # Reuse a single read-only ARCO store per token so repeated monthly/chunked
    # downloads do not reopen the public Zarr group for every task.
    return ArcoEra5Store(token=token)


def write_arco_subset(
    *,
    target: Path,
    variables: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: list[float],
    levels_hpa: list[int] | None = None,
    token: str = "anon",
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    print(
        f"[arco] open target={target.name} start={pd.Timestamp(start)} end={pd.Timestamp(end)} "
        f"levels={'surface' if levels_hpa is None else len(levels_hpa)}",
        flush=True,
    )
    store = _cached_store(token=token)
    t1 = perf_counter()
    ds = store.subset_dataset(
        variables=variables,
        start=start,
        end=end,
        area=area,
        levels_hpa=levels_hpa,
    )
    t2 = perf_counter()
    print(
        f"[arco] subset_ready target={target.name} open_s={t1 - t0:.1f} subset_s={t2 - t1:.1f}",
        flush=True,
    )
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    tmp_target.unlink(missing_ok=True)
    ds.to_netcdf(tmp_target)
    tmp_target.replace(target)
    t3 = perf_counter()
    print(
        f"[arco] write_done target={target.name} write_s={t3 - t2:.1f} total_s={t3 - t0:.1f}",
        flush=True,
    )
    ds.close()
    return target
