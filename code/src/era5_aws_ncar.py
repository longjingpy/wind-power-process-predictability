"""
Utilities for downloading local surface ERA5 subsets from the NSF NCAR AWS mirror.

References:
- Hersbach, H. et al. (2020), "The ERA5 global reanalysis", Quarterly Journal
  of the Royal Meteorological Society, 146(730), 1999-2049.
  https://doi.org/10.1002/qj.3803
- NSF NCAR, "Curated ECMWF Reanalysis 5 (ERA5) on AWS Open Data".
  https://registry.opendata.aws/nsf-ncar-era5/
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from time import perf_counter

import pandas as pd
import requests
import xarray as xr


AWS_NCAR_BUCKET = "https://nsf-ncar-era5.s3.amazonaws.com"
AWS_NCAR_SURFACE_KEYS = {
    "2m_temperature": ("128_167", "2t"),
    "2m_dewpoint_temperature": ("128_168", "2d"),
    "surface_pressure": ("128_134", "sp"),
    "100m_u_component_of_wind": ("228_246", "100u"),
    "100m_v_component_of_wind": ("228_247", "100v"),
}


def _month_end_day(year: int, month: int) -> int:
    return int((pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).day)


def _aws_ncar_surface_url(*, year: int, month: int, variable: str) -> str:
    code, short = AWS_NCAR_SURFACE_KEYS[variable]
    start = f"{year:04d}{month:02d}0100"
    end = f"{year:04d}{month:02d}{_month_end_day(year, month):02d}23"
    key = f"e5.oper.an.sfc/{year:04d}{month:02d}/e5.oper.an.sfc.{code}_{short}.ll025sc.{start}_{end}.nc"
    return f"{AWS_NCAR_BUCKET}/{key}"


def _download_http_file(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.unlink(missing_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(target)
    return target


def _coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name
    raise KeyError(f"Missing coordinate in AWS NCAR file. Tried: {candidates}")


def subset_surface_from_aws_ncar(
    *,
    target: Path,
    year: int,
    month: int,
    area: list[float],
    variables: list[str],
) -> Path:
    north, west, south, east = [float(v) for v in area]
    cache_root = target.parent / "_aws_ncar_cache" / f"{year:04d}{month:02d}"
    datasets: list[xr.Dataset] = []
    t0 = perf_counter()
    print(f"[aws_ncar] open target={target.name} month={year:04d}-{month:02d}", flush=True)
    try:
        for variable in variables:
            if variable not in AWS_NCAR_SURFACE_KEYS:
                raise KeyError(f"AWS NCAR surface variable mapping missing: {variable}")
            url = _aws_ncar_surface_url(year=year, month=month, variable=variable)
            source_path = cache_root / Path(url).name
            print(f"[aws_ncar] fetch {source_path.name}", flush=True)
            _download_http_file(url, source_path)
            ds = xr.open_dataset(source_path)
            lat_name = _coord_name(ds, ("latitude", "lat"))
            lon_name = _coord_name(ds, ("longitude", "lon"))
            time_name = _coord_name(ds, ("time", "valid_time"))
            lat_slice = (
                slice(north, south)
                if float(ds[lat_name][0]) > float(ds[lat_name][-1])
                else slice(south, north)
            )
            lon_vals = ds[lon_name]
            if float(lon_vals.max()) > 180.0:
                west_sel = west % 360.0
                east_sel = east % 360.0
            else:
                west_sel = west
                east_sel = east
            subset = ds[[list(ds.data_vars)[0]]].sel(
                {lat_name: lat_slice, lon_name: slice(west_sel, east_sel)}
            )
            var_name = list(subset.data_vars)[0]
            subset = subset.rename(
                {
                    var_name: variable,
                    lat_name: "latitude",
                    lon_name: "longitude",
                    time_name: "time",
                }
            )
            datasets.append(subset)
        t1 = perf_counter()
        print(
            f"[aws_ncar] subset_ready target={target.name} subset_s={t1 - t0:.1f}",
            flush=True,
        )
        merged = xr.merge(datasets, compat="override")
        tmp_target = target.with_suffix(target.suffix + ".tmp")
        tmp_target.unlink(missing_ok=True)
        merged.to_netcdf(tmp_target)
        merged.close()
        tmp_target.replace(target)
        t2 = perf_counter()
        print(
            f"[aws_ncar] write_done target={target.name} write_s={t2 - t1:.1f} total_s={t2 - t0:.1f}",
            flush=True,
        )
        return target
    finally:
        for ds in datasets:
            with contextlib.suppress(Exception):
                ds.close()
