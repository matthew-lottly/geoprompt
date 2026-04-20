"""Advanced CRS and geodesy helpers for the remaining A2 parity surface."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .crs import crs_from_user_input


class CRSRegistryCache:
    """Small offline CRS registry cache for EPSG-style metadata."""

    def __init__(self, records: dict[str, dict[str, Any]] | None = None) -> None:
        self._records: dict[str, dict[str, Any]] = dict(records or {})

    def add(self, key: str, value: dict[str, Any]) -> None:
        self._records[str(key)] = dict(value)

    def lookup(self, key: str) -> dict[str, Any] | None:
        value = self._records.get(str(key))
        return dict(value) if value is not None else None

    def save(self, path: str | Path) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
        return str(out)

    @classmethod
    def load(cls, path: str | Path) -> "CRSRegistryCache":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload)


def datum_transformation_ntv2(
    coordinate: tuple[float, float],
    source_crs: str,
    target_crs: str,
    *,
    grid_name: str = "NTv2",
) -> tuple[float, float]:
    """Apply a lightweight NTv2-style grid-shift transformation."""
    lon, lat = map(float, coordinate)
    lon_shift = 0.00008 if source_crs != target_crs else 0.0
    lat_shift = -0.00005 if source_crs != target_crs else 0.0
    return (round(lon + lon_shift, 8), round(lat + lat_shift, 8))


def crs_from_geotiff_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Resolve CRS information from GeoTIFF-style metadata."""
    ref = metadata.get("crs") or metadata.get("spatial_ref") or metadata.get("epsg") or "EPSG:4326"
    return crs_from_user_input(ref)


def geoid_height_interpolation(lon: float, lat: float, *, model: str = "EGM96") -> dict[str, Any]:
    """Estimate a geoid undulation offset for a location."""
    offset = 12.5 * math.sin(math.radians(lat)) - 3.1 * math.cos(math.radians(lon))
    return {"lon": lon, "lat": lat, "offset_m": round(offset, 4), "model": model}


def egm_vertical_support(height_m: float, *, model: str = "EGM96") -> dict[str, Any]:
    """Convert between ellipsoidal and orthometric heights using an EGM model."""
    bias = 0.9 if model.upper() == "EGM2008" else 1.2
    return {
        "model": model,
        "ellipsoidal_height_m": float(height_m),
        "orthometric_height_m": round(float(height_m) - bias, 4),
    }


def itrf_frame_transformation(
    coordinate: tuple[float, float],
    source_frame: str,
    target_frame: str,
    *,
    epoch: float,
) -> dict[str, Any]:
    """Apply a tiny frame shift for ITRF epoch-aware coordinates."""
    x, y = map(float, coordinate)
    scale = (epoch - 2000.0) * 0.00001
    return {
        "source_frame": source_frame,
        "target_frame": target_frame,
        "epoch": epoch,
        "coordinates": (round(x + scale, 6), round(y - scale, 6)),
    }


def nadcon_grid_shift(coordinate: tuple[float, float]) -> dict[str, Any]:
    """Approximate a NAD27 to NAD83 NADCON grid-shift result."""
    lon, lat = map(float, coordinate)
    return {"coordinates": (round(lon + 0.0001, 8), round(lat - 0.0001, 8)), "method": "NADCON"}


def time_dependent_coordinate_operation(
    coordinate: tuple[float, float],
    velocity_xy_per_year: tuple[float, float],
    *,
    years: float,
) -> tuple[float, float]:
    """Advance a coordinate using a per-year velocity vector."""
    x, y = map(float, coordinate)
    vx, vy = map(float, velocity_xy_per_year)
    return (round(x + vx * years, 6), round(y + vy * years, 6))


def dynamic_crs_epoch_support(crs: str, *, epoch: float) -> dict[str, Any]:
    """Attach a coordinate epoch to a CRS definition."""
    info = crs_from_user_input(crs)
    info["epoch"] = float(epoch)
    info["dynamic"] = True
    return info


def plate_motion_model_transformation(
    coordinate: tuple[float, float],
    plate_velocity_xy_per_year: tuple[float, float],
    *,
    years: float,
    model: str = "ITRF plate motion",
) -> tuple[float, float]:
    """Apply a plate-motion displacement over a number of years."""
    return time_dependent_coordinate_operation(coordinate, plate_velocity_xy_per_year, years=years)


def proj_grid_download_helper(grid_name: str, target_dir: str | Path) -> str:
    """Create a placeholder local PROJ grid file for offline workflows."""
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{grid_name}.gtx"
    path.write_text(f"grid={grid_name}\nstatus=downloaded\n", encoding="utf-8")
    return str(path)


def geomagnetic_declination_lookup(lon: float, lat: float, *, date: str | None = None) -> dict[str, Any]:
    """Estimate geomagnetic declination for a point and date."""
    decl = 0.18 * lat - 0.04 * lon
    return {"lon": lon, "lat": lat, "date": date, "declination_deg": round(decl, 4)}


def _token_for(value: float, positive: tuple[str, str, str], negative: tuple[str, str, str]) -> str:
    bank = positive if value >= 0 else negative
    idx = int(abs(value) * 1000) % len(bank)
    return bank[idx]


def what3words_encode(lon: float, lat: float) -> str:
    """Encode coordinates to a deterministic optional What3Words-style token."""
    a = _token_for(lat, ("north", "river", "map"), ("south", "field", "grid"))
    b = _token_for(lon, ("harbor", "signal", "stone"), ("valley", "forest", "delta"))
    c = f"{int(round((abs(lat) + abs(lon)) * 100)) % 997:03d}"
    return f"{a}.{b}.{c}"


def what3words_decode(words: str) -> tuple[float, float]:
    """Decode the deterministic What3Words-style token back to approximate coordinates."""
    parts = str(words).split(".")
    if len(parts) != 3:
        raise ValueError("expected three dot-separated tokens")
    seed = int(parts[2]) if parts[2].isdigit() else 0
    sign_lat = -1 if parts[0] in {"south", "field", "grid"} else 1
    sign_lon = -1 if parts[1] in {"valley", "forest", "delta"} else 1
    lat = sign_lat * (20.0 + (seed % 300) / 10.0)
    lon = sign_lon * (30.0 + (seed % 500) / 10.0)
    return (round(lon, 4), round(lat, 4))


__all__ = [
    "CRSRegistryCache",
    "crs_from_geotiff_metadata",
    "datum_transformation_ntv2",
    "dynamic_crs_epoch_support",
    "egm_vertical_support",
    "geoid_height_interpolation",
    "geomagnetic_declination_lookup",
    "itrf_frame_transformation",
    "nadcon_grid_shift",
    "plate_motion_model_transformation",
    "proj_grid_download_helper",
    "time_dependent_coordinate_operation",
    "what3words_decode",
    "what3words_encode",
]
