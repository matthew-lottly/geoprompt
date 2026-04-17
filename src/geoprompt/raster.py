"""Optional raster bridge utilities for GeoPrompt.

The helpers in this module intentionally provide a small, lightweight bridge
rather than trying to recreate a full raster platform. They support either:

- in-memory raster dictionaries of the form
  ``{"data": [[...]], "transform": (min_x, max_y, cell_width, cell_height)}``
- file paths readable by rasterio when that optional dependency is installed
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .frame import GeoPromptFrame
from .geometry import geometry_type, geometry_within
from .table import PromptTable


RasterLike = dict[str, Any] | str | Path


def _load_rasterio() -> Any:
    try:
        return importlib.import_module("rasterio")
    except ImportError as exc:
        raise RuntimeError(
            "Install raster support with 'pip install geoprompt[raster]' to read raster files."
        ) from exc


def _coerce_raster(raster: RasterLike) -> dict[str, Any]:
    if isinstance(raster, dict):
        if "data" not in raster or "transform" not in raster:
            raise ValueError("in-memory raster dicts must include 'data' and 'transform'")
        return raster

    path = Path(raster)
    rio = _load_rasterio()
    with rio.open(path) as src:
        band = src.read(1)
        transform = src.transform
        return {
            "data": band.tolist(),
            "transform": (transform.c, transform.f, float(transform.a), abs(float(transform.e))),
            "nodata": src.nodata,
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "crs": str(src.crs) if src.crs else None,
            "path": str(path),
        }


def inspect_raster(raster: RasterLike) -> dict[str, Any]:
    """Inspect a raster and return metadata and summary statistics."""
    info = _coerce_raster(raster)
    data = info["data"]
    values = [cell for row in data for cell in row if cell is not None]
    height = len(data)
    width = len(data[0]) if data else 0
    return {
        "width": info.get("width", width),
        "height": info.get("height", height),
        "band_count": info.get("band_count", 1),
        "nodata": info.get("nodata"),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": (sum(values) / len(values)) if values else None,
        "crs": info.get("crs"),
    }


def _sample_value(info: dict[str, Any], x: float, y: float) -> Any:
    data = info["data"]
    min_x, max_y, cell_width, cell_height = info["transform"]
    col = int((x - float(min_x)) / float(cell_width))
    row = int((float(max_y) - y) / float(cell_height))
    if row < 0 or col < 0:
        return None
    if row >= len(data) or (data and col >= len(data[0])):
        return None
    return data[row][col]


def sample_raster_points(
    raster: RasterLike,
    points: GeoPromptFrame,
    *,
    value_column: str = "raster_value",
) -> GeoPromptFrame:
    """Sample raster values at point locations."""
    info = _coerce_raster(raster)
    rows: list[dict[str, Any]] = []
    for row in points.to_records():
        geom = row[points.geometry_column]
        if geometry_type(geom) != "Point":
            raise TypeError("sample_raster_points requires point geometries")
        x, y = geom["coordinates"]
        new_row = dict(row)
        new_row[value_column] = _sample_value(info, float(x), float(y))
        rows.append(new_row)
    return GeoPromptFrame.from_records(rows, geometry=points.geometry_column, crs=points.crs)


def raster_slope_aspect(raster: RasterLike) -> dict[str, Any]:
    """Compute simple slope and aspect grids from a raster surface."""
    info = _coerce_raster(raster)
    data = info["data"]
    rows = len(data)
    cols = len(data[0]) if data else 0
    _, _, cell_width, cell_height = info["transform"]

    slope_grid: list[list[float]] = [[0.0] * cols for _ in range(rows)]
    aspect_grid: list[list[float]] = [[0.0] * cols for _ in range(rows)]

    if rows < 3 or cols < 3:
        return {"slope": slope_grid, "aspect": aspect_grid, "rows": rows, "cols": cols}

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            z1, z2, z3 = data[r - 1][c - 1], data[r - 1][c], data[r - 1][c + 1]
            z4, z5, z6 = data[r][c - 1], data[r][c], data[r][c + 1]
            z7, z8, z9 = data[r + 1][c - 1], data[r + 1][c], data[r + 1][c + 1]
            dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * float(cell_width))
            dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * float(cell_height))
            slope = (dzdx ** 2 + dzdy ** 2) ** 0.5
            aspect = 180.0 / 3.141592653589793 * __import__("math").atan2(dzdy, -dzdx)
            if aspect < 0:
                aspect += 360.0
            slope_grid[r][c] = slope
            aspect_grid[r][c] = aspect

    return {"slope": slope_grid, "aspect": aspect_grid, "rows": rows, "cols": cols}


def raster_hillshade(
    raster: RasterLike,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> dict[str, Any]:
    """Compute a basic hillshade grid from a raster surface."""
    import math

    terrain = raster_slope_aspect(raster)
    slope = terrain["slope"]
    aspect = terrain["aspect"]
    rows = terrain["rows"]
    cols = terrain["cols"]

    az_rad = math.radians(azimuth)
    alt_rad = math.radians(altitude)
    grid: list[list[float]] = [[0.0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            s = math.atan(slope[r][c]) if slope[r][c] is not None else 0.0
            a = math.radians(aspect[r][c]) if aspect[r][c] is not None else 0.0
            value = 255.0 * ((math.cos(alt_rad) * math.cos(s)) + (math.sin(alt_rad) * math.sin(s) * math.cos(az_rad - a)))
            grid[r][c] = max(0.0, min(255.0, value))

    return {"grid": grid, "rows": rows, "cols": cols}


def zonal_summary(
    raster: RasterLike,
    zones: GeoPromptFrame,
    *,
    zone_id_column: str = "zone_id",
) -> PromptTable:
    """Summarize raster values within polygon zones using cell-center sampling."""
    info = _coerce_raster(raster)
    data = info["data"]
    min_x, max_y, cell_width, cell_height = info["transform"]
    results: list[dict[str, Any]] = []

    for row in zones.to_records():
        geom = row[zones.geometry_column]
        values: list[float] = []
        for r_idx, raster_row in enumerate(data):
            for c_idx, val in enumerate(raster_row):
                center_x = float(min_x) + (c_idx + 0.5) * float(cell_width)
                center_y = float(max_y) - (r_idx + 0.5) * float(cell_height)
                pt = {"type": "Point", "coordinates": (center_x, center_y)}
                if geometry_within(pt, geom):
                    values.append(val)
        results.append({
            zone_id_column: row.get(zone_id_column),
            "count": len(values),
            "sum": sum(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": (sum(values) / len(values)) if values else None,
        })

    return PromptTable(results)


__all__ = [
    "inspect_raster",
    "raster_hillshade",
    "raster_slope_aspect",
    "sample_raster_points",
    "zonal_summary",
]
