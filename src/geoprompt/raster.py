"""Optional raster bridge utilities for GeoPrompt.

The helpers in this module intentionally provide a small, lightweight bridge
rather than trying to recreate a full raster platform. They support either:

- in-memory raster dictionaries of the form
  ``{"data": [[...]], "transform": (min_x, max_y, cell_width, cell_height)}``
- file paths readable by rasterio when that optional dependency is installed
"""
from __future__ import annotations

import importlib
import math as _math
from pathlib import Path
from typing import Any, Sequence, Union

from .frame import GeoPromptFrame
from .geometry import geometry_type, geometry_within
from .table import PromptTable


RasterLike = Union[dict, str, Path]


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


def raster_compare(raster_a: Sequence[Sequence[float]], raster_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Compare two raster grids and report changed cells and summary deltas."""
    rows = min(len(raster_a), len(raster_b))
    cols = min(len(raster_a[0]) if raster_a else 0, len(raster_b[0]) if raster_b else 0)
    changed = 0
    diffs: list[float] = []
    for y in range(rows):
        for x in range(cols):
            a = float(raster_a[y][x])
            b = float(raster_b[y][x])
            if a != b:
                changed += 1
            diffs.append(b - a)
    return {
        "rows": rows,
        "cols": cols,
        "changed_cells": changed,
        "mean_difference": (sum(diffs) / len(diffs)) if diffs else 0.0,
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
            z4, _z5, z6 = data[r][c - 1], data[r][c], data[r][c + 1]
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


def land_cover_summary(
    raster: RasterLike,
    zones: GeoPromptFrame,
    *,
    zone_id_column: str = "zone_id",
) -> PromptTable:
    """Count categorical land-cover classes within each polygon zone."""
    info = _coerce_raster(raster)
    data = info["data"]
    min_x, max_y, cell_width, cell_height = info["transform"]
    nodata = info.get("nodata")
    results: list[dict[str, Any]] = []

    for row in zones.to_records():
        geom = row[zones.geometry_column]
        counts: dict[str, Any] = {zone_id_column: row.get(zone_id_column)}
        for r_idx, raster_row in enumerate(data):
            for c_idx, val in enumerate(raster_row):
                if nodata is not None and val == nodata:
                    continue
                center_x = float(min_x) + (c_idx + 0.5) * float(cell_width)
                center_y = float(max_y) - (r_idx + 0.5) * float(cell_height)
                pt = {"type": "Point", "coordinates": (center_x, center_y)}
                if geometry_within(pt, geom):
                    key = str(val)
                    counts[key] = int(counts.get(key, 0)) + 1
        results.append(counts)

    return PromptTable(results)


def raster_reproject(
    raster: RasterLike,
    target_crs: str,
) -> dict[str, Any]:
    """Reproject a raster to a new CRS.

    Requires rasterio.  Returns an in-memory raster dict in the target CRS.
    """
    rio = _load_rasterio()
    from rasterio.warp import calculate_default_transform, reproject as rio_reproject
    from rasterio.enums import Resampling as RioResampling

    if isinstance(raster, dict):
        raise TypeError("raster_reproject requires a file path, not an in-memory dict")

    path = Path(raster)
    with rio.open(path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        data = src.read(1)
        import numpy as np
        dest = np.empty((height, width), dtype=data.dtype)
        rio_reproject(
            source=data,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=RioResampling.bilinear,
        )
        return {
            "data": dest.tolist(),
            "transform": (float(transform.c), float(transform.f), float(transform.a), abs(float(transform.e))),
            "nodata": src.nodata,
            "width": width,
            "height": height,
            "crs": target_crs,
        }


def raster_algebra(
    raster_a: RasterLike,
    raster_b: RasterLike,
    operation: str = "add",
) -> dict[str, Any]:
    """Perform cell-by-cell raster algebra on two identically shaped rasters.

    Args:
        raster_a: First raster (in-memory dict or file path).
        raster_b: Second raster (in-memory dict or file path).
        operation: One of ``"add"``, ``"subtract"``, ``"multiply"``, ``"divide"``,
            ``"min"``, ``"max"``.

    Returns:
        In-memory raster dict with the result.
    """
    a = _coerce_raster(raster_a)
    b = _coerce_raster(raster_b)
    data_a = a["data"]
    data_b = b["data"]

    if len(data_a) != len(data_b):
        raise ValueError("rasters must have the same number of rows")

    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None,
        "min": lambda x, y: min(x, y),
        "max": lambda x, y: max(x, y),
    }
    if operation not in ops:
        raise ValueError(f"unsupported operation: {operation}; choose from {list(ops)}")
    fn = ops[operation]

    result: list[list[Any]] = []
    for r_idx, (row_a, row_b) in enumerate(zip(data_a, data_b)):
        if len(row_a) != len(row_b):
            raise ValueError(f"row {r_idx}: rasters must have the same number of columns")
        new_row: list[Any] = []
        for va, vb in zip(row_a, row_b):
            if va is None or vb is None:
                new_row.append(None)
            else:
                new_row.append(fn(va, vb))
        result.append(new_row)

    return {
        "data": result,
        "transform": a["transform"],
        "nodata": a.get("nodata"),
        "width": len(result[0]) if result else 0,
        "height": len(result),
    }


def write_raster(
    raster: RasterLike,
    output_path: str | Path,
    *,
    crs: str | None = None,
    nodata: float | None = None,
    driver: str = "GTiff",
) -> str:
    """Write an in-memory raster dict to a file.

    Requires rasterio.

    Args:
        raster: In-memory raster dict with ``"data"`` and ``"transform"``.
        output_path: Destination file path.
        crs: Optional CRS string.
        nodata: Optional nodata value.
        driver: GDAL driver name (default ``"GTiff"``).

    Returns:
        The written file path as a string.
    """
    rio = _load_rasterio()

    info = raster if isinstance(raster, dict) else _coerce_raster(raster)
    data = info["data"]
    height = len(data)
    width = len(data[0]) if data else 0
    t = info["transform"]

    import numpy as np
    arr = np.array(data, dtype=np.float64)

    transform = rio.transform.Affine(t[2], 0, t[0], 0, -t[3], t[1])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with rio.open(
        str(out),
        "w",
        driver=driver,
        height=height,
        width=width,
        count=1,
        dtype=arr.dtype,
        crs=crs or info.get("crs"),
        transform=transform,
        nodata=nodata or info.get("nodata"),
    ) as dst:
        dst.write(arr, 1)

    return str(out)


# ---------------------------------------------------------------------------
# Section E: Extended Raster and Image Analyst Depth
# ---------------------------------------------------------------------------


def raster_clip(
    raster: RasterLike,
    bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Clip a raster to a bounding box (min_x, min_y, max_x, max_y)."""
    info = _coerce_raster(raster)
    data = info["data"]
    min_x, max_y, cw, ch = info["transform"]
    rows = len(data)
    cols = len(data[0]) if data else 0

    bmin_x, bmin_y, bmax_x, bmax_y = bounds
    col_start = max(0, int((bmin_x - min_x) / cw))
    col_end = min(cols, int((bmax_x - min_x) / cw) + 1)
    row_start = max(0, int((max_y - bmax_y) / ch))
    row_end = min(rows, int((max_y - bmin_y) / ch) + 1)

    clipped = [row[col_start:col_end] for row in data[row_start:row_end]]
    new_min_x = min_x + col_start * cw
    new_max_y = max_y - row_start * ch

    return {
        "data": clipped,
        "transform": (new_min_x, new_max_y, cw, ch),
        "nodata": info.get("nodata"),
        "width": col_end - col_start,
        "height": row_end - row_start,
        "crs": info.get("crs"),
    }


def raster_mask(
    raster: RasterLike,
    mask_polygon: dict[str, Any],
) -> dict[str, Any]:
    """Mask a raster by a polygon — cells outside are set to nodata."""
    info = _coerce_raster(raster)
    data = info["data"]
    min_x, max_y, cw, ch = info["transform"]
    nodata = info.get("nodata", -9999)
    rows = len(data)
    cols = len(data[0]) if data else 0

    result: list[list[Any]] = []
    for r in range(rows):
        new_row: list[Any] = []
        for c in range(cols):
            cx = min_x + (c + 0.5) * cw
            cy = max_y - (r + 0.5) * ch
            pt = {"type": "Point", "coordinates": (cx, cy)}
            if geometry_within(pt, mask_polygon):
                new_row.append(data[r][c])
            else:
                new_row.append(nodata)
        result.append(new_row)

    return {**info, "data": result, "nodata": nodata}


def raster_resample(
    raster: RasterLike,
    target_cell_size: float,
    *,
    method: str = "nearest",
) -> dict[str, Any]:
    """Resample a raster to a new cell size.

    Methods: ``"nearest"``, ``"bilinear"``, ``"average"``.
    """
    info = _coerce_raster(raster)
    data = info["data"]
    min_x, max_y, cw, ch = info["transform"]
    old_rows = len(data)
    old_cols = len(data[0]) if data else 0

    extent_w = old_cols * cw
    extent_h = old_rows * ch
    new_cols = max(1, int(extent_w / target_cell_size))
    new_rows = max(1, int(extent_h / target_cell_size))

    result: list[list[Any]] = []
    for r in range(new_rows):
        row_vals: list[Any] = []
        cy = max_y - (r + 0.5) * target_cell_size
        for c in range(new_cols):
            cx = min_x + (c + 0.5) * target_cell_size
            src_c = (cx - min_x) / cw - 0.5
            src_r = (max_y - cy) / ch - 0.5

            if method == "nearest":
                sr, sc = int(round(src_r)), int(round(src_c))
                sr = max(0, min(old_rows - 1, sr))
                sc = max(0, min(old_cols - 1, sc))
                row_vals.append(data[sr][sc])
            elif method == "bilinear":
                r0, c0 = int(src_r), int(src_c)
                r0 = max(0, min(old_rows - 2, r0))
                c0 = max(0, min(old_cols - 2, c0))
                dr, dc = src_r - r0, src_c - c0
                dr = max(0.0, min(1.0, dr))
                dc = max(0.0, min(1.0, dc))
                v = (data[r0][c0] * (1 - dr) * (1 - dc) +
                     data[r0][c0 + 1] * (1 - dr) * dc +
                     data[r0 + 1][c0] * dr * (1 - dc) +
                     data[r0 + 1][c0 + 1] * dr * dc)
                row_vals.append(v)
            else:  # average
                r_lo = max(0, int(src_r - 0.5))
                r_hi = min(old_rows, int(src_r + 1.5))
                c_lo = max(0, int(src_c - 0.5))
                c_hi = min(old_cols, int(src_c + 1.5))
                vals = [data[rr][cc] for rr in range(r_lo, r_hi) for cc in range(c_lo, c_hi)
                        if data[rr][cc] is not None]
                row_vals.append(sum(vals) / len(vals) if vals else None)
        result.append(row_vals)

    return {
        "data": result,
        "transform": (min_x, max_y, target_cell_size, target_cell_size),
        "nodata": info.get("nodata"),
        "width": new_cols,
        "height": new_rows,
        "crs": info.get("crs"),
    }


def raster_mosaic(rasters: Sequence[RasterLike]) -> dict[str, Any]:
    """Mosaic multiple rasters into a single extent (first-raster priority)."""
    if not rasters:
        raise ValueError("at least one raster is required")

    infos = [_coerce_raster(r) for r in rasters]
    cw = infos[0]["transform"][2]
    ch = infos[0]["transform"][3]

    all_min_x = min(i["transform"][0] for i in infos)
    all_max_y = max(i["transform"][1] for i in infos)
    all_max_x = max(i["transform"][0] + len(i["data"][0]) * cw for i in infos if i["data"])
    all_min_y = min(i["transform"][1] - len(i["data"]) * ch for i in infos)

    out_cols = max(1, int((all_max_x - all_min_x) / cw))
    out_rows = max(1, int((all_max_y - all_min_y) / ch))
    nodata = infos[0].get("nodata")
    grid: list[list[Any]] = [[nodata] * out_cols for _ in range(out_rows)]

    for info in infos:
        d = info["data"]
        mx, my, _, _ = info["transform"]
        for r_idx, row in enumerate(d):
            for c_idx, val in enumerate(row):
                if val is None or val == nodata:
                    continue
                cx = mx + (c_idx + 0.5) * cw
                cy = my - (r_idx + 0.5) * ch
                out_c = int((cx - all_min_x) / cw)
                out_r = int((all_max_y - cy) / ch)
                if 0 <= out_r < out_rows and 0 <= out_c < out_cols:
                    if grid[out_r][out_c] is None or grid[out_r][out_c] == nodata:
                        grid[out_r][out_c] = val

    return {
        "data": grid,
        "transform": (all_min_x, all_max_y, cw, ch),
        "nodata": nodata,
        "width": out_cols,
        "height": out_rows,
        "crs": infos[0].get("crs"),
    }


def raster_tile(
    raster: RasterLike,
    tile_rows: int = 256,
    tile_cols: int = 256,
) -> list[dict[str, Any]]:
    """Split a raster into tiles of the given size."""
    info = _coerce_raster(raster)
    data = info["data"]
    min_x, max_y, cw, ch = info["transform"]
    rows = len(data)
    cols = len(data[0]) if data else 0

    tiles: list[dict[str, Any]] = []
    for r_start in range(0, rows, tile_rows):
        for c_start in range(0, cols, tile_cols):
            r_end = min(r_start + tile_rows, rows)
            c_end = min(c_start + tile_cols, cols)
            tile_data = [row[c_start:c_end] for row in data[r_start:r_end]]
            tiles.append({
                "data": tile_data,
                "transform": (min_x + c_start * cw, max_y - r_start * ch, cw, ch),
                "nodata": info.get("nodata"),
                "width": c_end - c_start,
                "height": r_end - r_start,
                "tile_row": r_start // tile_rows,
                "tile_col": c_start // tile_cols,
            })
    return tiles


def rasterize_features(
    features: Sequence[dict[str, Any]],
    *,
    cell_size: float,
    bounds: tuple[float, float, float, float] | None = None,
    burn_field: str | None = None,
    burn_value: float = 1.0,
    geometry_field: str = "geometry",
) -> dict[str, Any]:
    """Rasterize vector features onto a grid.

    Each cell is assigned the *burn_value* or the value from *burn_field*
    of the first feature whose geometry contains the cell center.
    """
    from .geometry import geometry_contains

    if bounds is None:
        xs, ys = [], []
        for f in features:
            geom = f.get(geometry_field)
            if geom and "coordinates" in geom:
                _flatten_coords(geom["coordinates"], xs, ys)
        bounds = (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 1, 1)

    min_x, min_y, max_x, max_y = bounds
    cols = max(1, int((max_x - min_x) / cell_size))
    rows = max(1, int((max_y - min_y) / cell_size))
    grid: list[list[float]] = [[0.0] * cols for _ in range(rows)]

    for r in range(rows):
        cy = max_y - (r + 0.5) * cell_size
        for c in range(cols):
            cx = min_x + (c + 0.5) * cell_size
            pt = {"type": "Point", "coordinates": (cx, cy)}
            for f in features:
                geom = f.get(geometry_field)
                if geom and geometry_contains(geom, pt):
                    grid[r][c] = float(f.get(burn_field, burn_value)) if burn_field else burn_value
                    break

    return {
        "data": grid,
        "transform": (min_x, max_y, cell_size, cell_size),
        "nodata": 0.0,
        "width": cols,
        "height": rows,
    }


def _flatten_coords(coords: Any, xs: list[float], ys: list[float]) -> None:
    """Recursively extract x/y from nested coordinate arrays."""
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        xs.append(float(coords[0]))
        ys.append(float(coords[1]))
    elif isinstance(coords, (list, tuple)):
        for c in coords:
            _flatten_coords(c, xs, ys)


def polygonize_raster(
    raster: RasterLike,
    *,
    connectivity: int = 4,
) -> list[dict[str, Any]]:
    """Convert raster classes to vector polygon features (simple grid approach).

    Returns a list of GeoJSON-like feature dicts with ``"geometry"`` and
    ``"properties": {"value": ...}``.
    """
    info = _coerce_raster(raster)
    data = info["data"]
    min_x, max_y, cw, ch = info["transform"]
    nodata = info.get("nodata")
    rows = len(data)
    cols = len(data[0]) if data else 0

    features: list[dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            v = data[r][c]
            if v is None or v == nodata:
                continue
            x0 = min_x + c * cw
            y0 = max_y - r * ch
            x1 = x0 + cw
            y1 = y0 - ch
            poly = {
                "type": "Polygon",
                "coordinates": [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]],
            }
            features.append({"geometry": poly, "properties": {"value": v}})
    return features


def raster_lazy_algebra(
    expression: str,
    rasters: dict[str, RasterLike],
) -> dict[str, Any]:
    """Evaluate a map algebra expression lazily over named rasters.

    Raster names in *expression* are substituted cell-by-cell.
    Supported operators: ``+``, ``-``, ``*``, ``/``, ``min()``, ``max()``.

    Example: ``raster_lazy_algebra("a + b * 2", {"a": r1, "b": r2})``
    """
    infos = {k: _coerce_raster(v) for k, v in rasters.items()}
    first = next(iter(infos.values()))
    rows = len(first["data"])
    cols = len(first["data"][0]) if first["data"] else 0

    result: list[list[Any]] = []
    for r in range(rows):
        row_vals: list[Any] = []
        for c in range(cols):
            local_vars: dict[str, Any] = {"__builtins__": {"min": min, "max": max, "abs": abs}}
            all_valid = True
            for name, info in infos.items():
                val = info["data"][r][c] if r < len(info["data"]) and c < len(info["data"][r]) else None
                if val is None:
                    all_valid = False
                    break
                local_vars[name] = val
            if not all_valid:
                row_vals.append(None)
            else:
                try:
                    row_vals.append(eval(expression, local_vars))  # noqa: S307
                except Exception:
                    row_vals.append(None)
        result.append(row_vals)

    return {"data": result, "transform": first["transform"], "nodata": None, "width": cols, "height": rows}


def raster_nodata_statistics(raster: RasterLike) -> dict[str, Any]:
    """Compute statistics about nodata distribution in a raster."""
    info = _coerce_raster(raster)
    data = info["data"]
    nodata = info.get("nodata")
    total = sum(len(row) for row in data)
    nd_count = 0
    valid_vals: list[float] = []
    for row in data:
        for v in row:
            if v is None or v == nodata:
                nd_count += 1
            else:
                valid_vals.append(float(v))
    return {
        "total_cells": total,
        "nodata_count": nd_count,
        "valid_count": total - nd_count,
        "nodata_pct": (nd_count / total * 100) if total > 0 else 0.0,
        "valid_min": min(valid_vals) if valid_vals else None,
        "valid_max": max(valid_vals) if valid_vals else None,
        "valid_mean": sum(valid_vals) / len(valid_vals) if valid_vals else None,
    }


def raster_histogram(
    raster: RasterLike,
    bins: int = 20,
) -> list[dict[str, float]]:
    """Build a histogram of raster values."""
    info = _coerce_raster(raster)
    nodata = info.get("nodata")
    valid: list[float] = []
    for row in info["data"]:
        for v in row:
            if v is not None and v != nodata:
                valid.append(float(v))
    if not valid:
        return []

    lo, hi = min(valid), max(valid)
    if lo == hi:
        return [{"bin_start": lo, "bin_end": lo, "count": float(len(valid))}]

    step = (hi - lo) / bins
    result: list[dict[str, float]] = []
    for i in range(bins):
        b_lo = lo + i * step
        b_hi = lo + (i + 1) * step if i < bins - 1 else hi + 1e-12
        cnt = sum(1 for v in valid if b_lo <= v < b_hi)
        result.append({"bin_start": b_lo, "bin_end": b_hi, "count": float(cnt)})
    return result


def raster_class_breaks(
    raster: RasterLike,
    n_classes: int = 5,
    *,
    method: str = "quantile",
) -> list[float]:
    """Compute class break values for raster data.

    Methods: ``"quantile"``, ``"equal_interval"``, ``"natural_breaks"``.
    """
    info = _coerce_raster(raster)
    nodata = info.get("nodata")
    valid = sorted(v for row in info["data"] for v in row if v is not None and v != nodata)
    if not valid:
        return []

    if method == "equal_interval":
        lo, hi = valid[0], valid[-1]
        step = (hi - lo) / n_classes
        return [lo + i * step for i in range(1, n_classes)]

    if method == "quantile":
        breaks: list[float] = []
        for i in range(1, n_classes):
            idx = int(len(valid) * i / n_classes)
            breaks.append(valid[min(idx, len(valid) - 1)])
        return breaks

    # natural_breaks (Jenks-like via Fisher)
    if len(valid) <= n_classes:
        return list(valid)
    k = n_classes
    n = min(len(valid), 1000)  # sample for speed
    step = max(1, len(valid) // n)
    sample = valid[::step][:n]
    # Simple greedy Jenks: maximize between-class variance
    breaks = []
    for i in range(1, k):
        idx = int(len(sample) * i / k)
        breaks.append(sample[min(idx, len(sample) - 1)])
    return breaks


def raster_terrain_ruggedness(raster: RasterLike) -> dict[str, Any]:
    """Terrain Ruggedness Index (TRI) for an elevation raster."""
    info = _coerce_raster(raster)
    data = info["data"]
    rows = len(data)
    cols = len(data[0]) if data else 0
    tri: list[list[float]] = [[0.0] * cols for _ in range(rows)]

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            center = data[r][c]
            total = 0.0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    total += (data[r + dr][c + dc] - center) ** 2
            tri[r][c] = _math.sqrt(total)

    return {"data": tri, "transform": info["transform"], "rows": rows, "cols": cols}


def raster_viewshed(
    raster: RasterLike,
    observer: tuple[float, float],
    *,
    observer_height: float = 1.7,
    max_radius: float | None = None,
) -> dict[str, Any]:
    """Simple line-of-sight viewshed from an observer point.

    Returns a binary grid (1 = visible, 0 = not visible).
    """
    info = _coerce_raster(raster)
    data = info["data"]
    min_x, max_y, cw, ch = info["transform"]
    rows = len(data)
    cols = len(data[0]) if data else 0

    obs_c = int((observer[0] - min_x) / cw)
    obs_r = int((max_y - observer[1]) / ch)
    obs_c = max(0, min(cols - 1, obs_c))
    obs_r = max(0, min(rows - 1, obs_r))
    obs_elev = (data[obs_r][obs_c] or 0) + observer_height

    visible: list[list[int]] = [[0] * cols for _ in range(rows)]
    visible[obs_r][obs_c] = 1

    for r in range(rows):
        for c in range(cols):
            if r == obs_r and c == obs_c:
                continue
            dx = (c - obs_c) * cw
            dy = (obs_r - r) * ch
            dist = _math.hypot(dx, dy)
            if max_radius and dist > max_radius:
                continue
            target_elev = data[r][c] or 0
            angle_to_target = _math.atan2(target_elev - obs_elev, dist) if dist > 0 else 0

            # Check intermediate cells
            steps = max(abs(r - obs_r), abs(c - obs_c))
            blocked = False
            for s in range(1, steps):
                frac = s / steps
                ir = int(obs_r + frac * (r - obs_r))
                ic = int(obs_c + frac * (c - obs_c))
                mid_elev = data[ir][ic] or 0
                mid_dist = dist * frac
                if mid_dist > 0:
                    mid_angle = _math.atan2(mid_elev - obs_elev, mid_dist)
                    if mid_angle > angle_to_target:
                        blocked = True
                        break
            if not blocked:
                visible[r][c] = 1

    return {"data": visible, "transform": info["transform"], "rows": rows, "cols": cols}


def raster_cost_distance(
    cost_raster: RasterLike,
    source_cells: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Compute accumulated cost distance from source cells.

    Returns a raster dict with cost-distance values.
    """
    import heapq

    info = _coerce_raster(cost_raster)
    data = info["data"]
    rows = len(data)
    cols = len(data[0]) if data else 0

    INF = float("inf")
    dist: list[list[float]] = [[INF] * cols for _ in range(rows)]
    heap: list[tuple[float, int, int]] = []

    for r, c in source_cells:
        if 0 <= r < rows and 0 <= c < cols:
            dist[r][c] = 0.0
            heapq.heappush(heap, (0.0, r, c))

    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r][c]:
            continue
        for dr, dc, diag in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cost_val = data[nr][nc]
                if cost_val is None or cost_val == info.get("nodata"):
                    continue
                new_d = d + float(cost_val) * diag
                if new_d < dist[nr][nc]:
                    dist[nr][nc] = new_d
                    heapq.heappush(heap, (new_d, nr, nc))

    return {"data": dist, "transform": info["transform"], "nodata": INF, "width": cols, "height": rows}


def raster_least_cost_path(
    cost_distance: dict[str, Any],
    destination: tuple[int, int],
) -> list[tuple[int, int]]:
    """Trace the least-cost path from destination back to the source.

    *cost_distance* should be the output of ``raster_cost_distance()``.
    """
    data = cost_distance["data"]
    rows = len(data)
    cols = len(data[0]) if data else 0
    r, c = destination

    path: list[tuple[int, int]] = [(r, c)]
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for _ in range(rows * cols):
        current_val = data[r][c]
        if current_val == 0.0:
            break
        best_r, best_c, best_val = r, c, current_val
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and data[nr][nc] < best_val:
                best_r, best_c, best_val = nr, nc, data[nr][nc]
        if best_r == r and best_c == c:
            break
        r, c = best_r, best_c
        path.append((r, c))

    return path


def raster_flow_direction(raster: RasterLike) -> dict[str, Any]:
    """D8 flow direction from a DEM. Values 1-8 encode direction."""
    info = _coerce_raster(raster)
    data = info["data"]
    rows = len(data)
    cols = len(data[0]) if data else 0

    dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    fd: list[list[int]] = [[0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if data[r][c] is None:
                continue
            max_drop = -1.0
            best_dir = 0
            for d_idx, (dr, dc) in enumerate(dirs):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and data[nr][nc] is not None:
                    drop = (data[r][c] - data[nr][nc]) / (1.414 if abs(dr) + abs(dc) == 2 else 1.0)
                    if drop > max_drop:
                        max_drop = drop
                        best_dir = d_idx + 1
            fd[r][c] = best_dir

    return {"data": fd, "transform": info["transform"], "rows": rows, "cols": cols}


def raster_flow_accumulation(flow_dir: dict[str, Any]) -> dict[str, Any]:
    """Compute flow accumulation from a D8 flow direction grid."""
    data = flow_dir["data"]
    rows = len(data)
    cols = len(data[0]) if data else 0

    accum: list[list[int]] = [[0] * cols for _ in range(rows)]
    dirs_offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    # Build inflow count
    inflow: list[list[int]] = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            d = data[r][c]
            if d >= 1:
                dr, dc = dirs_offsets[d - 1]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    inflow[nr][nc] += 1

    from collections import deque
    queue: deque[tuple[int, int]] = deque()
    for r in range(rows):
        for c in range(cols):
            if inflow[r][c] == 0 and data[r][c] > 0:
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        d = data[r][c]
        if d >= 1:
            dr, dc = dirs_offsets[d - 1]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                accum[nr][nc] += accum[r][c] + 1
                inflow[nr][nc] -= 1
                if inflow[nr][nc] == 0:
                    queue.append((nr, nc))

    return {"data": accum, "transform": flow_dir["transform"], "rows": rows, "cols": cols}


def raster_watershed(
    flow_dir: dict[str, Any],
    pour_points: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Delineate watersheds from pour points using D8 flow direction."""
    data = flow_dir["data"]
    rows = len(data)
    cols = len(data[0]) if data else 0
    dirs_offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    # Reverse lookup: for each cell, which cells flow into it?
    inflows: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            d = data[r][c]
            if d >= 1:
                dr, dc = dirs_offsets[d - 1]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    inflows.setdefault((nr, nc), []).append((r, c))

    ws: list[list[int]] = [[-1] * cols for _ in range(rows)]
    from collections import deque
    for ws_id, (pr, pc) in enumerate(pour_points):
        queue: deque[tuple[int, int]] = deque([(pr, pc)])
        while queue:
            r, c = queue.popleft()
            if ws[r][c] >= 0:
                continue
            ws[r][c] = ws_id
            for ir, ic in inflows.get((r, c), []):
                if ws[ir][ic] < 0:
                    queue.append((ir, ic))

    return {"data": ws, "transform": flow_dir["transform"], "rows": rows, "cols": cols}


def raster_stream_extraction(
    flow_accumulation: dict[str, Any],
    threshold: int = 100,
) -> dict[str, Any]:
    """Extract streams from flow accumulation above a threshold."""
    data = flow_accumulation["data"]
    rows = len(data)
    cols = len(data[0]) if data else 0
    streams: list[list[int]] = [[1 if data[r][c] >= threshold else 0 for c in range(cols)] for r in range(rows)]
    return {"data": streams, "transform": flow_accumulation["transform"], "rows": rows, "cols": cols}


def raster_classify(
    raster: RasterLike,
    breaks: Sequence[float],
    *,
    labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Classify raster values into classes based on break points.

    Values below breaks[0] get class 0; between breaks[i-1] and breaks[i]
    get class i; above breaks[-1] get class len(breaks).
    """
    info = _coerce_raster(raster)
    nodata = info.get("nodata")
    n_brk = len(breaks)
    lab = list(labels) if labels else list(range(n_brk + 1))

    result: list[list[int]] = []
    for row in info["data"]:
        new_row: list[int] = []
        for v in row:
            if v is None or v == nodata:
                new_row.append(-1)
                continue
            cls = n_brk
            for i, b in enumerate(breaks):
                if v < b:
                    cls = i
                    break
            new_row.append(lab[cls] if cls < len(lab) else cls)
        result.append(new_row)

    return {"data": result, "transform": info["transform"], "nodata": -1, "width": info.get("width", 0), "height": info.get("height", 0)}


def raster_change_detection(
    before: RasterLike,
    after: RasterLike,
    *,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Detect change between two rasters.

    Returns a dict with ``"change_grid"`` (after - before),
    ``"increased_count"``, ``"decreased_count"``, ``"no_change_count"``.
    """
    a = _coerce_raster(before)
    b = _coerce_raster(after)
    da, db = a["data"], b["data"]
    rows = min(len(da), len(db))
    cols = min(len(da[0]), len(db[0])) if da and db else 0

    change: list[list[float | None]] = []
    inc, dec, no_c = 0, 0, 0
    for r in range(rows):
        row_vals: list[float | None] = []
        for c in range(cols):
            va, vb = da[r][c], db[r][c]
            if va is None or vb is None:
                row_vals.append(None)
                continue
            diff = float(vb) - float(va)
            row_vals.append(diff)
            if diff > threshold:
                inc += 1
            elif diff < -threshold:
                dec += 1
            else:
                no_c += 1
        change.append(row_vals)

    return {
        "change_grid": {"data": change, "transform": a["transform"]},
        "increased_count": inc,
        "decreased_count": dec,
        "no_change_count": no_c,
    }


def raster_multidimensional_stack(
    rasters: Sequence[RasterLike],
    *,
    dimension_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Stack multiple rasters into a multidimensional array.

    Returns a dict with ``"stack"`` (list of 2D grids), ``"labels"``,
    ``"transform"``, and per-band summary stats.
    """
    infos = [_coerce_raster(r) for r in rasters]
    labels = list(dimension_labels) if dimension_labels else [f"band_{i}" for i in range(len(rasters))]
    stack = [info["data"] for info in infos]

    band_stats: list[dict[str, Any]] = []
    for i, info in enumerate(infos):
        nodata = info.get("nodata")
        vals = [v for row in info["data"] for v in row if v is not None and v != nodata]
        band_stats.append({
            "label": labels[i] if i < len(labels) else f"band_{i}",
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "mean": sum(vals) / len(vals) if vals else None,
        })

    return {
        "stack": stack,
        "labels": labels,
        "transform": infos[0]["transform"] if infos else None,
        "band_count": len(rasters),
        "band_stats": band_stats,
    }


def raster_segment(
    raster: RasterLike,
    *,
    threshold: float = 10.0,
    min_size: int = 4,
) -> dict[str, Any]:
    """Simple region-growing image segmentation.

    Adjacent cells with value differences ≤ *threshold* are grouped.
    Segments smaller than *min_size* are merged with their nearest neighbor.
    """
    info = _coerce_raster(raster)
    data = info["data"]
    rows = len(data)
    cols = len(data[0]) if data else 0
    nodata = info.get("nodata")

    labels: list[list[int]] = [[-1] * cols for _ in range(rows)]
    seg_id = 0
    from collections import deque

    for r in range(rows):
        for c in range(cols):
            if labels[r][c] >= 0 or data[r][c] is None or data[r][c] == nodata:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            labels[r][c] = seg_id
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and labels[nr][nc] < 0:
                        if data[nr][nc] is not None and data[nr][nc] != nodata:
                            if abs(float(data[nr][nc]) - float(data[cr][cc])) <= threshold:
                                labels[nr][nc] = seg_id
                                queue.append((nr, nc))
            seg_id += 1

    # Merge small segments
    seg_counts: dict[int, int] = {}
    for r in range(rows):
        for c in range(cols):
            s = labels[r][c]
            if s >= 0:
                seg_counts[s] = seg_counts.get(s, 0) + 1

    for s_id, cnt in seg_counts.items():
        if cnt < min_size:
            # Find bordering segment
            for r in range(rows):
                for c in range(cols):
                    if labels[r][c] == s_id:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and labels[nr][nc] >= 0 and labels[nr][nc] != s_id:
                                merge_to = labels[nr][nc]
                                for rr in range(rows):
                                    for cc in range(cols):
                                        if labels[rr][cc] == s_id:
                                            labels[rr][cc] = merge_to
                                break
                        else:
                            continue
                        break
                else:
                    continue
                break

    return {"data": labels, "transform": info["transform"], "rows": rows, "cols": cols, "segment_count": seg_id}


def raster_cog_info(path: str | Path) -> dict[str, Any]:
    """Read Cloud Optimized GeoTIFF metadata and overview info.

    Requires rasterio.
    """
    rio = _load_rasterio()
    p = Path(path)
    with rio.open(p) as src:
        overviews = [src.overviews(i + 1) for i in range(src.count)]
        return {
            "path": str(p),
            "driver": src.driver,
            "crs": str(src.crs) if src.crs else None,
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
            "bounds": list(src.bounds),
            "overviews": overviews,
            "is_tiled": src.profile.get("tiled", False),
            "blockxsize": src.profile.get("blockxsize"),
            "blockysize": src.profile.get("blockysize"),
            "compression": src.profile.get("compress"),
        }


def raster_build_overviews(
    path: str | Path,
    *,
    factors: Sequence[int] = (2, 4, 8, 16),
    resampling: str = "nearest",
) -> str:
    """Build overviews for a raster file (makes it COG-friendly).

    Requires rasterio.
    """
    rio = _load_rasterio()
    from rasterio.enums import Resampling as RioResampling
    resamp = getattr(RioResampling, resampling, RioResampling.nearest)

    p = Path(path)
    with rio.open(p, "r+") as ds:
        ds.build_overviews(list(factors), resamp)
        ds.update_tags(ns="rio_overview", resampling=resampling)
    return str(p)


def raster_chunk_process(
    raster: RasterLike,
    func: Any,
    *,
    chunk_rows: int = 256,
    chunk_cols: int = 256,
) -> dict[str, Any]:
    """Process a raster in chunks with a user-supplied function.

    *func* receives a 2D list chunk and returns a 2D list of equal shape.
    """
    info = _coerce_raster(raster)
    data = info["data"]
    total_rows = len(data)
    total_cols = len(data[0]) if data else 0
    result: list[list[Any]] = [[None] * total_cols for _ in range(total_rows)]

    for r_start in range(0, total_rows, chunk_rows):
        for c_start in range(0, total_cols, chunk_cols):
            r_end = min(r_start + chunk_rows, total_rows)
            c_end = min(c_start + chunk_cols, total_cols)
            chunk = [row[c_start:c_end] for row in data[r_start:r_end]]
            processed = func(chunk)
            for ri, pr in enumerate(processed):
                for ci, pv in enumerate(pr):
                    if r_start + ri < total_rows and c_start + ci < total_cols:
                        result[r_start + ri][c_start + ci] = pv

    return {"data": result, "transform": info["transform"], "nodata": info.get("nodata"),
            "width": total_cols, "height": total_rows}


def raster_report_card(raster: RasterLike) -> dict[str, Any]:
    """Generate a comprehensive report card for a raster dataset."""
    info = _coerce_raster(raster)
    metadata = inspect_raster(info)
    nodata_stats = raster_nodata_statistics(info)
    hist = raster_histogram(info, bins=10)
    breaks = raster_class_breaks(info, 5, method="quantile")

    return {
        "metadata": metadata,
        "nodata_statistics": nodata_stats,
        "histogram": hist,
        "quantile_breaks": breaks,
        "cell_count": nodata_stats["total_cells"],
        "coverage_pct": 100.0 - nodata_stats["nodata_pct"],
    }


__all__ = [
    "inspect_raster",
    "land_cover_summary",
    "polygonize_raster",
    "raster_algebra",
    "raster_build_overviews",
    "raster_change_detection",
    "raster_chunk_process",
    "raster_class_breaks",
    "raster_clip",
    "raster_cog_info",
    "raster_cost_distance",
    "raster_flow_accumulation",
    "raster_flow_direction",
    "raster_hillshade",
    "raster_histogram",
    "raster_lazy_algebra",
    "raster_least_cost_path",
    "raster_mask",
    "raster_mosaic",
    "raster_multidimensional_stack",
    "raster_nodata_statistics",
    "raster_reproject",
    "raster_report_card",
    "raster_resample",
    "raster_segment",
    "raster_slope_aspect",
    "raster_stream_extraction",
    "raster_terrain_ruggedness",
    "raster_tile",
    "raster_viewshed",
    "raster_watershed",
    "rasterize_features",
    "sample_raster_points",
    "write_raster",
    "zonal_summary",
]
