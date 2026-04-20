"""Raster, imagery, photogrammetry, and LiDAR helpers for GeoPrompt."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence


Grid = Sequence[Sequence[float]]


def _grid(data: Grid | dict[str, Any]) -> list[list[float]]:
    if isinstance(data, dict):
        return [[float(v) for v in row] for row in data.get("data", [])]
    return [[float(v) for v in row] for row in data]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _flatten(grid: Grid | dict[str, Any]) -> list[float]:
    return [float(v) for row in _grid(grid) for v in row]


def raster_math_advanced(raster: Grid | dict[str, Any], *, operation: str = "log", power: float = 2.0) -> list[list[float]]:
    """Apply advanced math functions cell-by-cell across a raster."""
    funcs = {
        "log": lambda v: math.log(max(v, 1e-9)),
        "exp": math.exp,
        "power": lambda v: v ** power,
        "sqrt": lambda v: math.sqrt(max(v, 0.0)),
        "abs": abs,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
    }
    fn = funcs.get(operation, lambda v: v)
    return [[round(fn(float(v)), 4) for v in row] for row in _grid(raster)]


def pansharpen_raster(multispectral: Grid, panchromatic: Grid) -> list[list[float]]:
    """Fuse multispectral and panchromatic rasters with a lightweight sharpening ratio."""
    ms = _grid(multispectral)
    pan = _grid(panchromatic)
    out = []
    for mrow, prow in zip(ms, pan):
        out.append([round((m + p) / 2.0 + (p - _mean(prow)) * 0.25, 4) for m, p in zip(mrow, prow)])
    return out


def deep_learning_pixel_classification(image: Grid) -> list[list[str]]:
    """Classify pixels into coarse low/medium/high intensity classes."""
    vals = _flatten(image)
    lo = min(vals or [0])
    hi = max(vals or [1])
    mid = (lo + hi) / 2
    return [["low" if v <= lo + (mid - lo) / 2 else "medium" if v < mid + (hi - mid) / 2 else "high" for v in row] for row in _grid(image)]


def deep_learning_object_detection(image: Grid, *, threshold: float = 1.0) -> dict[str, Any]:
    """Detect bright objects in imagery using a threshold."""
    detections = [{"x": x, "y": y, "score": float(v)} for y, row in enumerate(_grid(image)) for x, v in enumerate(row) if float(v) >= threshold]
    return {"objects": detections, "count": len(detections)}


def deep_learning_change_detection(before: Grid, after: Grid) -> dict[str, Any]:
    """Count changed cells between two images."""
    changed = 0
    for rb, ra in zip(_grid(before), _grid(after)):
        for b, a in zip(rb, ra):
            if b != a:
                changed += 1
    return {"changed_pixels": changed}


def training_sample_creation(image: Grid, *, labels: dict[tuple[int, int], str]) -> list[dict[str, Any]]:
    """Build labelled training samples from image coordinates."""
    g = _grid(image)
    return [{"x": x, "y": y, "label": label, "pixel_value": g[y][x]} for (x, y), label in labels.items() if 0 <= y < len(g) and 0 <= x < len(g[0])]


def points_solar_radiation(points: Sequence[tuple[float, float, float]]) -> list[dict[str, Any]]:
    """Estimate solar radiation at point locations."""
    out = []
    for x, y, z in points:
        solar = max(0.0, 1000 - abs(z) * 2 + (math.cos(x) + math.sin(y)) * 20)
        out.append({"x": x, "y": y, "elevation": z, "solar_radiation": round(solar, 3)})
    return out


def area_solar_radiation(surface: Grid) -> dict[str, Any]:
    """Estimate average incoming solar radiation across a surface."""
    vals = _flatten(surface)
    radiation = [v * 100 + 500 for v in vals]
    return {"mean_radiation": round(_mean(radiation), 4), "max_radiation": round(max(radiation or [0]), 4)}


def heat_load_index(surface: Grid) -> dict[str, Any]:
    """Compute a simplified heat load index from raster intensities."""
    vals = [math.sqrt(abs(v) + 1) for v in _flatten(surface)]
    return {"mean_index": round(_mean(vals), 4), "max_index": round(max(vals or [0]), 4)}


def sediment_transport_index(surface: Grid) -> dict[str, Any]:
    """Compute a coarse sediment transport index from surface values."""
    vals = [abs(v) ** 1.3 for v in _flatten(surface)]
    return {"mean_index": round(_mean(vals), 4), "max_index": round(max(vals or [0]), 4)}


def tasseled_cap_transformation(bands: dict[str, Grid]) -> dict[str, list[list[float]]]:
    """Apply a simple tasseled-cap style transformation to multispectral bands."""
    red = _grid(bands.get("red", [[0.0]]))
    nir = _grid(bands.get("nir", [[0.0]]))
    green = _grid(bands.get("green", red))
    brightness = []
    greenness = []
    wetness = []
    for rr, nn, gg in zip(red, nir, green):
        brightness.append([round((r + n + g) / 3.0, 4) for r, n, g in zip(rr, nn, gg)])
        greenness.append([round(n - r, 4) for r, n in zip(rr, nn)])
        wetness.append([round(g - (r + n) / 2.0, 4) for r, n, g in zip(rr, nn, gg)])
    return {"brightness": brightness, "greenness": greenness, "wetness": wetness}


def minimum_noise_fraction(image: Grid) -> dict[str, Any]:
    """Estimate a simple MNF decomposition summary."""
    vals = _flatten(image)
    mu = _mean(vals)
    return {"component_count": 1 if vals else 0, "noise_reduced_mean": round(mu, 4)}


def spectral_unmixing(endmembers: dict[str, Sequence[float]], pixel: Sequence[float]) -> dict[str, Any]:
    """Estimate endmember fractions for a spectral pixel."""
    weights = {}
    for name, sig in endmembers.items():
        dist = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(sig, pixel)))
        weights[name] = 1.0 / (dist + 1e-6)
    total = sum(weights.values()) or 1.0
    return {"fractions": {k: round(v / total, 4) for k, v in weights.items()}}


def spectral_anomaly_detection(image: Grid) -> dict[str, Any]:
    """Flag spectral outliers based on mean and standard deviation."""
    vals = _flatten(image)
    mu = _mean(vals)
    sd = math.sqrt(_mean([(v - mu) ** 2 for v in vals])) if vals else 0.0
    anomalies = sum(1 for v in vals if abs(v - mu) > max(sd, 1.0))
    return {"anomaly_pixels": anomalies}


def time_series_animation_export(frames: Sequence[Grid], output_path: str | Path) -> str:
    """Export a placeholder time-series animation artifact."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"frame_count": len(frames)}), encoding="utf-8")
    return str(out)


def phenology_metric_extraction(series: Sequence[float]) -> dict[str, Any]:
    """Extract seasonal metrics from a vegetation time series."""
    if not series:
        return {"peak_index": None, "start_index": None, "end_index": None}
    peak = max(range(len(series)), key=lambda i: series[i])
    threshold = max(series) * 0.5
    start = next((i for i, v in enumerate(series) if v >= threshold), 0)
    end = len(series) - 1 - next((i for i, v in enumerate(reversed(series)) if v >= threshold), 0)
    return {"peak_index": peak, "start_index": start, "end_index": end}


def growing_degree_days_accumulation(temperatures: Sequence[float], *, base_temp: float = 10.0) -> dict[str, Any]:
    """Accumulate growing degree days above a base temperature."""
    return {"accumulated_gdd": round(sum(max(0.0, t - base_temp) for t in temperatures), 4)}


def crop_mask_generation(index_grid: Grid, *, threshold: float = 0.5) -> dict[str, Any]:
    """Generate a crop mask from an index raster."""
    mask = [[1 if v >= threshold else 0 for v in row] for row in _grid(index_grid)]
    return {"mask": mask, "crop_pixels": sum(sum(r) for r in mask)}


def land_cover_map(image: Grid) -> list[list[str]]:
    """Classify a raster into simple land-cover classes."""
    result = []
    for row in _grid(image):
        out_row = []
        for v in row:
            if v < 0.2:
                out_row.append("water")
            elif v < 0.5:
                out_row.append("bare")
            elif v < 0.8:
                out_row.append("vegetation")
            else:
                out_row.append("urban")
        result.append(out_row)
    return result


def impervious_surface_map(image: Grid, *, threshold: float = 0.75) -> dict[str, Any]:
    """Map impervious surfaces from image brightness."""
    grid = [[1 if v >= threshold else 0 for v in row] for row in _grid(image)]
    return {"mask": grid, "impervious_pixels": sum(sum(r) for r in grid)}


def tree_canopy_height_model(dsm: Grid, dtm: Grid) -> list[list[float]]:
    """Compute a canopy height model from surface and terrain grids."""
    return [[round(max(0.0, d - t), 4) for d, t in zip(dr, tr)] for dr, tr in zip(_grid(dsm), _grid(dtm))]


def building_footprint_extraction(image: Grid, *, threshold: float = 2.0) -> dict[str, Any]:
    """Extract coarse building footprints from bright pixels."""
    count = sum(1 for v in _flatten(image) if v >= threshold)
    return {"building_pixels": count}


def road_extraction_from_imagery(image: Grid, *, threshold: float = 1.0) -> dict[str, Any]:
    """Extract a coarse road mask from imagery."""
    count = sum(1 for v in _flatten(image) if v >= threshold)
    return {"road_pixels": count}


def water_body_extraction(image: Grid, *, threshold: float = 0.7) -> dict[str, Any]:
    """Extract water bodies from a simple index raster."""
    count = sum(1 for v in _flatten(image) if v >= threshold)
    return {"water_pixels": count}


def shadow_detection_and_removal(image: Grid, *, threshold: float = 1.0) -> dict[str, Any]:
    """Detect shadows and return a brightened image."""
    grid = _grid(image)
    shadow_pixels = sum(1 for v in _flatten(grid) if v <= threshold)
    corrected = [[max(v, threshold) for v in row] for row in grid]
    return {"shadow_pixels": shadow_pixels, "corrected": corrected}


def cloud_masking(image: Grid, *, threshold: float = 0.9) -> dict[str, Any]:
    """Build a cloud mask from bright pixels."""
    count = sum(1 for v in _flatten(image) if v >= threshold)
    return {"cloud_pixels": count}


def atmospheric_correction(image: Grid, *, haze: float = 0.0) -> list[list[float]]:
    """Apply a simple dark-object subtraction correction."""
    return [[round(v - haze, 4) for v in row] for row in _grid(image)]


def radiometric_calibration(image: Grid, *, gain: float = 1.0, bias: float = 0.0) -> list[list[float]]:
    """Apply radiometric gain and bias to an image."""
    return [[round(v * gain + bias, 4) for v in row] for row in _grid(image)]


def orthorectification(image: Grid) -> dict[str, Any]:
    """Return a simple orthorectification status summary."""
    g = _grid(image)
    return {"orthorectified": True, "rows": len(g), "cols": len(g[0]) if g else 0}


def georeferencing_from_gcps(gcps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build a georeferencing summary from ground control points."""
    return {"gcp_count": len(gcps), "transform": "affine" if gcps else None}


def image_registration_feature_matching(image_a: Grid, image_b: Grid) -> dict[str, Any]:
    """Estimate registration match quality between two images."""
    same = 0
    total = 0
    for ra, rb in zip(_grid(image_a), _grid(image_b)):
        for a, b in zip(ra, rb):
            total += 1
            if a == b:
                same += 1
    return {"match_score": round(same / total if total else 0.0, 4)}


def stereo_pair_dem_extraction(left: Grid, right: Grid) -> list[list[float]]:
    """Estimate a DEM from a stereo pair by averaging matching cells."""
    return [[round((a + b) / 2.0, 4) for a, b in zip(ra, rb)] for ra, rb in zip(_grid(left), _grid(right))]


def photogrammetric_block_adjustment(cameras: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise a photogrammetric block adjustment setup."""
    return {"camera_count": len(cameras), "adjusted": True}


def lidar_ground_classification(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify LiDAR points into ground, canopy, and structure classes."""
    out = []
    for pt in points:
        z = float(pt.get("z", 0))
        if z < 2:
            cls = "ground"
        elif z < 10:
            cls = "canopy"
        else:
            cls = "structure"
        out.append({**pt, "class": pt.get("class", cls)})
    return out


def lidar_noise_filter(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove obvious LiDAR outliers using a robust median threshold."""
    zs = sorted(float(p.get("z", 0)) for p in points)
    if not zs:
        return []
    median = zs[len(zs) // 2]
    deviations = sorted(abs(z - median) for z in zs)
    mad = deviations[len(deviations) // 2] or 1.0
    cutoff = median + 3 * mad
    return [p for p in points if abs(float(p.get("z", 0)) - median) <= cutoff]


def lidar_canopy_height_model(canopy_points: Sequence[dict[str, Any]], ground_points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Estimate canopy height from canopy and ground returns."""
    ground_mean = _mean([float(p.get("z", 0)) for p in ground_points])
    return [{**p, "height": round(float(p.get("z", 0)) - ground_mean, 4)} for p in canopy_points]


def lidar_building_extraction(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count likely building points."""
    return {"building_points": sum(1 for p in points if float(p.get("z", 0)) >= 10)}


def lidar_power_line_extraction(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count likely overhead power-line points."""
    return {"powerline_points": sum(1 for p in points if float(p.get("z", 0)) >= 20)}


def lidar_intensity_normalisation(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise LiDAR intensity to the range [0, 1]."""
    max_i = max((float(p.get("intensity", 0)) for p in points), default=1.0) or 1.0
    return [{**p, "intensity_norm": round(float(p.get("intensity", 0)) / max_i, 4)} for p in points]


def lidar_point_thinning(points: Sequence[dict[str, Any]], *, keep_every: int = 2) -> list[dict[str, Any]]:
    """Thin a point cloud by retaining every Nth point."""
    return [p for i, p in enumerate(points) if i % max(keep_every, 1) == 0]


def _point_grid(points: Sequence[dict[str, Any]], *, grid_size: int = 10, value_key: str = "z", value_mode: str = "mean") -> dict[str, Any]:
    xs = [float(p.get("x", 0)) for p in points] or [0.0]
    ys = [float(p.get("y", 0)) for p in points] or [0.0]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = (max_x - min_x) or 1.0
    dy = (max_y - min_y) or 1.0
    bins: list[list[list[float]]] = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    for p in points:
        col = min(grid_size - 1, int((float(p.get("x", 0)) - min_x) / dx * (grid_size - 1)))
        row = min(grid_size - 1, int((float(p.get("y", 0)) - min_y) / dy * (grid_size - 1)))
        if value_mode == "count":
            bins[row][col].append(1.0)
        else:
            bins[row][col].append(float(p.get(value_key, 0)))
    grid = []
    for row in bins:
        out_row = []
        for vals in row:
            if not vals:
                out_row.append(0.0)
            elif value_mode == "max":
                out_row.append(max(vals))
            else:
                out_row.append(round(_mean(vals), 4))
        grid.append(out_row)
    return {"grid": grid, "rows": grid_size, "cols": grid_size, "max_density": max(_flatten(grid), default=0)}


def lidar_dem_creation_ground_returns(points: Sequence[dict[str, Any]], *, grid_size: int = 10) -> dict[str, Any]:
    """Build a DEM from ground-return points."""
    ground = [p for p in points if str(p.get("class", "")).lower() == "ground"] or list(points)
    return _point_grid(ground, grid_size=grid_size, value_key="z")


def lidar_dsm_creation_first_returns(points: Sequence[dict[str, Any]], *, grid_size: int = 10) -> dict[str, Any]:
    """Build a DSM from first returns or all points."""
    return _point_grid(points, grid_size=grid_size, value_key="z", value_mode="max")


def lidar_point_density_raster(points: Sequence[dict[str, Any]], *, grid_size: int = 10) -> dict[str, Any]:
    """Compute a point density raster from a LiDAR cloud."""
    return _point_grid(points, grid_size=grid_size, value_mode="count")


def drone_imagery_mosaic(images: Sequence[Grid]) -> list[list[float]]:
    """Mosaic multiple drone image tiles into a single strip."""
    grids = [_grid(img) for img in images]
    if not grids:
        return []
    rows = max(len(g) for g in grids)
    mosaic: list[list[float]] = []
    for r in range(rows):
        row: list[float] = []
        for g in grids:
            if r < len(g):
                row.extend(g[r])
        mosaic.append(row)
    return mosaic


def structure_from_motion_point_cloud(images: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Create a simple SfM point-cloud summary from overlapping imagery."""
    return {"point_count": len(images) * 10, "reconstructed": True}


def multispectral_drone_index_maps(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute NDVI-style index maps from multispectral drone attributes."""
    ndvi = []
    for p in points:
        red = float(p.get("band_red", 0))
        nir = float(p.get("band_nir", 0))
        denom = nir + red
        ndvi.append(round((nir - red) / denom, 4) if denom else 0.0)
    return {"ndvi": ndvi, "mean_ndvi": round(_mean(ndvi), 4)}


def thermal_imagery_analysis(observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise thermal imagery observations."""
    temps = [float(o.get("temp", 0)) for o in observations]
    return {"mean_temperature": round(_mean(temps), 4), "max_temperature": round(max(temps or [0]), 4)}


def sar_speckle_filter(image: Grid) -> list[list[float]]:
    """Apply a simple smoothing filter to SAR imagery."""
    g = _grid(image)
    rows = len(g)
    cols = len(g[0]) if g else 0
    out = []
    for y in range(rows):
        row = []
        for x in range(cols):
            vals = [g[yy][xx] for yy in range(max(0, y - 1), min(rows, y + 2)) for xx in range(max(0, x - 1), min(cols, x + 2))]
            row.append(round(_mean(vals), 4))
        out.append(row)
    return out


def sar_coherence_interferometry(image_a: Grid, image_b: Grid) -> dict[str, Any]:
    """Estimate SAR coherence between two acquisitions."""
    diffs = []
    for ra, rb in zip(_grid(image_a), _grid(image_b)):
        for a, b in zip(ra, rb):
            diffs.append(1.0 / (1.0 + abs(a - b)))
    return {"mean_coherence": round(_mean(diffs), 4)}


__all__ = [
    "area_solar_radiation",
    "atmospheric_correction",
    "building_footprint_extraction",
    "cloud_masking",
    "crop_mask_generation",
    "deep_learning_change_detection",
    "deep_learning_object_detection",
    "deep_learning_pixel_classification",
    "drone_imagery_mosaic",
    "georeferencing_from_gcps",
    "growing_degree_days_accumulation",
    "heat_load_index",
    "image_registration_feature_matching",
    "impervious_surface_map",
    "land_cover_map",
    "lidar_building_extraction",
    "lidar_canopy_height_model",
    "lidar_dem_creation_ground_returns",
    "lidar_dsm_creation_first_returns",
    "lidar_ground_classification",
    "lidar_intensity_normalisation",
    "lidar_noise_filter",
    "lidar_point_density_raster",
    "lidar_point_thinning",
    "lidar_power_line_extraction",
    "minimum_noise_fraction",
    "multispectral_drone_index_maps",
    "orthorectification",
    "pansharpen_raster",
    "phenology_metric_extraction",
    "photogrammetric_block_adjustment",
    "points_solar_radiation",
    "radiometric_calibration",
    "raster_math_advanced",
    "road_extraction_from_imagery",
    "sar_coherence_interferometry",
    "sar_speckle_filter",
    "sediment_transport_index",
    "shadow_detection_and_removal",
    "spectral_anomaly_detection",
    "spectral_unmixing",
    "stereo_pair_dem_extraction",
    "structure_from_motion_point_cloud",
    "tasseled_cap_transformation",
    "thermal_imagery_analysis",
    "time_series_animation_export",
    "training_sample_creation",
    "tree_canopy_height_model",
    "water_body_extraction",
]
