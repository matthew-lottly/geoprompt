"""Lightweight 3D geometry helpers for the remaining A1 parity surface."""

from __future__ import annotations

import math
from typing import Any, Sequence


Coord3D = tuple[float, float, float]


def _to_3d(coord: Sequence[float]) -> Coord3D:
    if len(coord) >= 3:
        return (float(coord[0]), float(coord[1]), float(coord[2]))
    return (float(coord[0]), float(coord[1]), 0.0)


def _bbox3d(box: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    x1, y1, z1, x2, y2, z2 = map(float, box)
    return (min(x1, x2), min(y1, y2), min(z1, z2), max(x1, x2), max(y1, y2), max(z1, z2))


def _volume(box: Sequence[float]) -> float:
    x1, y1, z1, x2, y2, z2 = _bbox3d(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) * max(0.0, z2 - z1)


def true_curve_storage(
    center: Sequence[float],
    *,
    radius: float,
    start_angle: float,
    end_angle: float,
    segments: int = 16,
) -> dict[str, Any]:
    """Store a parametric arc as true-curve metadata plus sampled points."""
    cx, cy = float(center[0]), float(center[1])
    step = (end_angle - start_angle) / max(segments, 1)
    coords = []
    for i in range(segments + 1):
        angle = math.radians(start_angle + step * i)
        coords.append((round(cx + radius * math.cos(angle), 6), round(cy + radius * math.sin(angle), 6)))
    return {
        "type": "ParametricArc",
        "center": (cx, cy),
        "radius": float(radius),
        "start_angle": float(start_angle),
        "end_angle": float(end_angle),
        "coordinates": coords,
    }


def multi_patch_3d_solid_geometry(patches: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build a simple multipatch/solid summary from 3D patch rings."""
    vertices = []
    for patch in patches:
        for coord in patch.get("ring", []):
            vertices.append(_to_3d(coord))
    if not vertices:
        return {"type": "MultiPatch", "patch_count": 0, "solid": False, "volume": 0.0}
    xs, ys, zs = zip(*vertices)
    volume = (max(xs) - min(xs) or 1.0) * (max(ys) - min(ys) or 1.0) * (max(zs) - min(zs) or 1.0)
    return {
        "type": "MultiPatch",
        "patch_count": len(patches),
        "solid": True,
        "bounds": (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)),
        "volume": round(volume, 4),
    }


def extrude_polygon_to_3d_block(polygon: Sequence[Sequence[float]], *, height: float) -> dict[str, Any]:
    """Extrude a 2D polygon footprint into a 3D block."""
    base = [(_to_3d((x, y, 0.0))) for x, y in polygon]
    top = [(x, y, float(height)) for x, y, _ in base]
    return {
        "type": "ExtrudedBlock",
        "height": float(height),
        "base": base,
        "top": top,
        "wall_count": len(base),
    }


def intersection_3d(box_a: Sequence[float], box_b: Sequence[float]) -> dict[str, Any]:
    """Compute the axis-aligned 3D intersection between two boxes."""
    ax1, ay1, az1, ax2, ay2, az2 = _bbox3d(box_a)
    bx1, by1, bz1, bx2, by2, bz2 = _bbox3d(box_b)
    ix1, iy1, iz1 = max(ax1, bx1), max(ay1, by1), max(az1, bz1)
    ix2, iy2, iz2 = min(ax2, bx2), min(ay2, by2), min(az2, bz2)
    volume = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) * max(0.0, iz2 - iz1)
    return {"box": (ix1, iy1, iz1, ix2, iy2, iz2), "volume": round(volume, 4), "intersects": volume > 0}


def difference_3d(box_a: Sequence[float], box_b: Sequence[float]) -> dict[str, Any]:
    """Compute a remaining-volume summary for 3D difference."""
    inter = intersection_3d(box_a, box_b)
    total = _volume(box_a)
    remaining = max(0.0, total - float(inter["volume"]))
    return {"remaining_volume": round(remaining, 4), "original_volume": round(total, 4)}


def buffer_3d_sphere(point: Sequence[float], *, radius: float) -> dict[str, Any]:
    """Create a simple sphere buffer summary around a 3D point."""
    x, y, z = _to_3d(point)
    volume = 4.0 / 3.0 * math.pi * radius**3
    return {"type": "Sphere", "center": (x, y, z), "radius": float(radius), "volume": round(volume, 4)}


def near_3d_distance(origin: Sequence[float], candidates: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Find the nearest 3D point and distance."""
    ox, oy, oz = _to_3d(origin)
    best = None
    best_dist = float("inf")
    for candidate in candidates:
        cx, cy, cz = _to_3d(candidate)
        dist = math.sqrt((ox - cx) ** 2 + (oy - cy) ** 2 + (oz - cz) ** 2)
        if dist < best_dist:
            best = (cx, cy, cz)
            best_dist = dist
    return {"nearest": best, "distance": round(best_dist, 4)}


def skyline_analysis(buildings: Sequence[dict[str, Any]], *, observer: Sequence[float]) -> list[dict[str, Any]]:
    """Rank building features by skyline prominence."""
    ox, oy, oz = _to_3d(observer)
    ranked = []
    for item in buildings:
        height = float(item.get("height", 0.0))
        dist = math.sqrt((float(item.get("x", 0.0)) - ox) ** 2 + (float(item.get("y", 0.0)) - oy) ** 2 + 1.0)
        ranked.append({**item, "prominence": round(height / dist, 4)})
    return sorted(ranked, key=lambda row: (row.get("height", 0.0), row.get("prominence", 0.0)), reverse=True)


def line_of_sight_analysis(observer: Sequence[float], target: Sequence[float], obstacles: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Estimate whether a target is visible from an observer in 3D."""
    oz = _to_3d(observer)[2]
    tz = _to_3d(target)[2]
    line_z = (oz + tz) / 2.0
    max_obstacle = max((float(item.get("height", 0.0)) for item in (obstacles or [])), default=0.0)
    return {"visible": max_obstacle < line_z, "blocking_height": max_obstacle, "line_mid_height": round(line_z, 4)}


def shadow_volume_computation(
    footprint: Sequence[Sequence[float]],
    *,
    height: float,
    sun_altitude: float,
    sun_azimuth: float,
) -> dict[str, Any]:
    """Estimate shadow length and volume from a footprint and sun position."""
    shadow_length = float(height) / max(math.tan(math.radians(max(sun_altitude, 0.1))), 1e-6)
    footprint_area = 0.0
    points = list(footprint)
    if points and points[0] != points[-1]:
        points = points + [points[0]]
    for i in range(len(points) - 1):
        footprint_area += points[i][0] * points[i + 1][1] - points[i + 1][0] * points[i][1]
    footprint_area = abs(footprint_area) / 2.0
    return {
        "shadow_length": round(shadow_length, 4),
        "shadow_volume": round(footprint_area * shadow_length, 4),
        "sun_azimuth": float(sun_azimuth),
    }


def cross_section_profile_from_3d_line(line: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Extract cumulative distance/elevation values from a 3D line."""
    profile = []
    cumulative = 0.0
    last = None
    for coord in line:
        point = _to_3d(coord)
        if last is not None:
            cumulative += math.sqrt(sum((point[i] - last[i]) ** 2 for i in range(3)))
        profile.append({"distance": round(cumulative, 4), "elevation": point[2]})
        last = point
    return {"profile": profile, "length": round(cumulative, 4)}


def surface_volume_cut_fill(
    existing_surface: Sequence[Sequence[float]],
    design_surface: Sequence[Sequence[float]],
    *,
    cell_size: float = 1.0,
) -> dict[str, Any]:
    """Compute simple cut/fill volumes between two surfaces."""
    cut = 0.0
    fill = 0.0
    cell_area = float(cell_size) ** 2
    for existing_row, design_row in zip(existing_surface, design_surface):
        for existing, design in zip(existing_row, design_row):
            diff = float(design) - float(existing)
            if diff > 0:
                fill += diff * cell_area
            else:
                cut += abs(diff) * cell_area
    return {"cut_volume": round(cut, 4), "fill_volume": round(fill, 4), "net_volume": round(fill - cut, 4)}


def contour_polygon_fill_generation(levels: Sequence[float]) -> dict[str, Any]:
    """Generate fill bands between contour levels."""
    bands = []
    ordered = list(levels)
    for low, high in zip(ordered, ordered[1:]):
        bands.append({"low": float(low), "high": float(high), "label": f"{low}-{high}"})
    return {"bands": bands, "count": len(bands)}


__all__ = [
    "buffer_3d_sphere",
    "contour_polygon_fill_generation",
    "cross_section_profile_from_3d_line",
    "difference_3d",
    "extrude_polygon_to_3d_block",
    "intersection_3d",
    "line_of_sight_analysis",
    "multi_patch_3d_solid_geometry",
    "near_3d_distance",
    "shadow_volume_computation",
    "skyline_analysis",
    "surface_volume_cut_fill",
    "true_curve_storage",
]
