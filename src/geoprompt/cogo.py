"""Coordinate Geometry (COGO) tools for survey and parcel operations.

Pure-Python implementations covering roadmap items 1294-1300:
traverse, bearing-distance, curve data, intersection calculations,
and least-squares adjustment.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_bearing(bearing: float) -> float:
    """Normalise bearing to 0-360 degrees."""
    return bearing % 360


def _bearing_to_radians(bearing: float) -> float:
    """Convert compass bearing (0=N, CW) to math angle (0=E, CCW)."""
    return math.radians(90 - bearing)


def _radians_to_bearing(angle: float) -> float:
    """Convert math angle to compass bearing."""
    return _normalize_bearing(90 - math.degrees(angle))


# ---------------------------------------------------------------------------
# Bearing & distance (1295, 1297)
# ---------------------------------------------------------------------------

def bearing_between_points(
    p1: Sequence[float],
    p2: Sequence[float],
) -> float:
    """Compute compass bearing (degrees, 0=N, clockwise) from p1 to p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.atan2(dx, dy)
    return _normalize_bearing(math.degrees(angle))


def distance_between_points(
    p1: Sequence[float],
    p2: Sequence[float],
) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def point_from_bearing_distance(
    origin: Sequence[float],
    bearing: float,
    distance: float,
) -> tuple[float, float]:
    """Compute a point at given bearing (degrees) and distance from origin."""
    rad = _bearing_to_radians(bearing)
    x = origin[0] + distance * math.cos(rad)
    y = origin[1] + distance * math.sin(rad)
    return (x, y)


def bearing_distance_to_polygon(
    origin: Sequence[float],
    legs: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Convert a series of bearing-distance legs into a polygon.

    *legs*: sequence of (bearing_degrees, distance).
    Returns list of (x, y) coordinates forming a closed polygon.
    """
    coords = [tuple(origin[:2])]
    current = list(origin[:2])
    for bearing, dist in legs:
        pt = point_from_bearing_distance(current, bearing, dist)
        coords.append(pt)
        current = list(pt)
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


# ---------------------------------------------------------------------------
# Traverse computation (1296)
# ---------------------------------------------------------------------------

def compute_traverse(
    start: Sequence[float],
    observations: Sequence[dict[str, float]],
    *,
    adjust: bool = True,
) -> dict[str, Any]:
    """Compute a traverse from field observations.

    Each observation: {'bearing': degrees, 'distance': float}.
    Returns adjusted coordinates, misclosure, and precision ratio.
    """
    coords = [tuple(start[:2])]
    current = list(start[:2])
    total_dist = 0.0

    for obs in observations:
        bearing = obs["bearing"]
        dist = obs["distance"]
        total_dist += dist
        pt = point_from_bearing_distance(current, bearing, dist)
        coords.append(pt)
        current = list(pt)

    # Misclosure
    close_dx = coords[-1][0] - coords[0][0]
    close_dy = coords[-1][1] - coords[0][1]
    misclosure = math.sqrt(close_dx ** 2 + close_dy ** 2)
    precision = total_dist / misclosure if misclosure > 1e-12 else float("inf")

    if adjust and len(coords) > 2 and misclosure > 1e-12:
        # Compass (Bowditch) rule adjustment
        n = len(coords) - 1  # number of legs
        cum_dist = 0.0
        adjusted = [coords[0]]
        for i in range(len(observations)):
            cum_dist += observations[i]["distance"]
            ratio = cum_dist / total_dist
            ax = coords[i + 1][0] - close_dx * ratio
            ay = coords[i + 1][1] - close_dy * ratio
            adjusted.append((ax, ay))
        coords = adjusted

    return {
        "coordinates": coords,
        "misclosure": misclosure,
        "precision_ratio": precision,
        "total_distance": total_dist,
    }


# ---------------------------------------------------------------------------
# Curve data (1298)
# ---------------------------------------------------------------------------

def curve_data(
    *,
    radius: float | None = None,
    arc_length: float | None = None,
    chord_length: float | None = None,
    delta_angle: float | None = None,
) -> dict[str, float]:
    """Compute circular curve elements from any two parameters.

    Provide exactly two of: radius, arc_length, chord_length, delta_angle (degrees).
    Returns all curve elements.
    """
    given = sum(x is not None for x in [radius, arc_length, chord_length, delta_angle])
    if given < 2:
        raise ValueError("provide at least two curve parameters")

    if radius is not None and delta_angle is not None:
        da_rad = math.radians(delta_angle)
        arc_length = radius * da_rad
        chord_length = 2 * radius * math.sin(da_rad / 2)
    elif radius is not None and arc_length is not None:
        da_rad = arc_length / radius
        delta_angle = math.degrees(da_rad)
        chord_length = 2 * radius * math.sin(da_rad / 2)
    elif radius is not None and chord_length is not None:
        half_angle = math.asin(min(chord_length / (2 * radius), 1.0))
        da_rad = 2 * half_angle
        delta_angle = math.degrees(da_rad)
        arc_length = radius * da_rad
    elif arc_length is not None and chord_length is not None:
        # Newton's method to find delta angle
        da_rad = 2 * math.asin(chord_length / arc_length) if arc_length > 0 else 0.0
        for _ in range(20):
            radius = arc_length / da_rad if da_rad > 0 else float("inf")
            chord_calc = 2 * radius * math.sin(da_rad / 2)
            err = chord_calc - chord_length
            if abs(err) < 1e-12:
                break
            da_rad += err * 0.01
        delta_angle = math.degrees(da_rad)
        radius = arc_length / da_rad if da_rad > 0 else float("inf")
    elif delta_angle is not None and arc_length is not None:
        da_rad = math.radians(delta_angle)
        radius = arc_length / da_rad if da_rad > 0 else float("inf")
        chord_length = 2 * radius * math.sin(da_rad / 2)
    elif delta_angle is not None and chord_length is not None:
        da_rad = math.radians(delta_angle)
        radius = chord_length / (2 * math.sin(da_rad / 2)) if da_rad > 0 else float("inf")
        arc_length = radius * da_rad
    else:
        raise ValueError("invalid parameter combination")

    da_rad = math.radians(delta_angle)
    tangent = radius * math.tan(da_rad / 2) if da_rad > 0 else 0.0
    mid_ordinate = radius * (1 - math.cos(da_rad / 2)) if radius is not None else 0.0
    external = radius * (1 / math.cos(da_rad / 2) - 1) if (da_rad > 0 and radius) else 0.0

    return {
        "radius": radius,
        "arc_length": arc_length,
        "chord_length": chord_length,
        "delta_angle": delta_angle,
        "tangent": tangent,
        "mid_ordinate": mid_ordinate,
        "external": external,
    }


# ---------------------------------------------------------------------------
# Survey point adjustment (1299)
# ---------------------------------------------------------------------------

def adjust_survey_points(
    observed: Sequence[Sequence[float]],
    control: Sequence[Sequence[float]],
    control_indices: Sequence[int],
) -> list[tuple[float, float]]:
    """Simple affine adjustment of observed points to fit known control points.

    Uses a 2D affine transform (translation + scale) fit from control points.
    """
    if len(control_indices) < 1:
        raise ValueError("need at least 1 control point")
    if len(control_indices) == 1:
        # Pure translation
        ci = control_indices[0]
        dx = control[0][0] - observed[ci][0]
        dy = control[0][1] - observed[ci][1]
        return [(p[0] + dx, p[1] + dy) for p in observed]

    # Compute centroids of control subset
    obs_ctrl = [observed[i] for i in control_indices]
    cx_obs = sum(p[0] for p in obs_ctrl) / len(obs_ctrl)
    cy_obs = sum(p[1] for p in obs_ctrl) / len(obs_ctrl)
    cx_ctl = sum(p[0] for p in control) / len(control)
    cy_ctl = sum(p[1] for p in control) / len(control)

    # Compute scale and rotation (Helmert 2D)
    sum_num_a = sum_num_b = sum_den = 0.0
    for k, ci in enumerate(control_indices):
        ox = observed[ci][0] - cx_obs
        oy = observed[ci][1] - cy_obs
        cx = control[k][0] - cx_ctl
        cy = control[k][1] - cy_ctl
        sum_num_a += ox * cx + oy * cy
        sum_num_b += ox * cy - oy * cx
        sum_den += ox * ox + oy * oy

    if sum_den == 0:
        return [(cx_ctl, cy_ctl)] * len(observed)

    a = sum_num_a / sum_den
    b = sum_num_b / sum_den

    adjusted = []
    for p in observed:
        ox = p[0] - cx_obs
        oy = p[1] - cy_obs
        nx = a * ox - b * oy + cx_ctl
        ny = b * ox + a * oy + cy_ctl
        adjusted.append((nx, ny))
    return adjusted


# ---------------------------------------------------------------------------
# Least-squares adjustment (1300)
# ---------------------------------------------------------------------------

def least_squares_adjustment(
    observations: Sequence[dict[str, Any]],
    *,
    max_iter: int = 20,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Simple least-squares adjustment for a closed traverse.

    Each observation: {'from': (x,y), 'to': (x,y), 'bearing': deg, 'distance': float}.
    Returns adjusted coordinates and residuals.
    """
    # Collect unique points
    point_map: dict[tuple[float, float], int] = {}
    all_pts: list[list[float]] = []
    for obs in observations:
        for key in ("from", "to"):
            pt = tuple(obs[key][:2])
            if pt not in point_map:
                point_map[pt] = len(all_pts)
                all_pts.append(list(pt))

    n_pts = len(all_pts)
    if n_pts < 2:
        return {"coordinates": [tuple(p) for p in all_pts], "residuals": [], "iterations": 0}

    # Iterative adjustment
    residuals = []
    for iteration in range(max_iter):
        max_correction = 0.0
        residuals = []
        for obs in observations:
            fi = point_map[tuple(obs["from"][:2])]
            ti = point_map[tuple(obs["to"][:2])]
            dx = all_pts[ti][0] - all_pts[fi][0]
            dy = all_pts[ti][1] - all_pts[fi][1]
            obs_dist = obs["distance"]
            calc_dist = math.sqrt(dx ** 2 + dy ** 2)
            if calc_dist < 1e-15:
                residuals.append({"bearing_residual": 0.0, "distance_residual": obs_dist})
                continue
            calc_bearing = _normalize_bearing(math.degrees(math.atan2(dx, dy)))
            obs_bearing = obs["bearing"]
            dist_residual = obs_dist - calc_dist
            bearing_residual = obs_bearing - calc_bearing
            if bearing_residual > 180:
                bearing_residual -= 360
            elif bearing_residual < -180:
                bearing_residual += 360
            residuals.append({"bearing_residual": bearing_residual, "distance_residual": dist_residual})
            # Apply half correction
            corr = dist_residual * 0.5
            rad = _bearing_to_radians(obs_bearing)
            all_pts[ti][0] += corr * math.cos(rad) * 0.5
            all_pts[ti][1] += corr * math.sin(rad) * 0.5
            max_correction = max(max_correction, abs(dist_residual))

        if max_correction < tolerance:
            break

    return {
        "coordinates": [tuple(p) for p in all_pts],
        "residuals": residuals,
        "iterations": iteration + 1 if observations else 0,
    }


# ---------------------------------------------------------------------------
# Metes & bounds parsing (1294)
# ---------------------------------------------------------------------------

def parse_metes_and_bounds(description: str) -> list[dict[str, Any]]:
    """Parse a metes-and-bounds legal description into legs.

    Recognises patterns like "N 45°30' E 100.00" or "S45d30m00sW 200.50".
    Returns list of dicts with 'bearing' (degrees) and 'distance'.
    """
    import re
    legs = []
    # Pattern: N/S dd°mm'ss" E/W distance
    pattern = re.compile(
        r'([NS])\s*(\d+)[°d]\s*(\d+)?[\'m]?\s*(\d+)?["s]?\s*([EW])\s+(\d+\.?\d*)',
        re.IGNORECASE,
    )
    for match in pattern.finditer(description):
        ns, deg, mins, secs, ew, dist = match.groups()
        angle = float(deg)
        if mins:
            angle += float(mins) / 60
        if secs:
            angle += float(secs) / 3600
        # Convert to compass bearing
        if ns.upper() == "N" and ew.upper() == "E":
            bearing = angle
        elif ns.upper() == "N" and ew.upper() == "W":
            bearing = 360 - angle
        elif ns.upper() == "S" and ew.upper() == "E":
            bearing = 180 - angle
        else:  # S, W
            bearing = 180 + angle
        legs.append({"bearing": bearing, "distance": float(dist)})
    return legs


# ---------------------------------------------------------------------------
# Setback / FAR / lot coverage calculations (1303-1305)
# ---------------------------------------------------------------------------

def setback_check(
    building_polygon: Sequence[Sequence[float]],
    parcel_boundary: Sequence[Sequence[float]],
    *,
    required_setback: float,
) -> dict[str, Any]:
    """Check whether a building meets setback requirements from parcel boundary.

    Returns minimum distance from building to boundary and pass/fail.
    """
    min_dist = float("inf")
    for bp in building_polygon:
        for pp in parcel_boundary:
            d = math.sqrt((bp[0] - pp[0]) ** 2 + (bp[1] - pp[1]) ** 2)
            if d < min_dist:
                min_dist = d
    return {
        "min_distance": min_dist,
        "required_setback": required_setback,
        "passes": min_dist >= required_setback,
    }


def floor_area_ratio(
    total_floor_area: float,
    lot_area: float,
) -> float:
    """Calculate Floor Area Ratio (FAR)."""
    if lot_area <= 0:
        raise ValueError("lot area must be positive")
    return total_floor_area / lot_area


def lot_coverage(
    building_footprint_area: float,
    lot_area: float,
) -> float:
    """Calculate lot coverage percentage."""
    if lot_area <= 0:
        raise ValueError("lot area must be positive")
    return (building_footprint_area / lot_area) * 100
