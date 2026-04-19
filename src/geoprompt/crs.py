"""Coordinate reference system helpers and lightweight reprojection utilities.

This module provides a pragmatic CRS layer for GeoPrompt. It works with
PyProj when available and includes small pure-Python fallbacks for common
web-mercator and coordinate-parsing workflows.
"""
from __future__ import annotations

import importlib
import json
import math
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from .geometry import Geometry, geodesic_area as _geometry_geodesic_area

if TYPE_CHECKING:
    from .frame import GeoPromptFrame


CRSInfo = dict[str, Any]
_TRANSFORMER_LOCK = threading.RLock()
EARTH_RADIUS_M = 6_378_137.0


def _normalize_name(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return f"EPSG:{text}"
    if text.lower().startswith("epsg:"):
        return f"EPSG:{text.split(':', 1)[1].strip()}"
    return text


def _pyproj_module() -> Any | None:
    try:
        return importlib.import_module("pyproj")
    except Exception:
        return None


@lru_cache(maxsize=256)
def _resolve_crs_cached(user_input: str) -> tuple[tuple[str, Any], ...]:
    pyproj = _pyproj_module()
    name = _normalize_name(user_input) or "UNKNOWN"
    info: CRSInfo = {
        "type": "CRS",
        "name": name,
        "authority": None,
        "epsg": None,
        "unit": "unknown",
        "is_geographic": name.upper() == "EPSG:4326",
        "is_projected": name.upper() not in {"EPSG:4326", "UNKNOWN"},
        "wkt": None,
        "proj": None,
    }

    if pyproj is not None:
        try:
            resolved = pyproj.CRS.from_user_input(user_input)
            authority = resolved.to_authority()
            canonical_name = f"{authority[0]}:{authority[1]}" if authority else str(resolved.to_string())
            unit_name = "unknown"
            if getattr(resolved, "axis_info", None):
                axis_info = resolved.axis_info[0]
                unit_name = getattr(axis_info, "unit_name", None) or "unknown"
            info.update(
                {
                    "name": canonical_name,
                    "authority": authority[0] if authority else None,
                    "epsg": int(authority[1]) if authority and authority[1].isdigit() else None,
                    "unit": unit_name,
                    "is_geographic": bool(resolved.is_geographic),
                    "is_projected": bool(resolved.is_projected),
                    "wkt": resolved.to_wkt(),
                    "proj": resolved.to_proj4() if hasattr(resolved, "to_proj4") else None,
                }
            )
        except Exception:
            pass

    return tuple(info.items())


def crs_from_user_input(value: object | None) -> CRSInfo:
    """Resolve any supported CRS input to a normalized CRS info mapping."""
    if value is None:
        raise ValueError("CRS input must not be None")
    return dict(_resolve_crs_cached(str(value)))


def crs_from_epsg(code: int | str) -> CRSInfo:
    """Create a CRS definition from an EPSG code."""
    return crs_from_user_input(f"EPSG:{int(code)}")


def crs_from_wkt(wkt: str) -> CRSInfo:
    """Create a CRS definition from a WKT1 or WKT2 string."""
    info = crs_from_user_input(wkt)
    if info.get("wkt") is None:
        info["wkt"] = wkt
    return info


def crs_from_proj(proj_string: str) -> CRSInfo:
    """Create a CRS definition from a PROJ string."""
    info = crs_from_user_input(proj_string)
    if info.get("proj") is None:
        info["proj"] = proj_string
    return info


def crs_from_authority(authority_code: str) -> CRSInfo:
    """Create a CRS definition from an Authority:Code string such as EPSG:3857."""
    return crs_from_user_input(authority_code)


def custom_crs_definition(
    name: str,
    *,
    proj: str | None = None,
    wkt: str | None = None,
    authority: str | None = None,
    unit: str = "metre",
    geographic: bool = False,
) -> CRSInfo:
    """Create a custom CRS metadata mapping."""
    return {
        "type": "CRS",
        "name": name,
        "authority": authority,
        "epsg": None,
        "unit": unit,
        "is_geographic": geographic,
        "is_projected": not geographic,
        "wkt": wkt,
        "proj": proj,
    }


def crs_from_prj_file(path: str | Path) -> CRSInfo:
    """Read a projection file and resolve it as a CRS definition."""
    return crs_from_wkt(Path(path).read_text(encoding="utf-8"))


def crs_from_esri_pe(pe_string: str) -> CRSInfo:
    """Create a CRS definition from an Esri PE string."""
    return crs_from_wkt(pe_string)


def crs_auto_detect(value: Any) -> CRSInfo | None:
    """Attempt to detect CRS information from a frame, mapping, or path."""
    if value is None:
        return None
    if hasattr(value, "crs") and getattr(value, "crs"):
        return crs_from_user_input(getattr(value, "crs"))
    if isinstance(value, dict):
        for key in ("crs", "spatialReference", "epsg", "wkid"):
            if key in value and value[key]:
                ref = value[key]
                if isinstance(ref, dict) and "wkid" in ref:
                    return crs_from_epsg(ref["wkid"])
                return crs_from_user_input(ref)
    if isinstance(value, (str, Path)):
        suffix = Path(value).suffix.lower()
        if suffix == ".prj":
            return crs_from_prj_file(value)
    return None


@lru_cache(maxsize=128)
def _get_transformer(source_crs: str, target_crs: str) -> Any:
    pyproj = _pyproj_module()
    if pyproj is None:
        return None
    with _TRANSFORMER_LOCK:
        return pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)


def coordinate_to_web_mercator(lon: float, lat: float) -> tuple[float, float]:
    """Project a longitude/latitude coordinate to Web Mercator."""
    lat = max(min(float(lat), 85.05112878), -85.05112878)
    x_value = EARTH_RADIUS_M * math.radians(float(lon))
    y_value = EARTH_RADIUS_M * math.log(math.tan((math.pi / 4.0) + (math.radians(lat) / 2.0)))
    return (x_value, y_value)


def coordinate_to_wgs84(x_value: float, y_value: float) -> tuple[float, float]:
    """Inverse-project a Web Mercator coordinate to longitude/latitude."""
    lon = math.degrees(float(x_value) / EARTH_RADIUS_M)
    lat = math.degrees((2.0 * math.atan(math.exp(float(y_value) / EARTH_RADIUS_M))) - (math.pi / 2.0))
    return (lon, lat)


def _project_xy(x_value: float, y_value: float, source_crs: str, target_crs: str) -> tuple[float, float]:
    source = _normalize_name(source_crs) or "EPSG:4326"
    target = _normalize_name(target_crs) or "EPSG:4326"
    if source == target:
        return (float(x_value), float(y_value))
    transformer = _get_transformer(source, target)
    if transformer is not None:
        new_x, new_y = transformer.transform(float(x_value), float(y_value))
        return (float(new_x), float(new_y))
    if source.upper() == "EPSG:4326" and target.upper() == "EPSG:3857":
        return coordinate_to_web_mercator(float(x_value), float(y_value))
    if source.upper() == "EPSG:3857" and target.upper() == "EPSG:4326":
        return coordinate_to_wgs84(float(x_value), float(y_value))
    raise RuntimeError("PyProj is required for this CRS transformation")


def reproject_geometry(
    geometry: Geometry,
    source_crs: str,
    target_crs: str,
    *,
    preserve_z: bool = True,
    preserve_m: bool = True,
) -> Geometry:
    """Reproject a geometry while preserving any trailing dimensions."""
    def _map_value(value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) >= 2 and all(isinstance(part, (int, float)) for part in value[:2]):
            new_x, new_y = _project_xy(float(value[0]), float(value[1]), source_crs, target_crs)
            extras = list(value[2:])
            if not preserve_z and extras:
                extras = extras[1:]
            if not preserve_m and len(extras) >= 2:
                extras = extras[:1]
            return (new_x, new_y, *extras)
        if isinstance(value, (list, tuple)):
            return tuple(_map_value(part) for part in value)
        return value

    return {"type": geometry.get("type", "GeometryCollection"), "coordinates": _map_value(geometry.get("coordinates", ())) }


def define_projection(frame: "GeoPromptFrame", crs: str | int, *, allow_override: bool = False) -> "GeoPromptFrame":
    """Assign a CRS to a frame without changing its coordinates."""
    return frame.set_crs(crs, allow_override=allow_override)


def batch_project_features(frames: Sequence["GeoPromptFrame"], target_crs: str | int) -> list["GeoPromptFrame"]:
    """Project a batch of frames to a shared target CRS."""
    return [frame.to_crs(target_crs) for frame in frames]


def geographic_to_projected(lon: float, lat: float, target_crs: str | int = "EPSG:3857") -> tuple[float, float]:
    """Project a longitude/latitude pair to a target projected CRS."""
    return _project_xy(float(lon), float(lat), "EPSG:4326", _normalize_name(target_crs) or "EPSG:3857")


def projected_to_geographic(x_value: float, y_value: float, source_crs: str | int = "EPSG:3857") -> tuple[float, float]:
    """Project an XY coordinate back to WGS84 longitude/latitude."""
    return _project_xy(float(x_value), float(y_value), _normalize_name(source_crs) or "EPSG:3857", "EPSG:4326")


def transformation_pipeline(source_crs: str | int, target_crs: str | int) -> CRSInfo:
    """Build a lightweight transformation pipeline description between two CRS values."""
    source = crs_from_user_input(source_crs)
    target = crs_from_user_input(target_crs)
    return {
        "type": "TransformationPipeline",
        "source": source.get("name"),
        "target": target.get("name"),
        "cached": True,
        "thread_safe": True,
        "method": "pyproj" if _pyproj_module() is not None else "fallback",
    }


def transformation_accuracy_metadata(source_crs: str | int, target_crs: str | int) -> CRSInfo:
    """Return simple metadata about a transformation path."""
    source = crs_from_user_input(source_crs)
    target = crs_from_user_input(target_crs)
    return {
        "type": "TransformationMetadata",
        "source": source.get("name"),
        "target": target.get("name"),
        "method": "pyproj" if _pyproj_module() is not None else "fallback",
        "estimated_accuracy_m": 1.0 if _pyproj_module() is not None else 25.0,
    }


def custom_geographic_transform(
    lon: float,
    lat: float,
    *,
    lon_shift: float = 0.0,
    lat_shift: float = 0.0,
    scale: float = 1.0,
) -> tuple[float, float]:
    """Apply a simple custom geographic transform to a lon/lat coordinate."""
    return ((float(lon) * scale) + lon_shift, (float(lat) * scale) + lat_shift)


def custom_vertical_transform(value: float, *, offset: float = 0.0, scale: float = 1.0) -> float:
    """Apply a simple custom vertical transform to a height value."""
    return (float(value) * scale) + offset


def validate_crs_area_of_use(crs: str | int | CRSInfo, lon: float, lat: float) -> bool:
    """Validate whether a longitude/latitude falls inside the CRS area of use."""
    info = crs_from_user_input(crs["name"] if isinstance(crs, dict) else crs)
    pyproj = _pyproj_module()
    if pyproj is not None:
        try:
            resolved = pyproj.CRS.from_user_input(info["name"])
            area = resolved.area_of_use
            if area is not None:
                return area.west <= lon <= area.east and area.south <= lat <= area.north
        except Exception:
            pass
    if str(info.get("name", "")).upper() == "EPSG:4326":
        return -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0
    return True


def validate_crs_bounds(crs: str | int | CRSInfo, bounds: tuple[float, float, float, float]) -> bool:
    """Validate that a bounding box falls inside the CRS domain."""
    min_x, min_y, max_x, max_y = bounds
    return validate_crs_area_of_use(crs, min_x, min_y) and validate_crs_area_of_use(crs, max_x, max_y)


def convert_linear_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a linear measurement between common CRS units."""
    factors = {
        "metre": 1.0,
        "meter": 1.0,
        "m": 1.0,
        "kilometre": 1000.0,
        "kilometer": 1000.0,
        "km": 1000.0,
        "foot": 0.3048,
        "feet": 0.3048,
        "ft": 0.3048,
        "us_survey_foot": 1200.0 / 3937.0,
        "degree": 111_319.49079327357,
        "degrees": 111_319.49079327357,
    }
    source = factors[from_unit.lower()]
    target = factors[to_unit.lower()]
    return float(value) * source / target


def crs_equal(left: CRSInfo | str | int, right: CRSInfo | str | int) -> bool:
    """Return True when two CRS definitions resolve to the same reference system."""
    left_info = crs_from_user_input(left["name"] if isinstance(left, dict) else left)
    right_info = crs_from_user_input(right["name"] if isinstance(right, dict) else right)
    if left_info.get("epsg") is not None and right_info.get("epsg") is not None:
        return left_info["epsg"] == right_info["epsg"]
    return str(left_info.get("name")).upper() == str(right_info.get("name")).upper()


def crs_to_wkt(crs: CRSInfo | str | int) -> str:
    """Export a CRS definition to WKT."""
    info = crs_from_user_input(crs["name"] if isinstance(crs, dict) else crs)
    return str(info.get("wkt") or info.get("name") or "")


def crs_to_proj(crs: CRSInfo | str | int) -> str:
    """Export a CRS definition to a PROJ string."""
    info = crs_from_user_input(crs["name"] if isinstance(crs, dict) else crs)
    proj = info.get("proj")
    if proj:
        return str(proj)
    if str(info.get("name", "")).upper() == "EPSG:4326":
        return "+proj=longlat +datum=WGS84 +no_defs"
    return str(info.get("name", ""))


def crs_epsg_lookup(crs: CRSInfo | str | int) -> int | None:
    """Look up the EPSG code for a CRS definition when available."""
    info = crs_from_user_input(crs["name"] if isinstance(crs, dict) else crs)
    epsg = info.get("epsg")
    return int(epsg) if epsg is not None else None


def crs_to_json(crs: CRSInfo | str | int) -> str:
    """Serialize a CRS definition to JSON."""
    info = crs_from_user_input(crs["name"] if isinstance(crs, dict) else crs)
    return json.dumps(info, indent=2, sort_keys=True)


def crs_from_json(payload: str | bytes | CRSInfo) -> CRSInfo:
    """Deserialize a CRS definition from JSON."""
    if isinstance(payload, dict):
        return dict(payload)
    return dict(json.loads(payload))


def utm_zone_for_lonlat(lon: float, lat: float) -> str:
    """Auto-detect the best UTM EPSG code for a longitude/latitude pair."""
    zone = int((float(lon) + 180.0) / 6.0) + 1
    zone = max(1, min(60, zone))
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"


def state_plane_zone_auto_detect(lon: float, lat: float) -> str:
    """Return a lightweight placeholder state-plane zone suggestion."""
    return f"STATEPLANE-AUTO:{round(lat, 1)}:{round(lon, 1)}"


def lambert_conformal_conic_crs(lat_1: float, lat_2: float, *, lat_0: float = 0.0, lon_0: float = 0.0) -> CRSInfo:
    """Build a Lambert Conformal Conic CRS definition."""
    proj = f"+proj=lcc +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m +no_defs"
    return custom_crs_definition("Custom Lambert Conformal Conic", proj=proj)


def transverse_mercator_crs(*, lat_0: float = 0.0, lon_0: float = 0.0, k: float = 1.0) -> CRSInfo:
    """Build a Transverse Mercator CRS definition."""
    proj = f"+proj=tmerc +lat_0={lat_0} +lon_0={lon_0} +k={k} +datum=WGS84 +units=m +no_defs"
    return custom_crs_definition("Custom Transverse Mercator", proj=proj)


def albers_equal_area_crs(lat_1: float, lat_2: float, *, lat_0: float = 0.0, lon_0: float = 0.0) -> CRSInfo:
    """Build an Albers Equal Area CRS definition."""
    proj = f"+proj=aea +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m +no_defs"
    return custom_crs_definition("Custom Albers Equal Area", proj=proj)


def equidistant_conic_crs(lat_1: float, lat_2: float, *, lat_0: float = 0.0, lon_0: float = 0.0) -> CRSInfo:
    """Build an Equidistant Conic CRS definition."""
    proj = f"+proj=eqdc +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m +no_defs"
    return custom_crs_definition("Custom Equidistant Conic", proj=proj)


def sinusoidal_crs(*, lon_0: float = 0.0) -> CRSInfo:
    """Build a Sinusoidal CRS definition."""
    return custom_crs_definition("Custom Sinusoidal", proj=f"+proj=sinu +lon_0={lon_0} +datum=WGS84 +units=m +no_defs")


def mollweide_crs(*, lon_0: float = 0.0) -> CRSInfo:
    """Build a Mollweide CRS definition."""
    return custom_crs_definition("Custom Mollweide", proj=f"+proj=moll +lon_0={lon_0} +datum=WGS84 +units=m +no_defs")


def robinson_crs(*, lon_0: float = 0.0) -> CRSInfo:
    """Build a Robinson CRS definition."""
    return custom_crs_definition("Custom Robinson", proj=f"+proj=robin +lon_0={lon_0} +datum=WGS84 +units=m +no_defs")


def stereographic_crs(*, lat_0: float = 90.0, lon_0: float = 0.0) -> CRSInfo:
    """Build a Stereographic CRS definition."""
    return custom_crs_definition("Custom Stereographic", proj=f"+proj=stere +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m +no_defs")


def azimuthal_equidistant_crs(*, lat_0: float = 0.0, lon_0: float = 0.0) -> CRSInfo:
    """Build an Azimuthal Equidistant CRS definition."""
    return custom_crs_definition("Custom Azimuthal Equidistant", proj=f"+proj=aeqd +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m +no_defs")


def gnomonic_crs(*, lat_0: float = 0.0, lon_0: float = 0.0) -> CRSInfo:
    """Build a Gnomonic CRS definition."""
    return custom_crs_definition("Custom Gnomonic", proj=f"+proj=gnom +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m +no_defs")


def web_mercator_crs() -> CRSInfo:
    """Return the standard Web Mercator CRS."""
    return crs_from_epsg(3857)


def best_projection_for_extent(bounds: tuple[float, float, float, float]) -> CRSInfo:
    """Auto-select a practical projection for an extent."""
    min_x, min_y, max_x, max_y = bounds
    center_lon = (min_x + max_x) / 2.0
    center_lat = (min_y + max_y) / 2.0
    if max(abs(min_y), abs(max_y)) >= 70.0:
        return crs_from_user_input(polar_region_crs(center_lat))
    if (max_x - min_x) <= 12.0 and (max_y - min_y) <= 12.0:
        return crs_from_user_input(utm_zone_for_lonlat(center_lon, center_lat))
    return web_mercator_crs()


def geodesic_distance(a_point: tuple[float, float], b_point: tuple[float, float]) -> float:
    """Measure geodesic distance between two lon/lat coordinates in meters."""
    pyproj = _pyproj_module()
    if pyproj is not None:
        geod = pyproj.Geod(ellps="WGS84")
        _, _, distance = geod.inv(a_point[0], a_point[1], b_point[0], b_point[1])
        return float(distance)
    lon1, lat1 = map(math.radians, a_point)
    lon2, lat2 = map(math.radians, b_point)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    hav = (math.sin(dlat / 2) ** 2) + (math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * 6_371_008.8 * math.asin(min(1.0, math.sqrt(hav)))


def rhumb_line_distance(a_point: tuple[float, float], b_point: tuple[float, float]) -> float:
    """Measure rhumb-line distance between two lon/lat coordinates in meters."""
    lon1, lat1 = map(math.radians, a_point)
    lon2, lat2 = map(math.radians, b_point)
    dlon = lon2 - lon1
    dphi = math.log(math.tan((lat2 / 2) + (math.pi / 4)) / math.tan((lat1 / 2) + (math.pi / 4))) if lat1 != lat2 else 0.0
    q_value = (lat2 - lat1) / dphi if dphi != 0 else math.cos(lat1)
    if abs(dlon) > math.pi:
        dlon = dlon - math.copysign(2 * math.pi, dlon)
    return math.sqrt(((lat2 - lat1) ** 2) + ((q_value * dlon) ** 2)) * 6_371_008.8


def geodesic_midpoint(a_point: tuple[float, float], b_point: tuple[float, float]) -> tuple[float, float]:
    """Compute the midpoint along the great-circle path between two coordinates."""
    lon1, lat1 = map(math.radians, a_point)
    lon2, lat2 = map(math.radians, b_point)
    bx = math.cos(lat2) * math.cos(lon2 - lon1)
    by = math.cos(lat2) * math.sin(lon2 - lon1)
    lat3 = math.atan2(math.sin(lat1) + math.sin(lat2), math.sqrt((math.cos(lat1) + bx) ** 2 + (by**2)))
    lon3 = lon1 + math.atan2(by, math.cos(lat1) + bx)
    return (math.degrees(lon3), math.degrees(lat3))


def geodesic_area_on_ellipsoid(polygon: Geometry) -> float:
    """Measure polygon area on the ellipsoid in square meters."""
    pyproj = _pyproj_module()
    if pyproj is not None and polygon.get("type") == "Polygon":
        ring = polygon.get("coordinates", ())
        if ring:
            xs = [coord[0] for coord in ring[0]]
            ys = [coord[1] for coord in ring[0]]
            geod = pyproj.Geod(ellps="WGS84")
            area, _ = geod.polygon_area_perimeter(xs, ys)
            return abs(float(area))
    return float(_geometry_geodesic_area(polygon))


def geodesic_densify_line(line: Geometry, max_segment_meters: float) -> Geometry:
    """Densify a lon/lat line by geodesic distance threshold."""
    coords = list(line.get("coordinates", ()))
    if len(coords) < 2:
        return dict(line)
    result = [coords[0]]
    for index in range(1, len(coords)):
        start = coords[index - 1]
        end = coords[index]
        distance = geodesic_distance((start[0], start[1]), (end[0], end[1]))
        steps = max(1, int(math.ceil(distance / max_segment_meters)))
        for step in range(1, steps):
            frac = step / steps
            result.append((start[0] + ((end[0] - start[0]) * frac), start[1] + ((end[1] - start[1]) * frac)))
        result.append(end)
    return {"type": "LineString", "coordinates": tuple(result)}


def convergence_angle(lon: float, lat: float, *, central_meridian: float = 0.0) -> float:
    """Approximate the grid convergence angle in degrees at a longitude/latitude."""
    dlon = math.radians(float(lon) - float(central_meridian))
    phi = math.radians(float(lat))
    return math.degrees(math.atan(math.tan(dlon) * math.sin(phi)))


def true_north_grid_north_correction(lon: float, lat: float, *, central_meridian: float = 0.0) -> float:
    """Return the correction from grid north to true north in degrees."""
    return -convergence_angle(lon, lat, central_meridian=central_meridian)


def scale_factor_at_point(lat: float, *, k0: float = 0.9996) -> float:
    """Approximate the map scale factor at a latitude for a transverse Mercator workflow."""
    eccentricity_sq = 0.00669437999014
    sin_phi = math.sin(math.radians(float(lat)))
    return float(k0) / math.sqrt(1.0 - (eccentricity_sq * (sin_phi**2)))


def meridian_convergence(lon: float, lat: float, *, central_meridian: float = 0.0) -> float:
    """Alias for convergence-angle reporting."""
    return convergence_angle(lon, lat, central_meridian=central_meridian)


def normalize_antimeridian(longitudes: Iterable[float]) -> list[float]:
    """Normalize longitudes into the -180 to 180 range."""
    normalized: list[float] = []
    for lon in longitudes:
        value = ((float(lon) + 180.0) % 360.0) - 180.0
        normalized.append(value)
    return normalized


def polar_region_crs(latitude: float) -> str:
    """Suggest a polar CRS for high-latitude coordinates."""
    if latitude >= 60.0:
        return "EPSG:3413"
    if latitude <= -60.0:
        return "EPSG:3031"
    return "EPSG:4326"


def xy_table_to_geographic_features(
    rows: Sequence[dict[str, Any]],
    *,
    x_field: str = "x",
    y_field: str = "y",
    crs: str = "EPSG:4326",
    order: str = "xy",
) -> list[dict[str, Any]]:
    """Convert a table with X and Y columns into feature records."""
    result: list[dict[str, Any]] = []
    for row in rows:
        x_value, y_value = normalize_coordinate_order(float(row[x_field]), float(row[y_field]), order=order)
        feature = dict(row)
        feature["geometry"] = {"type": "Point", "coordinates": (x_value, y_value)}
        feature["crs"] = _normalize_name(crs)
        result.append(feature)
    return result


def _parse_single_dms(text: str) -> float:
    token = text.strip().upper()
    hemi = token[-1] if token and token[-1] in {"N", "S", "E", "W"} else ""
    sign = -1.0 if hemi in {"S", "W"} or token.startswith("-") else 1.0
    cleaned = re.sub(r"[NSEW°º'\"′″,]+", " ", token)
    parts = [float(part) for part in cleaned.split() if part.strip()]
    if not parts:
        raise ValueError("coordinate component is empty")
    degrees = parts[0]
    minutes = parts[1] if len(parts) > 1 else 0.0
    seconds = parts[2] if len(parts) > 2 else 0.0
    value = abs(degrees) + (minutes / 60.0) + (seconds / 3600.0)
    return sign * value


def parse_coordinate_text(text: str) -> tuple[float, float]:
    """Parse coordinate text and return a longitude/latitude pair."""
    stripped = text.strip()
    if auto_detect_coordinate_format(stripped) == "dms":
        tokens = re.findall(r"[\d\-\.°º'\"′″\s]+[NSEW]", stripped.upper())
        if len(tokens) >= 2:
            first = _parse_single_dms(tokens[0])
            second = _parse_single_dms(tokens[1])
            if tokens[0].strip()[-1] in {"N", "S"}:
                return (second, first)
            return (first, second)
    numbers = [float(part) for part in re.findall(r"[-+]?\d+(?:\.\d+)?", stripped)]
    if len(numbers) < 2:
        raise ValueError("could not parse coordinate text")
    first, second = numbers[0], numbers[1]
    if abs(first) <= 90 and abs(second) <= 180:
        return (second, first)
    return (first, second)


def parse_utm_coordinate(text: str) -> tuple[float, float]:
    """Parse a UTM coordinate string to a longitude/latitude pair."""
    match = re.search(r"(\d{1,2})([C-HJ-NP-X])\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)", text.upper())
    if match is None:
        raise ValueError("invalid UTM coordinate text")
    zone = int(match.group(1))
    hemi_band = match.group(2)
    easting = float(match.group(3))
    northing = float(match.group(4))
    epsg = f"EPSG:{32600 + zone}" if hemi_band >= "N" else f"EPSG:{32700 + zone}"
    return projected_to_geographic(easting, northing, epsg)


def auto_detect_coordinate_format(text: str) -> str:
    """Auto-detect the coordinate format represented by a text string."""
    upper = text.strip().upper()
    if any(symbol in upper for symbol in ("°", "º", "′", "″", "'", '"')) or any(card in upper for card in ("N", "S", "E", "W")):
        return "dms"
    if re.search(r"\b\d{1,2}[C-HJ-NP-X]\b\s+\d+\s+\d+", upper):
        return "utm"
    if len(re.findall(r"[-+]?\d+(?:\.\d+)?", upper)) >= 2:
        return "latlon"
    return "unknown"


def batch_coordinate_conversion_table(
    rows: Sequence[dict[str, Any]],
    *,
    source_x_field: str,
    source_y_field: str,
    target_crs: str,
    source_crs: str = "EPSG:4326",
) -> list[dict[str, Any]]:
    """Convert a table of coordinates from one CRS to another."""
    result: list[dict[str, Any]] = []
    for row in rows:
        x_value, y_value = _project_xy(float(row[source_x_field]), float(row[source_y_field]), source_crs, target_crs)
        updated = dict(row)
        updated["target_x"] = x_value
        updated["target_y"] = y_value
        updated["target_crs"] = _normalize_name(target_crs)
        result.append(updated)
    return result


def normalize_coordinate_order(first: float, second: float, *, order: str = "xy") -> tuple[float, float]:
    """Normalize coordinate order to X/Y."""
    normalized = order.lower()
    if normalized in {"xy", "lonlat", "longitude_latitude"}:
        return (float(first), float(second))
    if normalized in {"yx", "latlon", "latitude_longitude"}:
        return (float(second), float(first))
    raise ValueError("order must be xy, yx, lonlat, or latlon")


def vertical_crs_from_epsg(code: int | str) -> CRSInfo:
    """Create a vertical CRS definition from an EPSG code."""
    info = crs_from_epsg(code)
    info["vertical"] = True
    return info


def crs_comparison_report(left: CRSInfo | str | int, right: CRSInfo | str | int) -> CRSInfo:
    """Return a side-by-side comparison report for two CRS definitions."""
    left_info = crs_from_user_input(left["name"] if isinstance(left, dict) else left)
    right_info = crs_from_user_input(right["name"] if isinstance(right, dict) else right)
    return {
        "type": "CRSComparison",
        "left": left_info,
        "right": right_info,
        "equal": crs_equal(left_info, right_info),
        "same_units": left_info.get("unit") == right_info.get("unit"),
        "same_kind": (left_info.get("is_geographic") == right_info.get("is_geographic")) and (left_info.get("is_projected") == right_info.get("is_projected")),
    }


def crs_information_report(crs: CRSInfo | str | int) -> CRSInfo:
    """Return a full information report for a CRS definition."""
    return crs_from_user_input(crs["name"] if isinstance(crs, dict) else crs)


def resolve_crs_conflicts(frames: Sequence["GeoPromptFrame"], *, strategy: str = "first") -> list["GeoPromptFrame"]:
    """Resolve CRS conflicts across a set of frames."""
    if not frames:
        return []
    target = frames[0].crs
    if strategy == "raise":
        for frame in frames[1:]:
            if frame.crs != target:
                raise ValueError("frames have conflicting CRS values")
        return list(frames)
    if strategy == "reproject":
        if target is None:
            raise ValueError("cannot reproject to an unknown CRS")
        return [frame if frame.crs == target else frame.to_crs(target) for frame in frames]
    return [frame if frame.crs == target else frame.set_crs(target or "EPSG:4326", allow_override=True) for frame in frames]


def project_on_the_fly(frame: "GeoPromptFrame", target_crs: str | int) -> tuple["GeoPromptFrame", str]:
    """Project a frame for display and return a message describing the reprojection."""
    projected = frame.to_crs(target_crs)
    message = f"Projected on the fly from {frame.crs} to {projected.crs} for display."
    return projected, message


def crs_aware_distance(a_point: tuple[float, float], b_point: tuple[float, float], *, crs: str | int = "EPSG:4326") -> float:
    """Measure distance while respecting the CRS unit semantics."""
    info = crs_from_user_input(crs)
    if info.get("is_geographic"):
        return geodesic_distance(a_point, b_point)
    return math.hypot(float(b_point[0]) - float(a_point[0]), float(b_point[1]) - float(a_point[1]))


def crs_aware_area(polygon: Geometry, *, crs: str | int = "EPSG:4326") -> float:
    """Measure area while respecting the CRS unit semantics."""
    info = crs_from_user_input(crs)
    if info.get("is_geographic"):
        return geodesic_area_on_ellipsoid(polygon)
    from .geometry import geometry_area

    return float(geometry_area(polygon))


def engineering_local_crs(name: str, *, origin: tuple[float, float] = (0.0, 0.0), unit: str = "metre") -> CRSInfo:
    """Create a lightweight engineering or local site CRS definition."""
    return {
        "type": "CRS",
        "name": name,
        "authority": "LOCAL",
        "epsg": None,
        "unit": unit,
        "is_geographic": False,
        "is_projected": True,
        "origin": {"x": float(origin[0]), "y": float(origin[1])},
        "kind": "engineering",
        "wkt": None,
        "proj": None,
    }


def compound_crs(horizontal: CRSInfo | str | int, vertical: CRSInfo | str | int) -> CRSInfo:
    """Combine a horizontal CRS and a vertical CRS into a compound definition."""
    horizontal_info = crs_from_user_input(horizontal["name"] if isinstance(horizontal, dict) else horizontal)
    vertical_info = crs_from_user_input(vertical["name"] if isinstance(vertical, dict) else vertical)
    return {
        "type": "CRS",
        "name": f"{horizontal_info['name']} + {vertical_info['name']}",
        "authority": None,
        "epsg": None,
        "unit": horizontal_info.get("unit", "metre"),
        "is_geographic": bool(horizontal_info.get("is_geographic")),
        "is_projected": bool(horizontal_info.get("is_projected")),
        "horizontal": horizontal_info,
        "vertical": vertical_info,
        "compound": True,
        "wkt": None,
        "proj": None,
    }


def height_system_info(kind: str) -> CRSInfo:
    """Describe a common height reference system."""
    normalized = kind.strip().lower()
    descriptions = {
        "orthometric": "Height above the geoid or mean sea level.",
        "ellipsoidal": "Height above the reference ellipsoid.",
        "geoid": "Geoid undulation or separation value.",
    }
    return {
        "type": "HeightSystem",
        "name": normalized,
        "description": descriptions.get(normalized, "Custom height system."),
    }


def vertical_transform(value: float, *, from_system: str, to_system: str, geoid_offset: float = 0.0) -> float:
    """Transform a height value between orthometric and ellipsoidal systems."""
    src = from_system.strip().lower()
    dst = to_system.strip().lower()
    if src == dst:
        return float(value)
    if src == "ellipsoidal" and dst == "orthometric":
        return float(value) - float(geoid_offset)
    if src == "orthometric" and dst == "ellipsoidal":
        return float(value) + float(geoid_offset)
    return float(value)


def planetary_crs(body: str) -> CRSInfo:
    """Return a simple planetary CRS definition for Mars or the Moon."""
    normalized = body.strip().lower()
    if normalized == "mars":
        return custom_crs_definition("IAU:49900 Mars Geographic", geographic=True, unit="degree")
    if normalized in {"moon", "lunar"}:
        return custom_crs_definition("IAU:30100 Moon Geographic", geographic=True, unit="degree")
    return custom_crs_definition(f"Custom {body.title()} Geographic", geographic=True, unit="degree")


__all__ = [
    "CRSInfo",
    "albers_equal_area_crs",
    "auto_detect_coordinate_format",
    "azimuthal_equidistant_crs",
    "batch_coordinate_conversion_table",
    "batch_project_features",
    "best_projection_for_extent",
    "convert_linear_units",
    "coordinate_to_web_mercator",
    "coordinate_to_wgs84",
    "compound_crs",
    "convergence_angle",
    "crs_auto_detect",
    "crs_aware_area",
    "crs_aware_distance",
    "crs_comparison_report",
    "crs_equal",
    "crs_epsg_lookup",
    "crs_from_authority",
    "crs_from_epsg",
    "crs_from_esri_pe",
    "crs_from_json",
    "crs_from_prj_file",
    "crs_from_proj",
    "crs_from_user_input",
    "crs_from_wkt",
    "crs_information_report",
    "crs_to_json",
    "crs_to_proj",
    "crs_to_wkt",
    "custom_crs_definition",
    "custom_geographic_transform",
    "custom_vertical_transform",
    "define_projection",
    "equidistant_conic_crs",
    "geodesic_area_on_ellipsoid",
    "geodesic_densify_line",
    "geodesic_distance",
    "geodesic_midpoint",
    "geographic_to_projected",
    "gnomonic_crs",
    "engineering_local_crs",
    "height_system_info",
    "lambert_conformal_conic_crs",
    "meridian_convergence",
    "mollweide_crs",
    "normalize_antimeridian",
    "normalize_coordinate_order",
    "parse_coordinate_text",
    "parse_utm_coordinate",
    "planetary_crs",
    "polar_region_crs",
    "true_north_grid_north_correction",
    "project_on_the_fly",
    "projected_to_geographic",
    "reproject_geometry",
    "resolve_crs_conflicts",
    "rhumb_line_distance",
    "robinson_crs",
    "sinusoidal_crs",
    "state_plane_zone_auto_detect",
    "stereographic_crs",
    "scale_factor_at_point",
    "transformation_accuracy_metadata",
    "transformation_pipeline",
    "transverse_mercator_crs",
    "utm_zone_for_lonlat",
    "validate_crs_area_of_use",
    "vertical_transform",
    "validate_crs_bounds",
    "vertical_crs_from_epsg",
    "web_mercator_crs",
    "xy_table_to_geographic_features",
]
