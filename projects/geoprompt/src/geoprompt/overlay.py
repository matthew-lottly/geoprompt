from __future__ import annotations

import importlib
from typing import Any

from .geometry import Geometry, geometry_type


def _load_shapely() -> tuple[Any, Any, Any]:
    try:
        shapely_geometry = importlib.import_module("shapely.geometry")
        shapely_ops = importlib.import_module("shapely.ops")
    except ImportError as exc:
        raise RuntimeError("Install overlay support with 'pip install -e .[overlay]' before using clip or overlay operations.") from exc

    return shapely_geometry, shapely_geometry.shape, shapely_ops.unary_union


def geometry_to_geojson(geometry: Geometry) -> dict[str, Any]:
    geometry_kind = geometry_type(geometry)
    coordinates = geometry["coordinates"]
    if geometry_kind == "Point":
        return {"type": "Point", "coordinates": list(coordinates)}
    if geometry_kind == "LineString":
        return {"type": "LineString", "coordinates": [list(coord) for coord in coordinates]}
    if geometry_kind == "Polygon":
        return {"type": "Polygon", "coordinates": [[list(coord) for coord in coordinates]]}
    raise TypeError(f"unsupported geometry type: {geometry_kind}")


def geometry_to_shapely(geometry: Geometry) -> Any:
    _, shape, _ = _load_shapely()
    return shape(geometry_to_geojson(geometry))


def geometry_from_shapely(value: Any) -> list[Geometry]:
    if value.is_empty:
        return []

    geometry_kind = value.geom_type
    if geometry_kind == "Point":
        return [{"type": "Point", "coordinates": (float(value.x), float(value.y))}]
    if geometry_kind == "LineString":
        return [{"type": "LineString", "coordinates": tuple((float(x_value), float(y_value)) for x_value, y_value in value.coords)}]
    if geometry_kind == "Polygon":
        return [
            {
                "type": "Polygon",
                "coordinates": tuple((float(x_value), float(y_value)) for x_value, y_value in value.exterior.coords),
            }
        ]
    if geometry_kind.startswith("Multi") or geometry_kind == "GeometryCollection":
        geometries: list[Geometry] = []
        for child in value.geoms:
            geometries.extend(geometry_from_shapely(child))
        return geometries
    raise TypeError(f"unsupported Shapely geometry type: {geometry_kind}")


def clip_geometries(geometries: list[Geometry], mask_geometries: list[Geometry]) -> list[list[Geometry]]:
    _, _, unary_union = _load_shapely()
    mask_shape = unary_union([geometry_to_shapely(geometry) for geometry in mask_geometries])
    clipped: list[list[Geometry]] = []
    for geometry in geometries:
        result = geometry_to_shapely(geometry).intersection(mask_shape)
        clipped.append(geometry_from_shapely(result))
    return clipped


def overlay_intersections(left_geometries: list[Geometry], right_geometries: list[Geometry]) -> list[tuple[int, int, list[Geometry]]]:
    intersections: list[tuple[int, int, list[Geometry]]] = []
    for left_index, left_geometry in enumerate(left_geometries):
        left_shape = geometry_to_shapely(left_geometry)
        for right_index, right_geometry in enumerate(right_geometries):
            intersection = left_shape.intersection(geometry_to_shapely(right_geometry))
            exploded = geometry_from_shapely(intersection)
            if exploded:
                intersections.append((left_index, right_index, exploded))
    return intersections


__all__ = [
    "clip_geometries",
    "geometry_from_shapely",
    "geometry_to_geojson",
    "geometry_to_shapely",
    "overlay_intersections",
]