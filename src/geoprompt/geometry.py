"""Pure Python geometry predicates and transformations.

Provides Point-in-Polygon, line/polygon intersection, area/length calculation,
and centroid computation without external geometry libraries. All algorithms
implement standard computational geometry patterns.
"""
from __future__ import annotations

from typing import Any, Callable

from .equations import DistanceMethod, coordinate_distance


Coordinate = tuple[float, float]
Geometry = dict[str, object]


def _normalize_coordinate(value: Any) -> Coordinate:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError("coordinate must contain exactly two numeric values")
    return (float(value[0]), float(value[1]))


def _normalize_ring(values: Any) -> tuple[Coordinate, ...]:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        raise TypeError("polygon ring must contain at least three coordinates")
    ring = tuple(_normalize_coordinate(value) for value in values)
    if ring[0] != ring[-1]:
        ring = (*ring, ring[0])
    return ring


def _normalize_line(values: Any) -> tuple[Coordinate, ...]:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise TypeError("line geometry must contain at least two coordinates")
    return tuple(_normalize_coordinate(coord) for coord in values)


def _normalize_polygon(values: Any) -> tuple[Coordinate, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise TypeError("polygon geometry must contain at least one ring")
    first_value = values[0]
    if isinstance(first_value, (list, tuple)) and len(first_value) == 2 and all(isinstance(item, (int, float)) for item in first_value):
        return _normalize_ring(values)
    return _normalize_ring(first_value)


def normalize_geometry(value: Any) -> Geometry:
    if isinstance(value, dict):
        geometry_type = value.get("type")
        coordinates = value.get("coordinates")
        if geometry_type == "Point":
            return {"type": "Point", "coordinates": _normalize_coordinate(coordinates)}
        if geometry_type == "MultiPoint":
            if not isinstance(coordinates, (list, tuple)) or not coordinates:
                raise TypeError("multi-point geometry must contain at least one coordinate")
            return {
                "type": "MultiPoint",
                "coordinates": tuple(_normalize_coordinate(coord) for coord in coordinates),
            }
        if geometry_type == "LineString":
            return {
                "type": "LineString",
                "coordinates": _normalize_line(coordinates),
            }
        if geometry_type == "MultiLineString":
            if not isinstance(coordinates, (list, tuple)) or not coordinates:
                raise TypeError("multi-line geometry must contain at least one line")
            return {
                "type": "MultiLineString",
                "coordinates": tuple(_normalize_line(line) for line in coordinates),
            }
        if geometry_type == "Polygon":
            return {"type": "Polygon", "coordinates": _normalize_polygon(coordinates)}
        if geometry_type == "MultiPolygon":
            if not isinstance(coordinates, (list, tuple)) or not coordinates:
                raise TypeError("multi-polygon geometry must contain at least one polygon")
            return {
                "type": "MultiPolygon",
                "coordinates": tuple(_normalize_polygon(polygon) for polygon in coordinates),
            }
        raise TypeError("unsupported geometry type")

    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
        return {"type": "Point", "coordinates": _normalize_coordinate(value)}

    raise TypeError("geometry must be a point tuple or a GeoJSON-like geometry mapping")


def geometry_type(geometry: Geometry) -> str:
    return str(geometry["type"])


def _component_geometries(geometry: Geometry) -> tuple[Geometry, ...]:
    geometry_kind = geometry_type(geometry)
    coordinates = geometry["coordinates"]
    if geometry_kind in {"Point", "LineString", "Polygon"}:
        return (geometry,)
    if geometry_kind == "MultiPoint":
        return tuple({"type": "Point", "coordinates": coordinate} for coordinate in coordinates)  # type: ignore[arg-type]
    if geometry_kind == "MultiLineString":
        return tuple({"type": "LineString", "coordinates": tuple(line)} for line in coordinates)  # type: ignore[arg-type]
    if geometry_kind == "MultiPolygon":
        return tuple({"type": "Polygon", "coordinates": tuple(polygon)} for polygon in coordinates)  # type: ignore[arg-type]
    raise TypeError("unsupported geometry type")


def geometry_vertices(geometry: Geometry) -> tuple[Coordinate, ...]:
    geometry_kind = geometry_type(geometry)
    coordinates = geometry["coordinates"]
    if geometry_kind == "Point":
        return (coordinates,)  # type: ignore[return-value]
    if geometry_kind in {"LineString", "Polygon", "MultiPoint"}:
        return tuple(coordinates)  # type: ignore[return-value]
    if geometry_kind == "MultiLineString":
        return tuple(vertex for line in coordinates for vertex in line)  # type: ignore[return-value]
    if geometry_kind == "MultiPolygon":
        return tuple(vertex for polygon in coordinates for vertex in polygon)  # type: ignore[return-value]
    raise TypeError("unsupported geometry type")


def transform_geometry(geometry: Geometry, transform: Callable[[Coordinate], Coordinate]) -> Geometry:
    geometry_kind = geometry_type(geometry)
    if geometry_kind == "Point":
        return {"type": "Point", "coordinates": transform(geometry_vertices(geometry)[0])}
    if geometry_kind == "MultiPoint":
        return {
            "type": "MultiPoint",
            "coordinates": tuple(transform(vertex) for vertex in geometry_vertices(geometry)),
        }
    if geometry_kind == "LineString":
        return {"type": "LineString", "coordinates": tuple(transform(vertex) for vertex in geometry_vertices(geometry))}
    if geometry_kind == "MultiLineString":
        return {
            "type": "MultiLineString",
            "coordinates": tuple(
                tuple(transform(vertex) for vertex in part["coordinates"])
                for part in _component_geometries(geometry)
            ),
        }
    if geometry_kind == "Polygon":
        return {"type": "Polygon", "coordinates": tuple(transform(vertex) for vertex in geometry_vertices(geometry))}
    if geometry_kind == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": tuple(
                tuple(transform(vertex) for vertex in part["coordinates"])
                for part in _component_geometries(geometry)
            ),
        }
    raise TypeError("unsupported geometry type")


def geometry_bounds(geometry: Geometry) -> tuple[float, float, float, float]:
    vertices = geometry_vertices(geometry)
    xs = [coord[0] for coord in vertices]
    ys = [coord[1] for coord in vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def geometry_intersects_bounds(
    geometry: Geometry,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> bool:
    geometry_min_x, geometry_min_y, geometry_max_x, geometry_max_y = geometry_bounds(geometry)
    return not (
        geometry_max_x < min_x
        or geometry_min_x > max_x
        or geometry_max_y < min_y
        or geometry_min_y > max_y
    )


def geometry_within_bounds(
    geometry: Geometry,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> bool:
    geometry_min_x, geometry_min_y, geometry_max_x, geometry_max_y = geometry_bounds(geometry)
    return (
        geometry_min_x >= min_x
        and geometry_max_x <= max_x
        and geometry_min_y >= min_y
        and geometry_max_y <= max_y
    )


def _segments(vertices: tuple[Coordinate, ...]) -> tuple[tuple[Coordinate, Coordinate], ...]:
    return tuple((vertices[index - 1], vertices[index]) for index in range(1, len(vertices)))


def _cross_product(origin: Coordinate, middle: Coordinate, destination: Coordinate) -> float:
    return (middle[0] - origin[0]) * (destination[1] - origin[1]) - (middle[1] - origin[1]) * (destination[0] - origin[0])


def _point_on_segment(point: Coordinate, start: Coordinate, end: Coordinate, tolerance: float = 1e-9) -> bool:
    cross = _cross_product(start, point, end)
    if abs(cross) > tolerance:
        return False
    min_x, max_x = sorted((start[0], end[0]))
    min_y, max_y = sorted((start[1], end[1]))
    return min_x - tolerance <= point[0] <= max_x + tolerance and min_y - tolerance <= point[1] <= max_y + tolerance


def _segments_intersect(
    first_start: Coordinate,
    first_end: Coordinate,
    second_start: Coordinate,
    second_end: Coordinate,
    tolerance: float = 1e-9,
) -> bool:
    orientation_one = _cross_product(first_start, first_end, second_start)
    orientation_two = _cross_product(first_start, first_end, second_end)
    orientation_three = _cross_product(second_start, second_end, first_start)
    orientation_four = _cross_product(second_start, second_end, first_end)

    if ((orientation_one > tolerance and orientation_two < -tolerance) or (orientation_one < -tolerance and orientation_two > tolerance)) and (
        (orientation_three > tolerance and orientation_four < -tolerance)
        or (orientation_three < -tolerance and orientation_four > tolerance)
    ):
        return True

    return (
        _point_on_segment(second_start, first_start, first_end, tolerance=tolerance)
        or _point_on_segment(second_end, first_start, first_end, tolerance=tolerance)
        or _point_on_segment(first_start, second_start, second_end, tolerance=tolerance)
        or _point_on_segment(first_end, second_start, second_end, tolerance=tolerance)
    )


def _point_in_polygon(point: Coordinate, polygon: Geometry) -> bool:
    ring = geometry_vertices(polygon)
    for start, end in _segments(ring):
        if _point_on_segment(point, start, end):
            return True

    inside = False
    for index in range(len(ring) - 1):
        start_x, start_y = ring[index]
        end_x, end_y = ring[index + 1]
        intersects = (start_y > point[1]) != (end_y > point[1])
        if not intersects:
            continue
        edge_x = ((end_x - start_x) * (point[1] - start_y) / (end_y - start_y)) + start_x
        if point[0] <= edge_x:
            inside = not inside
    return inside


def geometry_intersects(origin: Geometry, destination: Geometry) -> bool:
    origin_parts = _component_geometries(origin)
    destination_parts = _component_geometries(destination)
    if len(origin_parts) > 1 or len(destination_parts) > 1:
        return any(
            geometry_intersects(origin_part, destination_part)
            for origin_part in origin_parts
            for destination_part in destination_parts
        )

    origin_type = geometry_type(origin)
    destination_type = geometry_type(destination)

    if origin_type == "Point" and destination_type == "Point":
        return geometry_vertices(origin)[0] == geometry_vertices(destination)[0]
    if origin_type == "Point" and destination_type == "LineString":
        point = geometry_vertices(origin)[0]
        return any(_point_on_segment(point, start, end) for start, end in _segments(geometry_vertices(destination)))
    if origin_type == "LineString" and destination_type == "Point":
        return geometry_intersects(destination, origin)
    if origin_type == "Point" and destination_type == "Polygon":
        return _point_in_polygon(geometry_vertices(origin)[0], destination)
    if origin_type == "Polygon" and destination_type == "Point":
        return geometry_intersects(destination, origin)
    if origin_type == "LineString" and destination_type == "LineString":
        return any(
            _segments_intersect(first_start, first_end, second_start, second_end)
            for first_start, first_end in _segments(geometry_vertices(origin))
            for second_start, second_end in _segments(geometry_vertices(destination))
        )
    if origin_type == "LineString" and destination_type == "Polygon":
        destination_ring = geometry_vertices(destination)
        return any(
            _segments_intersect(first_start, first_end, second_start, second_end)
            for first_start, first_end in _segments(geometry_vertices(origin))
            for second_start, second_end in _segments(destination_ring)
        ) or any(_point_in_polygon(vertex, destination) for vertex in geometry_vertices(origin))
    if origin_type == "Polygon" and destination_type == "LineString":
        return geometry_intersects(destination, origin)
    if origin_type == "Polygon" and destination_type == "Polygon":
        origin_ring = geometry_vertices(origin)
        destination_ring = geometry_vertices(destination)
        return any(
            _segments_intersect(first_start, first_end, second_start, second_end)
            for first_start, first_end in _segments(origin_ring)
            for second_start, second_end in _segments(destination_ring)
        ) or _point_in_polygon(origin_ring[0], destination) or _point_in_polygon(destination_ring[0], origin)

    raise TypeError("unsupported geometry type")


def geometry_within(candidate: Geometry, container: Geometry) -> bool:
    candidate_parts = _component_geometries(candidate)
    container_parts = _component_geometries(container)
    if len(candidate_parts) > 1:
        return all(geometry_within(part, container) for part in candidate_parts)
    if len(container_parts) > 1:
        return any(geometry_within(candidate, part) for part in container_parts)

    candidate_type = geometry_type(candidate)
    container_type = geometry_type(container)

    if container_type == "Polygon":
        vertices = geometry_vertices(candidate)
        if candidate_type == "Polygon":
            vertices = vertices[:-1]
        return all(_point_in_polygon(vertex, container) for vertex in vertices)
    if candidate_type == "Point" and container_type == "LineString":
        point = geometry_vertices(candidate)[0]
        return any(_point_on_segment(point, start, end) for start, end in _segments(geometry_vertices(container)))
    if candidate_type == "Point" and container_type == "Point":
        return geometry_vertices(candidate)[0] == geometry_vertices(container)[0]
    if candidate_type == "LineString" and container_type == "LineString":
        return all(
            any(_point_on_segment(vertex, start, end) for start, end in _segments(geometry_vertices(container)))
            for vertex in geometry_vertices(candidate)
        )
    if candidate_type == "Polygon" and container_type == "Polygon":
        return all(_point_in_polygon(vertex, container) for vertex in geometry_vertices(candidate)[:-1])
    return False


def geometry_contains(container: Geometry, candidate: Geometry) -> bool:
    return geometry_within(candidate, container)


def geometry_length(geometry: Geometry) -> float:
    geometry_kind = geometry_type(geometry)
    if geometry_kind in {"Point", "MultiPoint"}:
        return 0.0
    if geometry_kind in {"MultiLineString", "MultiPolygon"}:
        return sum(geometry_length(part) for part in _component_geometries(geometry))
    vertices = geometry_vertices(geometry)
    if len(vertices) < 2:
        return 0.0
    return sum(coordinate_distance(vertices[index - 1], vertices[index]) for index in range(1, len(vertices)))


def geometry_area(geometry: Geometry) -> float:
    geometry_kind = geometry_type(geometry)
    if geometry_kind == "MultiPolygon":
        return sum(geometry_area(part) for part in _component_geometries(geometry))
    if geometry_kind != "Polygon":
        return 0.0
    ring = geometry_vertices(geometry)
    area = 0.0
    for index in range(len(ring) - 1):
        x1, y1 = ring[index]
        x2, y2 = ring[index + 1]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def geometry_centroid(geometry: Geometry) -> Coordinate:
    geometry_kind = geometry_type(geometry)
    if geometry_kind == "Point":
        return geometry["coordinates"]  # type: ignore[return-value]

    if geometry_kind == "MultiPoint":
        vertices = geometry_vertices(geometry)
        return (
            sum(coord[0] for coord in vertices) / len(vertices),
            sum(coord[1] for coord in vertices) / len(vertices),
        )

    if geometry_kind == "LineString":
        vertices = geometry_vertices(geometry)
        total_length = 0.0
        centroid_x = 0.0
        centroid_y = 0.0
        for index in range(1, len(vertices)):
            start = vertices[index - 1]
            end = vertices[index]
            segment_length = coordinate_distance(start, end)
            if segment_length == 0.0:
                continue
            total_length += segment_length
            centroid_x += ((start[0] + end[0]) / 2.0) * segment_length
            centroid_y += ((start[1] + end[1]) / 2.0) * segment_length
        if total_length != 0.0:
            return (centroid_x / total_length, centroid_y / total_length)

    if geometry_kind == "MultiLineString":
        parts = _component_geometries(geometry)
        weights = [geometry_length(part) for part in parts]
        total_weight = sum(weights)
        if total_weight != 0.0:
            centroids = [geometry_centroid(part) for part in parts]
            return (
                sum(centroid[0] * weight for centroid, weight in zip(centroids, weights)) / total_weight,
                sum(centroid[1] * weight for centroid, weight in zip(centroids, weights)) / total_weight,
            )

    if geometry_kind == "Polygon":
        ring = geometry_vertices(geometry)
        area_factor = 0.0
        centroid_x = 0.0
        centroid_y = 0.0
        for index in range(len(ring) - 1):
            x1, y1 = ring[index]
            x2, y2 = ring[index + 1]
            cross = (x1 * y2) - (x2 * y1)
            area_factor += cross
            centroid_x += (x1 + x2) * cross
            centroid_y += (y1 + y2) * cross
        if area_factor != 0.0:
            factor = 1.0 / (3.0 * area_factor)
            return (centroid_x * factor, centroid_y * factor)

    if geometry_kind == "MultiPolygon":
        parts = _component_geometries(geometry)
        weights = [geometry_area(part) for part in parts]
        total_weight = sum(weights)
        if total_weight != 0.0:
            centroids = [geometry_centroid(part) for part in parts]
            return (
                sum(centroid[0] * weight for centroid, weight in zip(centroids, weights)) / total_weight,
                sum(centroid[1] * weight for centroid, weight in zip(centroids, weights)) / total_weight,
            )

    vertices = geometry_vertices(geometry)
    return (
        sum(coord[0] for coord in vertices) / len(vertices),
        sum(coord[1] for coord in vertices) / len(vertices),
    )


def geometry_distance(origin: Geometry, destination: Geometry, method: DistanceMethod = "euclidean") -> float:
    return coordinate_distance(geometry_centroid(origin), geometry_centroid(destination), method=method)


__all__ = [
    "Coordinate",
    "Geometry",
    "geometry_area",
    "geometry_bounds",
    "geometry_centroid",
    "geometry_contains",
    "geometry_intersects_bounds",
    "geometry_intersects",
    "geometry_distance",
    "geometry_length",
    "geometry_type",
    "geometry_vertices",
    "geometry_within",
    "geometry_within_bounds",
    "normalize_geometry",
    "transform_geometry",
]