"""Pure Python geometry predicates and transformations.

Provides Point-in-Polygon, line/polygon intersection, area/length calculation,
and centroid computation without external geometry libraries. All algorithms
implement standard computational geometry patterns.
"""
from __future__ import annotations

import importlib
import math
from typing import Any, Callable, Sequence

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


def _has_consecutive_duplicate_vertices(vertices: tuple[Coordinate, ...]) -> bool:
    return any(vertices[index] == vertices[index - 1] for index in range(1, len(vertices)))


def _dedupe_consecutive_vertices(vertices: tuple[Coordinate, ...]) -> tuple[Coordinate, ...]:
    deduped: list[Coordinate] = []
    for vertex in vertices:
        if not deduped or deduped[-1] != vertex:
            deduped.append(vertex)
    return tuple(deduped)


def _line_has_self_intersection(vertices: tuple[Coordinate, ...]) -> bool:
    segments = _segments(vertices)
    for index, (first_start, first_end) in enumerate(segments):
        for other_index in range(index + 1, len(segments)):
            if abs(index - other_index) <= 1:
                continue
            second_start, second_end = segments[other_index]
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return True
    return False


def _polygon_has_self_intersection(ring: tuple[Coordinate, ...]) -> bool:
    segments = _segments(ring)
    last_index = len(segments) - 1
    for index, (first_start, first_end) in enumerate(segments):
        for other_index in range(index + 1, len(segments)):
            if abs(index - other_index) <= 1 or (index == 0 and other_index == last_index):
                continue
            second_start, second_end = segments[other_index]
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return True
    return False


def validate_geometry(geometry: Geometry) -> dict[str, object]:
    normalized = normalize_geometry(geometry)
    geometry_kind = geometry_type(normalized)
    issues: list[str] = []

    if geometry_kind == "Point":
        pass
    elif geometry_kind == "MultiPoint":
        vertices = geometry_vertices(normalized)
        if len(set(vertices)) < len(vertices):
            issues.append("duplicate_vertices")
    elif geometry_kind == "LineString":
        vertices = geometry_vertices(normalized)
        if _has_consecutive_duplicate_vertices(vertices):
            issues.append("duplicate_vertices")
        if len(set(vertices)) < 2:
            issues.append("insufficient_distinct_vertices")
        if geometry_length(normalized) == 0.0:
            issues.append("zero_length")
        if len(vertices) >= 4 and _line_has_self_intersection(vertices):
            issues.append("self_intersection")
    elif geometry_kind == "Polygon":
        ring = geometry_vertices(normalized)
        if _has_consecutive_duplicate_vertices(ring):
            issues.append("duplicate_vertices")
        if len(set(ring[:-1])) < 3:
            issues.append("insufficient_distinct_vertices")
        if geometry_area(normalized) == 0.0:
            issues.append("zero_area")
        if len(ring) >= 4 and _polygon_has_self_intersection(ring):
            issues.append("self_intersection")
    elif geometry_kind in {"MultiLineString", "MultiPolygon"}:
        part_reports = [validate_geometry(part) for part in _component_geometries(normalized)]
        seen: set[str] = set()
        for report in part_reports:
            for issue in report["issues"]:  # type: ignore[index]
                if issue not in seen:
                    seen.add(issue)
                    issues.append(issue)

    suggestions: list[str] = []
    if "duplicate_vertices" in issues:
        suggestions.append("Use repair to remove repeated vertices.")
    if "insufficient_distinct_vertices" in issues or "zero_length" in issues or "zero_area" in issues:
        suggestions.append("Review the source coordinates and repair collapsed geometry parts.")
    if "self_intersection" in issues:
        suggestions.append("Use repair or simplify the polygon to remove self intersections.")

    return {
        "geometry_type": geometry_kind,
        "is_valid": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "suggested_fix": " ".join(suggestions) if suggestions else "No repair needed.",
    }


def repair_geometry(geometry: Geometry) -> Geometry:
    normalized = normalize_geometry(geometry)
    geometry_kind = geometry_type(normalized)

    if geometry_kind == "Point":
        return normalized
    if geometry_kind == "MultiPoint":
        seen: set[Coordinate] = set()
        coordinates: list[Coordinate] = []
        for vertex in geometry_vertices(normalized):
            if vertex in seen:
                continue
            seen.add(vertex)
            coordinates.append(vertex)
        return {"type": "MultiPoint", "coordinates": tuple(coordinates)}
    if geometry_kind == "LineString":
        vertices = _dedupe_consecutive_vertices(geometry_vertices(normalized))
        return {"type": "LineString", "coordinates": vertices}
    if geometry_kind == "MultiLineString":
        repaired_lines = []
        for part in _component_geometries(normalized):
            repaired = repair_geometry(part)
            coordinates = repaired["coordinates"]
            if len(coordinates) >= 2:  # type: ignore[arg-type]
                repaired_lines.append(tuple(coordinates))  # type: ignore[arg-type]
        return {"type": "MultiLineString", "coordinates": tuple(repaired_lines)}
    if geometry_kind == "Polygon":
        ring = _dedupe_consecutive_vertices(geometry_vertices(normalized))
        if ring and ring[0] != ring[-1]:
            ring = (*ring, ring[0])
        return {"type": "Polygon", "coordinates": ring}
    if geometry_kind == "MultiPolygon":
        repaired_polygons = []
        for part in _component_geometries(normalized):
            repaired = repair_geometry(part)
            repaired_polygons.append(tuple(repaired["coordinates"]))  # type: ignore[arg-type]
        return {"type": "MultiPolygon", "coordinates": tuple(repaired_polygons)}

    raise TypeError("unsupported geometry type")


def geometry_distance(origin: Geometry, destination: Geometry, method: DistanceMethod = "euclidean") -> float:
    return coordinate_distance(geometry_centroid(origin), geometry_centroid(destination), method=method)


def geometry_equals(origin: Geometry, destination: Geometry) -> bool:
    """Return ``True`` when two geometries are structurally identical."""
    return normalize_geometry(origin) == normalize_geometry(destination)


def geometry_disjoint(origin: Geometry, destination: Geometry) -> bool:
    """Return ``True`` when two geometries do not intersect."""
    return not geometry_intersects(origin, destination)


def geometry_covers(origin: Geometry, destination: Geometry) -> bool:
    """Return ``True`` when *origin* spatially covers *destination*."""
    return geometry_contains(origin, destination) or geometry_equals(origin, destination)


def geometry_covered_by(origin: Geometry, destination: Geometry) -> bool:
    """Return ``True`` when *origin* is covered by *destination*."""
    return geometry_covers(destination, origin)


def representative_point(geometry: Geometry) -> Geometry:
    """Return a point suitable for labeling or representative display."""
    centroid = geometry_centroid(geometry)
    point = {"type": "Point", "coordinates": centroid}
    if geometry_type(geometry) == "Polygon" and not geometry_within(point, geometry):
        first = geometry_vertices(geometry)[0]
        return {"type": "Point", "coordinates": first}
    return point


def _collapse_geometry_parts(parts: list[Geometry]) -> Geometry:
    if not parts:
        raise ValueError("geometry operation produced an empty result")
    if len(parts) == 1:
        return parts[0]
    part_types = {part["type"] for part in parts}
    if part_types == {"Point"}:
        return {"type": "MultiPoint", "coordinates": tuple(part["coordinates"] for part in parts)}
    if part_types == {"LineString"}:
        return {"type": "MultiLineString", "coordinates": tuple(part["coordinates"] for part in parts)}
    if part_types == {"Polygon"}:
        return {"type": "MultiPolygon", "coordinates": tuple(part["coordinates"] for part in parts)}
    return parts[0]


def geometry_boundary(geometry: Geometry) -> Geometry:
    """Return the boundary geometry for a supported input geometry."""
    normalized = normalize_geometry(geometry)
    kind = geometry_type(normalized)

    if kind == "Polygon":
        return {"type": "LineString", "coordinates": geometry_vertices(normalized)}
    if kind == "MultiPolygon":
        return {
            "type": "MultiLineString",
            "coordinates": tuple(part["coordinates"] for part in _component_geometries(normalized)),
        }
    if kind == "LineString":
        vertices = geometry_vertices(normalized)
        return {"type": "MultiPoint", "coordinates": (vertices[0], vertices[-1])}
    if kind == "MultiLineString":
        endpoints: list[Coordinate] = []
        for part in _component_geometries(normalized):
            vertices = geometry_vertices(part)
            endpoints.extend([vertices[0], vertices[-1]])
        return {"type": "MultiPoint", "coordinates": tuple(endpoints)}
    if kind == "Point":
        return {"type": "MultiPoint", "coordinates": ()}
    raise TypeError("unsupported geometry type")


def translate_geometry(geometry: Geometry, *, dx: float = 0.0, dy: float = 0.0) -> Geometry:
    """Translate a geometry by the given offset."""
    return transform_geometry(geometry, lambda coord: (coord[0] + float(dx), coord[1] + float(dy)))


def geometry_union_all(geometries: Sequence[Geometry]) -> Geometry:
    """Union a collection of geometries into a single combined geometry."""
    if not geometries:
        raise ValueError("geometries must not be empty")
    from .overlay import _load_shapely, geometry_from_shapely, geometry_to_shapely

    _, _, unary_union, _ = _load_shapely()
    merged = unary_union([geometry_to_shapely(normalize_geometry(geometry)) for geometry in geometries])
    return _collapse_geometry_parts(geometry_from_shapely(merged))


def geometry_polygonize(lines: Sequence[Geometry]) -> list[Geometry]:
    """Polygonize a set of input line geometries using Shapely."""
    if not lines:
        return []
    from .overlay import geometry_from_shapely, geometry_to_shapely

    ops = importlib.import_module("shapely.ops")
    shapes = [geometry_to_shapely(normalize_geometry(line)) for line in lines]
    polygons = ops.polygonize(shapes)
    results: list[Geometry] = []
    for polygon in polygons:
        results.extend(geometry_from_shapely(polygon))
    return results


def geometry_linemerge(geometries: Sequence[Geometry]) -> Geometry:
    """Merge connected line segments into a single line or multi-line."""
    if not geometries:
        raise ValueError("geometries must not be empty")
    from .overlay import geometry_from_shapely, geometry_to_shapely

    ops = importlib.import_module("shapely.ops")
    shapes = [geometry_to_shapely(normalize_geometry(geometry)) for geometry in geometries]
    merged = ops.linemerge(shapes)
    return _collapse_geometry_parts(geometry_from_shapely(merged))


def geometry_offset_curve(
    geometry: Geometry,
    distance: float,
    *,
    join_style: str = "round",
) -> Geometry:
    """Build a left/right offset curve from a line geometry."""
    from .overlay import geometry_from_shapely, geometry_to_shapely

    join_lookup = {"round": 1, "mitre": 2, "bevel": 3, "miter": 2}
    normalized = normalize_geometry(geometry)
    if geometry_type(normalized) != "LineString":
        raise TypeError("geometry_offset_curve requires a LineString geometry")
    line = geometry_to_shapely(normalized)
    result = line.offset_curve(distance, join_style=join_lookup.get(join_style.lower(), 1))
    return _collapse_geometry_parts(geometry_from_shapely(result))


def scale_geometry(
    geometry: Geometry,
    *,
    xfact: float = 1.0,
    yfact: float | None = None,
    origin: Coordinate = (0.0, 0.0),
) -> Geometry:
    """Scale a geometry around an origin."""
    y_scale = float(xfact if yfact is None else yfact)
    ox, oy = float(origin[0]), float(origin[1])
    x_scale = float(xfact)
    return transform_geometry(
        geometry,
        lambda coord: (ox + ((coord[0] - ox) * x_scale), oy + ((coord[1] - oy) * y_scale)),
    )


def rotate_geometry(
    geometry: Geometry,
    *,
    angle_degrees: float,
    origin: Coordinate = (0.0, 0.0),
) -> Geometry:
    """Rotate a geometry counterclockwise around an origin."""
    ox, oy = float(origin[0]), float(origin[1])
    angle = math.radians(float(angle_degrees))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    def _rotate(coord: Coordinate) -> Coordinate:
        tx = coord[0] - ox
        ty = coord[1] - oy
        return (
            ox + (tx * cos_a) - (ty * sin_a),
            oy + (tx * sin_a) + (ty * cos_a),
        )

    return transform_geometry(geometry, _rotate)


def skew_geometry(
    geometry: Geometry,
    *,
    x_angle_degrees: float = 0.0,
    y_angle_degrees: float = 0.0,
    origin: Coordinate = (0.0, 0.0),
) -> Geometry:
    """Skew a geometry along the x and/or y axis around an origin."""
    ox, oy = float(origin[0]), float(origin[1])
    tan_x = math.tan(math.radians(float(x_angle_degrees)))
    tan_y = math.tan(math.radians(float(y_angle_degrees)))

    def _skew(coord: Coordinate) -> Coordinate:
        tx = coord[0] - ox
        ty = coord[1] - oy
        return (
            ox + tx + (tan_x * ty),
            oy + ty + (tan_y * tx),
        )

    return transform_geometry(geometry, _skew)


def geometry_touches(origin: Geometry, destination: Geometry) -> bool:
    """Return ``True`` when *origin* and *destination* share boundary points but their interiors do not intersect.

    Two geometries "touch" if they have at least one boundary point in common
    but no interior points in common.
    """
    origin_parts = _component_geometries(origin)
    destination_parts = _component_geometries(destination)
    if len(origin_parts) > 1 or len(destination_parts) > 1:
        return any(
            geometry_touches(op, dp)
            for op in origin_parts
            for dp in destination_parts
        )

    if not geometry_intersects(origin, destination):
        return False

    origin_type = geometry_type(origin)
    destination_type = geometry_type(destination)

    # Point-Point: touching means identical (boundary is the point itself)
    if origin_type == "Point" and destination_type == "Point":
        return geometry_vertices(origin)[0] == geometry_vertices(destination)[0]

    # Point-LineString: touches if on an endpoint only
    if origin_type == "Point" and destination_type == "LineString":
        pt = geometry_vertices(origin)[0]
        line_verts = geometry_vertices(destination)
        return pt in (line_verts[0], line_verts[-1]) and not _interior_point_on_line(pt, line_verts)

    if origin_type == "LineString" and destination_type == "Point":
        return geometry_touches(destination, origin)

    # Point-Polygon: touches if on boundary ring
    if origin_type == "Point" and destination_type == "Polygon":
        pt = geometry_vertices(origin)[0]
        ring = geometry_vertices(destination)
        return any(_point_on_segment(pt, s, e) for s, e in _segments(ring)) and not _strict_point_in_polygon(pt, destination)

    if origin_type == "Polygon" and destination_type == "Point":
        return geometry_touches(destination, origin)

    # LineString-LineString: share boundary but not interior
    if origin_type == "LineString" and destination_type == "LineString":
        o_verts = geometry_vertices(origin)
        d_verts = geometry_vertices(destination)
        o_endpoints = {o_verts[0], o_verts[-1]}
        d_endpoints = {d_verts[0], d_verts[-1]}
        shared_endpoints = o_endpoints & d_endpoints
        if not shared_endpoints:
            return False
        # Check no interior crossing
        o_segs = _segments(o_verts)
        d_segs = _segments(d_verts)
        for os_s, os_e in o_segs:
            for ds_s, ds_e in d_segs:
                if _segments_cross_properly(os_s, os_e, ds_s, ds_e):
                    return False
        return True

    # Polygon-Polygon: share boundary but interiors disjoint
    if origin_type == "Polygon" and destination_type == "Polygon":
        o_ring = geometry_vertices(origin)
        d_ring = geometry_vertices(destination)
        # If any vertex of one is strictly inside the other, interiors overlap
        if any(_strict_point_in_polygon(v, destination) for v in o_ring[:-1]):
            return False
        if any(_strict_point_in_polygon(v, origin) for v in d_ring[:-1]):
            return False
        return True

    # LineString-Polygon: line touches polygon boundary only
    if origin_type == "LineString" and destination_type == "Polygon":
        o_verts = geometry_vertices(origin)
        if any(_strict_point_in_polygon(v, destination) for v in o_verts):
            return False
        return True

    if origin_type == "Polygon" and destination_type == "LineString":
        return geometry_touches(destination, origin)

    return False


def geometry_crosses(origin: Geometry, destination: Geometry) -> bool:
    """Return ``True`` when *origin* crosses *destination*.

    Two geometries cross if they share interior points but neither is
    contained in the other.  Crossing is meaningful for line/line and
    line/polygon combinations.
    """
    origin_parts = _component_geometries(origin)
    destination_parts = _component_geometries(destination)
    if len(origin_parts) > 1 or len(destination_parts) > 1:
        return any(
            geometry_crosses(op, dp)
            for op in origin_parts
            for dp in destination_parts
        )

    origin_type = geometry_type(origin)
    destination_type = geometry_type(destination)

    # Line-Line: cross if segments properly cross at least once
    if origin_type == "LineString" and destination_type == "LineString":
        o_segs = _segments(geometry_vertices(origin))
        d_segs = _segments(geometry_vertices(destination))
        return any(
            _segments_cross_properly(os_s, os_e, ds_s, ds_e)
            for os_s, os_e in o_segs
            for ds_s, ds_e in d_segs
        )

    # Line-Polygon: line crosses polygon if part is inside and part is outside
    if origin_type == "LineString" and destination_type == "Polygon":
        verts = geometry_vertices(origin)
        has_inside = any(_strict_point_in_polygon(v, destination) for v in verts)
        ring = geometry_vertices(destination)
        has_outside = any(not _point_in_polygon(v, destination) for v in verts)
        return has_inside and has_outside

    if origin_type == "Polygon" and destination_type == "LineString":
        return geometry_crosses(destination, origin)

    return False


def geometry_overlaps(origin: Geometry, destination: Geometry) -> bool:
    """Return ``True`` when *origin* and *destination* overlap.

    Two geometries of the same dimension overlap if they share some but
    not all interior points.
    """
    origin_parts = _component_geometries(origin)
    destination_parts = _component_geometries(destination)
    if len(origin_parts) > 1 or len(destination_parts) > 1:
        return any(
            geometry_overlaps(op, dp)
            for op in origin_parts
            for dp in destination_parts
        )

    origin_type = geometry_type(origin)
    destination_type = geometry_type(destination)

    # Overlaps applies to same-dimension geometries
    # Polygon-Polygon
    if origin_type == "Polygon" and destination_type == "Polygon":
        o_ring = geometry_vertices(origin)
        d_ring = geometry_vertices(destination)
        o_in_d = any(_strict_point_in_polygon(v, destination) for v in o_ring[:-1])
        d_in_o = any(_strict_point_in_polygon(v, origin) for v in d_ring[:-1])
        o_all_in_d = all(_point_in_polygon(v, destination) for v in o_ring[:-1])
        d_all_in_o = all(_point_in_polygon(v, origin) for v in d_ring[:-1])
        return (o_in_d or d_in_o) and not o_all_in_d and not d_all_in_o

    # LineString-LineString: share some interior but neither contained in other
    if origin_type == "LineString" and destination_type == "LineString":
        if not geometry_intersects(origin, destination):
            return False
        if geometry_within(origin, destination) or geometry_within(destination, origin):
            return False
        # They intersect and neither is contained — they overlap
        return True

    return False


def _interior_point_on_line(point: Coordinate, vertices: tuple[Coordinate, ...]) -> bool:
    """Return True if *point* lies on the interior (not endpoints) of a line."""
    for i in range(1, len(vertices)):
        start, end = vertices[i - 1], vertices[i]
        if _point_on_segment(point, start, end) and point != start and point != end:
            return True
    return False


def _strict_point_in_polygon(point: Coordinate, polygon: Geometry) -> bool:
    """Return True if *point* is strictly inside *polygon* (not on boundary)."""
    ring = geometry_vertices(polygon)
    for start, end in _segments(ring):
        if _point_on_segment(point, start, end):
            return False

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


def _segments_cross_properly(
    first_start: Coordinate,
    first_end: Coordinate,
    second_start: Coordinate,
    second_end: Coordinate,
    tolerance: float = 1e-9,
) -> bool:
    """Return True if two segments cross at an interior point (proper crossing)."""
    o1 = _cross_product(first_start, first_end, second_start)
    o2 = _cross_product(first_start, first_end, second_end)
    o3 = _cross_product(second_start, second_end, first_start)
    o4 = _cross_product(second_start, second_end, first_end)

    return (
        ((o1 > tolerance and o2 < -tolerance) or (o1 < -tolerance and o2 > tolerance))
        and ((o3 > tolerance and o4 < -tolerance) or (o3 < -tolerance and o4 > tolerance))
    )


def geometry_convex_hull(geometry: Geometry) -> Geometry:
    """Return the convex hull of a geometry.

    Uses the Graham scan algorithm for pure-Python computation.
    """
    verts = geometry_vertices(geometry)
    if len(verts) < 3:
        if len(verts) == 2:
            return {"type": "LineString", "coordinates": tuple(verts)}
        if len(verts) == 1:
            return {"type": "Point", "coordinates": verts[0]}
        return geometry

    # Graham scan
    pts = sorted(set(verts))
    if len(pts) < 3:
        if len(pts) == 2:
            return {"type": "LineString", "coordinates": tuple(pts)}
        return {"type": "Point", "coordinates": pts[0]}

    def _cross(o: Coordinate, a: Coordinate, b: Coordinate) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Coordinate] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Coordinate] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    hull.append(hull[0])
    return {"type": "Polygon", "coordinates": tuple(hull)}


def geometry_envelope(geometry: Geometry) -> Geometry:
    """Return the bounding rectangle (envelope) of a geometry."""
    min_x, min_y, max_x, max_y = geometry_bounds(geometry)
    coords = (
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y),
    )
    return {"type": "Polygon", "coordinates": coords}


def geometry_is_empty(geometry: Geometry) -> bool:
    """Return True if the geometry has no coordinates or is an empty collection."""
    gtype = geometry_type(geometry)
    coords = geometry.get("coordinates")
    if gtype == "GeometryCollection":
        geoms = geometry.get("geometries", [])
        return len(geoms) == 0
    if coords is None:
        return True
    if isinstance(coords, (list, tuple)) and len(coords) == 0:
        return True
    return False


def geometry_validity_reason(geometry: Geometry) -> str:
    """Return a human-readable reason if the geometry is invalid, or 'Valid' if OK."""
    report = validate_geometry(geometry)
    if report.get("is_valid"):
        return "Valid"
    issues = report.get("issues", [])
    if issues:
        return "; ".join(str(i) for i in issues)
    return "Invalid (unknown reason)"


def geometry_snap(geometry: Geometry, reference: Geometry, tolerance: float) -> Geometry:
    """Snap vertices of *geometry* to nearby vertices of *reference* within *tolerance*.

    Returns a new geometry with snapped coordinates.
    """
    ref_verts = geometry_vertices(reference)
    gtype = geometry_type(geometry)
    coords = geometry.get("coordinates")

    def _snap_coord(c: Coordinate) -> Coordinate:
        best = c
        best_d = tolerance
        for rv in ref_verts:
            d = math.hypot(c[0] - rv[0], c[1] - rv[1])
            if d < best_d:
                best = rv
                best_d = d
        return best

    if gtype == "Point":
        return {"type": "Point", "coordinates": _snap_coord(tuple(coords))}  # type: ignore[arg-type]
    if gtype == "LineString":
        new_coords = tuple(_snap_coord(tuple(c)) for c in coords)  # type: ignore[union-attr]
        return {"type": "LineString", "coordinates": new_coords}
    if gtype == "Polygon":
        ring = _normalize_polygon(coords)
        new_ring = tuple(_snap_coord(c) for c in ring)
        return {"type": "Polygon", "coordinates": new_ring}
    return geometry


def geometry_split(geometry: Geometry, splitter: Geometry) -> list[Geometry]:
    """Split a geometry by a splitting geometry.

    Delegates to Shapely's ``split`` when available.  Returns a list of
    result geometries (may be a single-element list if no split occurred).
    """
    try:
        overlay = importlib.import_module("shapely.ops")
        shp_geom = importlib.import_module("shapely.geometry")
    except ImportError:
        return [geometry]

    def _to_shapely(g: Geometry) -> Any:
        gt = geometry_type(g)
        c = g["coordinates"]
        if gt == "Point":
            return shp_geom.Point(c)
        if gt == "LineString":
            return shp_geom.LineString(c)
        if gt == "Polygon":
            return shp_geom.Polygon(c)
        raise TypeError(f"unsupported type for split: {gt}")

    def _from_shapely(s: Any) -> list[Geometry]:
        if s.is_empty:
            return []
        gt = s.geom_type
        if gt == "Point":
            return [{"type": "Point", "coordinates": (float(s.x), float(s.y))}]
        if gt == "LineString":
            return [{"type": "LineString", "coordinates": tuple((float(x), float(y)) for x, y in s.coords)}]
        if gt == "Polygon":
            return [{"type": "Polygon", "coordinates": tuple((float(x), float(y)) for x, y in s.exterior.coords)}]
        if hasattr(s, "geoms"):
            result: list[Geometry] = []
            for child in s.geoms:
                result.extend(_from_shapely(child))
            return result
        return []

    try:
        parts = overlay.split(_to_shapely(geometry), _to_shapely(splitter))
        return _from_shapely(parts)
    except Exception:
        return [geometry]


def geometry_voronoi(points: Sequence[Geometry]) -> list[Geometry]:
    """Build Voronoi polygons from a set of point geometries.

    Delegates to ``scipy.spatial.Voronoi``.  Returns a list of polygon
    geometries (one per input point where finite).
    """
    if len(points) < 2:
        return []

    try:
        spatial = importlib.import_module("scipy.spatial")
    except ImportError:
        return []

    coords = []
    for pt in points:
        if geometry_type(pt) != "Point":
            continue
        c = pt["coordinates"]
        coords.append([float(c[0]), float(c[1])])

    if len(coords) < 2:
        return []

    vor = spatial.Voronoi(coords)
    polygons: list[Geometry] = []
    for region_idx in vor.point_region:
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        ring = [tuple(vor.vertices[i]) for i in region]
        ring.append(ring[0])
        polygons.append({"type": "Polygon", "coordinates": tuple(ring)})
    return polygons


def geometry_delaunay(points: Sequence[Geometry]) -> list[Geometry]:
    """Build Delaunay triangles from a set of point geometries.

    Delegates to ``scipy.spatial.Delaunay``.
    """
    if len(points) < 3:
        return []

    try:
        spatial = importlib.import_module("scipy.spatial")
    except ImportError:
        return []

    coords = []
    for pt in points:
        if geometry_type(pt) != "Point":
            continue
        c = pt["coordinates"]
        coords.append([float(c[0]), float(c[1])])

    if len(coords) < 3:
        return []

    tri = spatial.Delaunay(coords)
    triangles: list[Geometry] = []
    for simplex in tri.simplices:
        ring = [tuple(coords[i]) for i in simplex]
        ring.append(ring[0])
        triangles.append({"type": "Polygon", "coordinates": tuple(ring)})
    return triangles


__all__ = [
    "Coordinate",
    "Geometry",
    "geometry_area",
    "geometry_bounds",
    "geometry_boundary",
    "geometry_centroid",
    "geometry_contains",
    "geometry_convex_hull",
    "geometry_delaunay",
    "geometry_envelope",
    "geometry_is_empty",
    "geometry_linemerge",
    "geometry_offset_curve",
    "geometry_polygonize",
    "geometry_covered_by",
    "geometry_covers",
    "geometry_crosses",
    "geometry_disjoint",
    "geometry_equals",
    "geometry_intersects_bounds",
    "geometry_intersects",
    "geometry_distance",
    "geometry_length",
    "geometry_overlaps",
    "geometry_snap",
    "geometry_split",
    "geometry_touches",
    "geometry_type",
    "geometry_union_all",
    "geometry_validity_reason",
    "geometry_vertices",
    "geometry_voronoi",
    "geometry_within",
    "geometry_within_bounds",
    "normalize_geometry",
    "repair_geometry",
    "representative_point",
    "rotate_geometry",
    "scale_geometry",
    "skew_geometry",
    "transform_geometry",
    "translate_geometry",
    "validate_geometry",
]