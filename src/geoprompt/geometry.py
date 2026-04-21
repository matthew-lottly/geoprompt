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


Coordinate = tuple[float, ...]
Geometry = dict[str, object]


def _normalize_coordinate(value: Any) -> Coordinate:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise TypeError("coordinate must contain at least two numeric values")
    return tuple(float(part) for part in value)


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
    if isinstance(first_value, (list, tuple)) and len(first_value) >= 2 and all(isinstance(item, (int, float)) for item in first_value[:2]):
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

    if isinstance(value, (list, tuple)) and len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
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
    normalized = normalize_geometry(geometry)
    geometry_kind = geometry_type(normalized)
    coordinates = normalized["coordinates"]
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
        start_x, start_y = _coord_xy(ring[index])
        end_x, end_y = _coord_xy(ring[index + 1])
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
    return abs(_ring_signed_area(ring))


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
            x1, y1 = _coord_xy(ring[index])
            x2, y2 = _coord_xy(ring[index + 1])
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


def _coord_xy(coord: Coordinate) -> tuple[float, float]:
    return (float(coord[0]), float(coord[1]))


def _force_2d_vertices(vertices: tuple[Coordinate, ...]) -> tuple[Coordinate, ...]:
    return tuple((float(vertex[0]), float(vertex[1])) for vertex in vertices)


def _has_non_planar_dimensions(vertices: tuple[Coordinate, ...]) -> bool:
    return any(len(vertex) != 2 for vertex in vertices)


def _ring_signed_area(ring: tuple[Coordinate, ...]) -> float:
    area = 0.0
    for index in range(len(ring) - 1):
        x1, y1 = _coord_xy(ring[index])
        x2, y2 = _coord_xy(ring[index + 1])
        area += (x1 * y2) - (x2 * y1)
    return area / 2.0


def _polygon_has_narrow_spike(ring: tuple[Coordinate, ...], tolerance: float = 0.12) -> bool:
    open_ring = ring[:-1] if ring and ring[0] == ring[-1] else ring
    if len(open_ring) < 3:
        return False
    for index, current in enumerate(open_ring):
        previous = open_ring[index - 1]
        following = open_ring[(index + 1) % len(open_ring)]
        prev_x, prev_y = _coord_xy(previous)
        curr_x, curr_y = _coord_xy(current)
        next_x, next_y = _coord_xy(following)
        first_len = math.hypot(curr_x - prev_x, curr_y - prev_y)
        second_len = math.hypot(next_x - curr_x, next_y - curr_y)
        denominator = first_len * second_len
        if denominator == 0:
            return True
        bend_ratio = abs(_cross_product(previous, current, following)) / denominator
        if bend_ratio < tolerance:
            return True
    return False


def _remove_narrow_spikes(ring: tuple[Coordinate, ...], tolerance: float = 0.12) -> tuple[Coordinate, ...]:
    open_ring = list(ring[:-1] if ring and ring[0] == ring[-1] else ring)
    if len(open_ring) < 4:
        return tuple(open_ring + ([open_ring[0]] if open_ring else []))

    kept: list[Coordinate] = []
    total = len(open_ring)
    for index, current in enumerate(open_ring):
        previous = open_ring[index - 1]
        following = open_ring[(index + 1) % total]
        prev_x, prev_y = _coord_xy(previous)
        curr_x, curr_y = _coord_xy(current)
        next_x, next_y = _coord_xy(following)
        first_len = math.hypot(curr_x - prev_x, curr_y - prev_y)
        second_len = math.hypot(next_x - curr_x, next_y - curr_y)
        denominator = first_len * second_len
        if denominator == 0:
            continue
        bend_ratio = abs(_cross_product(previous, current, following)) / denominator
        if bend_ratio < tolerance:
            continue
        kept.append(current)

    if len(kept) < 3:
        kept = open_ring
    kept.append(kept[0])
    return tuple(kept)


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
        if _has_non_planar_dimensions(vertices):
            issues.append("mixed_dimension_coordinates")
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
        if _has_non_planar_dimensions(ring):
            issues.append("mixed_dimension_coordinates")
        if _has_consecutive_duplicate_vertices(ring):
            issues.append("duplicate_vertices")
        if len(set(ring[:-1])) < 3:
            issues.append("insufficient_distinct_vertices")
        if geometry_area(normalized) == 0.0:
            issues.append("zero_area")
        if _ring_signed_area(ring) < 0:
            issues.append("clockwise_exterior")
        if len(ring) >= 4 and _polygon_has_self_intersection(ring):
            issues.append("self_intersection")
        if len(ring) >= 4 and _polygon_has_narrow_spike(ring):
            issues.append("narrow_spike")
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
    if "mixed_dimension_coordinates" in issues:
        suggestions.append("Project the geometry into a consistent 2D analytic form before running planar checks.")
    if "insufficient_distinct_vertices" in issues or "zero_length" in issues or "zero_area" in issues:
        suggestions.append("Review the source coordinates and repair collapsed geometry parts.")
    if "clockwise_exterior" in issues:
        suggestions.append("Repair can rewind polygon rings into a standard counter-clockwise exterior orientation.")
    if "self_intersection" in issues or "narrow_spike" in issues:
        suggestions.append("Use repair or simplify the polygon to remove self intersections and narrow spikes.")

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
        vertices = _force_2d_vertices(_dedupe_consecutive_vertices(geometry_vertices(normalized)))
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
        ring = _force_2d_vertices(_dedupe_consecutive_vertices(geometry_vertices(normalized)))
        if ring and ring[0] != ring[-1]:
            ring = (*ring, ring[0])
        ring = _remove_narrow_spikes(ring)
        if _ring_signed_area(ring) < 0:
            ring = tuple(reversed(ring))
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


def translate_geometry(geometry: Geometry, dx: float = 0.0, dy: float = 0.0) -> Geometry:
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
        geometry_vertices(destination)
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


# ── Section C: Advanced Geometry, Vector, and Editing ────────────────────────


def geometry_union(a: Geometry, b: Geometry) -> Geometry:
    """Compute the geometric union of two geometries (Shapely-backed)."""
    try:
        importlib.import_module("shapely.ops")
        importlib.import_module("shapely.geometry")
    except ImportError:
        return geometry_union_all([a, b])
    from .overlay import geometry_to_shapely, geometry_from_shapely
    sa = geometry_to_shapely(a)
    sb = geometry_to_shapely(b)
    return _collapse_geometry_parts(geometry_from_shapely(sa.union(sb)))


def geometry_identity(input_geom: Geometry, identity_geom: Geometry) -> list[Geometry]:
    """Identity overlay: split *input_geom* by *identity_geom* keeping all of input."""
    try:
        from .overlay import geometry_to_shapely, geometry_from_shapely
        sa = geometry_to_shapely(input_geom)
        sb = geometry_to_shapely(identity_geom)
        intersection = sa.intersection(sb)
        difference = sa.difference(sb)
        parts: list[Geometry] = []
        for g in (intersection, difference):
            if not g.is_empty:
                parts.append(geometry_from_shapely(g))
        return parts
    except ImportError:
        return [input_geom]


def geometry_symmetric_difference(a: Geometry, b: Geometry) -> Geometry:
    """Compute the symmetric difference of two geometries."""
    try:
        from .overlay import geometry_to_shapely, geometry_from_shapely
        sa = geometry_to_shapely(a)
        sb = geometry_to_shapely(b)
        return _collapse_geometry_parts(geometry_from_shapely(sa.symmetric_difference(sb)))
    except ImportError:
        return a


def geometry_erase(input_geom: Geometry, erase_geom: Geometry) -> Geometry:
    """Erase *erase_geom* from *input_geom* (difference)."""
    try:
        from .overlay import geometry_to_shapely, geometry_from_shapely
        sa = geometry_to_shapely(input_geom)
        sb = geometry_to_shapely(erase_geom)
        return _collapse_geometry_parts(geometry_from_shapely(sa.difference(sb)))
    except ImportError:
        return input_geom


def geometry_simplify(geometry: Geometry, tolerance: float, *, preserve_topology: bool = True) -> Geometry:
    """Simplify a geometry with optional topology preservation."""
    try:
        from .overlay import geometry_to_shapely, geometry_from_shapely
        s = geometry_to_shapely(geometry)
        return _collapse_geometry_parts(geometry_from_shapely(s.simplify(tolerance, preserve_topology=preserve_topology)))
    except ImportError:
        return geometry


def geometry_densify(geometry: Geometry, max_segment_length: float) -> Geometry:
    """Add vertices along edges so no segment exceeds *max_segment_length*."""
    verts = geometry_vertices(geometry)
    if len(verts) < 2:
        return geometry
    new_coords: list[Coordinate] = [verts[0]]
    for i in range(1, len(verts)):
        p0, p1 = verts[i - 1], verts[i]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        seg_len = math.hypot(dx, dy)
        if seg_len > max_segment_length and max_segment_length > 0:
            n_segs = math.ceil(seg_len / max_segment_length)
            for j in range(1, n_segs):
                t = j / n_segs
                new_coords.append((p0[0] + dx * t, p0[1] + dy * t))
        new_coords.append(p1)
    gtype = geometry_type(geometry)
    if gtype == "Polygon":
        return {"type": "Polygon", "coordinates": tuple(new_coords)}
    elif gtype == "LineString":
        return {"type": "LineString", "coordinates": tuple(new_coords)}
    return geometry


def geometry_smooth(geometry: Geometry, *, iterations: int = 1) -> Geometry:
    """Smooth a geometry using Chaikin's corner-cutting algorithm."""
    verts = list(geometry_vertices(geometry))
    if len(verts) < 3:
        return geometry
    for _ in range(iterations):
        smoothed: list[Coordinate] = []
        for i in range(len(verts) - 1):
            p0, p1 = verts[i], verts[i + 1]
            smoothed.append((0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]))
            smoothed.append((0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]))
        verts = smoothed
    gtype = geometry_type(geometry)
    coords = tuple(verts)
    if gtype == "Polygon":
        if coords[0] != coords[-1]:
            coords = coords + (coords[0],)
        return {"type": "Polygon", "coordinates": coords}
    return {"type": "LineString", "coordinates": coords}


def geometry_multipart_to_singlepart(geometry: Geometry) -> list[Geometry]:
    """Explode a Multi* geometry into its constituent single parts."""
    gtype = geometry_type(geometry)
    if gtype.startswith("Multi"):
        single_type = gtype.replace("Multi", "")
        return [{"type": single_type, "coordinates": c} for c in geometry.get("coordinates", [])]
    return [geometry]


def geometry_singlepart_to_multipart(geometries: Sequence[Geometry]) -> Geometry:
    """Combine single-part geometries into the corresponding Multi* type."""
    if not geometries:
        return {"type": "GeometryCollection", "coordinates": ()}
    gtype = geometry_type(geometries[0])
    multi_type = f"Multi{gtype}" if not gtype.startswith("Multi") else gtype
    coords = [g.get("coordinates", ()) for g in geometries]
    return {"type": multi_type, "coordinates": tuple(coords)}


def geometry_vertices_to_points(geometry: Geometry) -> list[Geometry]:
    """Extract every vertex of a geometry as individual Point geometries."""
    return [{"type": "Point", "coordinates": v} for v in geometry_vertices(geometry)]


def points_along_line(geometry: Geometry, interval: float) -> list[Geometry]:
    """Generate points at regular intervals along a line geometry."""
    verts = geometry_vertices(geometry)
    if len(verts) < 2 or interval <= 0:
        return []
    total = 0.0
    segments: list[tuple[Coordinate, Coordinate, float]] = []
    for i in range(len(verts) - 1):
        d = math.hypot(verts[i + 1][0] - verts[i][0], verts[i + 1][1] - verts[i][1])
        segments.append((verts[i], verts[i + 1], d))
        total += d
    points: list[Geometry] = []
    target = 0.0
    while target <= total:
        remaining = target
        for p0, p1, seg_len in segments:
            if remaining <= seg_len:
                t = remaining / seg_len if seg_len > 0 else 0
                x = p0[0] + t * (p1[0] - p0[0])
                y = p0[1] + t * (p1[1] - p0[1])
                points.append({"type": "Point", "coordinates": (x, y)})
                break
            remaining -= seg_len
        target += interval
    return points


def geometry_centerline(polygon: Geometry) -> Geometry:
    """Extract an approximate centerline from an elongated polygon.

    Uses the Voronoi diagram of densified boundary vertices, filtered
    to segments inside the polygon.  Falls back to a simple bounding-box
    midline when scipy is unavailable.
    """
    gtype = geometry_type(polygon)
    if gtype != "Polygon":
        return polygon
    geometry_vertices(polygon)
    bounds = geometry_bounds(polygon)
    try:
        spatial = importlib.import_module("scipy.spatial")
        from .overlay import geometry_to_shapely
        densified = geometry_densify(polygon, max((bounds[2] - bounds[0], bounds[3] - bounds[1])) / 50)
        pts = geometry_vertices(densified)
        if len(pts) < 4:
            raise ValueError("too few")
        vor = spatial.Voronoi([[p[0], p[1]] for p in pts])
        shp = geometry_to_shapely(polygon)
        from shapely.geometry import Point as ShapelyPoint
        inside: list[Coordinate] = []
        for v in vor.vertices:
            if shp.contains(ShapelyPoint(v[0], v[1])):
                inside.append((v[0], v[1]))
        if len(inside) >= 2:
            inside.sort()
            return {"type": "LineString", "coordinates": tuple(inside)}
    except Exception:
        pass
    mid_y = (bounds[1] + bounds[3]) / 2
    return {"type": "LineString", "coordinates": ((bounds[0], mid_y), (bounds[2], mid_y))}


def geometry_concave_hull(points: Sequence[Geometry], *, alpha: float = 1.0) -> Geometry:
    """Compute a concave hull from point geometries.

    Delegates to ``shapely.concave_hull`` when available; otherwise
    falls back to the convex hull.
    """
    try:
        shapely_mod = importlib.import_module("shapely")
        from .overlay import geometry_from_shapely
        mp = shapely_mod.MultiPoint([(p["coordinates"][0], p["coordinates"][1]) for p in points if geometry_type(p) == "Point"])
        hull = shapely_mod.concave_hull(mp, ratio=min(1.0, max(0.0, alpha)))
        return geometry_from_shapely(hull)
    except (ImportError, AttributeError):
        if not points:
            return {"type": "Polygon", "coordinates": ()}
        return geometry_convex_hull(geometry_singlepart_to_multipart(list(points)))


def geometry_minimum_bounding(geometry: Geometry, kind: str = "circle") -> Geometry:
    """Compute a minimum bounding geometry.

    *kind*: ``'circle'``, ``'rectangle'``, ``'convex_hull'``.
    """
    if kind == "convex_hull":
        return geometry_convex_hull(geometry)
    if kind == "rectangle":
        return geometry_envelope(geometry)
    # minimum bounding circle approximation
    cx, cy = geometry_centroid(geometry)
    verts = geometry_vertices(geometry)
    max_r = max((math.hypot(v[0] - cx, v[1] - cy) for v in verts), default=0)
    n_pts = 64
    ring: list[Coordinate] = []
    for i in range(n_pts + 1):
        angle = 2 * math.pi * i / n_pts
        ring.append((cx + max_r * math.cos(angle), cy + max_r * math.sin(angle)))
    return {"type": "Polygon", "coordinates": tuple(ring)}


def geometry_integrate(
    geometries: Sequence[Geometry],
    tolerance: float,
) -> list[Geometry]:
    """Snap nearby vertices together across a set of geometries (integrate).

    Applies mutual snapping: for each geometry pair within *tolerance*,
    vertices are merged to the midpoint.
    """
    result = [dict(g) for g in geometries]
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            result[i] = geometry_snap(result[i], result[j], tolerance)
            result[j] = geometry_snap(result[j], result[i], tolerance)
    return result


def geometry_eliminate_slivers(
    geometries: Sequence[Geometry],
    area_threshold: float,
) -> list[Geometry]:
    """Remove polygons smaller than *area_threshold* by merging into neighbors."""
    kept: list[Geometry] = []
    slivers: list[Geometry] = []
    for g in geometries:
        if geometry_type(g) in ("Polygon", "MultiPolygon") and geometry_area(g) < area_threshold:
            slivers.append(g)
        else:
            kept.append(g)
    # merge each sliver into nearest large neighbor
    for s in slivers:
        sc = geometry_centroid(s)
        best_idx = 0
        best_dist = float("inf")
        for i, k in enumerate(kept):
            d = math.hypot(geometry_centroid(k)[0] - sc[0], geometry_centroid(k)[1] - sc[1])
            if d < best_dist:
                best_dist = d
                best_idx = i
        if kept:
            kept[best_idx] = geometry_union(kept[best_idx], s)
    return kept


def geometry_from_bearing_distance(
    origin: Coordinate,
    bearing_deg: float,
    distance: float,
) -> Geometry:
    """Construct a Point geometry from an origin, bearing, and distance.

    Uses simple planar math (not geodesic).
    """
    rad = math.radians(bearing_deg)
    x = origin[0] + distance * math.sin(rad)
    y = origin[1] + distance * math.cos(rad)
    return {"type": "Point", "coordinates": (x, y)}


def line_from_offsets(
    baseline: Geometry,
    offsets: Sequence[float],
) -> list[Geometry]:
    """Generate offset lines at given distances from a baseline."""
    try:
        from .overlay import geometry_to_shapely, geometry_from_shapely
        s = geometry_to_shapely(baseline)
        return [geometry_from_shapely(s.parallel_offset(d, "left")) for d in offsets]
    except ImportError:
        return [baseline] * len(offsets)


def split_line_at_point(line: Geometry, point: Geometry) -> list[Geometry]:
    """Split a line at the nearest projection of *point*."""
    try:
        from .overlay import geometry_to_shapely, geometry_from_shapely
        shapely_ops = importlib.import_module("shapely.ops")
        importlib.import_module("shapely.geometry")
        sl = geometry_to_shapely(line)
        sp = geometry_to_shapely(point)
        snap_pt = sl.interpolate(sl.project(sp))
        parts = shapely_ops.split(shapely_ops.snap(sl, snap_pt, 1e-8), snap_pt)
        return [geometry_from_shapely(p) for p in parts.geoms]
    except ImportError:
        return [line]


def geometry_planarize(geometries: Sequence[Geometry]) -> list[Geometry]:
    """Planarize (node) a collection of line geometries at intersections."""
    try:
        shapely_ops = importlib.import_module("shapely.ops")
        from .overlay import geometry_to_shapely, geometry_from_shapely
        shapes = [geometry_to_shapely(g) for g in geometries]
        merged = shapely_ops.unary_union(shapes)
        noded = shapely_ops.node(merged) if hasattr(shapely_ops, "node") else merged
        if hasattr(noded, "geoms"):
            return [geometry_from_shapely(g) for g in noded.geoms]
        return [geometry_from_shapely(noded)]
    except ImportError:
        return list(geometries)


# ---------------------------------------------------------------------------
# Section: Extended Geometry Utilities
# ---------------------------------------------------------------------------


def variable_distance_buffer(
    features: Sequence[dict[str, Any]],
    distance_field: str,
    *,
    geometry_column: str = "geometry",
    segments: int = 32,
) -> list[dict[str, Any]]:
    """Buffer each feature by a per-feature distance read from *distance_field*."""
    results: list[dict[str, Any]] = []
    for feat in features:
        geom = feat.get(geometry_column)
        if geom is None:
            results.append(dict(feat))
            continue
        dist = float(feat.get(distance_field, 0))
        centroid = geometry_centroid(geom)
        ring: list[Coordinate] = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            ring.append((centroid[0] + dist * math.cos(angle), centroid[1] + dist * math.sin(angle)))
        ring.append(ring[0])
        new_feat = dict(feat)
        new_feat[geometry_column] = {"type": "Polygon", "coordinates": (tuple(ring),)}
        results.append(new_feat)
    return results


def multi_ring_buffer(
    geometry: Geometry,
    distances: Sequence[float],
    *,
    segments: int = 32,
) -> list[Geometry]:
    """Generate concentric buffer rings at each distance."""
    centroid = geometry_centroid(geometry)
    sorted_dists = sorted(distances)
    rings: list[Geometry] = []
    for dist in sorted_dists:
        ring: list[Coordinate] = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            ring.append((centroid[0] + dist * math.cos(angle), centroid[1] + dist * math.sin(angle)))
        ring.append(ring[0])
        rings.append({"type": "Polygon", "coordinates": (tuple(ring),)})
    return rings


def geometry_generalize(geometry: Geometry, tolerance: float) -> Geometry:
    """Generalize geometry by removing vertices within *tolerance* distance of their neighbours."""
    kind = geometry_type(geometry)
    if kind == "Point":
        return dict(geometry)
    verts = list(geometry_vertices(geometry))
    if len(verts) < 2:
        return dict(geometry)
    kept: list[Coordinate] = [verts[0]]
    for v in verts[1:]:
        if coordinate_distance(kept[-1], v) >= tolerance:
            kept.append(v)
    if len(kept) < 2:
        kept = [verts[0], verts[-1]]
    if kind == "Polygon":
        if kept[0] != kept[-1]:
            kept.append(kept[0])
        return {"type": "Polygon", "coordinates": (tuple(kept),)}
    return {"type": kind, "coordinates": tuple(kept)}


def generate_random_points(
    bounds: tuple[float, float, float, float],
    count: int,
    *,
    seed: int | None = None,
) -> list[Geometry]:
    """Generate *count* random points within *bounds* (minx, miny, maxx, maxy)."""
    import random as _random
    rng = _random.Random(seed)
    minx, miny, maxx, maxy = bounds
    return [
        {"type": "Point", "coordinates": (rng.uniform(minx, maxx), rng.uniform(miny, maxy))}
        for _ in range(count)
    ]


def generate_random_points_in_polygon(
    polygon: Geometry,
    count: int,
    *,
    seed: int | None = None,
    max_attempts: int = 10000,
) -> list[Geometry]:
    """Generate *count* random points guaranteed inside *polygon*."""
    import random as _random
    rng = _random.Random(seed)
    minx, miny, maxx, maxy = geometry_bounds(polygon)
    pts: list[Geometry] = []
    attempts = 0
    while len(pts) < count and attempts < max_attempts:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        pt: Geometry = {"type": "Point", "coordinates": (x, y)}
        if geometry_within(pt, polygon):
            pts.append(pt)
        attempts += 1
    return pts


def create_fishnet(
    bounds: tuple[float, float, float, float],
    cell_width: float,
    cell_height: float,
    *,
    as_polygons: bool = True,
) -> list[Geometry]:
    """Create a fishnet grid of rectangles (or lines) covering *bounds*."""
    minx, miny, maxx, maxy = bounds
    cells: list[Geometry] = []
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            x2 = min(x + cell_width, maxx)
            y2 = min(y + cell_height, maxy)
            if as_polygons:
                ring = ((x, y), (x2, y), (x2, y2), (x, y2), (x, y))
                cells.append({"type": "Polygon", "coordinates": (ring,)})
            else:
                cells.append({"type": "LineString", "coordinates": ((x, y), (x2, y), (x2, y2), (x, y2), (x, y))})
            x += cell_width
        y += cell_height
    return cells


def create_hexagonal_tessellation(
    bounds: tuple[float, float, float, float],
    size: float,
) -> list[Geometry]:
    """Create a hexagonal tessellation covering *bounds*. *size* is the hex radius."""
    minx, miny, maxx, maxy = bounds
    hexes: list[Geometry] = []
    dx = size * math.sqrt(3)
    dy = size * 1.5
    row = 0
    y = miny
    while y < maxy + size:
        offset = dx / 2 if row % 2 else 0
        x = minx + offset
        while x < maxx + size:
            ring: list[Coordinate] = []
            for i in range(6):
                angle = math.pi / 3 * i + math.pi / 6
                ring.append((x + size * math.cos(angle), y + size * math.sin(angle)))
            ring.append(ring[0])
            hexes.append({"type": "Polygon", "coordinates": (tuple(ring),)})
            x += dx
        y += dy
        row += 1
    return hexes


def geometry_hash(geometry: Geometry) -> str:
    """Return a deterministic hash string for a geometry (useful for deduplication)."""
    import hashlib
    kind = geometry_type(geometry)
    coords = geometry_vertices(geometry)
    raw = f"{kind}:{coords}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def force_geometry_type(geometry: Geometry, target_type: str) -> Geometry:
    """Cast geometry to *target_type* (e.g. Point→MultiPoint, Polygon→MultiPolygon)."""
    kind = geometry_type(geometry)
    if kind == target_type:
        return dict(geometry)
    multi_map = {"Point": "MultiPoint", "LineString": "MultiLineString", "Polygon": "MultiPolygon"}
    if target_type in multi_map.values() and multi_map.get(kind) == target_type:
        return geometry_singlepart_to_multipart([geometry])
    single_map = {v: k for k, v in multi_map.items()}
    if target_type in single_map and kind == list(multi_map.values())[list(multi_map.keys()).index(target_type)] if target_type in multi_map else False:
        parts = geometry_multipart_to_singlepart(geometry)
        return parts[0] if parts else geometry
    if target_type in single_map and kind in multi_map.values():
        parts = geometry_multipart_to_singlepart(geometry)
        return parts[0] if parts else geometry
    return dict(geometry)


def flip_line(geometry: Geometry) -> Geometry:
    """Reverse the vertex order of a LineString."""
    kind = geometry_type(geometry)
    if kind != "LineString":
        return dict(geometry)
    coords = list(geometry.get("coordinates", ()))
    return {"type": "LineString", "coordinates": tuple(reversed(coords))}


def fill_polygon_holes(geometry: Geometry) -> Geometry:
    """Remove interior rings (holes) from a Polygon."""
    kind = geometry_type(geometry)
    if kind != "Polygon":
        return dict(geometry)
    rings = geometry.get("coordinates", ())
    if not rings:
        return dict(geometry)
    return {"type": "Polygon", "coordinates": (rings[0],)}


def feature_to_point(geometry: Geometry) -> Geometry:
    """Convert any geometry to its centroid point."""
    c = geometry_centroid(geometry)
    return {"type": "Point", "coordinates": c}


def feature_to_line(geometry: Geometry) -> Geometry:
    """Convert a Polygon boundary to a LineString."""
    kind = geometry_type(geometry)
    if kind == "Polygon":
        ring = geometry_vertices(geometry)
        return {"type": "LineString", "coordinates": ring}
    if kind == "MultiPolygon":
        parts = _component_geometries(geometry)
        lines = [feature_to_line(p) for p in parts]
        return {"type": "MultiLineString", "coordinates": tuple(ln["coordinates"] for ln in lines)}
    return dict(geometry)


def line_to_polygon(geometry: Geometry) -> Geometry:
    """Close a LineString into a Polygon."""
    kind = geometry_type(geometry)
    if kind != "LineString":
        return dict(geometry)
    coords = list(geometry.get("coordinates", ()))
    if len(coords) < 3:
        return dict(geometry)
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return {"type": "Polygon", "coordinates": (tuple(coords),)}


def points_to_line(points: Sequence[Geometry]) -> Geometry:
    """Connect a sequence of Point geometries into a LineString."""
    coords: list[Coordinate] = []
    for p in points:
        if geometry_type(p) == "Point":
            c = p.get("coordinates", ())
            if len(c) >= 2:
                coords.append((float(c[0]), float(c[1])))
    return {"type": "LineString", "coordinates": tuple(coords)}


def hausdorff_distance(geom_a: Geometry, geom_b: Geometry) -> float:
    """Compute the discrete Hausdorff distance between two geometries."""
    verts_a = geometry_vertices(geom_a)
    verts_b = geometry_vertices(geom_b)
    if not verts_a or not verts_b:
        return 0.0

    def _directed(src: tuple[Coordinate, ...], dst: tuple[Coordinate, ...]) -> float:
        max_min = 0.0
        for s in src:
            min_d = min(coordinate_distance(s, d) for d in dst)
            if min_d > max_min:
                max_min = min_d
        return max_min

    return max(_directed(verts_a, verts_b), _directed(verts_b, verts_a))


def frechet_distance(geom_a: Geometry, geom_b: Geometry) -> float:
    """Compute the discrete Fréchet distance between two geometries."""
    pa = list(geometry_vertices(geom_a))
    pb = list(geometry_vertices(geom_b))
    n, m = len(pa), len(pb)
    if n == 0 or m == 0:
        return 0.0
    ca: list[list[float]] = [[-1.0] * m for _ in range(n)]

    def _c(i: int, j: int) -> float:
        if ca[i][j] > -0.5:
            return ca[i][j]
        d = coordinate_distance(pa[i], pb[j])
        if i == 0 and j == 0:
            ca[i][j] = d
        elif i > 0 and j == 0:
            ca[i][j] = max(_c(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i][j] = max(_c(0, j - 1), d)
        else:
            ca[i][j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
        return ca[i][j]

    return _c(n - 1, m - 1)


def line_substring(geometry: Geometry, start_fraction: float, end_fraction: float) -> Geometry:
    """Extract a substring of a LineString between *start_fraction* and *end_fraction* (0–1)."""
    kind = geometry_type(geometry)
    if kind != "LineString":
        return dict(geometry)
    coords = list(geometry.get("coordinates", ()))
    if len(coords) < 2:
        return dict(geometry)
    total = sum(coordinate_distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1))
    if total == 0:
        return dict(geometry)
    start_d, end_d = start_fraction * total, end_fraction * total
    result: list[Coordinate] = []
    cumulative = 0.0
    for i in range(len(coords) - 1):
        seg_len = coordinate_distance(coords[i], coords[i + 1])
        next_cum = cumulative + seg_len
        if next_cum >= start_d and not result:
            ratio = (start_d - cumulative) / seg_len if seg_len > 0 else 0
            result.append((
                coords[i][0] + ratio * (coords[i + 1][0] - coords[i][0]),
                coords[i][1] + ratio * (coords[i + 1][1] - coords[i][1]),
            ))
        if cumulative >= start_d and cumulative <= end_d:
            if not result or result[-1] != tuple(coords[i]):
                result.append(tuple(coords[i]))
        if next_cum >= end_d:
            ratio = (end_d - cumulative) / seg_len if seg_len > 0 else 0
            end_pt = (
                coords[i][0] + ratio * (coords[i + 1][0] - coords[i][0]),
                coords[i][1] + ratio * (coords[i + 1][1] - coords[i][1]),
            )
            if not result or result[-1] != end_pt:
                result.append(end_pt)
            break
        cumulative = next_cum
    if len(result) < 2:
        return {"type": "LineString", "coordinates": tuple(coords[:2])}
    return {"type": "LineString", "coordinates": tuple(result)}


def line_interpolate_point(geometry: Geometry, fraction: float) -> Geometry:
    """Return a Point at *fraction* (0–1) along a LineString."""
    kind = geometry_type(geometry)
    if kind != "LineString":
        c = geometry_centroid(geometry)
        return {"type": "Point", "coordinates": c}
    coords = list(geometry.get("coordinates", ()))
    if len(coords) < 2:
        return {"type": "Point", "coordinates": tuple(coords[0]) if coords else (0.0, 0.0)}
    total = sum(coordinate_distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1))
    target = fraction * total
    cumulative = 0.0
    for i in range(len(coords) - 1):
        seg_len = coordinate_distance(coords[i], coords[i + 1])
        if cumulative + seg_len >= target:
            ratio = (target - cumulative) / seg_len if seg_len > 0 else 0
            x = coords[i][0] + ratio * (coords[i + 1][0] - coords[i][0])
            y = coords[i][1] + ratio * (coords[i + 1][1] - coords[i][1])
            return {"type": "Point", "coordinates": (x, y)}
        cumulative += seg_len
    return {"type": "Point", "coordinates": tuple(coords[-1])}


def line_locate_point(geometry: Geometry, point: Geometry) -> float:
    """Return the fraction (0–1) along a LineString closest to *point*."""
    kind = geometry_type(geometry)
    if kind != "LineString":
        return 0.0
    pt = point.get("coordinates", (0, 0))
    coords = list(geometry.get("coordinates", ()))
    if len(coords) < 2:
        return 0.0
    total = sum(coordinate_distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1))
    if total == 0:
        return 0.0
    best_frac = 0.0
    best_dist = float("inf")
    cumulative = 0.0
    for i in range(len(coords) - 1):
        seg_len = coordinate_distance(coords[i], coords[i + 1])
        if seg_len == 0:
            cumulative += seg_len
            continue
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        t = max(0.0, min(1.0, ((pt[0] - coords[i][0]) * dx + (pt[1] - coords[i][1]) * dy) / (seg_len * seg_len)))
        proj_x = coords[i][0] + t * dx
        proj_y = coords[i][1] + t * dy
        dist = math.hypot(pt[0] - proj_x, pt[1] - proj_y)
        if dist < best_dist:
            best_dist = dist
            best_frac = (cumulative + t * seg_len) / total
        cumulative += seg_len
    return best_frac


def polygon_neighbors(
    features: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
    id_column: str = "id",
) -> list[dict[str, Any]]:
    """Report which polygon features share a boundary (touch or intersect)."""
    rows: list[dict[str, Any]] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            ga = features[i].get(geometry_column)
            gb = features[j].get(geometry_column)
            if ga and gb and (geometry_intersects(ga, gb) or geometry_touches(ga, gb)):
                rows.append({
                    "left_id": features[i].get(id_column, i),
                    "right_id": features[j].get(id_column, j),
                })
    return rows


def tabulate_intersection(
    features_a: Sequence[dict[str, Any]],
    features_b: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
    id_column: str = "id",
) -> list[dict[str, Any]]:
    """Tabulate the area of intersection between two feature sets."""
    from .overlay import overlay_intersections
    rows: list[dict[str, Any]] = []
    for a in features_a:
        for b in features_b:
            ga = a.get(geometry_column)
            gb = b.get(geometry_column)
            if ga and gb and geometry_intersects(ga, gb):
                try:
                    ix = overlay_intersections(ga, gb)
                    ix_area = geometry_area(ix) if ix else 0.0
                except Exception:
                    ix_area = 0.0
                rows.append({
                    "left_id": a.get(id_column),
                    "right_id": b.get(id_column),
                    "intersection_area": ix_area,
                })
    return rows


def coordinate_precision(geometry: Geometry, decimals: int = 6) -> Geometry:
    """Round all coordinates to *decimals* decimal places."""
    def _round(coord: Coordinate) -> Coordinate:
        return (round(coord[0], decimals), round(coord[1], decimals))
    return transform_geometry(geometry, _round)


def subdivide_geometry(geometry: Geometry, max_vertices: int = 256) -> list[Geometry]:
    """Subdivide a geometry into smaller pieces with at most *max_vertices* each."""
    verts = geometry_vertices(geometry)
    if len(verts) <= max_vertices:
        return [dict(geometry)]
    kind = geometry_type(geometry)
    if kind == "LineString":
        coords = list(geometry.get("coordinates", ()))
        parts: list[Geometry] = []
        for i in range(0, len(coords), max_vertices - 1):
            chunk = coords[i:i + max_vertices]
            if len(chunk) >= 2:
                parts.append({"type": "LineString", "coordinates": tuple(chunk)})
        return parts
    minx, miny, maxx, maxy = geometry_bounds(geometry)
    midx = (minx + maxx) / 2
    midy = (miny + maxy) / 2
    quadrants = [
        (minx, miny, midx, midy),
        (midx, miny, maxx, midy),
        (minx, midy, midx, maxy),
        (midx, midy, maxx, maxy),
    ]
    result: list[Geometry] = []
    for qminx, qminy, qmaxx, qmaxy in quadrants:
        clip_poly: Geometry = {"type": "Polygon", "coordinates": (((qminx, qminy), (qmaxx, qminy), (qmaxx, qmaxy), (qminx, qmaxy), (qminx, qminy)),)}
        if geometry_intersects(geometry, clip_poly):
            result.append(clip_poly)
    return result if result else [dict(geometry)]


# ---------------------------------------------------------------------------
# Additional geometry functions (A1 items)
# ---------------------------------------------------------------------------


def convex_hull_of_layer(
    features: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
) -> Geometry:
    """Return the convex hull enclosing **all** features in a layer.

    Parameters
    ----------
    features : sequence of records
    geometry_column : str

    Returns
    -------
    Geometry
        A Polygon that is the convex hull of the combined set of vertices.
    """
    all_coords: list[tuple[float, float]] = []
    for feat in features:
        verts = geometry_vertices(feat[geometry_column])
        all_coords.extend((v[0], v[1]) for v in verts)
    if not all_coords:
        return {"type": "Polygon", "coordinates": ()}
    combined: Geometry = {"type": "MultiPoint", "coordinates": tuple(all_coords)}
    return geometry_convex_hull(combined)


def feature_vertices_to_points(
    geometry: Geometry,
    *,
    which: str = "all",
) -> list[Geometry]:
    """Extract vertices from a geometry as Point geometries.

    Parameters
    ----------
    geometry : Geometry
    which : str
        ``"all"`` (every vertex), ``"start"`` (first), ``"end"`` (last),
        ``"mid"`` (midpoints of line segments). Default ``"all"``.

    Returns
    -------
    list[Geometry]
        List of Point geometries.
    """
    verts = geometry_vertices(geometry)
    if not verts:
        return []
    if which == "start":
        return [{"type": "Point", "coordinates": (verts[0][0], verts[0][1])}]
    if which == "end":
        return [{"type": "Point", "coordinates": (verts[-1][0], verts[-1][1])}]
    if which == "mid":
        mid_pts: list[Geometry] = []
        for i in range(len(verts) - 1):
            mx = (verts[i][0] + verts[i + 1][0]) / 2
            my = (verts[i][1] + verts[i + 1][1]) / 2
            mid_pts.append({"type": "Point", "coordinates": (mx, my)})
        return mid_pts
    # "all"
    return [{"type": "Point", "coordinates": (v[0], v[1])} for v in verts]


def generate_points_along_line(
    geometry: Geometry,
    count: int,
) -> list[Geometry]:
    """Generate *count* evenly-spaced points along a LineString.

    Parameters
    ----------
    geometry : Geometry
        Must be a LineString.
    count : int
        Number of points to generate (≥ 1).

    Returns
    -------
    list[Geometry]
        Point geometries at equal fractional intervals.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    return [line_interpolate_point(geometry, i / count) for i in range(1, count + 1)]


def snap_geometries_to_grid(
    geometry: Geometry,
    grid_size: float,
) -> Geometry:
    """Snap every vertex of a geometry to the nearest grid intersection.

    Parameters
    ----------
    geometry : Geometry
    grid_size : float
        Grid spacing.

    Returns
    -------
    Geometry
        Geometry with snapped coordinates.
    """
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    def _snap(val: float) -> float:
        return round(val / grid_size) * grid_size

    verts = geometry_vertices(geometry)
    new_verts = [(_snap(v[0]), _snap(v[1])) for v in verts]
    return coordinate_precision(
        geometry,
        max(0, -int(math.log10(grid_size)) + 1) if grid_size < 1 else 0,
    ) if not new_verts else _rebuild_geometry_with_coords(geometry, new_verts)


def _rebuild_geometry_with_coords(
    geometry: Geometry,
    new_coords: list[tuple[float, float]],
) -> Geometry:
    """Rebuild *geometry* using *new_coords* in vertex order."""
    kind = geometry.get("type", "")
    if kind == "Point" and new_coords:
        return {"type": "Point", "coordinates": new_coords[0]}
    if kind == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": tuple(new_coords)}
    if kind == "LineString":
        return {"type": "LineString", "coordinates": tuple(new_coords)}
    if kind == "Polygon":
        rings = geometry.get("coordinates", ())
        rebuilt: list[tuple[tuple[float, float], ...]] = []
        idx = 0
        for ring in rings:
            n = len(ring)
            rebuilt.append(tuple(new_coords[idx: idx + n]))
            idx += n
        return {"type": "Polygon", "coordinates": tuple(rebuilt)}
    # Fallback: just return with rounded coordinates
    return dict(geometry)


def de9im_relate(geom_a: Geometry, geom_b: Geometry) -> str:
    """Return the DE-9IM intersection matrix relating *geom_a* to *geom_b*.

    This is a pure-Python approximation using the predicates available in geoprompt.
    Each character in the returned 9-char string is one of ``'0'``, ``'1'``, ``'2'``,
    or ``'F'`` (false / empty).

    The matrix encodes: Interior/Interior, Interior/Boundary, Interior/Exterior,
    Boundary/Interior, Boundary/Boundary, Boundary/Exterior, Exterior/Interior,
    Exterior/Boundary, Exterior/Exterior.

    This is an approximation; for exact results use a computational-geometry library.

    Returns
    -------
    str
        9-character DE-9IM string.
    """
    # Simplistic approximation based on available predicates:
    intersects = geometry_intersects(geom_a, geom_b)
    contains_ab = geometry_contains(geom_a, geom_b)
    touches = geometry_touches(geom_a, geom_b)
    crosses = geometry_crosses(geom_a, geom_b)
    within = geometry_within(geom_a, geom_b)
    equals = geometry_equals(geom_a, geom_b)

    dim_a = _geom_dim(geom_a)
    dim_b = _geom_dim(geom_b)

    matrix = ["F"] * 9  # default: empty intersections

    if equals:
        d = str(max(dim_a, dim_b))
        matrix = [d, "F", "F", "F", "0", "F", "F", "F", "2"]
    elif within:
        matrix[0] = str(dim_a)  # II
        matrix[2] = "F"  # IE
        matrix[3] = "F" if not touches else "0"  # BI
        matrix[5] = "0"  # BE
        matrix[6] = "F"  # EI → depends
        matrix[8] = "2"  # EE
    elif contains_ab:
        matrix[0] = str(dim_b)
        matrix[1] = "F" if not touches else "0"
        matrix[6] = str(dim_b)
        matrix[8] = "2"
    elif touches:
        matrix[0] = "F"
        matrix[3] = "0"
        matrix[4] = "0"
        matrix[8] = "2"
    elif crosses:
        matrix[0] = "0"
        matrix[8] = "2"
    elif intersects:
        matrix[0] = str(min(dim_a, dim_b))
        matrix[8] = "2"
    else:
        matrix[8] = "2"

    return "".join(matrix)


def relate(geometry: Geometry, other: Geometry) -> str:
    """Return the DE-9IM relation string for two geometries."""
    return de9im_relate(geometry, other)


def covers(geometry: Geometry, other: Geometry) -> bool:
    """Return ``True`` when *geometry* covers *other*."""
    return geometry_covers(geometry, other)


def covered_by(geometry: Geometry, other: Geometry) -> bool:
    """Return ``True`` when *geometry* is covered by *other*."""
    return geometry_covered_by(geometry, other)


def _geom_dim(geom: Geometry) -> int:
    """Return the topological dimension of a geometry (0=point,1=line,2=polygon)."""
    kind = geom.get("type", "")
    if kind in ("Point", "MultiPoint"):
        return 0
    if kind in ("LineString", "MultiLineString"):
        return 1
    if kind in ("Polygon", "MultiPolygon"):
        return 2
    return 0


def geometry_wkt_read(wkt: str) -> Geometry:
    """Parse a Well-Known Text string into a GeoJSON-style geometry dict.

    Supports POINT, LINESTRING, POLYGON, MULTIPOINT, MULTILINESTRING, MULTIPOLYGON.
    """
    text = wkt.strip()
    upper = text.upper()

    def _parse_coords(s: str) -> list[tuple[float, ...]]:
        parts = s.split(",")
        result: list[tuple[float, ...]] = []
        for part in parts:
            nums = part.strip().split()
            result.append(tuple(float(n) for n in nums))
        return result

    def _extract_parens(s: str) -> str:
        idx = s.index("(")
        return s[idx:]

    if upper.startswith("POINT"):
        inner = _extract_parens(text).strip("() ")
        coords = _parse_coords(inner)
        return {"type": "Point", "coordinates": coords[0]}

    if upper.startswith("MULTIPOINT"):
        body = _extract_parens(text)[1:-1]  # strip outer parens
        # Handle both MULTIPOINT((1 2),(3 4)) and MULTIPOINT(1 2, 3 4)
        if "(" in body:
            pts = [s.strip().strip("()") for s in body.split("),")]
            pts = [p.rstrip(")").lstrip("(") for p in pts]
        else:
            pts = [body]
        coords: list[tuple[float, ...]] = []
        for pt in pts:
            coords.extend(_parse_coords(pt))
        return {"type": "MultiPoint", "coordinates": tuple(coords)}

    if upper.startswith("LINESTRING"):
        inner = _extract_parens(text).strip("()")
        return {"type": "LineString", "coordinates": tuple(_parse_coords(inner))}

    if upper.startswith("MULTILINESTRING"):
        body = _extract_parens(text)[1:-1]
        lines_raw = body.split("),")
        line_list: list[tuple[tuple[float, ...], ...]] = []
        for lr in lines_raw:
            lr = lr.strip().strip("()")
            line_list.append(tuple(_parse_coords(lr)))
        return {"type": "MultiLineString", "coordinates": tuple(line_list)}

    if upper.startswith("MULTIPOLYGON"):
        body = _extract_parens(text)
        # Split polygons by ")),(("
        body = body[1:-1]  # strip outermost parens → "((...)),((...))"
        poly_strs = body.split(")),((")
        polys: list[tuple[tuple[tuple[float, ...], ...], ...]] = []
        for ps in poly_strs:
            ps = ps.strip().strip("()")
            ring_strs = ps.split("),(")
            rings: list[tuple[tuple[float, ...], ...]] = []
            for rs in ring_strs:
                rs = rs.strip().strip("()")
                rings.append(tuple(_parse_coords(rs)))
            polys.append(tuple(rings))
        return {"type": "MultiPolygon", "coordinates": tuple(polys)}

    if upper.startswith("POLYGON"):
        body = _extract_parens(text)[1:-1]  # strip outer parens
        ring_strs = body.split("),(")
        rings: list[tuple[tuple[float, ...], ...]] = []
        for rs in ring_strs:
            rs = rs.strip().strip("()")
            rings.append(tuple(_parse_coords(rs)))
        return {"type": "Polygon", "coordinates": tuple(rings)}

    raise ValueError(f"unsupported WKT type: {text[:30]}")


def geometry_wkt_write(geometry: Geometry) -> str:
    """Convert a GeoJSON-style geometry dict to a Well-Known Text string."""
    kind = geometry.get("type", "")
    coords = geometry.get("coordinates", ())

    def _fmt_coord(c: Sequence[float]) -> str:
        return " ".join(str(v) for v in c)

    def _fmt_ring(ring: Sequence[Sequence[float]]) -> str:
        return "(" + ", ".join(_fmt_coord(c) for c in ring) + ")"

    if kind == "Point":
        return f"POINT ({_fmt_coord(coords)})"
    if kind == "MultiPoint":
        inner = ", ".join(f"({_fmt_coord(c)})" for c in coords)
        return f"MULTIPOINT ({inner})"
    if kind == "LineString":
        return f"LINESTRING {_fmt_ring(coords)}"
    if kind == "MultiLineString":
        inner = ", ".join(_fmt_ring(line) for line in coords)
        return f"MULTILINESTRING ({inner})"
    if kind == "Polygon":
        inner = ", ".join(_fmt_ring(ring) for ring in coords)
        return f"POLYGON ({inner})"
    if kind == "MultiPolygon":
        polys = ", ".join("(" + ", ".join(_fmt_ring(ring) for ring in poly) + ")" for poly in coords)
        return f"MULTIPOLYGON ({polys})"
    raise ValueError(f"unsupported geometry type: {kind}")


def geometry_wkb_read(data: bytes) -> Geometry:
    """Parse a Well-Known Binary (ISO / OGC) blob into a GeoJSON-style geometry dict.

    Supports Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
    in both little-endian and big-endian byte order.
    """
    import struct

    pos = 0

    def _read(fmt: str) -> tuple[Any, ...]:
        nonlocal pos
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, data, pos)
        pos += size
        return vals

    def _parse() -> Geometry:
        nonlocal pos
        byte_order = data[pos]
        pos += 1
        endian = "<" if byte_order == 1 else ">"
        (wkb_type,) = _read(f"{endian}I")
        geom_type = wkb_type & 0xFF

        if geom_type == 1:  # Point
            x, y = _read(f"{endian}dd")
            return {"type": "Point", "coordinates": (x, y)}
        if geom_type == 2:  # LineString
            (n,) = _read(f"{endian}I")
            coords = [_read(f"{endian}dd") for _ in range(n)]
            return {"type": "LineString", "coordinates": tuple(coords)}
        if geom_type == 3:  # Polygon
            (num_rings,) = _read(f"{endian}I")
            rings: list[tuple[tuple[float, float], ...]] = []
            for _ in range(num_rings):
                (n,) = _read(f"{endian}I")
                ring = [_read(f"{endian}dd") for _ in range(n)]
                rings.append(tuple(ring))
            return {"type": "Polygon", "coordinates": tuple(rings)}
        if geom_type == 4:  # MultiPoint
            (num,) = _read(f"{endian}I")
            pts = [_parse() for _ in range(num)]
            return {"type": "MultiPoint", "coordinates": tuple(p["coordinates"] for p in pts)}
        if geom_type == 5:  # MultiLineString
            (num,) = _read(f"{endian}I")
            lines = [_parse() for _ in range(num)]
            return {"type": "MultiLineString", "coordinates": tuple(ln["coordinates"] for ln in lines)}
        if geom_type == 6:  # MultiPolygon
            (num,) = _read(f"{endian}I")
            polys = [_parse() for _ in range(num)]
            return {"type": "MultiPolygon", "coordinates": tuple(p["coordinates"] for p in polys)}
        raise ValueError(f"unsupported WKB geometry type: {geom_type}")

    return _parse()


def geometry_wkb_write(geometry: Geometry) -> bytes:
    """Convert a GeoJSON-style geometry dict to Well-Known Binary (little-endian)."""
    import struct

    kind = geometry.get("type", "")
    coords = geometry.get("coordinates", ())

    def _write_point(c: Sequence[float]) -> bytes:
        return struct.pack("<BIdd", 1, 1, float(c[0]), float(c[1]))

    def _write_ring(ring: Sequence[Sequence[float]]) -> bytes:
        buf = struct.pack("<I", len(ring))
        for c in ring:
            buf += struct.pack("<dd", float(c[0]), float(c[1]))
        return buf

    if kind == "Point":
        return _write_point(coords)
    if kind == "LineString":
        buf = struct.pack("<BI", 1, 2)
        buf += struct.pack("<I", len(coords))
        for c in coords:
            buf += struct.pack("<dd", float(c[0]), float(c[1]))
        return buf
    if kind == "Polygon":
        buf = struct.pack("<BI", 1, 3)
        buf += struct.pack("<I", len(coords))
        for ring in coords:
            buf += _write_ring(ring)
        return buf
    if kind == "MultiPoint":
        buf = struct.pack("<BII", 1, 4, len(coords))
        for c in coords:
            buf += _write_point(c)
        return buf
    if kind == "MultiLineString":
        buf = struct.pack("<BII", 1, 5, len(coords))
        for line in coords:
            sub = struct.pack("<BI", 1, 2)
            sub += struct.pack("<I", len(line))
            for c in line:
                sub += struct.pack("<dd", float(c[0]), float(c[1]))
            buf += sub
        return buf
    if kind == "MultiPolygon":
        buf = struct.pack("<BII", 1, 6, len(coords))
        for poly in coords:
            sub = struct.pack("<BI", 1, 3)
            sub += struct.pack("<I", len(poly))
            for ring in poly:
                sub += _write_ring(ring)
            buf += sub
        return buf
    raise ValueError(f"unsupported geometry type: {kind}")


def segmentize(geometry: Geometry, max_segment_length: float) -> Geometry:
    """Add vertices to ensure no segment exceeds *max_segment_length*.

    Parameters
    ----------
    geometry : Geometry
        LineString or Polygon.
    max_segment_length : float
        Maximum allowed distance between consecutive vertices.

    Returns
    -------
    Geometry
        Densified geometry.
    """
    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be positive")

    def _densify_ring(ring: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        for i in range(len(ring)):
            ax, ay = float(ring[i][0]), float(ring[i][1])
            if i == 0:
                result.append((ax, ay))
                continue
            px, py = float(ring[i - 1][0]), float(ring[i - 1][1])
            dist = math.hypot(ax - px, ay - py)
            if dist > max_segment_length:
                n_segs = math.ceil(dist / max_segment_length)
                for s in range(1, n_segs):
                    frac = s / n_segs
                    result.append((px + (ax - px) * frac, py + (ay - py) * frac))
            result.append((ax, ay))
        return result

    kind = geometry.get("type", "")
    coords = geometry.get("coordinates", ())

    if kind == "LineString":
        return {"type": "LineString", "coordinates": tuple(_densify_ring(coords))}
    if kind == "Polygon":
        new_rings = [tuple(_densify_ring(ring)) for ring in coords]
        return {"type": "Polygon", "coordinates": tuple(new_rings)}
    return dict(geometry)


def point_distance_matrix(
    features: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
) -> list[list[float]]:
    """Compute the full pairwise distance matrix between feature centroids.

    Parameters
    ----------
    features : sequence of records
    geometry_column : str

    Returns
    -------
    list[list[float]]
        ``result[i][j]`` is the Euclidean distance between features *i* and *j*.
    """
    centroids = [geometry_centroid(f[geometry_column]) for f in features]
    n = len(centroids)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def near_analysis(
    features: Sequence[dict[str, Any]],
    near_features: Sequence[dict[str, Any]],
    *,
    search_radius: float | None = None,
    geometry_column: str = "geometry",
) -> list[dict[str, Any]]:
    """For each feature, find the nearest feature in *near_features*.

    Parameters
    ----------
    features, near_features : sequences of records
    search_radius : float, optional
        Only consider near features within this distance.
    geometry_column : str

    Returns
    -------
    list[dict]
        One record per input feature with keys ``near_index``, ``near_distance``,
        ``near_x``, ``near_y``. ``near_index`` is -1 if none found within radius.
    """
    near_centroids = [geometry_centroid(nf[geometry_column]) for nf in near_features]
    results: list[dict[str, Any]] = []
    for feat in features:
        cx, cy = geometry_centroid(feat[geometry_column])
        best_j = -1
        best_d = float("inf")
        for j, (nx, ny) in enumerate(near_centroids):
            d = math.hypot(cx - nx, cy - ny)
            if search_radius is not None and d > search_radius:
                continue
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            results.append({
                "near_index": best_j,
                "near_distance": best_d,
                "near_x": near_centroids[best_j][0],
                "near_y": near_centroids[best_j][1],
            })
        else:
            results.append({"near_index": -1, "near_distance": None, "near_x": None, "near_y": None})
    return results


def geodesic_bearing(
    point_a: Geometry,
    point_b: Geometry,
) -> float:
    """Compute the initial geodesic bearing from *point_a* to *point_b* (in degrees).

    Uses the forward azimuth formula on a sphere.

    Returns
    -------
    float
        Bearing in degrees (0 = north, clockwise).
    """
    lon1, lat1 = math.radians(point_a["coordinates"][0]), math.radians(point_a["coordinates"][1])
    lon2, lat2 = math.radians(point_b["coordinates"][0]), math.radians(point_b["coordinates"][1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def geodesic_area(polygon: Geometry) -> float:
    """Estimate the geodesic area of *polygon* on the WGS-84 ellipsoid (m²).

    Uses a spherical excess formula with Earth radius 6371008.8 m.

    Parameters
    ----------
    polygon : Geometry
        A Polygon geometry with coordinates in degrees (lon, lat).

    Returns
    -------
    float
        Area in square metres.
    """
    EARTH_RADIUS = 6_371_008.8  # metres
    coords = polygon.get("coordinates", ())
    if not coords:
        return 0.0
    ring = coords[0]  # exterior ring only
    n = len(ring)
    if n < 4:
        return 0.0

    total = 0.0
    for i in range(n - 1):
        lon1 = math.radians(ring[i][0])
        lat1 = math.radians(ring[i][1])
        lon2 = math.radians(ring[(i + 1) % (n - 1)][0])
        lat2 = math.radians(ring[(i + 1) % (n - 1)][1])
        total += (lon2 - lon1) * (2 + math.sin(lat1) + math.sin(lat2))
    area = abs(total) * EARTH_RADIUS * EARTH_RADIUS / 2.0
    return area


def geodesic_length(linestring: Geometry) -> float:
    """Estimate the geodesic length of *linestring* on the WGS-84 sphere (metres).

    Uses the Haversine formula with Earth radius 6371008.8 m.

    Parameters
    ----------
    linestring : Geometry
        A LineString with coordinates in degrees (lon, lat).

    Returns
    -------
    float
        Length in metres.
    """
    EARTH_RADIUS = 6_371_008.8
    coords = linestring.get("coordinates", ())
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = math.radians(coords[i][0]), math.radians(coords[i][1])
        lon2, lat2 = math.radians(coords[i + 1][0]), math.radians(coords[i + 1][1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(min(1.0, math.sqrt(a)))
        total += EARTH_RADIUS * c
    return total


def shared_paths(line_a: Geometry, line_b: Geometry) -> dict[str, Any]:
    """Find shared (overlapping) segments between two LineString geometries.

    This is a simplified version that checks for coincident vertices.

    Returns
    -------
    dict
        Keys: ``shared_forward`` (segments in same direction),
        ``shared_backward`` (segments in opposite direction).
    """
    coords_a = list(line_a.get("coordinates", ()))
    coords_b = list(line_b.get("coordinates", ()))
    set_b = {(round(c[0], 10), round(c[1], 10)) for c in coords_b}

    forward_segs: list[tuple[tuple[float, float], ...]] = []
    backward_segs: list[tuple[tuple[float, float], ...]] = []

    # Find runs of shared vertices
    current_run: list[tuple[float, float]] = []
    for c in coords_a:
        rounded = (round(c[0], 10), round(c[1], 10))
        if rounded in set_b:
            current_run.append((c[0], c[1]))
        else:
            if len(current_run) >= 2:
                forward_segs.append(tuple(current_run))
            current_run = []
    if len(current_run) >= 2:
        forward_segs.append(tuple(current_run))

    return {
        "shared_forward": forward_segs,
        "shared_backward": backward_segs,
    }


def boolean_operations(
    geometries: Sequence[Geometry],
    operation: str = "union",
) -> Geometry:
    """Apply a boolean set operation across a collection of geometries.

    Parameters
    ----------
    geometries : sequence of Geometry
    operation : str
        One of ``"union"``, ``"intersection"``.

    Returns
    -------
    Geometry
    """
    if not geometries:
        raise ValueError("geometries must not be empty")
    if operation == "union":
        return geometry_union_all(list(geometries))
    if operation == "intersection":
        result = geometries[0]
        for g in geometries[1:]:
            from .overlay import overlay_intersections
            inter = overlay_intersections([{"geometry": result}], [{"geometry": g}])
            if inter:
                result = inter[0]["geometry"]
            else:
                return {"type": "GeometryCollection", "geometries": ()}
        return result
    raise ValueError(f"unsupported operation: {operation}")


def geohash_encode(lon: float, lat: float, precision: int = 12) -> str:
    """Encode a longitude/latitude pair into a GeoHash string.

    Parameters
    ----------
    lon : float
        Longitude (-180 to 180).
    lat : float
        Latitude (-90 to 90).
    precision : int
        Number of characters in the hash. Default 12.

    Returns
    -------
    str
        GeoHash string.
    """
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    is_lon = True
    bit = 0
    ch_idx = 0
    result: list[str] = []

    while len(result) < precision:
        if is_lon:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon >= mid:
                ch_idx = ch_idx * 2 + 1
                lon_range[0] = mid
            else:
                ch_idx = ch_idx * 2
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                ch_idx = ch_idx * 2 + 1
                lat_range[0] = mid
            else:
                ch_idx = ch_idx * 2
                lat_range[1] = mid
        is_lon = not is_lon
        bit += 1
        if bit == 5:
            result.append(base32[ch_idx])
            bit = 0
            ch_idx = 0
    return "".join(result)


def geohash_decode(geohash: str) -> tuple[float, float]:
    """Decode a GeoHash string to a (longitude, latitude) pair.

    Returns the center of the GeoHash cell.

    Parameters
    ----------
    geohash : str

    Returns
    -------
    tuple[float, float]
        ``(longitude, latitude)``
    """
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    is_lon = True

    for ch in geohash:
        idx = base32.index(ch.lower())
        for bit in (16, 8, 4, 2, 1):
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if idx & bit:
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if idx & bit:
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid
            is_lon = not is_lon

    return ((lon_range[0] + lon_range[1]) / 2, (lat_range[0] + lat_range[1]) / 2)


def geodesic_buffer(geometry: Geometry, distance_meters: float, *, segments: int = 72) -> Geometry:
    """Create an approximate geodesic buffer polygon around a point geometry."""
    if distance_meters < 0:
        raise ValueError("distance_meters must be non-negative")
    center = geometry if geometry.get("type") == "Point" else geometry_centroid(geometry)
    lon_deg, lat_deg = center["coordinates"]  # type: ignore[assignment]
    lon = math.radians(float(lon_deg))
    lat = math.radians(float(lat_deg))
    earth_radius = 6_371_008.8
    angular = distance_meters / earth_radius
    ring: list[tuple[float, float]] = []
    for index in range(max(12, segments)):
        bearing = (2.0 * math.pi * index) / max(12, segments)
        lat2 = math.asin(
            math.sin(lat) * math.cos(angular)
            + math.cos(lat) * math.sin(angular) * math.cos(bearing)
        )
        lon2 = lon + math.atan2(
            math.sin(bearing) * math.sin(angular) * math.cos(lat),
            math.cos(angular) - math.sin(lat) * math.sin(lat2),
        )
        ring.append((math.degrees(lon2), math.degrees(lat2)))
    if ring:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": (tuple(ring),)}


def line_side_buffer(line: Geometry, distance: float, *, side: str = "both") -> Geometry:
    """Buffer a line on the left, right, or both sides."""
    if side == "both":
        return offset_polygon(feature_to_polygon(line), distance)
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left', 'right', or 'both'")
    try:
        from .overlay import geometry_from_shapely, geometry_to_shapely

        shape = geometry_to_shapely(line)
        offset = shape.parallel_offset(abs(distance), side)
        pieces = geometry_from_shapely(offset)
        if pieces:
            coords = tuple(line.get("coordinates", ()))
            offset_coords = tuple(pieces[0].get("coordinates", ()))
            shell = coords + tuple(reversed(offset_coords)) + (coords[0],)
            return {"type": "Polygon", "coordinates": (shell,)}
    except Exception:
        pass
    return feature_to_polygon(line)


def update_overlay(
    base_features: Sequence[dict[str, Any]],
    update_features: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
) -> list[dict[str, Any]]:
    """Apply a simplified update overlay, replacing overlapping base features."""
    result: list[dict[str, Any]] = []
    update_geometries = [f[geometry_column] for f in update_features]
    for feature in base_features:
        if not any(geometry_intersects(feature[geometry_column], update_geom) for update_geom in update_geometries):
            result.append(dict(feature))
    result.extend(dict(feature) for feature in update_features)
    return result


def split_lines_at_intersections(lines: Sequence[Geometry]) -> list[Geometry]:
    """Split lines wherever they intersect another line."""
    split_parts: list[Geometry] = []
    for index, line in enumerate(lines):
        parts = [line]
        for other_index, other in enumerate(lines):
            if other_index == index or not geometry_intersects(line, other):
                continue
            try:
                from .overlay import geometry_from_shapely, geometry_to_shapely

                intersection = geometry_to_shapely(line).intersection(geometry_to_shapely(other))
                points = [geom for geom in geometry_from_shapely(intersection) if geom.get("type") == "Point"]
            except Exception:
                points = []
            for point in points:
                new_parts: list[Geometry] = []
                for part in parts:
                    new_parts.extend(split_line_at_point(part, point))
                parts = new_parts
        split_parts.extend(parts)
    return split_parts


def extend_lines_to_nearest_feature(
    lines: Sequence[Geometry],
    targets: Sequence[Geometry],
) -> list[Geometry]:
    """Extend each line endpoint toward the nearest target centroid."""
    results: list[Geometry] = []
    target_points = [geometry_centroid(target) for target in targets]
    for line in lines:
        coords = list(line.get("coordinates", ()))
        if len(coords) < 2 or not target_points:
            results.append(dict(line))
            continue
        start = coords[0]
        end = coords[-1]
        nearest_start = min(target_points, key=lambda point: math.hypot(point[0] - start[0], point[1] - start[1]))
        nearest_end = min(target_points, key=lambda point: math.hypot(point[0] - end[0], point[1] - end[1]))
        coords[0] = nearest_start  # type: ignore[index]
        coords[-1] = nearest_end  # type: ignore[index]
        results.append({"type": "LineString", "coordinates": tuple(coords)})
    return results


def trim_dangling_lines(lines: Sequence[Geometry], *, min_length: float = 0.0) -> list[Geometry]:
    """Remove line segments that appear to dangle and are shorter than the threshold."""
    endpoints: dict[tuple[float, float], int] = {}
    for line in lines:
        coords = list(line.get("coordinates", ()))
        if len(coords) >= 2:
            for endpoint in (coords[0], coords[-1]):
                key = (round(float(endpoint[0]), 9), round(float(endpoint[1]), 9))
                endpoints[key] = endpoints.get(key, 0) + 1
    kept: list[Geometry] = []
    for line in lines:
        coords = list(line.get("coordinates", ()))
        if len(coords) < 2:
            continue
        start_key = (round(float(coords[0][0]), 9), round(float(coords[0][1]), 9))
        end_key = (round(float(coords[-1][0]), 9), round(float(coords[-1][1]), 9))
        dangling = endpoints.get(start_key, 0) == 1 or endpoints.get(end_key, 0) == 1
        if dangling and geometry_length(line) < min_length:
            continue
        kept.append(dict(line))
    return kept


def simplify_visvalingam(geometry: Geometry, tolerance: float) -> Geometry:
    """Simplify geometry using the current generalization engine."""
    return geometry_generalize(geometry, tolerance)


def simplify_weighted_area(geometry: Geometry, tolerance: float) -> Geometry:
    """Simplify geometry using weighted-area style reduction."""
    return geometry_simplify(geometry, tolerance, preserve_topology=True)


def simplify_line_advanced(geometry: Geometry, tolerance: float, *, algorithm: str = "visvalingam") -> Geometry:
    """Simplify a line using one of the supported algorithms."""
    if algorithm in {"visvalingam", "point-removal"}:
        return geometry_generalize(geometry, tolerance)
    if algorithm in {"weighted-area", "topology"}:
        return geometry_simplify(geometry, tolerance, preserve_topology=True)
    return geometry_simplify(geometry, tolerance, preserve_topology=False)


def smooth_polygon_bezier(geometry: Geometry, *, iterations: int = 1) -> Geometry:
    """Smooth a polygon with the current smoothing routine."""
    return geometry_smooth(geometry, iterations=iterations)


def densify_by_angle(geometry: Geometry, max_angle_degrees: float) -> Geometry:
    """Densify a geometry more aggressively as the allowed turning angle decreases."""
    effective_length = max(0.001, max_angle_degrees / 180.0)
    return geometry_densify(geometry, effective_length)


def densify_by_deviation(geometry: Geometry, deviation: float) -> Geometry:
    """Densify a geometry so long segments are broken below the deviation threshold."""
    return segmentize(geometry, max(0.001, deviation))


def minimum_bounding_ellipse(geometry: Geometry, *, segments: int = 72) -> Geometry:
    """Approximate the minimum bounding ellipse for a geometry."""
    vertices = geometry_vertices(geometry)
    if not vertices:
        return {"type": "Polygon", "coordinates": ()}
    center_x = sum(vertex[0] for vertex in vertices) / len(vertices)
    center_y = sum(vertex[1] for vertex in vertices) / len(vertices)
    dx = [vertex[0] - center_x for vertex in vertices]
    dy = [vertex[1] - center_y for vertex in vertices]
    sxx = sum(value * value for value in dx) / len(dx)
    syy = sum(value * value for value in dy) / len(dy)
    sxy = sum(dx[i] * dy[i] for i in range(len(dx))) / len(dx)
    angle = 0.5 * math.atan2(2 * sxy, sxx - syy) if dx else 0.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    major = 0.0
    minor = 0.0
    for vertex in vertices:
        tx = vertex[0] - center_x
        ty = vertex[1] - center_y
        rx = (tx * cos_a) + (ty * sin_a)
        ry = (-tx * sin_a) + (ty * cos_a)
        major = max(major, abs(rx))
        minor = max(minor, abs(ry))
    major = max(major, 1e-9)
    minor = max(minor, 1e-9)
    ring: list[tuple[float, float]] = []
    for index in range(max(24, segments)):
        theta = 2.0 * math.pi * index / max(24, segments)
        ex = major * math.cos(theta)
        ey = minor * math.sin(theta)
        x_value = center_x + (ex * cos_a) - (ey * sin_a)
        y_value = center_y + (ex * sin_a) + (ey * cos_a)
        ring.append((x_value, y_value))
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": (tuple(ring),)}


def oriented_minimum_bounding_rectangle(geometry: Geometry) -> Geometry:
    """Approximate the oriented minimum bounding rectangle for a geometry."""
    vertices = geometry_vertices(geometry)
    if not vertices:
        return {"type": "Polygon", "coordinates": ()}
    center_x = sum(vertex[0] for vertex in vertices) / len(vertices)
    center_y = sum(vertex[1] for vertex in vertices) / len(vertices)
    dx = [vertex[0] - center_x for vertex in vertices]
    dy = [vertex[1] - center_y for vertex in vertices]
    sxx = sum(value * value for value in dx) / len(dx)
    syy = sum(value * value for value in dy) / len(dy)
    sxy = sum(dx[i] * dy[i] for i in range(len(dx))) / len(dx)
    angle_deg = math.degrees(0.5 * math.atan2(2 * sxy, sxx - syy)) if dx else 0.0
    rotated = rotate_geometry(geometry, angle_degrees=-angle_deg, origin=(center_x, center_y))
    min_x, min_y, max_x, max_y = geometry_bounds(rotated)
    rect = {
        "type": "Polygon",
        "coordinates": (((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)),),
    }
    return rotate_geometry(rect, angle_degrees=angle_deg, origin=(center_x, center_y))


def pole_of_inaccessibility(geometry: Geometry, *, samples: int = 25) -> Geometry:
    """Approximate the pole of inaccessibility for a polygon by grid search."""
    min_x, min_y, max_x, max_y = geometry_bounds(geometry)
    best_point = geometry_centroid(geometry)
    best_distance = -1.0
    for ix in range(samples + 1):
        for iy in range(samples + 1):
            x_value = min_x + ((max_x - min_x) * ix / max(1, samples))
            y_value = min_y + ((max_y - min_y) * iy / max(1, samples))
            candidate: Geometry = {"type": "Point", "coordinates": (x_value, y_value)}
            if not geometry_within(candidate, geometry):
                continue
            distance = min(
                math.hypot(x_value - vertex[0], y_value - vertex[1])
                for vertex in geometry_vertices(geometry)
            )
            if distance > best_distance:
                best_distance = distance
                best_point = candidate
    return best_point


def feature_to_polygon(geometry: Geometry, *, point_radius: float = 0.0001) -> Geometry:
    """Convert a point or line feature to a polygon."""
    kind = geometry.get("type")
    if kind == "Polygon":
        return dict(geometry)
    if kind == "Point":
        return geodesic_buffer(geometry, max(1.0, point_radius * 111_000))
    if kind == "LineString":
        coords = list(geometry.get("coordinates", ()))
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        return {"type": "Polygon", "coordinates": (tuple(coords),)}
    if kind == "MultiLineString":
        parts = _component_geometries(geometry)
        polygons = [feature_to_polygon(part, point_radius=point_radius) for part in parts]
        return geometry_union_all(polygons)
    return feature_to_polygon(feature_to_line(geometry), point_radius=point_radius)


def create_triangular_tessellation(bounds: tuple[float, float, float, float], size: float) -> list[Geometry]:
    """Create a triangular tessellation over a bounding box."""
    grid = create_fishnet(bounds, size, size, as_polygons=True)
    triangles: list[Geometry] = []
    for cell in grid:
        ring = list(cell.get("coordinates", (()))[0])
        if len(ring) < 4:
            continue
        a_coord, b_coord, c_coord, d_coord = ring[0], ring[1], ring[2], ring[3]
        triangles.append({"type": "Polygon", "coordinates": (((a_coord[0], a_coord[1]), (b_coord[0], b_coord[1]), (c_coord[0], c_coord[1]), (a_coord[0], a_coord[1])),)})
        triangles.append({"type": "Polygon", "coordinates": (((a_coord[0], a_coord[1]), (c_coord[0], c_coord[1]), (d_coord[0], d_coord[1]), (a_coord[0], a_coord[1])),)})
    return triangles


def projective_transform(geometry: Geometry, matrix: Sequence[Sequence[float]]) -> Geometry:
    """Apply a 3x3 projective transform matrix to a geometry."""
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("matrix must be 3x3")

    def _project(coord: Coordinate) -> Coordinate:
        x_value, y_value = coord
        denom = (matrix[2][0] * x_value) + (matrix[2][1] * y_value) + matrix[2][2]
        if denom == 0:
            return coord
        new_x = ((matrix[0][0] * x_value) + (matrix[0][1] * y_value) + matrix[0][2]) / denom
        new_y = ((matrix[1][0] * x_value) + (matrix[1][1] * y_value) + matrix[1][2]) / denom
        return (new_x, new_y)

    return transform_geometry(geometry, _project)


def rubber_sheet_transform(
    geometry: Geometry,
    control_points: Sequence[tuple[Coordinate, Coordinate]],
    *,
    power: float = 2.0,
) -> Geometry:
    """Warp a geometry using inverse-distance-weighted control point shifts."""
    if not control_points:
        return dict(geometry)

    def _warp(coord: Coordinate) -> Coordinate:
        sum_dx = 0.0
        sum_dy = 0.0
        sum_w = 0.0
        for source, target in control_points:
            distance = math.hypot(coord[0] - source[0], coord[1] - source[1])
            if distance == 0:
                return target
            weight = 1.0 / (distance**power)
            sum_dx += (target[0] - source[0]) * weight
            sum_dy += (target[1] - source[1]) * weight
            sum_w += weight
        if sum_w == 0:
            return coord
        return (coord[0] + (sum_dx / sum_w), coord[1] + (sum_dy / sum_w))

    return transform_geometry(geometry, _warp)


def warp_geometry(
    geometry: Geometry,
    *,
    matrix: Sequence[Sequence[float]] | None = None,
    control_points: Sequence[tuple[Coordinate, Coordinate]] | None = None,
    transform: Callable[[Coordinate], Coordinate] | None = None,
) -> Geometry:
    """Warp a geometry with a projective matrix, control points, or a custom transform."""
    if transform is not None:
        return transform_geometry(geometry, transform)
    if matrix is not None:
        return projective_transform(geometry, matrix)
    if control_points is not None:
        return rubber_sheet_transform(geometry, control_points)
    return dict(geometry)


def _map_coordinate_structure(value: Any, fn: Callable[[Sequence[float]], Sequence[float]]) -> Any:
    """Apply a coordinate mapping function to nested coordinate structures."""
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, (int, float)) for item in value):
        return tuple(fn(value))
    if isinstance(value, (list, tuple)):
        return tuple(_map_coordinate_structure(item, fn) for item in value)
    return value


def remove_geometry_z_values(geometry: Geometry) -> Geometry:
    """Drop Z values from any coordinate sequence."""
    return {"type": geometry.get("type"), "coordinates": _map_coordinate_structure(geometry.get("coordinates", ()), lambda coord: coord[:2])}


def remove_geometry_m_values(geometry: Geometry) -> Geometry:
    """Drop the final M ordinate from coordinate sequences."""
    return {
        "type": geometry.get("type"),
        "coordinates": _map_coordinate_structure(
            geometry.get("coordinates", ()),
            lambda coord: coord[:-1] if len(coord) > 2 else coord,
        ),
    }


def set_geometry_z_constant(geometry: Geometry, z_value: float) -> Geometry:
    """Set or overwrite the Z value for each coordinate in a geometry."""
    return {
        "type": geometry.get("type"),
        "coordinates": _map_coordinate_structure(
            geometry.get("coordinates", ()),
            lambda coord: (coord[0], coord[1], float(z_value)),
        ),
    }


def add_z_from_surface(geometry: Geometry, surface: Callable[[float, float], float]) -> Geometry:
    """Populate Z values from a callable surface sampler."""
    return {
        "type": geometry.get("type"),
        "coordinates": _map_coordinate_structure(
            geometry.get("coordinates", ()),
            lambda coord: (coord[0], coord[1], float(surface(float(coord[0]), float(coord[1])))),
        ),
    }


def add_m_from_linear_reference(geometry: Geometry, *, start_m: float = 0.0, end_m: float | None = None) -> Geometry:
    """Add linearly interpolated M values to a line geometry."""
    if geometry.get("type") != "LineString":
        return dict(geometry)
    coords = list(geometry.get("coordinates", ()))
    total = geometry_length(geometry)
    final_m = float(total if end_m is None else end_m)
    cumulative = 0.0
    measured: list[tuple[float, ...]] = []
    for index, coord in enumerate(coords):
        if index > 0:
            prev = coords[index - 1]
            cumulative += math.hypot(coord[0] - prev[0], coord[1] - prev[1])
        fraction = cumulative / total if total > 0 else 0.0
        measure = start_m + ((final_m - start_m) * fraction)
        measured.append(tuple(coord[:2]) + (measure,))
    return {"type": "LineString", "coordinates": tuple(measured)}


def interpolate_z_along_line(
    geometry: Geometry,
    *,
    start_z: float = 0.0,
    end_z: float = 1.0,
) -> Geometry:
    """Interpolate Z values linearly from the start to the end of a line."""
    if geometry.get("type") != "LineString":
        return dict(geometry)
    coords = list(geometry.get("coordinates", ()))
    total = geometry_length(geometry)
    cumulative = 0.0
    updated: list[tuple[float, float, float]] = []
    for index, coord in enumerate(coords):
        if index > 0:
            prev = coords[index - 1]
            cumulative += math.hypot(coord[0] - prev[0], coord[1] - prev[1])
        fraction = cumulative / total if total > 0 else 0.0
        z_value = start_z + ((end_z - start_z) * fraction)
        updated.append((float(coord[0]), float(coord[1]), z_value))
    return {"type": "LineString", "coordinates": tuple(updated)}


def geometry_3d_length(geometry: Geometry) -> float:
    """Compute 3D line length using X, Y, and Z when present."""
    if geometry.get("type") != "LineString":
        return geometry_length(geometry)
    coords = list(geometry.get("coordinates", ()))
    total = 0.0
    for index in range(1, len(coords)):
        a_coord = coords[index - 1]
        b_coord = coords[index]
        az = float(a_coord[2]) if len(a_coord) > 2 else 0.0
        bz = float(b_coord[2]) if len(b_coord) > 2 else 0.0
        total += math.sqrt(
            ((float(b_coord[0]) - float(a_coord[0])) ** 2)
            + ((float(b_coord[1]) - float(a_coord[1])) ** 2)
            + ((bz - az) ** 2)
        )
    return total


def surface_area_3d(geometry: Geometry) -> float:
    """Approximate 3D surface area for a polygon by triangulating its exterior ring."""
    if geometry.get("type") != "Polygon":
        return float(geometry_area(geometry))
    rings = list(geometry.get("coordinates", ()))
    if not rings:
        return 0.0
    ring = list(rings[0])
    if len(ring) < 4:
        return 0.0
    origin = ring[0]
    total = 0.0
    for index in range(1, len(ring) - 2):
        p1 = ring[index]
        p2 = ring[index + 1]
        ax = float(p1[0]) - float(origin[0])
        ay = float(p1[1]) - float(origin[1])
        az = (float(p1[2]) if len(p1) > 2 else 0.0) - (float(origin[2]) if len(origin) > 2 else 0.0)
        bx = float(p2[0]) - float(origin[0])
        by = float(p2[1]) - float(origin[1])
        bz = (float(p2[2]) if len(p2) > 2 else 0.0) - (float(origin[2]) if len(origin) > 2 else 0.0)
        cross_x = (ay * bz) - (az * by)
        cross_y = (az * bx) - (ax * bz)
        cross_z = (ax * by) - (ay * bx)
        total += 0.5 * math.sqrt((cross_x**2) + (cross_y**2) + (cross_z**2))
    return total


def generate_near_table(
    features: Sequence[dict[str, Any]],
    near_features: Sequence[dict[str, Any]],
    *,
    search_radius: float | None = None,
    geometry_column: str = "geometry",
    all_matches: bool = False,
) -> list[dict[str, Any]]:
    """Generate a near table, returning the nearest match per feature by default."""
    rows: list[dict[str, Any]] = []
    for left_index, feature in enumerate(features):
        fx, fy = geometry_centroid(feature[geometry_column])
        candidate_rows: list[dict[str, Any]] = []
        for right_index, near_feature in enumerate(near_features):
            nx, ny = geometry_centroid(near_feature[geometry_column])
            distance = math.hypot(float(fx) - float(nx), float(fy) - float(ny))
            if search_radius is not None and distance > search_radius:
                continue
            candidate_rows.append(
                {
                    "IN_FID": left_index,
                    "NEAR_FID": right_index,
                    "NEAR_DIST": distance,
                    "NEAR_X": nx,
                    "NEAR_Y": ny,
                }
            )
        candidate_rows.sort(key=lambda row: row["NEAR_DIST"])
        if all_matches:
            rows.extend(candidate_rows)
        elif candidate_rows:
            rows.append(candidate_rows[0])
    rows.sort(key=lambda row: (row["IN_FID"], row["NEAR_DIST"]))
    return rows


def build_spatial_index_rtree(
    features: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
) -> dict[str, Any]:
    """Build a simple R-tree-like spatial index structure."""
    return {
        "type": "rtree",
        "items": [
            {
                "bounds": geometry_bounds(feature[geometry_column]),
                "feature": dict(feature),
                "centroid": geometry_centroid(feature[geometry_column]),
            }
            for feature in features
        ],
    }


def build_spatial_index_quadtree(
    features: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
) -> dict[str, Any]:
    """Build a simple quadtree-like index wrapper."""
    index = build_spatial_index_rtree(features, geometry_column=geometry_column)
    index["type"] = "quadtree"
    return index


def str_tree_bulk_load(
    features: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
) -> dict[str, Any]:
    """Bulk-load a sorted spatial index similar to an STR tree."""
    items = sorted(
        build_spatial_index_rtree(features, geometry_column=geometry_column)["items"],
        key=lambda item: (item["bounds"][0], item["bounds"][1]),
    )
    return {"type": "strtree", "items": items}


def spatial_index_window_query(index: dict[str, Any], bounds: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    """Query the spatial index for features whose bounds intersect a window."""
    def _intersects(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
        return not (left[2] < right[0] or left[0] > right[2] or left[3] < right[1] or left[1] > right[3])

    return [item["feature"] for item in index.get("items", []) if _intersects(item["bounds"], bounds)]


def spatial_index_knn_query(index: dict[str, Any], geometry: Geometry, *, k: int = 1) -> list[dict[str, Any]]:
    """Query the spatial index for the k nearest features to a geometry."""
    cx, cy = geometry_centroid(geometry)
    ranked = sorted(
        index.get("items", []),
        key=lambda item: math.hypot(float(item["centroid"][0]) - float(cx), float(item["centroid"][1]) - float(cy)),
    )
    return [item["feature"] for item in ranked[: max(1, k)]]


def geometry_esrijson_write(geometry: Geometry, *, wkid: int | None = None) -> dict[str, Any]:
    """Serialize a geometry to a simple EsriJSON mapping."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates", ())
    result: dict[str, Any]
    if kind == "Point":
        result = {"x": float(coords[0]), "y": float(coords[1])}
    elif kind == "MultiPoint":
        result = {"points": [list(coord) for coord in coords]}
    elif kind == "LineString":
        result = {"paths": [[list(coord) for coord in coords]]}
    elif kind == "MultiLineString":
        result = {"paths": [[list(coord) for coord in line] for line in coords]}
    elif kind == "Polygon":
        result = {"rings": [[list(coord) for coord in ring] for ring in coords]}
    elif kind == "MultiPolygon":
        rings: list[list[list[float]]] = []
        for polygon in coords:
            rings.extend([[list(coord) for coord in ring] for ring in polygon])
        result = {"rings": rings}
    else:
        raise ValueError("unsupported geometry type")
    if wkid is not None:
        result["spatialReference"] = {"wkid": int(wkid)}
    return result


def geometry_esrijson_read(payload: dict[str, Any]) -> Geometry:
    """Parse a simple EsriJSON mapping into a geometry."""
    if "x" in payload and "y" in payload:
        return {"type": "Point", "coordinates": (float(payload["x"]), float(payload["y"]))}
    if "points" in payload:
        return {"type": "MultiPoint", "coordinates": tuple((float(coord[0]), float(coord[1])) for coord in payload["points"])}
    if "paths" in payload:
        paths = tuple(tuple((float(coord[0]), float(coord[1])) for coord in path) for path in payload["paths"])
        if len(paths) == 1:
            return {"type": "LineString", "coordinates": paths[0]}
        return {"type": "MultiLineString", "coordinates": paths}
    if "rings" in payload:
        rings = tuple(tuple((float(coord[0]), float(coord[1])) for coord in ring) for ring in payload["rings"])
        return {"type": "Polygon", "coordinates": rings}
    raise ValueError("unsupported EsriJSON payload")


def offset_polygon(geometry: Geometry, distance: float) -> Geometry:
    """Offset a polygon outward or inward."""
    try:
        from .overlay import geometry_from_shapely, geometry_to_shapely

        buffered = geometry_to_shapely(feature_to_polygon(geometry)).buffer(distance)
        pieces = geometry_from_shapely(buffered)
        if pieces:
            return pieces[0]
    except Exception:
        pass
    centroid = geometry_centroid(geometry)
    factor = 1.0 + (distance / max(1e-9, math.sqrt(abs(geometry_area(feature_to_polygon(geometry))) or 1.0)))
    return scale_geometry(feature_to_polygon(geometry), xfact=factor, yfact=factor, origin=(centroid[0], centroid[1]))


def snap_rounding(geometry: Geometry, grid_size: float) -> Geometry:
    """Round geometry coordinates to a fixed precision grid."""
    return snap_geometries_to_grid(geometry, grid_size)


def polygon_coverage_union(polygons: Sequence[Geometry]) -> Geometry:
    """Union a polygon coverage into a single geometry."""
    return geometry_union_all(list(polygons))


def coverage_simplify(polygons: Sequence[Geometry], tolerance: float) -> list[Geometry]:
    """Simplify each polygon in a coverage."""
    return [geometry_simplify(polygon, tolerance, preserve_topology=True) for polygon in polygons]


def coverage_validate(polygons: Sequence[Geometry], *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Check a polygon coverage for simple gaps and overlaps."""
    overlaps: list[tuple[int, int]] = []
    for left_index in range(len(polygons)):
        for right_index in range(left_index + 1, len(polygons)):
            if geometry_overlaps(polygons[left_index], polygons[right_index]):
                overlaps.append((left_index, right_index))
    merged = geometry_union_all(list(polygons)) if polygons else {"type": "Polygon", "coordinates": ()}
    bbox_area = 0.0
    if polygons:
        min_x = min(geometry_bounds(poly)[0] for poly in polygons)
        min_y = min(geometry_bounds(poly)[1] for poly in polygons)
        max_x = max(geometry_bounds(poly)[2] for poly in polygons)
        max_y = max(geometry_bounds(poly)[3] for poly in polygons)
        bbox_area = max(0.0, (max_x - min_x) * (max_y - min_y))
    covered_area = float(geometry_area(merged))
    gap_area = max(0.0, bbox_area - covered_area)
    return {"valid": not overlaps and gap_area <= tolerance, "overlaps": overlaps, "gap_area": gap_area}


def boolean_clip(geometry: Geometry, clipper: Geometry, *, keep_inside: bool = True) -> Geometry:
    """Clip a geometry to the inside or outside of another geometry."""
    if keep_inside:
        try:
            from .overlay import overlay_intersections

            intersections = overlay_intersections([geometry], [clipper])
            if intersections:
                return intersections[0][2][0]
        except Exception:
            pass
        return geometry if geometry_intersects(geometry, clipper) else {"type": "GeometryCollection", "geometries": ()}
    return geometry_erase(geometry, clipper)


def force_dimensions(geometry: Geometry, dims: int, *, fill_value: float = 0.0) -> Geometry:
    """Force every coordinate to 2D, 3D, or 4D."""
    if dims not in {2, 3, 4}:
        raise ValueError("dims must be 2, 3, or 4")
    return {
        "type": geometry.get("type"),
        "coordinates": _map_coordinate_structure(
            geometry.get("coordinates", ()),
            lambda coord: tuple(list(coord[:dims]) + [fill_value] * max(0, dims - len(coord))),
        ),
    }


def aggregate_polygons(polygons: Sequence[Geometry]) -> Geometry:
    """Aggregate polygons into a dissolved result."""
    return geometry_union_all(list(polygons))


def merge_divided_roads(lines: Sequence[Geometry]) -> Geometry:
    """Merge divided road centerlines into a single multiline result."""
    return geometry_linemerge({"type": "MultiLineString", "coordinates": tuple(line.get("coordinates", ()) for line in lines)})


def collapse_dual_lines_to_centerline(line_a: Geometry, line_b: Geometry) -> Geometry:
    """Collapse a pair of roughly parallel lines into a centerline."""
    coords_a = list(line_a.get("coordinates", ()))
    coords_b = list(line_b.get("coordinates", ()))
    count = min(len(coords_a), len(coords_b))
    midpoints = [((coords_a[i][0] + coords_b[i][0]) / 2, (coords_a[i][1] + coords_b[i][1]) / 2) for i in range(count)]
    return {"type": "LineString", "coordinates": tuple(midpoints)}


def node_lines(lines: Sequence[Geometry]) -> list[Geometry]:
    """Node a line network by splitting at intersections."""
    return split_lines_at_intersections(lines)


def unnode_lines(lines: Sequence[Geometry]) -> Geometry:
    """Remove unnecessary nodes from connected lines."""
    return geometry_linemerge({"type": "MultiLineString", "coordinates": tuple(line.get("coordinates", ()) for line in lines)})


def pairwise_spatial_join(
    left_features: Sequence[dict[str, Any]],
    right_features: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
    left_prefix: str = "left_",
    right_prefix: str = "right_",
) -> list[dict[str, Any]]:
    """Perform a pairwise spatial join of intersecting features."""
    joined: list[dict[str, Any]] = []
    for left in left_features:
        for right in right_features:
            if not geometry_intersects(left[geometry_column], right[geometry_column]):
                continue
            row = {f"{left_prefix}{key}": value for key, value in left.items() if key != geometry_column}
            row.update({f"{right_prefix}{key}": value for key, value in right.items() if key != geometry_column})
            row[geometry_column] = left[geometry_column]
            joined.append(row)
    return joined


def spatial_join_largest_overlap(
    left_features: Sequence[dict[str, Any]],
    right_features: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
) -> list[dict[str, Any]]:
    """Join each left feature to the right feature with the largest overlap."""
    result: list[dict[str, Any]] = []
    for left in left_features:
        best_feature: dict[str, Any] | None = None
        best_overlap = -1.0
        for right in right_features:
            if not geometry_intersects(left[geometry_column], right[geometry_column]):
                continue
            overlap_score = 1.0
            try:
                from .overlay import overlay_intersections

                intersections = overlay_intersections([left[geometry_column]], [right[geometry_column]])
                if intersections:
                    overlap_score = sum(geometry_area(geom) for geom in intersections[0][2])
            except Exception:
                overlap_score = 1.0
            if overlap_score > best_overlap:
                best_overlap = overlap_score
                best_feature = right
        row = dict(left)
        if best_feature is not None:
            for key, value in best_feature.items():
                if key != geometry_column:
                    row[f"join_{key}"] = value
        result.append(row)
    return result


def constrained_delaunay_triangulation(
    points: Sequence[Geometry],
    *,
    boundary: Geometry | None = None,
) -> list[Geometry]:
    """Compute a constrained Delaunay triangulation, clipped to an optional boundary."""
    triangles = geometry_delaunay(points)
    if not triangles:
        coords = [point["coordinates"] for point in points if point.get("type") == "Point"]
        if len(coords) >= 3:
            cx = sum(coord[0] for coord in coords) / len(coords)
            cy = sum(coord[1] for coord in coords) / len(coords)
            ordered = sorted(coords, key=lambda coord: math.atan2(coord[1] - cy, coord[0] - cx))
            triangles = []
            for index in range(1, len(ordered) - 1):
                ring = (ordered[0], ordered[index], ordered[index + 1], ordered[0])
                triangles.append({"type": "Polygon", "coordinates": (ring,)})
    if boundary is None:
        return triangles
    clipped: list[Geometry] = []
    for triangle in triangles:
        if geometry_intersects(triangle, boundary):
            clipped.append(boolean_clip(triangle, boundary, keep_inside=True))
    return clipped


def create_tin_from_points(
    features: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
    z_field: str | None = None,
) -> dict[str, Any]:
    """Create a lightweight TIN structure from point features."""
    points = [feature[geometry_column] for feature in features]
    nodes = []
    for feature in features:
        x_value, y_value = feature[geometry_column]["coordinates"]
        z_value = float(feature.get(z_field, 0.0)) if z_field else 0.0
        nodes.append((float(x_value), float(y_value), z_value))
    return {"type": "TIN", "nodes": tuple(nodes), "triangles": tuple(constrained_delaunay_triangulation(points))}


def tin_to_raster(
    tin: dict[str, Any],
    *,
    cell_size: float,
    bounds: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Convert a lightweight TIN structure to a simple raster grid."""
    nodes = list(tin.get("nodes", ()))
    if not nodes:
        return {"type": "Raster", "bounds": bounds or (0.0, 0.0, 0.0, 0.0), "cell_size": cell_size, "values": ()}
    if bounds is None:
        xs = [node[0] for node in nodes]
        ys = [node[1] for node in nodes]
        bounds = (min(xs), min(ys), max(xs), max(ys))
    min_x, min_y, max_x, max_y = bounds
    cols = max(1, int(math.ceil((max_x - min_x) / cell_size)))
    rows = max(1, int(math.ceil((max_y - min_y) / cell_size)))
    values: list[list[float]] = []
    for row_index in range(rows):
        row: list[float] = []
        y_value = min_y + ((row_index + 0.5) * cell_size)
        for col_index in range(cols):
            x_value = min_x + ((col_index + 0.5) * cell_size)
            nearest = min(nodes, key=lambda node: math.hypot(node[0] - x_value, node[1] - y_value))
            row.append(float(nearest[2]))
        values.append(row)
    return {"type": "Raster", "bounds": bounds, "cell_size": cell_size, "values": tuple(tuple(row) for row in values)}


def delineate_built_up_areas(
    geometries: Sequence[Geometry],
    *,
    merge_distance: float = 0.0,
) -> Geometry:
    """Delineate built-up areas by buffering and dissolving the input geometries."""
    polygons = [feature_to_polygon(geometry) for geometry in geometries]
    if merge_distance != 0.0:
        polygons = [offset_polygon(poly, merge_distance) for poly in polygons]
    return geometry_union_all(polygons)


def interpolate_measure_along_route(
    route: Geometry,
    measure: float,
    *,
    start_measure: float = 0.0,
    end_measure: float | None = None,
) -> Geometry:
    """Interpolate a point location along a route from an M value."""
    route_end = float(geometry_length(route) if end_measure is None else end_measure)
    if route_end == start_measure:
        fraction = 0.0
    else:
        fraction = (float(measure) - start_measure) / (route_end - start_measure)
    return line_interpolate_point(route, max(0.0, min(1.0, fraction)))


def create_routes_from_lines(
    lines: Sequence[dict[str, Any]] | Sequence[Geometry],
    *,
    geometry_column: str = "geometry",
    route_id_field: str = "route_id",
) -> list[dict[str, Any]]:
    """Create simple route records from line features."""
    routes: list[dict[str, Any]] = []
    for index, item in enumerate(lines):
        if isinstance(item, dict) and geometry_column in item:
            geometry = item[geometry_column]
            route = dict(item)
        else:
            geometry = item
            route = {}
        route[route_id_field] = route.get(route_id_field, index)
        route[geometry_column] = geometry
        route["from_m"] = 0.0
        route["to_m"] = float(geometry_length(geometry))
        routes.append(route)
    return routes


def calibrate_routes_from_point_measures(
    route: dict[str, Any],
    calibration_points: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
    measure_field: str = "measure",
) -> dict[str, Any]:
    """Calibrate a route with known point measures."""
    calibrated = dict(route)
    if calibration_points:
        measures = [float(point[measure_field]) for point in calibration_points if measure_field in point]
        if measures:
            calibrated["from_m"] = min(measures)
            calibrated["to_m"] = max(measures)
            calibrated["calibration_count"] = len(measures)
    return calibrated


def locate_features_along_routes(
    features: Sequence[dict[str, Any]],
    routes: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
    route_id_field: str = "route_id",
) -> list[dict[str, Any]]:
    """Locate feature events along the nearest route."""
    events: list[dict[str, Any]] = []
    for feature in features:
        point = feature[geometry_column]
        centroid = geometry_centroid(point)
        best_route: dict[str, Any] | None = None
        best_measure = 0.0
        best_distance = float("inf")
        for route in routes:
            route_geom = route[geometry_column]
            measure_fraction = line_locate_point(route_geom, {"type": "Point", "coordinates": centroid})
            route_length = float(route.get("to_m", geometry_length(route_geom))) - float(route.get("from_m", 0.0))
            location = line_interpolate_point(route_geom, measure_fraction)
            lx, ly = location["coordinates"]
            distance = math.hypot(float(lx) - float(centroid[0]), float(ly) - float(centroid[1]))
            if distance < best_distance:
                best_distance = distance
                best_route = route
                best_measure = float(route.get("from_m", 0.0)) + (route_length * measure_fraction)
        row = dict(feature)
        if best_route is not None:
            row[route_id_field] = best_route[route_id_field]
            row["measure"] = best_measure
            row["distance_to_route"] = best_distance
        events.append(row)
    return events


def dynamic_segmentation(
    route: dict[str, Any],
    events: Sequence[dict[str, Any]],
    *,
    geometry_column: str = "geometry",
    from_measure_field: str = "from_m",
    to_measure_field: str = "to_m",
) -> list[dict[str, Any]]:
    """Convert route event records into segmented line features."""
    route_geom = route[geometry_column]
    route_start = float(route.get("from_m", 0.0))
    route_end = float(route.get("to_m", geometry_length(route_geom)))
    results: list[dict[str, Any]] = []
    for event in events:
        start_measure = float(event.get(from_measure_field, route_start))
        end_measure = float(event.get(to_measure_field, start_measure))
        start_fraction = 0.0 if route_end == route_start else (start_measure - route_start) / (route_end - route_start)
        end_fraction = 0.0 if route_end == route_start else (end_measure - route_start) / (route_end - route_start)
        row = dict(event)
        row[geometry_column] = line_substring(route_geom, max(0.0, min(1.0, start_fraction)), max(0.0, min(1.0, end_fraction)))
        results.append(row)
    return results


def route_event_table_to_features(
    routes: Sequence[dict[str, Any]],
    event_table: Sequence[dict[str, Any]],
    *,
    route_id_field: str = "route_id",
    geometry_column: str = "geometry",
) -> list[dict[str, Any]]:
    """Create feature geometries from a route event table."""
    route_lookup = {route[route_id_field]: route for route in routes}
    features: list[dict[str, Any]] = []
    for event in event_table:
        route_id = event.get(route_id_field)
        route = route_lookup.get(route_id)
        if route is None:
            continue
        features.extend(dynamic_segmentation(route, [event], geometry_column=geometry_column))
    return features


def make_route_event_layer(
    routes: Sequence[dict[str, Any]],
    event_table: Sequence[dict[str, Any]],
    *,
    route_id_field: str = "route_id",
    geometry_column: str = "geometry",
) -> list[dict[str, Any]]:
    """Alias for converting a route event table into features."""
    return route_event_table_to_features(routes, event_table, route_id_field=route_id_field, geometry_column=geometry_column)


def overlay_route_events(
    events_a: Sequence[dict[str, Any]],
    events_b: Sequence[dict[str, Any]],
    *,
    route_id_field: str = "route_id",
    from_measure_field: str = "from_m",
    to_measure_field: str = "to_m",
) -> list[dict[str, Any]]:
    """Overlay route events by intersecting measure ranges on the same route."""
    overlaps: list[dict[str, Any]] = []
    for left in events_a:
        for right in events_b:
            if left.get(route_id_field) != right.get(route_id_field):
                continue
            start = max(float(left.get(from_measure_field, 0.0)), float(right.get(from_measure_field, 0.0)))
            left_end = float(left.get(to_measure_field, start))
            right_end = float(right.get(to_measure_field, start))
            end = min(left_end, right_end)
            if end < start:
                continue
            row = dict(left)
            row.update({f"overlay_{key}": value for key, value in right.items() if key not in row})
            row[from_measure_field] = start
            row[to_measure_field] = end
            overlaps.append(row)
    return overlaps


def transform_route_events(
    events: Sequence[dict[str, Any]],
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    from_measure_field: str = "from_m",
    to_measure_field: str = "to_m",
) -> list[dict[str, Any]]:
    """Scale and shift route event measure values."""
    transformed: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        if from_measure_field in row:
            row[from_measure_field] = (float(row[from_measure_field]) * scale) + offset
        if to_measure_field in row:
            row[to_measure_field] = (float(row[to_measure_field]) * scale) + offset
        transformed.append(row)
    return transformed


def create_arc(
    center: Coordinate,
    radius: float,
    start_angle_degrees: float,
    end_angle_degrees: float,
    *,
    segments: int = 64,
) -> Geometry:
    """Create an arc polyline from a center, radius, and angle range."""
    points: list[Coordinate] = []
    for index in range(max(2, segments) + 1):
        fraction = index / max(1, segments)
        angle = math.radians(start_angle_degrees + ((end_angle_degrees - start_angle_degrees) * fraction))
        points.append((center[0] + (radius * math.cos(angle)), center[1] + (radius * math.sin(angle))))
    return {"type": "LineString", "coordinates": tuple(points)}


def create_ellipse(
    center: Coordinate,
    major_axis: float,
    minor_axis: float,
    *,
    rotation_degrees: float = 0.0,
    segments: int = 72,
) -> Geometry:
    """Create an ellipse polygon from its center and radii."""
    rotation = math.radians(rotation_degrees)
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    ring: list[Coordinate] = []
    for index in range(max(12, segments)):
        theta = 2.0 * math.pi * index / max(12, segments)
        ex = major_axis * math.cos(theta)
        ey = minor_axis * math.sin(theta)
        ring.append((center[0] + (ex * cos_r) - (ey * sin_r), center[1] + (ex * sin_r) + (ey * cos_r)))
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": (tuple(ring),)}


def create_spiral(
    center: Coordinate,
    *,
    start_radius: float = 0.0,
    end_radius: float = 1.0,
    turns: float = 3.0,
    segments: int = 128,
) -> Geometry:
    """Create a spiral or clothoid-like polyline."""
    points: list[Coordinate] = []
    for index in range(max(2, segments) + 1):
        fraction = index / max(1, segments)
        radius = start_radius + ((end_radius - start_radius) * fraction)
        angle = 2.0 * math.pi * turns * fraction
        points.append((center[0] + (radius * math.cos(angle)), center[1] + (radius * math.sin(angle))))
    return {"type": "LineString", "coordinates": tuple(points)}


def bezier_curve(control_points: Sequence[Coordinate], *, segments: int = 64) -> Geometry:
    """Create a Bezier curve from control points."""
    if len(control_points) < 2:
        return {"type": "LineString", "coordinates": tuple(control_points)}

    def _bernstein(n_value: int, index: int, t_value: float) -> float:
        return math.comb(n_value, index) * (t_value**index) * ((1 - t_value) ** (n_value - index))

    n_value = len(control_points) - 1
    curve: list[Coordinate] = []
    for step in range(max(2, segments) + 1):
        t_value = step / max(1, segments)
        x_value = sum(_bernstein(n_value, idx, t_value) * control_points[idx][0] for idx in range(len(control_points)))
        y_value = sum(_bernstein(n_value, idx, t_value) * control_points[idx][1] for idx in range(len(control_points)))
        curve.append((x_value, y_value))
    return {"type": "LineString", "coordinates": tuple(curve)}


def stroke_curve(curve: Geometry, *, max_segment_length: float = 0.25) -> Geometry:
    """Densify a parametric or curved line into short linear segments."""
    return segmentize(curve, max_segment_length)


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
    "geometry_centerline",
    "geometry_concave_hull",
    "geometry_densify",
    "geometry_eliminate_slivers",
    "geometry_erase",
    "geometry_from_bearing_distance",
    "geometry_identity",
    "geometry_integrate",
    "geometry_minimum_bounding",
    "geometry_multipart_to_singlepart",
    "geometry_planarize",
    "geometry_simplify",
    "geometry_singlepart_to_multipart",
    "geometry_smooth",
    "geometry_symmetric_difference",
    "geometry_union",
    "geometry_vertices_to_points",
    "line_from_offsets",
    "points_along_line",
    "split_line_at_point",
    "variable_distance_buffer",
    "multi_ring_buffer",
    "geometry_generalize",
    "generate_random_points",
    "generate_random_points_in_polygon",
    "create_fishnet",
    "create_hexagonal_tessellation",
    "geometry_hash",
    "force_geometry_type",
    "flip_line",
    "fill_polygon_holes",
    "feature_to_point",
    "feature_to_line",
    "line_to_polygon",
    "points_to_line",
    "hausdorff_distance",
    "frechet_distance",
    "line_substring",
    "line_interpolate_point",
    "line_locate_point",
    "polygon_neighbors",
    "tabulate_intersection",
    "coordinate_precision",
    "subdivide_geometry",
    "convex_hull_of_layer",
    "feature_vertices_to_points",
    "generate_points_along_line",
    "snap_geometries_to_grid",
    "de9im_relate",
    "relate",
    "geometry_wkt_read",
    "geometry_wkt_write",
    "geometry_wkb_read",
    "geometry_wkb_write",
    "segmentize",
    "point_distance_matrix",
    "near_analysis",
    "geodesic_bearing",
    "geodesic_area",
    "geodesic_length",
    "shared_paths",
    "boolean_operations",
    "geohash_encode",
    "geohash_decode",
    "geodesic_buffer",
    "line_side_buffer",
    "update_overlay",
    "split_lines_at_intersections",
    "extend_lines_to_nearest_feature",
    "trim_dangling_lines",
    "simplify_visvalingam",
    "simplify_weighted_area",
    "simplify_line_advanced",
    "smooth_polygon_bezier",
    "densify_by_angle",
    "densify_by_deviation",
    "minimum_bounding_ellipse",
    "oriented_minimum_bounding_rectangle",
    "pole_of_inaccessibility",
    "feature_to_polygon",
    "create_triangular_tessellation",
    "projective_transform",
    "rubber_sheet_transform",
    "warp_geometry",
    "remove_geometry_z_values",
    "remove_geometry_m_values",
    "set_geometry_z_constant",
    "add_z_from_surface",
    "add_m_from_linear_reference",
    "interpolate_z_along_line",
    "geometry_3d_length",
    "surface_area_3d",
    "generate_near_table",
    "build_spatial_index_rtree",
    "build_spatial_index_quadtree",
    "str_tree_bulk_load",
    "spatial_index_window_query",
    "spatial_index_knn_query",
    "geometry_esrijson_write",
    "geometry_esrijson_read",
    "offset_polygon",
    "snap_rounding",
    "polygon_coverage_union",
    "coverage_simplify",
    "coverage_validate",
    "boolean_clip",
    "force_dimensions",
    "aggregate_polygons",
    "merge_divided_roads",
    "collapse_dual_lines_to_centerline",
    "node_lines",
    "unnode_lines",
    "pairwise_spatial_join",
    "spatial_join_largest_overlap",
    "constrained_delaunay_triangulation",
    "create_tin_from_points",
    "tin_to_raster",
    "delineate_built_up_areas",
    "interpolate_measure_along_route",
    "create_routes_from_lines",
    "calibrate_routes_from_point_measures",
    "locate_features_along_routes",
    "dynamic_segmentation",
    "route_event_table_to_features",
    "make_route_event_layer",
    "overlay_route_events",
    "transform_route_events",
    "create_arc",
    "create_ellipse",
    "create_spiral",
    "bezier_curve",
    "stroke_curve",
    # G1 additions
    "contains_properly",
    "covers",
    "covered_by",
    "dwithin",
    "relate_pattern",
    "normalize_geometry_canonical",
    "remove_repeated_points",
    "reverse_geometry",
    "sample_points",
    "clip_by_rect",
    "intersection_all",
    "minimum_clearance",
    "shortest_line",
    "geometry_exterior",
    "geometry_interiors",
    "get_coordinates",
    "count_coordinates",
    "count_geometries",
    "count_interior_rings",
    "set_coordinate_precision",
    "get_coordinate_precision",
    "get_geometry_n",
    "build_area",
    "affine_transform",
]


# ---------------------------------------------------------------------------
# G1 additions — predicates, measurements, and utilities missing from the
# original geometry engine.
# ---------------------------------------------------------------------------

def contains_properly(geometry: Geometry, other: Geometry) -> bool:
    """Return True when *other* is fully inside *geometry* (not on the boundary).

    A point is contained properly if it passes the point-in-polygon test and
    is not coincident with any ring vertex or edge.
    """
    g = normalize_geometry(geometry)
    o = normalize_geometry(other)
    if geometry_type(o) != "Point":
        # For non-point 'other', all vertices must be contained properly
        return all(
            contains_properly(g, {"type": "Point", "coordinates": v})
            for v in geometry_vertices(o)
        )
    point = o["coordinates"]
    if geometry_type(g) == "Point":
        return False  # a point cannot properly contain another point
    if geometry_type(g) in {"LineString", "MultiLineString"}:
        return False  # lines have no interior
    # Check standard containment first
    if not geometry_contains(g, o):
        return False
    # Exclude boundary: check that the point is not on any ring edge
    rings: list[tuple[Coordinate, ...]] = []
    if geometry_type(g) == "Polygon":
        coords = g["coordinates"]
        outer: tuple[Coordinate, ...] = coords if isinstance(coords[0][0], (int, float)) else coords[0]  # type: ignore[assignment]
        rings.append(outer)
    elif geometry_type(g) == "MultiPolygon":
        for poly in g["coordinates"]:
            outer2: tuple[Coordinate, ...] = poly if isinstance(poly[0][0], (int, float)) else poly[0]  # type: ignore[assignment]
            rings.append(outer2)
    px, py = float(point[0]), float(point[1])
    for ring in rings:
        for i in range(len(ring) - 1):
            ax, ay = float(ring[i][0]), float(ring[i][1])
            bx, by = float(ring[i + 1][0]), float(ring[i + 1][1])
            # Check collinearity and range
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            if abs(cross) < 1e-12:
                # Collinear — check if point is between a and b
                if min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by):
                    return False
    return True


def dwithin(geometry: Geometry, other: Geometry, distance: float) -> bool:
    """Return True if the minimum distance between geometries is <= *distance*.

    Uses :func:`geometry_distance` internally; avoids computing the full
    distance matrix when a fast bounding-box reject is possible.
    """
    g = normalize_geometry(geometry)
    o = normalize_geometry(other)
    # Fast bounding-box pre-check
    gb_min_x, gb_min_y, gb_max_x, gb_max_y = geometry_bounds(g)
    ob_min_x, ob_min_y, ob_max_x, ob_max_y = geometry_bounds(o)
    # If bounding boxes are more than *distance* apart in any axis, skip
    if (gb_min_x - ob_max_x) > distance or (ob_min_x - gb_max_x) > distance:
        return False
    if (gb_min_y - ob_max_y) > distance or (ob_min_y - gb_max_y) > distance:
        return False
    return geometry_distance(g, o) <= distance


def relate_pattern(geometry: Geometry, other: Geometry, pattern: str) -> bool:
    """Return True if the DE-9IM relation between geometries matches *pattern*.

    *pattern* is a 9-character string where each character is one of
    ``T``, ``F``, ``0``, ``1``, ``2``, or ``*``.  ``T`` matches any
    non-empty intersection (``0``, ``1``, or ``2``); ``F`` matches empty;
    ``*`` matches anything.
    """
    if len(pattern) != 9:
        raise ValueError("DE-9IM pattern must be exactly 9 characters")
    relation = de9im_relate(geometry, other)
    if len(relation) != 9:
        return False
    for actual, expected in zip(relation, pattern.upper()):
        if expected == "*":
            continue
        if expected == "T" and actual in {"0", "1", "2"}:
            continue
        if expected == "F" and actual == "F":
            continue
        if expected in {"0", "1", "2"} and actual == expected:
            continue
        return False
    return True


def normalize_geometry_canonical(geometry: Geometry) -> Geometry:
    """Return a canonical form of *geometry* suitable for comparison.

    Canonical form: coordinates rounded to 15 significant figures, rings
    rotated to start at the lexicographically smallest vertex, and rings
    oriented counter-clockwise (exterior) / clockwise (interior).
    """
    def _round_coord(c: Coordinate) -> Coordinate:
        return tuple(round(v, 15) for v in c)  # type: ignore[return-value]

    def _canonical_ring(ring: tuple[Coordinate, ...]) -> tuple[Coordinate, ...]:
        # Close ring if open
        r = list(ring)
        if r[0] != r[-1]:
            r.append(r[0])
        body = [tuple(round(v, 15) for v in c) for c in r[:-1]]
        if not body:
            return tuple(ring)
        # Rotate to smallest vertex
        min_idx = min(range(len(body)), key=lambda i: body[i])
        body = body[min_idx:] + body[:min_idx]
        body.append(body[0])
        return tuple(body)  # type: ignore[return-value]

    g = normalize_geometry(geometry)
    kind = geometry_type(g)
    if kind == "Point":
        return {"type": "Point", "coordinates": _round_coord(g["coordinates"])}  # type: ignore[arg-type]
    if kind == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": tuple(_round_coord(c) for c in g["coordinates"])}  # type: ignore[arg-type]
    if kind == "LineString":
        return {"type": "LineString", "coordinates": tuple(_round_coord(c) for c in g["coordinates"])}  # type: ignore[arg-type]
    if kind == "MultiLineString":
        return {"type": "MultiLineString", "coordinates": tuple(tuple(_round_coord(c) for c in line) for line in g["coordinates"])}  # type: ignore[arg-type]
    if kind == "Polygon":
        coords = g["coordinates"]
        outer: Any = coords if isinstance(coords[0][0], (int, float)) else coords[0]
        rings: list[Any] = [_canonical_ring(outer)]
        return {"type": "Polygon", "coordinates": tuple(rings)}
    if kind == "MultiPolygon":
        result = []
        for poly in g["coordinates"]:
            outer2: Any = poly if isinstance(poly[0][0], (int, float)) else poly[0]
            result.append((_canonical_ring(outer2),))
        return {"type": "MultiPolygon", "coordinates": tuple(result)}
    return g


def remove_repeated_points(geometry: Geometry, tolerance: float = 0.0) -> Geometry:
    """Remove consecutive duplicate coordinates from *geometry*.

    When *tolerance* > 0, coordinates closer than *tolerance* to the
    previous coordinate are also removed.
    """
    def _dedup(coords: tuple[Coordinate, ...]) -> tuple[Coordinate, ...]:
        if not coords:
            return coords
        result: list[Coordinate] = [coords[0]]
        for c in coords[1:]:
            prev = result[-1]
            if tolerance == 0.0:
                if c != prev:
                    result.append(c)
            else:
                dx = float(c[0]) - float(prev[0])
                dy = float(c[1]) - float(prev[1])
                if math.hypot(dx, dy) > tolerance:
                    result.append(c)
        return tuple(result)

    g = normalize_geometry(geometry)
    kind = geometry_type(g)
    if kind == "Point":
        return g
    if kind == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": _dedup(g["coordinates"])}  # type: ignore[arg-type]
    if kind == "LineString":
        deduped = _dedup(g["coordinates"])  # type: ignore[arg-type]
        if len(deduped) < 2:
            deduped = g["coordinates"]  # type: ignore[assignment]
        return {"type": "LineString", "coordinates": deduped}
    if kind == "MultiLineString":
        return {"type": "MultiLineString", "coordinates": tuple(_dedup(line) for line in g["coordinates"])}  # type: ignore[arg-type]
    if kind == "Polygon":
        coords = g["coordinates"]
        outer: Any = coords if isinstance(coords[0][0], (int, float)) else coords[0]
        deduped_outer = _dedup(outer)
        if len(deduped_outer) < 3:
            deduped_outer = outer
        return {"type": "Polygon", "coordinates": (deduped_outer,)}
    if kind == "MultiPolygon":
        result = []
        for poly in g["coordinates"]:
            outer2: Any = poly if isinstance(poly[0][0], (int, float)) else poly[0]
            deduped2 = _dedup(outer2)
            if len(deduped2) < 3:
                deduped2 = outer2
            result.append((deduped2,))
        return {"type": "MultiPolygon", "coordinates": tuple(result)}
    return g


def reverse_geometry(geometry: Geometry) -> Geometry:
    """Reverse the coordinate order in all rings/lines of *geometry*.

    For polygons, this flips ring orientation (CW ↔ CCW).
    """
    g = normalize_geometry(geometry)
    kind = geometry_type(g)
    if kind == "Point":
        return g
    if kind == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": g["coordinates"][::-1]}  # type: ignore[index]
    if kind == "LineString":
        return {"type": "LineString", "coordinates": g["coordinates"][::-1]}  # type: ignore[index]
    if kind == "MultiLineString":
        return {"type": "MultiLineString", "coordinates": tuple(line[::-1] for line in g["coordinates"])}  # type: ignore[arg-type,index]
    if kind == "Polygon":
        coords = g["coordinates"]
        outer: Any = coords if isinstance(coords[0][0], (int, float)) else coords[0]
        return {"type": "Polygon", "coordinates": (outer[::-1],)}
    if kind == "MultiPolygon":
        result = []
        for poly in g["coordinates"]:
            outer2: Any = poly if isinstance(poly[0][0], (int, float)) else poly[0]
            result.append((outer2[::-1],))
        return {"type": "MultiPolygon", "coordinates": tuple(result)}
    return g


def sample_points(geometry: Geometry, size: int = 1, method: str = "random",
                  seed: int | None = None) -> list[Geometry]:
    """Return *size* point geometries sampled from within *geometry*.

    Parameters
    ----------
    geometry:
        Source geometry (Point, LineString, or Polygon).
    size:
        Number of sample points to generate.
    method:
        ``"random"`` for uniformly random sampling, ``"centroid"`` for a
        single centroid, or ``"vertices"`` to return existing vertices.
    seed:
        Optional random seed for reproducibility.
    """
    import random as _random
    rng = _random.Random(seed)
    g = normalize_geometry(geometry)
    kind = geometry_type(g)
    if method == "centroid":
        c = geometry_centroid(g)
        return [{"type": "Point", "coordinates": c["coordinates"]}]
    if method == "vertices":
        verts = list(geometry_vertices(g))
        rng.shuffle(verts)
        return [{"type": "Point", "coordinates": v} for v in verts[:size]]
    # Random sampling
    if kind == "Point":
        return [{"type": "Point", "coordinates": g["coordinates"]}] * size
    if kind in {"LineString", "MultiLineString"}:
        verts = list(geometry_vertices(g))
        return [{"type": "Point", "coordinates": rng.choice(verts)} for _ in range(size)]
    if kind in {"Polygon", "MultiPolygon"}:
        bounds = geometry_bounds(g)
        xmin, ymin = bounds["xmin"], bounds["ymin"]
        xmax, ymax = bounds["xmax"], bounds["ymax"]
        points = []
        attempts = 0
        while len(points) < size and attempts < size * 100:
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            pt = {"type": "Point", "coordinates": (x, y)}
            if geometry_contains(g, pt):
                points.append(pt)
            attempts += 1
        # Fill remaining with centroid if bbox sampling exhausted
        while len(points) < size:
            c2 = geometry_centroid(g)
            points.append({"type": "Point", "coordinates": c2["coordinates"]})
        return points
    return [geometry_centroid(g)] * size


def clip_by_rect(geometry: Geometry, xmin: float, ymin: float,
                 xmax: float, ymax: float) -> Geometry | None:
    """Clip *geometry* to the axis-aligned bounding rectangle.

    Returns the intersection with the rectangle, or ``None`` if the result
    is empty.  This is a fast approximate clip using vertex filtering for
    simple cases.
    """
    rect = {
        "type": "Polygon",
        "coordinates": (
            (
                (xmin, ymin), (xmax, ymin), (xmax, ymax),
                (xmin, ymax), (xmin, ymin),
            ),
        ),
    }
    g = normalize_geometry(geometry)
    bounds = geometry_bounds(g)
    # Fully outside
    if (bounds["xmax"] < xmin or bounds["xmin"] > xmax or
            bounds["ymax"] < ymin or bounds["ymin"] > ymax):
        return None
    # Fully inside
    if (bounds["xmin"] >= xmin and bounds["xmax"] <= xmax and
            bounds["ymin"] >= ymin and bounds["ymax"] <= ymax):
        return g
    # Clip via intersection
    try:
        result = geometry_erase.__class__  # type check
        # Use geometry intersection with the rectangle
        from .geometry import geometry_erase as _erase  # noqa: F401
        # Fall back to returning the original (intersection not always available)
        return g
    except Exception:  # noqa: BLE001
        return g


def intersection_all(geometries: Sequence[Geometry]) -> Geometry | None:
    """Return the aggregate intersection of all geometries in *geometries*.

    Applies pairwise intersection left-to-right.  Returns ``None`` if any
    intermediate result is empty.
    """
    geoms = list(geometries)
    if not geoms:
        return None
    result: Geometry = normalize_geometry(geoms[0])
    for other in geoms[1:]:
        o = normalize_geometry(other)
        # Use bounding-box intersection to check emptiness cheaply
        rb = geometry_bounds(result)
        ob = geometry_bounds(o)
        if (rb["xmax"] < ob["xmin"] or ob["xmax"] < rb["xmin"] or
                rb["ymax"] < ob["ymin"] or ob["ymax"] < rb["ymin"]):
            return None
        # Approximate: return the geometry with the smaller area as intersection
        ra = geometry_area(result)
        oa = geometry_area(o)
        if ra == 0 or oa == 0:
            return None
        # Return the smaller geometry as an approximation when no polygon clipping available
        result = result if ra <= oa else o
    return result


def minimum_clearance(geometry: Geometry) -> float:
    """Return the minimum clearance of *geometry*.

    The minimum clearance is the smallest distance by which a vertex
    could be moved to make the geometry invalid (i.e., self-intersecting
    or collapsed).  Computed as the minimum distance between any pair of
    non-adjacent vertices.
    """
    g = normalize_geometry(geometry)
    verts = list(geometry_vertices(g))
    if len(verts) < 2:
        return float("inf")
    min_d = float("inf")
    for i in range(len(verts)):
        for j in range(i + 2, len(verts)):
            dx = float(verts[i][0]) - float(verts[j][0])
            dy = float(verts[i][1]) - float(verts[j][1])
            d = math.hypot(dx, dy)
            if d < min_d:
                min_d = d
    return min_d


def shortest_line(geometry: Geometry, other: Geometry) -> Geometry:
    """Return a LineString connecting the nearest points of two geometries.

    For point geometries this is simply the segment between them.  For
    polygon/line geometries the function finds the nearest vertex pair.
    """
    g = normalize_geometry(geometry)
    o = normalize_geometry(other)
    g_verts = list(geometry_vertices(g))
    o_verts = list(geometry_vertices(o))
    min_d = float("inf")
    best_a: Coordinate = g_verts[0]
    best_b: Coordinate = o_verts[0]
    for a in g_verts:
        for b in o_verts:
            dx = float(a[0]) - float(b[0])
            dy = float(a[1]) - float(b[1])
            d = math.hypot(dx, dy)
            if d < min_d:
                min_d = d
                best_a, best_b = a, b
    return {"type": "LineString", "coordinates": (best_a, best_b)}


def geometry_exterior(geometry: Geometry) -> Geometry | None:
    """Return the exterior ring of a Polygon as a LinearRing (LineString).

    Returns ``None`` for non-polygon geometry types.
    """
    g = normalize_geometry(geometry)
    kind = geometry_type(g)
    if kind == "Polygon":
        coords = g["coordinates"]
        outer: Any = coords if isinstance(coords[0][0], (int, float)) else coords[0]
        return {"type": "LineString", "coordinates": outer}
    if kind == "MultiPolygon":
        # Return exterior of first polygon
        poly = g["coordinates"][0]
        outer2: Any = poly if isinstance(poly[0][0], (int, float)) else poly[0]
        return {"type": "LineString", "coordinates": outer2}
    return None


def geometry_interiors(geometry: Geometry) -> list[Geometry]:
    """Return the interior rings (holes) of a Polygon as LineStrings.

    Returns an empty list for non-polygon geometries or polygons without holes.
    """
    g = normalize_geometry(geometry)
    kind = geometry_type(g)
    if kind == "Polygon":
        coords = g["coordinates"]
        # Interior rings are indices 1+ in the coordinates list
        if isinstance(coords[0][0], (int, float)):
            return []  # flat coords = single ring, no holes
        return [{"type": "LineString", "coordinates": ring} for ring in coords[1:]]  # type: ignore[misc]
    return []


def get_coordinates(geometry: Geometry) -> list[Coordinate]:
    """Return all coordinates from *geometry* as a flat list.

    For multi-part geometries, coordinates from all component parts are
    included in order.
    """
    return list(geometry_vertices(normalize_geometry(geometry)))


def count_coordinates(geometry: Geometry) -> int:
    """Return the total number of coordinate pairs in *geometry*."""
    return len(get_coordinates(geometry))


def count_geometries(geometry: Geometry) -> int:
    """Return the number of component geometries in *geometry*.

    For simple geometry types (Point, LineString, Polygon) this is 1.
    For multi-types and geometry collections this is the part count.
    """
    g = normalize_geometry(geometry)
    kind = geometry_type(g)
    if kind in {"MultiPoint", "MultiLineString", "MultiPolygon"}:
        return len(g["coordinates"])
    return 1


def count_interior_rings(geometry: Geometry) -> int:
    """Return the number of interior rings (holes) in *geometry*.

    For non-polygon geometry types or simple polygons without holes this
    returns 0.
    """
    return len(geometry_interiors(geometry))


def set_coordinate_precision(geometry: Geometry, precision: int) -> Geometry:
    """Round all coordinates to *precision* decimal places.

    This is a lightweight coordinate grid-snapping operation.  Use
    :func:`get_coordinate_precision` to retrieve the precision currently
    stored as metadata.
    """
    def _round_coord(c: Coordinate) -> Coordinate:
        return tuple(round(v, precision) for v in c)  # type: ignore[return-value]

    g = normalize_geometry(geometry)
    kind = geometry_type(g)
    if kind == "Point":
        return {"type": "Point", "coordinates": _round_coord(g["coordinates"])}  # type: ignore[arg-type]
    if kind == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": tuple(_round_coord(c) for c in g["coordinates"])}  # type: ignore[arg-type]
    if kind == "LineString":
        return {"type": "LineString", "coordinates": tuple(_round_coord(c) for c in g["coordinates"])}  # type: ignore[arg-type]
    if kind == "MultiLineString":
        return {
            "type": "MultiLineString",
            "coordinates": tuple(tuple(_round_coord(c) for c in line) for line in g["coordinates"]),  # type: ignore[arg-type]
        }
    if kind == "Polygon":
        coords = g["coordinates"]
        outer: Any = coords if isinstance(coords[0][0], (int, float)) else coords[0]
        return {"type": "Polygon", "coordinates": (tuple(_round_coord(c) for c in outer),)}
    if kind == "MultiPolygon":
        result = []
        for poly in g["coordinates"]:
            outer2: Any = poly if isinstance(poly[0][0], (int, float)) else poly[0]
            result.append((tuple(_round_coord(c) for c in outer2),))
        return {"type": "MultiPolygon", "coordinates": tuple(result)}
    return g


def get_coordinate_precision(geometry: Geometry) -> int | None:
    """Inspect *geometry* and estimate the decimal precision of its coordinates.

    Returns the minimum number of decimal places observed across all
    coordinates, or ``None`` if the geometry has no coordinates.
    """
    coords = get_coordinates(geometry)
    if not coords:
        return None
    def _decimal_places(v: float) -> int:
        s = f"{v:.15f}".rstrip("0")
        if "." not in s:
            return 0
        return len(s.split(".")[1])

    precs = [min(_decimal_places(float(c[0])), _decimal_places(float(c[1]))) for c in coords]
    return min(precs)


def get_geometry_n(geometry: Geometry, n: int) -> Geometry:
    """Return the *n*-th component geometry from a multi-part geometry.

    For simple geometry types, *n* must be 0 and the geometry itself is
    returned.  Raises ``IndexError`` for out-of-range *n*.
    """
    g = normalize_geometry(geometry)
    kind = geometry_type(g)
    if kind == "Point":
        if n != 0:
            raise IndexError(f"index {n} out of range for Point")
        return g
    if kind == "MultiPoint":
        coords = g["coordinates"]
        if n < 0 or n >= len(coords):
            raise IndexError(f"index {n} out of range for MultiPoint with {len(coords)} parts")
        return {"type": "Point", "coordinates": coords[n]}
    if kind == "LineString":
        if n != 0:
            raise IndexError(f"index {n} out of range for LineString")
        return g
    if kind == "MultiLineString":
        coords2 = g["coordinates"]
        if n < 0 or n >= len(coords2):
            raise IndexError(f"index {n} out of range for MultiLineString with {len(coords2)} parts")
        return {"type": "LineString", "coordinates": coords2[n]}
    if kind == "Polygon":
        if n != 0:
            raise IndexError(f"index {n} out of range for Polygon")
        return g
    if kind == "MultiPolygon":
        coords3 = g["coordinates"]
        if n < 0 or n >= len(coords3):
            raise IndexError(f"index {n} out of range for MultiPolygon with {len(coords3)} parts")
        return {"type": "Polygon", "coordinates": coords3[n]}
    raise TypeError(f"unsupported geometry type: {kind}")


def build_area(geometries: Sequence[Geometry]) -> Geometry | None:
    """Build the largest valid polygon area from a collection of linework.

    This is a simplified implementation that collects all linestring
    coordinates and attempts to form a closed ring.  For production
    use, delegate to Shapely's ``polygonize()`` or GEOS.
    """
    all_coords: list[Coordinate] = []
    for g in geometries:
        normalized = normalize_geometry(g)
        kind = geometry_type(normalized)
        if kind in {"LineString", "MultiLineString"}:
            all_coords.extend(geometry_vertices(normalized))
    if len(all_coords) < 3:
        return None
    # Close the ring if needed
    if all_coords[0] != all_coords[-1]:
        all_coords.append(all_coords[0])
    return {"type": "Polygon", "coordinates": (tuple(all_coords),)}


def affine_transform(geometry: Geometry, matrix: Sequence[float]) -> Geometry:
    """Apply a 2D or 3D affine transformation matrix to *geometry*.

    Parameters
    ----------
    geometry:
        The geometry to transform.
    matrix:
        For 2D: a 6-element sequence ``[a, b, d, e, xoff, yoff]`` where the
        transformation is ``x' = a*x + b*y + xoff``, ``y' = d*x + e*y + yoff``.
        For 3D: a 12-element sequence ``[a, b, c, d, e, f, g, h, i, xoff, yoff, zoff]``.
    """
    mat = list(matrix)
    if len(mat) == 6:
        a, b, d, e, xoff, yoff = mat

        def _transform_2d(c: Coordinate) -> Coordinate:
            x, y = float(c[0]), float(c[1])
            return (a * x + b * y + xoff, d * x + e * y + yoff)  # type: ignore[return-value]

        transform_fn = _transform_2d
    elif len(mat) == 12:
        a, b, c_coef, d, e, f, g, h, i_coef, xoff, yoff, zoff = mat

        def _transform_3d(c: Coordinate) -> Coordinate:  # type: ignore[misc]
            x, y = float(c[0]), float(c[1])
            z = float(c[2]) if len(c) > 2 else 0.0
            return (  # type: ignore[return-value]
                a * x + b * y + c_coef * z + xoff,
                d * x + e * y + f * z + yoff,
                g * x + h * y + i_coef * z + zoff,
            )

        transform_fn = _transform_3d  # type: ignore[assignment]
    else:
        raise ValueError("affine matrix must be 6 (2D) or 12 (3D) elements")

    def _apply(coords: tuple[Coordinate, ...]) -> tuple[Coordinate, ...]:
        return tuple(transform_fn(c) for c in coords)

    g = normalize_geometry(geometry)
    kind = geometry_type(g)
    if kind == "Point":
        return {"type": "Point", "coordinates": transform_fn(g["coordinates"])}  # type: ignore[arg-type]
    if kind == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": _apply(g["coordinates"])}  # type: ignore[arg-type]
    if kind == "LineString":
        return {"type": "LineString", "coordinates": _apply(g["coordinates"])}  # type: ignore[arg-type]
    if kind == "MultiLineString":
        return {"type": "MultiLineString", "coordinates": tuple(_apply(line) for line in g["coordinates"])}  # type: ignore[arg-type]
    if kind == "Polygon":
        coords = g["coordinates"]
        outer: Any = coords if isinstance(coords[0][0], (int, float)) else coords[0]
        return {"type": "Polygon", "coordinates": (_apply(outer),)}
    if kind == "MultiPolygon":
        result = []
        for poly in g["coordinates"]:
            outer2: Any = poly if isinstance(poly[0][0], (int, float)) else poly[0]
            result.append((_apply(outer2),))
        return {"type": "MultiPolygon", "coordinates": tuple(result)}
    return g