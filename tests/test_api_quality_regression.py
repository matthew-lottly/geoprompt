from __future__ import annotations

import math

import pytest

from geoprompt.equations import coordinate_distance
from geoprompt.geometry import (
    geometry_area,
    geometry_bounds,
    geometry_distance,
    geometry_intersects,
    geometry_within,
)


POINTS = [
    {"type": "Point", "coordinates": [0.0, 0.0]},
    {"type": "Point", "coordinates": [3.0, 4.0]},
    {"type": "Point", "coordinates": [1.5, -2.5]},
]

POLYGONS = [
    {
        "type": "Polygon",
        "coordinates": [[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]],
    },
    {
        "type": "Polygon",
        "coordinates": [[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0), (1.0, 1.0)]],
    },
]


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((0.0, 0.0), (0.0, 0.0), 0.0),
        ((0.0, 0.0), (3.0, 4.0), 5.0),
        ((-1.0, -1.0), (2.0, 3.0), 5.0),
    ],
)
def test_coordinate_distance_edge_cases(a, b, expected) -> None:
    assert math.isclose(coordinate_distance(a, b), expected, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(coordinate_distance(a, b), coordinate_distance(b, a), rel_tol=1e-9, abs_tol=1e-9)


def test_geometry_bounds_and_area_are_stable() -> None:
    bounds = geometry_bounds(POLYGONS[0])
    assert bounds[0] == 0.0
    assert bounds[3] == 2.0
    assert math.isclose(geometry_area(POLYGONS[0]), 4.0, rel_tol=1e-9, abs_tol=1e-9)


def test_geometry_predicates_on_locked_corpus() -> None:
    assert geometry_intersects(POLYGONS[0], POLYGONS[1]) is True
    outside_point = {"type": "Point", "coordinates": [-1.0, -1.0]}
    assert geometry_within(outside_point, POLYGONS[0]) is False
    inside_point = {"type": "Point", "coordinates": [1.0, 1.0]}
    assert geometry_within(inside_point, POLYGONS[0]) is True


def test_geometry_distance_zero_for_same_geometry() -> None:
    point = {"type": "Point", "coordinates": [5.0, 5.0]}
    assert math.isclose(geometry_distance(point, point), 0.0, rel_tol=1e-9, abs_tol=1e-9)


def test_optional_shapely_parity_on_golden_corpus() -> None:
    shapely = pytest.importorskip("shapely.geometry")

    polygon = shapely.Polygon(POLYGONS[0]["coordinates"][0])
    point = shapely.Point(1.0, 1.0)

    assert math.isclose(geometry_area(POLYGONS[0]), polygon.area, rel_tol=1e-9, abs_tol=1e-9)
    assert geometry_intersects(POLYGONS[0], POLYGONS[1]) == polygon.intersects(shapely.Polygon(POLYGONS[1]["coordinates"][0]))
    assert geometry_within({"type": "Point", "coordinates": [1.0, 1.0]}, POLYGONS[0]) == point.within(polygon)
