from __future__ import annotations

import math
import random

from geoprompt.geometry import geometry_area, geometry_distance, geometry_simplify, translate_geometry


def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": [x, y]}


def _square(x: float, y: float, size: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[
            (x, y),
            (x + size, y),
            (x + size, y + size),
            (x, y + size),
            (x, y),
        ]],
    }


def test_randomized_distance_symmetry_and_identity() -> None:
    random.seed(42)
    for _ in range(100):
        x1, y1 = random.uniform(-100, 100), random.uniform(-100, 100)
        x2, y2 = random.uniform(-100, 100), random.uniform(-100, 100)
        a = _point(x1, y1)
        b = _point(x2, y2)
        assert math.isclose(geometry_distance(a, b), geometry_distance(b, a), rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(geometry_distance(a, a), 0.0, rel_tol=1e-9, abs_tol=1e-9)


def test_randomized_translation_preserves_area() -> None:
    random.seed(7)
    for _ in range(50):
        x, y = random.uniform(-50, 50), random.uniform(-50, 50)
        size = random.uniform(0.5, 10.0)
        poly = _square(x, y, size)
        moved = translate_geometry(poly, random.uniform(-20, 20), random.uniform(-20, 20))
        assert math.isclose(geometry_area(poly), geometry_area(moved), rel_tol=1e-9, abs_tol=1e-9)


def test_zero_tolerance_simplify_is_area_stable() -> None:
    poly = _square(0, 0, 3)
    simplified = geometry_simplify(poly, 0.0)
    assert math.isclose(geometry_area(poly), geometry_area(simplified), rel_tol=1e-9, abs_tol=1e-9)
