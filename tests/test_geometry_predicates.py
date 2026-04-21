from __future__ import annotations

import pytest

import geoprompt as gp


def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": (x, y)}


def _line(*coords: tuple[float, float]) -> dict:
    return {"type": "LineString", "coordinates": tuple(coords)}


def _polygon(*coords: tuple[float, float]) -> dict:
    ring = list(coords)
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": (tuple(ring),)}


def test_named_predicate_wrappers_delegate_to_geometry_engine() -> None:
    square = _polygon((0, 0), (4, 0), (4, 4), (0, 4))
    inner_point = _point(1, 1)
    edge_point = _point(0, 2)
    other_square = _polygon((2, 2), (6, 2), (6, 6), (2, 6))

    assert gp.covers(square, inner_point)
    assert gp.covered_by(inner_point, square)
    assert gp.contains_properly(square, inner_point)
    assert not gp.contains_properly(square, edge_point)
    assert gp.dwithin(inner_point, edge_point, 3.0)
    assert gp.relate(square, inner_point) == gp.de9im_relate(square, inner_point)
    assert gp.relate_pattern(square, inner_point, "0FFFFF0F2")
    assert gp.geometry_overlaps(square, other_square)


def test_crosses_touches_and_overlaps_on_canonical_cases() -> None:
    horizontal = _line((0, 0), (4, 0))
    vertical = _line((2, -2), (2, 2))
    touching = _line((4, 0), (6, 0))
    square = _polygon((0, 0), (4, 0), (4, 4), (0, 4))
    overlap_a = _polygon((0, 0), (4, 0), (4, 4), (0, 4))
    overlap_b = _polygon((2, 2), (6, 2), (6, 6), (2, 6))

    assert gp.geometry_crosses(horizontal, vertical)
    assert gp.geometry_touches(horizontal, touching)
    assert gp.geometry_crosses(vertical, square)
    assert gp.geometry_overlaps(overlap_a, overlap_b)


def test_predicate_parity_against_shapely_on_many_geometry_pairs() -> None:
    shapely_geometry = pytest.importorskip("shapely.geometry")
    shape = shapely_geometry.shape

    geometries = [
        _point(0, 0),
        _point(2, 2),
        _point(5, 5),
        _line((0, 0), (4, 0)),
        _line((2, -2), (2, 2)),
        _polygon((0, 0), (4, 0), (4, 4), (0, 4)),
        _polygon((2, 2), (6, 2), (6, 6), (2, 6)),
        _polygon((5, 5), (7, 5), (7, 7), (5, 7)),
    ]

    pairs_checked = 0
    for left in geometries:
        for right in geometries:
            left_shape = shape(left)
            right_shape = shape(right)
            assert gp.geometry_intersects(left, right) == left_shape.intersects(right_shape)
            assert gp.geometry_disjoint(left, right) == left_shape.disjoint(right_shape)
            assert gp.covers(left, right) == left_shape.covers(right_shape)
            pairs_checked += 1

    assert pairs_checked >= 50

    strict_contains_pairs = [
        (_polygon((0, 0), (4, 0), (4, 4), (0, 4)), _point(1, 1)),
        (_polygon((0, 0), (4, 0), (4, 4), (0, 4)), _polygon((1, 1), (2, 1), (2, 2), (1, 2))),
        (_polygon((2, 2), (6, 2), (6, 6), (2, 6)), _point(5, 5)),
    ]
    for left, right in strict_contains_pairs:
        left_shape = shape(left)
        right_shape = shape(right)
        assert gp.geometry_contains(left, right) == left_shape.contains(right_shape)
        assert gp.geometry_within(right, left) == right_shape.within(left_shape)
