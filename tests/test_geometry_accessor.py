from __future__ import annotations

from geoprompt.frame import GeoPromptFrame


def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": [x, y]}


def _polygon(coords: list) -> dict:
    return {"type": "Polygon", "coordinates": [coords]}


def test_geom_accessor_area_length_centroid_and_bounds() -> None:
    frame = GeoPromptFrame([
        {"id": 1, "geometry": _polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])},
        {"id": 2, "geometry": _point(5, 5)},
    ])

    areas = frame.geom.area()
    lengths = frame.geom.length()
    centroids = frame.geom.centroid()
    bounds = frame.geom.bounds()

    assert areas[0] == 4.0
    assert areas[1] == 0.0
    assert lengths[0] > 0.0
    assert centroids[0] == (1.0, 1.0)
    assert bounds[0] == (0.0, 0.0, 2.0, 2.0)


def test_geom_accessor_validity_summary() -> None:
    frame = GeoPromptFrame([
        {"id": 1, "geometry": _polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])},
    ])

    report = frame.geom.validity()
    assert report[0]["is_valid"] is True
