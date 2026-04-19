from __future__ import annotations

from geoprompt.geometry import (
    bezier_curve,
    build_spatial_index_rtree,
    constrained_delaunay_triangulation,
    create_arc,
    create_ellipse,
    create_routes_from_lines,
    create_spiral,
    generate_near_table,
    geometry_esrijson_read,
    geometry_esrijson_write,
    geodesic_buffer,
    route_event_table_to_features,
    spatial_index_knn_query,
    spatial_index_window_query,
)


def test_geodesic_buffer_returns_polygon() -> None:
    point = {"type": "Point", "coordinates": (-111.93, 40.77)}
    buffered = geodesic_buffer(point, 1000, segments=24)

    assert buffered["type"] == "Polygon"
    assert len(buffered["coordinates"][0]) >= 24


def test_generate_near_table_finds_matches() -> None:
    features = [
        {"id": 1, "geometry": {"type": "Point", "coordinates": (0, 0)}},
        {"id": 2, "geometry": {"type": "Point", "coordinates": (2, 0)}},
    ]
    others = [
        {"id": 10, "geometry": {"type": "Point", "coordinates": (0.5, 0)}},
        {"id": 11, "geometry": {"type": "Point", "coordinates": (2.5, 0)}},
    ]

    table = generate_near_table(features, others, search_radius=2.0)

    assert len(table) == 2
    assert table[0]["IN_FID"] == 0
    assert table[0]["NEAR_FID"] == 0
    assert table[1]["NEAR_FID"] == 1


def test_spatial_index_queries_work() -> None:
    features = [
        {"id": 1, "geometry": {"type": "Point", "coordinates": (0, 0)}},
        {"id": 2, "geometry": {"type": "Point", "coordinates": (5, 5)}},
        {"id": 3, "geometry": {"type": "Point", "coordinates": (10, 10)}},
    ]

    index = build_spatial_index_rtree(features)
    hits = spatial_index_window_query(index, (4, 4, 6, 6))
    knn = spatial_index_knn_query(index, {"type": "Point", "coordinates": (4.9, 5.1)}, k=2)

    assert [hit["id"] for hit in hits] == [2]
    assert knn[0]["id"] == 2
    assert len(knn) == 2


def test_esrijson_roundtrip() -> None:
    geometry = {
        "type": "Polygon",
        "coordinates": (((0, 0), (1, 0), (1, 1), (0, 1), (0, 0)),),
    }

    esri = geometry_esrijson_write(geometry)
    restored = geometry_esrijson_read(esri)

    assert esri["rings"]
    assert restored["type"] == "Polygon"
    assert restored["coordinates"][0][0] == (0.0, 0.0)


def test_route_events_and_curve_helpers() -> None:
    routes = create_routes_from_lines(
        [{"route_id": "R1", "geometry": {"type": "LineString", "coordinates": ((0, 0), (10, 0))}}]
    )
    features = route_event_table_to_features(
        routes,
        [{"route_id": "R1", "from_m": 2, "to_m": 6, "name": "segment"}],
    )

    arc = create_arc((0, 0), 1, 0, 90, segments=8)
    ellipse = create_ellipse((0, 0), 2, 1, segments=16)
    spiral = create_spiral((0, 0), end_radius=2, turns=2, segments=16)
    curve = bezier_curve(((0, 0), (1, 2), (2, 0)), segments=12)

    assert len(features) == 1
    assert features[0]["geometry"]["type"] == "LineString"
    assert arc["type"] == "LineString"
    assert ellipse["type"] == "Polygon"
    assert spiral["type"] == "LineString"
    assert curve["type"] == "LineString"


def test_constrained_delaunay_fallback_returns_triangles() -> None:
    points = [
        {"type": "Point", "coordinates": (0, 0)},
        {"type": "Point", "coordinates": (1, 0)},
        {"type": "Point", "coordinates": (0, 1)},
    ]

    triangles = constrained_delaunay_triangulation(points)

    assert len(triangles) >= 1
    assert triangles[0]["type"] == "Polygon"
