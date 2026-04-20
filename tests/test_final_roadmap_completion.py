from __future__ import annotations

from pathlib import Path

import geoprompt as gp


ROOT = Path(__file__).resolve().parents[1]

FINAL_FUNCTIONS = [
    "static_linking_option",
    "binary_wheel_build_targets",
    "data_encryption_in_transit_tls",
    "couchdb_spatial_views",
    "neo4j_spatial_integration",
    "tigergraph_spatial_graph",
    "tile38_real_time_geofencing",
    "h3_hexagonal_indexing",
    "s2_geometry_cell_indexing",
    "uber_h3_pandas_bridge",
    "quadkey_bing_maps_tiling",
    "discrete_global_grid_system",
    "rstar_tree_spatial_index_wrapper",
    "ball_tree_spatial_index",
]


def test_final_public_surface_presence() -> None:
    for name in FINAL_FUNCTIONS:
        assert hasattr(gp, name), name


def test_a8_build_and_tls_helpers() -> None:
    static = gp.static_linking_option(platform_name="linux", libraries=["proj", "geos"])
    assert static["mode"] == "static"
    assert "proj" in static["libraries"]

    wheels = gp.binary_wheel_build_targets(["3.10", "3.11", "3.12", "3.13"])
    assert "manylinux" in wheels["platforms"]
    assert "macos" in wheels["platforms"]
    assert "windows" in wheels["platforms"]

    tls = gp.data_encryption_in_transit_tls("https://example.com/api", minimum_version="1.3")
    assert tls["tls"] is True
    assert tls["minimum_version"] == "1.3"

    wheel_workflow = ROOT / ".github" / "workflows" / "wheel-build.yml"
    assert wheel_workflow.exists()


def test_a11_remaining_database_and_index_bridges() -> None:
    couch = gp.couchdb_spatial_views({"docs": [{"id": 1}, {"id": 2}]})
    assert couch["backend"] == "CouchDB"
    assert couch["count"] == 2

    neo = gp.neo4j_spatial_integration([("A", "B")])
    tiger = gp.tigergraph_spatial_graph([("A", "B"), ("B", "C")])
    assert neo["backend"] == "Neo4j"
    assert tiger["backend"] == "TigerGraph"

    fence = gp.tile38_real_time_geofencing((1.0, 1.0), [{"id": "zone-1", "bounds": (0.0, 0.0, 2.0, 2.0)}])
    assert fence["matched_ids"] == ["zone-1"]

    h3 = gp.h3_hexagonal_indexing(47.61, -122.33, resolution=6)
    assert h3.startswith("h3|")

    s2 = gp.s2_geometry_cell_indexing(47.61, -122.33, level=12)
    assert s2.startswith("s2|")

    bridge = gp.uber_h3_pandas_bridge([{"lat": 47.61, "lon": -122.33}], lat_column="lat", lon_column="lon")
    assert bridge[0]["h3_index"].startswith("h3|")

    quadkey = gp.quadkey_bing_maps_tiling(3, 5, zoom=4)
    assert isinstance(quadkey, str)
    assert len(quadkey) == 4

    dggs = gp.discrete_global_grid_system(47.61, -122.33, resolution=5)
    assert dggs["scheme"] == "DGGS"

    points = [(0.0, 0.0), (3.0, 4.0), (1.0, 1.0)]
    rstar = gp.rstar_tree_spatial_index_wrapper(points, query_point=(0.9, 0.9), k=2)
    assert len(rstar["nearest"]) == 2

    ball = gp.ball_tree_spatial_index(points, query_point=(0.0, 0.0), k=1)
    assert ball["nearest"][0]["point"] == (0.0, 0.0)
