from __future__ import annotations

import json

import pytest

from geoprompt.frame import GeoPromptFrame
from geoprompt.io import read_data, write_data


def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": [x, y]}


def _sample_frame() -> GeoPromptFrame:
    return GeoPromptFrame.from_records(
        [
            {"site_id": "a", "status": "open", "value": 1, "geometry": _point(0, 0)},
            {"site_id": "b", "status": "closed", "value": 2, "geometry": _point(5, 5)},
        ],
        crs="EPSG:4326",
    )


def test_read_data_supports_attribute_filter_and_geometry_mask(tmp_path):
    path = tmp_path / "features.geojson"
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"site_id": "a", "status": "open", "value": 1},
                "geometry": _point(0, 0),
            },
            {
                "type": "Feature",
                "properties": {"site_id": "b", "status": "closed", "value": 2},
                "geometry": _point(5, 5),
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    mask = {
        "type": "Polygon",
        "coordinates": [[(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]],
    }

    frame = read_data(
        path,
        where={"status": "open"},
        geometry_mask=mask,
        use_columns=["site_id", "status"],
    )

    assert len(frame) == 1
    assert frame[0]["site_id"] == "a"
    assert "value" not in frame[0]


def test_write_data_supports_append_and_overwrite_modes(tmp_path):
    path = tmp_path / "records.csv"

    first = GeoPromptFrame.from_records([
        {"site_id": "a", "geometry": _point(0, 0)},
    ])
    second = GeoPromptFrame.from_records([
        {"site_id": "b", "geometry": _point(1, 1)},
    ])
    replacement = GeoPromptFrame.from_records([
        {"site_id": "c", "geometry": _point(2, 2)},
    ])

    write_data(path, first, mode="overwrite")
    write_data(path, second, mode="append")

    appended = read_data(path, geometry_column="geometry")
    assert [row["site_id"] for row in appended] == ["a", "b"]

    write_data(path, replacement, mode="overwrite")
    overwritten = read_data(path, geometry_column="geometry")
    assert [row["site_id"] for row in overwritten] == ["c"]


def test_frame_pandas_roundtrip_and_dataframe_protocol():
    pd = pytest.importorskip("pandas")

    frame = _sample_frame()
    dataframe = frame.to_pandas()
    assert list(dataframe["site_id"]) == ["a", "b"]

    restored = GeoPromptFrame.from_pandas(dataframe, geometry_column="geometry", crs="EPSG:4326")
    assert len(restored) == 2
    assert restored[1]["status"] == "closed"

    interchange = frame.__dataframe__()
    assert interchange is not None


def test_frame_arrow_roundtrip():
    pytest.importorskip("pyarrow")

    frame = _sample_frame()
    table = frame.to_arrow()
    restored = GeoPromptFrame.from_arrow(table, geometry_column="geometry", crs="EPSG:4326")

    assert len(restored) == 2
    assert restored[0]["site_id"] == "a"


def test_to_folium_map_supports_cluster_and_controls():
    pytest.importorskip("folium")

    from geoprompt.viz import to_folium_map

    frame = _sample_frame()
    fmap = to_folium_map(
        frame,
        cluster_points=True,
        fullscreen=True,
        minimap=True,
        measure_control=True,
        add_layer_control=True,
        custom_tile_url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    )

    html = fmap.get_root().render()
    assert "OpenStreetMap" in html or "tile.openstreetmap.org" in html


def test_external_bridge_command_builders():
    from geoprompt.geoprocessing import (
        gdal_command,
        grass_command,
        postgis_function_sql,
        qgis_process_command,
        saga_command,
        whitebox_command,
    )

    assert gdal_command("info", "input.tif") == ["gdal", "info", "input.tif"]
    assert qgis_process_command("native:buffer", {"INPUT": "roads.gpkg"})[:2] == ["qgis_process", "run"]
    assert whitebox_command("BreachDepressions", dem="dem.tif")[0] == "whitebox_tools"
    assert grass_command("v.buffer", input="roads", output="roads_buf")[0] == "grass"
    assert saga_command("shapes_polygons", "Polygon Dissolve", INPUT="zones.shp")[0] == "saga_cmd"

    sql = postgis_function_sql("ST_Buffer", "assets", distance=25)
    assert "ST_Buffer" in sql
    assert "FROM assets" in sql


def test_frame_spatial_index_lifecycle():
    frame = _sample_frame()

    index = frame.build_spatial_index(cell_size=2.0)
    assert frame.spatial_index is index
    assert index.query(-1, -1, 1, 1)

    frame.clear_spatial_index()
    assert frame.spatial_index is None
