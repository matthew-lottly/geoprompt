from __future__ import annotations

import json

import pytest

from geoprompt.frame import GeoPromptFrame
from geoprompt import io as io_module
from geoprompt.io import read_data, read_file, read_stac_catalog, read_wfs, to_file, write_data


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


def test_read_file_and_to_file_aliases_use_extension_auto_detection(tmp_path):
    path = tmp_path / "alias_roundtrip.geojson"
    frame = _sample_frame()

    written = to_file(frame, path)
    assert written == path

    restored = read_file(path)
    assert len(restored) == 2
    assert restored[0]["site_id"] == "a"


def test_unified_io_supports_feather_via_extension(tmp_path):
    pytest.importorskip("pyarrow")

    path = tmp_path / "records.feather"
    frame = _sample_frame()

    write_data(path, frame)
    restored = read_data(path)

    assert len(restored) == 2
    assert restored[1]["site_id"] == "b"


def test_unified_io_supports_gml_roundtrip(tmp_path):
    path = tmp_path / "records.gml"
    frame = _sample_frame()

    write_data(path, frame)
    restored = read_data(path)

    assert len(restored) == 2
    assert {row["site_id"] for row in restored} == {"a", "b"}


def test_read_data_dispatches_filegdb_extension(monkeypatch, tmp_path):
    calls: list[str] = []

    def _fake_read_with_geopandas(path, **kwargs):
        calls.append(str(path))
        return GeoPromptFrame.from_records([
            {"site_id": "gdb", "geometry": _point(1, 1)},
        ])

    monkeypatch.setattr(io_module, "_read_with_geopandas", _fake_read_with_geopandas)

    gdb_path = tmp_path / "example.gdb"
    gdb_path.mkdir()
    frame = read_data(gdb_path)
    assert len(frame) == 1
    assert frame[0]["site_id"] == "gdb"
    assert calls


def test_read_stac_catalog_non_recursive_sets_empty_links(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog = {
        "type": "Catalog",
        "id": "demo",
        "links": [{"rel": "child", "href": "child.json"}],
    }
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    result = read_stac_catalog(catalog_path, recursive=False)
    assert result["id"] == "demo"
    assert result["resolved_links"] == []


def test_read_wfs_builds_rows_from_geojson_payload(monkeypatch):
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            payload = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"site_id": "w1"},
                        "geometry": _point(0, 0),
                    }
                ],
            }
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(io_module, "urlopen", lambda *args, **kwargs: _FakeResponse())
    frame = read_wfs("https://example.com/wfs", "layer_name", max_features=5)

    assert len(frame) == 1
    assert frame[0]["site_id"] == "w1"


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
