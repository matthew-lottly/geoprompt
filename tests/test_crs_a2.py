from __future__ import annotations

from geoprompt.crs import (
    auto_detect_coordinate_format,
    batch_project_features,
    coordinate_to_web_mercator,
    coordinate_to_wgs84,
    crs_equal,
    crs_from_epsg,
    crs_from_json,
    crs_from_proj,
    crs_from_wkt,
    crs_to_json,
    define_projection,
    geodesic_midpoint,
    parse_coordinate_text,
    project_on_the_fly,
    utm_zone_for_lonlat,
    xy_table_to_geographic_features,
)
from geoprompt.frame import GeoPromptFrame


def test_crs_creation_and_roundtrip() -> None:
    crs = crs_from_epsg(4326)
    assert crs["name"] == "EPSG:4326"
    assert crs_equal(crs, crs_from_json(crs_to_json(crs)))


def test_crs_input_parsers() -> None:
    proj = crs_from_proj("+proj=longlat +datum=WGS84 +no_defs")
    wkt = crs_from_wkt('GEOGCS["WGS 84",DATUM["WGS_1984"]]')

    assert proj["type"] == "CRS"
    assert wkt["type"] == "CRS"


def test_web_mercator_helpers() -> None:
    x_value, y_value = coordinate_to_web_mercator(-111.93, 40.77)
    lon, lat = coordinate_to_wgs84(x_value, y_value)

    assert abs(lon + 111.93) < 0.01
    assert abs(lat - 40.77) < 0.01


def test_utm_zone_and_coordinate_parsing() -> None:
    assert utm_zone_for_lonlat(-111.93, 40.77) == "EPSG:32612"
    assert auto_detect_coordinate_format("40°46'12\"N 111°53'24\"W") in {"dms", "latlon"}
    lon, lat = parse_coordinate_text("40°46'12\"N, 111°53'24\"W")
    assert lon < 0
    assert lat > 0


def test_xy_table_to_features_and_batch_project() -> None:
    rows = [{"id": 1, "x": -111.93, "y": 40.77}]
    features = xy_table_to_geographic_features(rows, x_field="x", y_field="y")
    frame = GeoPromptFrame.from_records(features, crs="EPSG:4326")
    projected = batch_project_features([frame], "EPSG:3857")

    assert features[0]["geometry"]["type"] == "Point"
    assert projected[0].crs == "EPSG:3857"


def test_geodesic_midpoint() -> None:
    lon, lat = geodesic_midpoint((-111.93, 40.77), (-111.83, 40.87))

    assert -111.93 < lon < -111.83
    assert 40.77 < lat < 40.87


def test_define_projection_and_preserve_z() -> None:
    frame = GeoPromptFrame.from_records(
        [{"id": 1, "geometry": {"type": "Point", "coordinates": (-111.93, 40.77, 1500.0)}}]
    )
    defined = define_projection(frame, "EPSG:4326")
    projected, message = project_on_the_fly(defined, "EPSG:3857")

    coords = projected.to_records()[0]["geometry"]["coordinates"]
    assert projected.crs == "EPSG:3857"
    assert len(coords) >= 3
    assert coords[2] == 1500.0
    assert "Projected on the fly" in message
