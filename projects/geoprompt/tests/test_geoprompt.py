import json
from pathlib import Path

from geoprompt import geometry_centroid
from geoprompt.demo import build_demo_report
from geoprompt.equations import area_similarity, corridor_strength, directional_alignment, euclidean_distance, haversine_distance, prompt_decay, prompt_interaction
from geoprompt.io import frame_to_geojson, read_features, read_geojson, read_points, write_geojson


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_read_points_and_centroid() -> None:
    frame = read_points(PROJECT_ROOT / "data" / "sample_points.json")

    assert len(frame) == 5
    assert frame.columns[0] == "site_id"
    centroid = frame.centroid()
    assert round(centroid[0], 3) == -111.928
    assert round(centroid[1], 3) == 40.684


def test_distance_and_decay_equations() -> None:
    distance_value = euclidean_distance((-111.92, 40.78), (-111.96, 40.71))
    geographic_distance = haversine_distance((-111.92, 40.78), (-111.96, 40.71))

    assert round(distance_value, 4) == 0.0806
    assert round(geographic_distance, 3) == 8.482
    assert round(prompt_decay(distance_value=distance_value, scale=0.14, power=1.6), 4) == 0.4830
    assert round(prompt_interaction(0.71, 0.88, distance_value=distance_value, scale=0.16, power=1.5), 4) == 0.3388


def test_neighborhood_pressure_and_anchor_influence() -> None:
    frame = read_points(PROJECT_ROOT / "data" / "sample_points.json")

    pressure = frame.neighborhood_pressure(weight_column="demand_index", scale=0.14, power=1.6)
    anchor = frame.anchor_influence(weight_column="priority_index", anchor="north-hub", scale=0.14, power=1.4)

    assert len(pressure) == len(frame)
    assert pressure[1] == max(pressure)
    assert anchor[0] == max(anchor)


def test_mixed_geometries_and_corridor_strength() -> None:
    frame = read_features(PROJECT_ROOT / "data" / "sample_features.json")

    assert sorted(set(frame.geometry_types())) == ["LineString", "Point", "Polygon"]
    lengths = frame.geometry_lengths()
    areas = frame.geometry_areas()
    corridor = frame.corridor_accessibility(weight_column="capacity_index", anchor="north-hub-point", scale=0.18, power=1.4)

    assert max(lengths) > 0.14
    assert max(areas) > 0.008
    assert len(corridor) == len(frame)
    assert corridor[0] == 0.0


def test_line_centroid_uses_segment_length_weighting() -> None:
    centroid = geometry_centroid(
        {
            "type": "LineString",
            "coordinates": [(0.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        }
    )

    assert centroid == (1.5, 0.0)


def test_geojson_round_trip_and_nearest_neighbors(tmp_path: Path) -> None:
    frame = read_features(PROJECT_ROOT / "data" / "sample_features.json", crs="EPSG:4326")
    geojson_path = write_geojson(tmp_path / "sample_features.geojson", frame)
    reloaded = read_geojson(geojson_path)
    neighbors = reloaded.nearest_neighbors(k=1)
    geographic_neighbors = reloaded.nearest_neighbors(k=1, distance_method="haversine")
    feature_collection = frame_to_geojson(reloaded)

    assert len(reloaded) == len(frame)
    assert len(neighbors) == len(frame)
    assert len(geographic_neighbors) == len(frame)
    assert neighbors[0]["rank"] == 1
    assert geographic_neighbors[0]["distance_method"] == "haversine"
    assert feature_collection["type"] == "FeatureCollection"
    assert feature_collection["crs"]["properties"]["name"] == "EPSG:4326"
    assert len(feature_collection["features"]) == len(frame)


def test_query_bounds_modes() -> None:
    frame = read_features(PROJECT_ROOT / "data" / "sample_features.json")

    intersects = frame.query_bounds(min_x=-111.97, min_y=40.68, max_x=-111.84, max_y=40.79, mode="intersects")
    within = frame.query_bounds(min_x=-111.97, min_y=40.68, max_x=-111.84, max_y=40.79, mode="within")
    centroid = frame.query_bounds(min_x=-111.97, min_y=40.68, max_x=-111.84, max_y=40.79, mode="centroid")

    assert len(intersects) >= len(within)
    assert len(intersects) >= len(centroid)
    assert sorted(record["site_id"] for record in within) == [
        "central-yard-point",
        "north-hub-point",
    ]


def test_read_geojson_feature_collection(tmp_path: Path) -> None:
    geojson_path = tmp_path / "feature_collection.geojson"
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [
            {
                "type": "Feature",
                "id": "point-a",
                "properties": {"name": "Point A", "demand_index": 1.0},
                "geometry": {"type": "Point", "coordinates": [-111.92, 40.78]},
            },
            {
                "type": "Feature",
                "id": "point-b",
                "properties": {"name": "Point B", "demand_index": 0.8},
                "geometry": {"type": "Point", "coordinates": [-111.96, 40.71]},
            },
        ],
    }
    geojson_path.write_text(json.dumps(payload), encoding="utf-8")

    frame = read_geojson(geojson_path)

    assert len(frame) == 2
    assert frame.crs == "EPSG:4326"
    assert sorted(frame.geometry_types()) == ["Point", "Point"]


def test_set_crs_and_reproject() -> None:
    frame = read_features(PROJECT_ROOT / "data" / "sample_features.json", crs="EPSG:4326")

    projected = frame.to_crs("EPSG:3857")
    first_point = projected.head(1)[0]["geometry"]["coordinates"]

    assert projected.crs == "EPSG:3857"
    assert abs(first_point[0]) > 10000000
    assert abs(first_point[1]) > 1000000


def test_spatial_join_contains_regions() -> None:
    regions = read_features(PROJECT_ROOT / "data" / "benchmark_regions.json", crs="EPSG:4326")
    features = read_features(PROJECT_ROOT / "data" / "benchmark_features.json", crs="EPSG:4326")

    joined = regions.spatial_join(features, predicate="intersects")
    left_joined = regions.spatial_join(features, predicate="intersects", how="left")
    pairs = {f"{row['region_id']}->{row['site_id']}" for row in joined}

    assert "northwest-sector->alpha-point" in pairs
    assert "northeast-sector->beta-point" in pairs
    assert "southwest-sector->gamma-point" in pairs
    assert "southeast-sector->delta-point" in pairs
    assert all("remote-point" not in pair for pair in pairs)
    assert len(left_joined) >= len(joined)


def test_clip_and_overlay_intersections() -> None:
    regions = read_features(PROJECT_ROOT / "data" / "benchmark_regions.json", crs="EPSG:4326")
    features = read_features(PROJECT_ROOT / "data" / "benchmark_features.json", crs="EPSG:4326")

    clipped = features.clip(regions.query_bounds(min_x=-112.02, min_y=40.62, max_x=-111.92, max_y=40.82))
    intersections = regions.overlay_intersections(features)

    assert len(clipped) > 0
    assert any(record["site_id"] == "north-connector-line" for record in clipped)
    assert any(record["region_id"] == "northwest-sector" and record["site_id"] == "alpha-point" for record in intersections)
    assert any(record["region_id"] == "southeast-sector" and record["site_id"] == "delta-point" for record in intersections)


def test_dissolve_regions_by_band() -> None:
    regions = read_features(PROJECT_ROOT / "data" / "benchmark_regions.json", crs="EPSG:4326")

    dissolved = regions.dissolve(by="region_band", aggregations={"region_name": "count"})
    records = sorted(dissolved.to_records(), key=lambda item: str(item["region_band"]))

    assert len(records) == 2
    assert records[0]["region_band"] == "north"
    assert records[0]["region_name"] == 2
    assert records[1]["region_band"] == "south"
    assert records[1]["region_name"] == 2
    assert all(record["geometry"]["type"] == "Polygon" for record in records)


def test_area_similarity_equation() -> None:
    similarity = area_similarity(origin_area=0.010, destination_area=0.009, distance_value=0.05, scale=0.2, power=1.2)
    corridor = corridor_strength(weight=0.95, corridor_length=0.15, distance_value=0.08, scale=0.18, power=1.4)

    assert round(similarity, 4) == 0.6886
    assert round(corridor, 4) == 0.0793


def test_directional_alignment() -> None:
    alignment = directional_alignment((-111.96, 40.71), (-111.78, 40.66), preferred_bearing=90.0)

    assert alignment > 0.7


def test_build_demo_report(tmp_path: Path) -> None:
    report = build_demo_report(
        input_path=PROJECT_ROOT / "data" / "sample_features.json",
        output_dir=tmp_path,
    )

    assert report["package"] == "geoprompt"
    assert len(report["records"]) == 6
    assert len(report["top_interactions"]) == 5
    assert len(report["top_area_similarity"]) == 5
    assert len(report["top_nearest_neighbors"]) == 6
    assert len(report["top_geographic_neighbors"]) == 6
    assert report["summary"]["geometry_types"] == ["LineString", "Point", "Polygon"]
    assert report["summary"]["crs"] == "EPSG:4326"
    assert report["summary"]["projected_bounds_3857"]["min_x"] < -12000000
    assert report["summary"]["valley_window_feature_count"] == 3
    assert (tmp_path / "charts" / "neighborhood-pressure-review.png").exists()