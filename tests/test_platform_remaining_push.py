from __future__ import annotations

from pathlib import Path

import pytest

from geoprompt.frame import GeoPromptFrame
from geoprompt.io import read_service_url


def test_read_service_url_paginates_arcgis_payload(monkeypatch):
    from geoprompt import io as gio

    calls: list[str] = []

    def fake_read(url: str, timeout: float = 30.0, headers=None):
        calls.append(url)
        if "resultOffset=2" in url:
            return {
                "features": [
                    {"attributes": {"id": 3, "name": "c"}, "geometry": {"x": 2, "y": 2}},
                ],
                "spatialReference": {"wkid": 4326},
            }
        return {
            "features": [
                {"attributes": {"id": 1, "name": "a"}, "geometry": {"x": 0, "y": 0}},
                {"attributes": {"id": 2, "name": "b"}, "geometry": {"x": 1, "y": 1}},
            ],
            "exceededTransferLimit": True,
            "spatialReference": {"wkid": 4326},
        }

    monkeypatch.setattr(gio, "_read_json_source", fake_read)

    frame = read_service_url(
        "https://example.com/FeatureServer/0/query",
        out_fields=["id", "name"],
        where="1=1",
        page_size=2,
    )

    assert len(frame) == 3
    records = frame.to_records()
    assert records[0]["id"] == 1
    assert any("outFields=id,name" in call for call in calls)
    assert any("resultOffset=2" in call for call in calls)


def test_plotly_dashboard_builds_figure():
    pytest.importorskip("plotly")
    from geoprompt.viz import plotly_scenario_dashboard

    fig = plotly_scenario_dashboard(
        {"cost": 100, "coverage": 0.8},
        {"cost": 80, "coverage": 0.9},
        higher_is_better=["coverage"],
        title="Scenario",
    )
    assert fig is not None
    assert len(fig.data) >= 2


def test_raster_sampling_and_zonal_summary():
    from geoprompt.raster import inspect_raster, sample_raster_points, zonal_summary

    raster = {
        "data": [
            [1, 2],
            [3, 4],
        ],
        "transform": (0.0, 2.0, 1.0, 1.0),
        "nodata": None,
    }

    info = inspect_raster(raster)
    assert info["width"] == 2
    assert info["height"] == 2
    assert info["min"] == 1
    assert info["max"] == 4

    points = GeoPromptFrame(
        [
            {"id": "a", "geometry": {"type": "Point", "coordinates": (0.5, 1.5)}},
            {"id": "b", "geometry": {"type": "Point", "coordinates": (1.5, 0.5)}},
        ],
        geometry_column="geometry",
    )
    sampled = sample_raster_points(raster, points, value_column="elev")
    records = sampled.to_records()
    assert records[0]["elev"] == 1
    assert records[1]["elev"] == 4

    zones = GeoPromptFrame(
        [
            {"zone": "z1", "geometry": {"type": "Polygon", "coordinates": [(0, 1), (1, 1), (1, 2), (0, 2)]}},
            {"zone": "z2", "geometry": {"type": "Polygon", "coordinates": [(1, 0), (2, 0), (2, 1), (1, 1)]}},
        ],
        geometry_column="geometry",
    )
    summary = zonal_summary(raster, zones, zone_id_column="zone")
    rows = summary.to_records()
    assert rows[0]["mean"] == 1
    assert rows[1]["mean"] == 4


def test_topology_validate_and_snap():
    from geoprompt.topology import snap_points, validate_topology_rules

    points = GeoPromptFrame(
        [
            {"id": "a", "geometry": {"type": "Point", "coordinates": (0.0, 0.0)}},
            {"id": "b", "geometry": {"type": "Point", "coordinates": (0.05, 0.02)}},
            {"id": "c", "geometry": {"type": "Point", "coordinates": (2.0, 2.0)}},
        ],
        geometry_column="geometry",
    )
    snapped = snap_points(points, tolerance=0.1)
    snapped_rows = snapped.to_records()
    assert snapped_rows[0]["geometry"]["coordinates"] == snapped_rows[1]["geometry"]["coordinates"]

    polys = GeoPromptFrame(
        [
            {"id": 1, "geometry": {"type": "Polygon", "coordinates": [(0, 0), (2, 0), (2, 2), (0, 2)]}},
            {"id": 2, "geometry": {"type": "Polygon", "coordinates": [(1, 1), (3, 1), (3, 3), (1, 3)]}},
        ],
        geometry_column="geometry",
    )
    report = validate_topology_rules(polys, rule="must_not_overlap")
    assert report["valid"] is False
    assert report["violation_count"] == 1


def test_benchmark_history_table_and_export(tmp_path: Path):
    from geoprompt.compare import benchmark_history_table, export_benchmark_history

    (tmp_path / "bench-a.json").write_text(
        '{"version": "0.1.7", "summary": {"all_checks_passed": true}, "datasets": [{"dataset": "demo", "benchmarks": [{"operation": "demo.geoprompt.clip", "median_seconds": 1.0}, {"operation": "demo.reference.clip", "median_seconds": 2.0}]}]}',
        encoding="utf-8",
    )
    (tmp_path / "bench-b.json").write_text(
        '{"version": "0.1.8", "summary": {"all_checks_passed": true}, "datasets": [{"dataset": "demo", "benchmarks": [{"operation": "demo.geoprompt.clip", "median_seconds": 0.8}, {"operation": "demo.reference.clip", "median_seconds": 2.4}]}]}',
        encoding="utf-8",
    )

    table = benchmark_history_table(tmp_path)
    rows = table.to_records()
    assert len(rows) == 2
    assert rows[0]["version"] in {"0.1.7", "0.1.8"}

    exported = export_benchmark_history(tmp_path)
    assert (tmp_path / "benchmark_history.html").exists()
    assert "html" in exported


def test_frame_pipe_and_map_column():
    frame = GeoPromptFrame(
        [
            {"id": "a", "value": 2, "geometry": {"type": "Point", "coordinates": (0, 0)}},
            {"id": "b", "value": 3, "geometry": {"type": "Point", "coordinates": (1, 1)}},
        ],
        geometry_column="geometry",
    )
    mapped = frame.map_column("value", lambda v: v * 10)
    piped = mapped.pipe(lambda f: f.rename_columns({"value": "score"}))
    rows = piped.to_records()
    assert rows[0]["score"] == 20
    assert rows[1]["score"] == 30


def test_merge_indicator_and_validation():
    left = GeoPromptFrame(
        [
            {"id": "a", "geometry": {"type": "Point", "coordinates": (0, 0)}},
            {"id": "b", "geometry": {"type": "Point", "coordinates": (1, 1)}},
        ],
        geometry_column="geometry",
    )
    right = GeoPromptFrame(
        [
            {"id": "a", "name": "alpha", "geometry": {"type": "Point", "coordinates": (0, 0)}},
        ],
        geometry_column="geometry",
    )
    merged = left.merge(right, on="id", how="left", indicator=True)
    rows = merged.to_records()
    assert rows[0]["_merge"] == "both"
    assert rows[1]["_merge"] == "left_only"

    dup_right = GeoPromptFrame(
        [
            {"id": "a", "name": "x", "geometry": {"type": "Point", "coordinates": (0, 0)}},
            {"id": "a", "name": "y", "geometry": {"type": "Point", "coordinates": (0, 0)}},
        ],
        geometry_column="geometry",
    )
    with pytest.raises(ValueError):
        left.merge(dup_right, on="id", validate="one_to_one")


def test_spatial_weights_semivariogram_and_lag():
    from geoprompt.stats import semivariogram, spatial_lag, spatial_weights_matrix

    centroids = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    values = [10.0, 12.0, 18.0, 20.0]

    weights = spatial_weights_matrix(centroids, k=1, row_standardize=True)
    assert len(weights) == 4
    assert abs(sum(weights[0]) - 1.0) < 1e-9

    lag = spatial_lag(values, centroids, k=1)
    assert len(lag) == 4
    assert lag[0] == 12.0

    semi = semivariogram(centroids, values, bins=3)
    assert semi[0]["pair_count"] >= 1
    assert "semivariance" in semi[0]


def test_topology_rules_cover_and_self_intersection():
    from geoprompt.topology import validate_topology_rules

    lines = GeoPromptFrame(
        [
            {"id": 1, "geometry": {"type": "LineString", "coordinates": [(0, 0), (1, 1), (0, 1), (1, 0)]}},
        ],
        geometry_column="geometry",
    )
    report = validate_topology_rules(lines, rule="must_not_self_intersect")
    assert report["valid"] is False

    polys = GeoPromptFrame(
        [
            {"id": "inner", "geometry": {"type": "Polygon", "coordinates": [(1, 1), (2, 1), (2, 2), (1, 2)]}},
            {"id": "outside", "geometry": {"type": "Polygon", "coordinates": [(9, 9), (10, 9), (10, 10), (9, 10)]}},
        ],
        geometry_column="geometry",
    )
    cover = GeoPromptFrame(
        [
            {"id": "cover", "geometry": {"type": "Polygon", "coordinates": [(0, 0), (5, 0), (5, 5), (0, 5)]}},
        ],
        geometry_column="geometry",
    )
    covered_report = validate_topology_rules(polys, rule="must_be_covered_by", other=cover)
    assert covered_report["valid"] is False
    assert covered_report["violation_count"] == 1


def test_workspace_manifest_and_provenance_export(tmp_path: Path):
    from geoprompt.workspace import build_workspace_manifest, export_provenance_bundle

    manifest = build_workspace_manifest(
        name="demo-project",
        datasets=[{"name": "sites", "path": "data/sites.geojson", "crs": "EPSG:4326"}],
        steps=["load", "join", "export"],
        outputs=["outputs/report.html"],
    )
    assert manifest["name"] == "demo-project"
    assert manifest["dataset_count"] == 1

    written = export_provenance_bundle(tmp_path, manifest)
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "manifest.md").exists()
    assert "json" in written


def test_raster_terrain_metrics():
    from geoprompt.raster import raster_slope_aspect, raster_hillshade

    raster = {
        "data": [
            [100, 110, 120],
            [100, 110, 120],
            [100, 110, 120],
        ],
        "transform": (0.0, 3.0, 1.0, 1.0),
        "nodata": None,
    }

    terrain = raster_slope_aspect(raster)
    assert terrain["rows"] == 3
    assert terrain["cols"] == 3
    assert terrain["slope"][1][1] > 0

    shade = raster_hillshade(raster)
    assert shade["rows"] == 3
    assert 0 <= shade["grid"][1][1] <= 255


def test_temporal_resample_and_rolling_stats():
    from geoprompt.temporal import resample_time_series, rolling_window_stats, sort_by_time

    rows = [
        {"ts": "2026-01-01", "value": 10},
        {"ts": "2026-01-02", "value": 20},
        {"ts": "2026-01-05", "value": 30},
    ]
    ordered = sort_by_time(rows, "ts")
    assert ordered[0]["ts"] == "2026-01-01"

    daily = resample_time_series(rows, time_column="ts", value_column="value", freq="day")
    assert len(daily) == 3
    assert daily[0]["sum"] == 10

    rolling = rolling_window_stats(ordered, value_column="value", window=2)
    assert rolling[1]["rolling_mean"] == 15
    assert rolling[2]["rolling_mean"] == 25
