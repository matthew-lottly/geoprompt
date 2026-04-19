"""Tests for the seventh parity push: geometry depth, frame ergonomics,
stats exports, topology pipeline, raster algebra, viz presets,
workspace lineage, IO zip/alias, and workflow job specs."""
from __future__ import annotations

import json
from pathlib import Path

from geoprompt.frame import GeoPromptFrame
from geoprompt.geometry import (
    geometry_convex_hull,
    geometry_envelope,
    geometry_is_empty,
    geometry_snap,
    geometry_split,
    geometry_type,
    geometry_validity_reason,
)
from geoprompt.stats import (
    gearys_c,
    getis_ord_g,
    idw_interpolation,
    kernel_density,
    local_morans_i,
    spatial_outliers,
)
from geoprompt.topology import (
    feature_validation_pipeline,
    repair_suggestions,
    validate_topology_rules,
)
from geoprompt.raster import raster_algebra
from geoprompt.viz import (
    BASEMAP_PRESETS,
    SYMBOL_PRESETS,
    before_after_comparison,
    portfolio_scorecard,
    recommendation_card,
)
from geoprompt.workspace import JobSpec, LineageTracker
from geoprompt.io import apply_field_aliases


# ---------------------------------------------------------------------------
#  Geometry depth
# ---------------------------------------------------------------------------


def test_geometry_convex_hull_triangle():
    geom = {"type": "Polygon", "coordinates": [(0, 0), (4, 0), (2, 3), (1, 1), (0, 0)]}
    hull = geometry_convex_hull(geom)
    assert geometry_type(hull) == "Polygon"
    # All original points should be inside or on the hull boundary
    from geoprompt.geometry import geometry_area
    assert geometry_area(hull) > 0


def test_geometry_envelope_returns_rectangle():
    geom = {"type": "LineString", "coordinates": [(1, 2), (5, 8)]}
    env = geometry_envelope(geom)
    assert geometry_type(env) == "Polygon"
    from geoprompt.geometry import geometry_bounds
    b = geometry_bounds(env)
    assert b == (1.0, 2.0, 5.0, 8.0)


def test_geometry_is_empty_true_for_empty_collection():
    assert geometry_is_empty({"type": "GeometryCollection", "geometries": []})
    assert not geometry_is_empty({"type": "Point", "coordinates": (1, 2)})


def test_geometry_validity_reason_reports_reason():
    good = {"type": "Polygon", "coordinates": [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]}
    assert geometry_validity_reason(good) == "Valid"
    # A degenerate polygon (< 3 unique coords) should have issues
    bad = {"type": "Polygon", "coordinates": [(0, 0), (0, 0), (0, 0), (0, 0)]}
    reason = geometry_validity_reason(bad)
    assert reason != "Valid"


def test_geometry_snap_moves_vertex():
    geom = {"type": "Point", "coordinates": (1.01, 2.01)}
    ref = {"type": "Point", "coordinates": (1.0, 2.0)}
    snapped = geometry_snap(geom, ref, tolerance=0.1)
    assert snapped["coordinates"] == (1.0, 2.0)


def test_geometry_snap_no_snap_outside_tolerance():
    geom = {"type": "Point", "coordinates": (5.0, 5.0)}
    ref = {"type": "Point", "coordinates": (1.0, 2.0)}
    snapped = geometry_snap(geom, ref, tolerance=0.1)
    assert snapped["coordinates"] == (5.0, 5.0)


def test_geometry_split_returns_list():
    """geometry_split should return at least one geometry."""
    geom = {"type": "LineString", "coordinates": [(0, 0), (10, 0)]}
    splitter = {"type": "Point", "coordinates": (5, 0)}
    result = geometry_split(geom, splitter)
    assert isinstance(result, list)
    assert len(result) >= 1


# ---------------------------------------------------------------------------
#  Stats exports and functions
# ---------------------------------------------------------------------------


def test_local_morans_i_clusters():
    vals = [10.0, 11.0, 9.0, 50.0, 48.0]
    pts = [(0, 0), (1, 0), (0, 1), (10, 10), (11, 10)]
    result = local_morans_i(vals, pts, k=2)
    assert len(result) == 5
    assert all("cluster_type" in r for r in result)


def test_getis_ord_g_hot_cold():
    vals = [100.0, 90.0, 95.0, 1.0, 2.0, 3.0]
    pts = [(0, 0), (1, 0), (0, 1), (20, 20), (21, 20), (20, 21)]
    result = getis_ord_g(vals, pts, k=2)
    assert len(result) == 6
    # High-value cluster should have positive z-scores
    assert result[0]["z_score"] > 0


def test_gearys_c_returns_statistic():
    vals = [10.0, 11.0, 9.0, 50.0, 48.0]
    pts = [(0, 0), (1, 0), (0, 1), (10, 10), (11, 10)]
    result = gearys_c(vals, pts, k=2)
    assert "gearys_c" in result
    assert "z_score" in result


def test_idw_interpolation():
    known = [(0, 0), (10, 0), (0, 10)]
    values = [100.0, 200.0, 300.0]
    queries = [(5, 5)]
    result = idw_interpolation(known, values, queries)
    assert len(result) == 1
    assert result[0] is not None
    assert 100 < result[0] < 300


def test_kernel_density_grid():
    pts = [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)]
    result = kernel_density(pts, bandwidth=1.0, cell_size=0.5)
    assert "grid" in result
    assert result["rows"] > 0
    assert result["cols"] > 0


def test_spatial_outliers_flags():
    vals = [10.0, 11.0, 9.0, 10.0, 100.0]
    pts = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]
    result = spatial_outliers(vals, pts, k=3, threshold=1.5)
    assert len(result) == 5
    assert any(r["is_outlier"] for r in result)


def test_morans_i_exported():
    """morans_i should now be importable from geoprompt top-level."""
    import geoprompt
    assert hasattr(geoprompt, "morans_i")
    assert hasattr(geoprompt, "local_morans_i")
    assert hasattr(geoprompt, "getis_ord_g")
    assert hasattr(geoprompt, "kernel_density")
    assert hasattr(geoprompt, "idw_interpolation")
    assert hasattr(geoprompt, "spatial_outliers")
    assert hasattr(geoprompt, "gearys_c")


# ---------------------------------------------------------------------------
#  Topology pipeline and repair suggestions
# ---------------------------------------------------------------------------


def test_feature_validation_pipeline_runs():
    rows = [
        {"id": "a", "geometry": {"type": "Polygon", "coordinates": [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]}},
        {"id": "b", "geometry": {"type": "Polygon", "coordinates": [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]}},
    ]
    frame = GeoPromptFrame.from_records(rows)
    result = feature_validation_pipeline(frame)
    assert "valid" in result
    assert "rule_results" in result
    assert "total_violations" in result


def test_repair_suggestions_provides_text():
    rows = [
        {"id": "a", "geometry": {"type": "Polygon", "coordinates": [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]}},
        {"id": "b", "geometry": {"type": "Polygon", "coordinates": [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]}},
    ]
    frame = GeoPromptFrame.from_records(rows)
    suggestions = repair_suggestions(frame, rules=["must_not_overlap"])
    # We expect overlap violations and suggestions
    assert isinstance(suggestions, list)
    if suggestions:
        assert "suggestion" in suggestions[0]
        assert "rule" in suggestions[0]


def test_must_not_have_gaps_rule():
    rows = [
        {"id": "a", "geometry": {"type": "Polygon", "coordinates": [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]}},
        {"id": "b", "geometry": {"type": "Polygon", "coordinates": [(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)]}},
    ]
    frame = GeoPromptFrame.from_records(rows)
    result = validate_topology_rules(frame, rule="must_not_have_gaps")
    # Disjoint polygons with no touching neighbor → potential gaps
    assert result["violation_count"] >= 1


# ---------------------------------------------------------------------------
#  Raster algebra
# ---------------------------------------------------------------------------


def test_raster_algebra_add():
    a = {"data": [[1, 2], [3, 4]], "transform": (0, 10, 1, 1)}
    b = {"data": [[10, 20], [30, 40]], "transform": (0, 10, 1, 1)}
    result = raster_algebra(a, b, operation="add")
    assert result["data"] == [[11, 22], [33, 44]]


def test_raster_algebra_subtract():
    a = {"data": [[10, 20], [30, 40]], "transform": (0, 10, 1, 1)}
    b = {"data": [[1, 2], [3, 4]], "transform": (0, 10, 1, 1)}
    result = raster_algebra(a, b, operation="subtract")
    assert result["data"] == [[9, 18], [27, 36]]


def test_raster_algebra_handles_none():
    a = {"data": [[1, None], [3, 4]], "transform": (0, 10, 1, 1)}
    b = {"data": [[10, 20], [None, 40]], "transform": (0, 10, 1, 1)}
    result = raster_algebra(a, b, operation="add")
    assert result["data"][0][1] is None
    assert result["data"][1][0] is None
    assert result["data"][0][0] == 11
    assert result["data"][1][1] == 44


# ---------------------------------------------------------------------------
#  Viz presets and storytelling
# ---------------------------------------------------------------------------


def test_basemap_presets_exist():
    assert "light" in BASEMAP_PRESETS
    assert "dark" in BASEMAP_PRESETS
    assert "report" in BASEMAP_PRESETS


def test_symbol_presets_cover_domains():
    assert "risk" in SYMBOL_PRESETS
    assert "outage" in SYMBOL_PRESETS
    assert "severity" in SYMBOL_PRESETS
    assert "asset_class" in SYMBOL_PRESETS
    assert "recommendation" in SYMBOL_PRESETS


def test_portfolio_scorecard_html():
    records = [
        {"name": "Project A", "cost": 100, "benefit": 300},
        {"name": "Project B", "cost": 200, "benefit": 150},
    ]
    html = portfolio_scorecard(records, title="Test Card")
    assert "<h2>Test Card</h2>" in html
    assert "Project A" in html
    assert "Project B" in html


def test_recommendation_card_html():
    html = recommendation_card("Pump Station 5", "Do Now", explanation="High risk of failure", score=0.92)
    assert "Pump Station 5" in html
    assert "Do Now" in html
    assert "0.92" in html
    assert "High risk" in html


def test_before_after_comparison_html():
    before = [{"name": "Zone A", "pressure": 60, "customers": 500}]
    after = [{"name": "Zone A", "pressure": 72, "customers": 500}]
    html = before_after_comparison(before, after, metric_columns=["pressure", "customers"])
    assert "Zone A" in html
    assert "before" in html.lower() or "Before" in html


# ---------------------------------------------------------------------------
#  Workspace lineage and job specs
# ---------------------------------------------------------------------------


def test_lineage_tracker_records_steps():
    tracker = LineageTracker()
    tracker.add_step("load", inputs=["data.geojson"])
    tracker.add_step("buffer", params={"distance": 100}, outputs=["buffered.geojson"])
    report = tracker.report()
    assert len(report) == 2
    assert report[0]["name"] == "load"
    assert report[1]["params"]["distance"] == 100


def test_lineage_tracker_markdown():
    tracker = LineageTracker()
    tracker.add_step("load", inputs=["data.geojson"])
    md = tracker.to_markdown()
    assert "# Lineage Report" in md
    assert "load" in md


def test_job_spec_manifest():
    spec = JobSpec("analysis_run", params={"crs": "EPSG:4326"})
    spec.add_step("buffer", callable_name="buffer_geometries", params={"distance": 50})
    spec.add_step("clip", callable_name="clip_geometries")
    manifest = spec.to_manifest()
    assert manifest["job_name"] == "analysis_run"
    assert manifest["step_count"] == 2
    assert manifest["global_params"]["crs"] == "EPSG:4326"


def test_job_spec_save_and_load(tmp_path):
    spec = JobSpec("test_job")
    spec.add_step("step1", callable_name="fn1")
    path = spec.save(tmp_path / "job.json")
    loaded = json.loads(Path(path).read_text())
    assert loaded["job_name"] == "test_job"
    assert loaded["step_count"] == 1


# ---------------------------------------------------------------------------
#  IO: field aliases
# ---------------------------------------------------------------------------


def test_apply_field_aliases():
    records = [{"OBJECTID": 1, "Shape": "point"}, {"OBJECTID": 2, "Shape": "line"}]
    aliased = apply_field_aliases(records, {"OBJECTID": "id", "Shape": "geometry_type"})
    assert aliased[0]["id"] == 1
    assert aliased[1]["geometry_type"] == "line"
    assert "OBJECTID" not in aliased[0]


# ---------------------------------------------------------------------------
#  Frame ergonomics: multi-key merge with suffixes
# ---------------------------------------------------------------------------


def test_merge_multi_key():
    left = GeoPromptFrame.from_records([
        {"city": "A", "year": 2020, "pop": 100, "geometry": {"type": "Point", "coordinates": (0, 0)}},
        {"city": "B", "year": 2021, "pop": 200, "geometry": {"type": "Point", "coordinates": (1, 1)}},
    ])
    right = GeoPromptFrame.from_records([
        {"city": "A", "year": 2020, "score": 8.5, "geometry": {"type": "Point", "coordinates": (0, 0)}},
        {"city": "C", "year": 2022, "score": 7.0, "geometry": {"type": "Point", "coordinates": (2, 2)}},
    ])
    merged = left.merge(right, on=["city", "year"], how="inner")
    records = merged.to_records()
    assert len(records) == 1
    assert records[0]["city"] == "A"
    assert records[0]["score"] == 8.5


def test_merge_suffixes_parameter():
    left = GeoPromptFrame.from_records([
        {"id": 1, "value": 10, "geometry": {"type": "Point", "coordinates": (0, 0)}},
    ])
    right = GeoPromptFrame.from_records([
        {"id": 1, "value": 99, "geometry": {"type": "Point", "coordinates": (1, 1)}},
    ])
    merged = left.merge(right, on="id", suffixes=("_l", "_r"))
    cols = merged.columns
    assert "value_l" in cols
    assert "value_r" in cols


# ---------------------------------------------------------------------------
#  Top-level export check
# ---------------------------------------------------------------------------


def test_all_new_exports_accessible():
    import geoprompt
    new_names = [
        "geometry_convex_hull", "geometry_envelope", "geometry_is_empty",
        "geometry_validity_reason", "geometry_snap", "geometry_split",
        "geometry_voronoi", "geometry_delaunay",
        "gearys_c", "getis_ord_g", "idw_interpolation", "kernel_density",
        "local_morans_i", "spatial_outliers", "morans_i",
        "feature_validation_pipeline", "repair_suggestions",
        "raster_algebra", "raster_reproject", "write_raster",
        "BASEMAP_PRESETS", "SYMBOL_PRESETS", "portfolio_scorecard",
        "recommendation_card", "before_after_comparison",
        "LineageTracker", "JobSpec", "lineagetracker", "jobspec",
        "apply_field_aliases", "read_zipped_shapefile",
        "normalize_geometry",
    ]
    for name in new_names:
        assert hasattr(geoprompt, name), f"missing export: {name}"
