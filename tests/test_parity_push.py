"""Tests for the features implemented in the latest parity push."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from geoprompt.frame import GeoPromptFrame
from geoprompt.table import PromptTable
from geoprompt.geometry import (
    geometry_touches,
    geometry_crosses,
    geometry_overlaps,
    normalize_geometry,
)
from geoprompt.io import schema_report, validate_schema
from geoprompt.network.utility import (
    capital_planning_prioritization,
    facility_siting_score,
    pressure_scenario_sweep,
    dependency_graph_overlay,
)
from geoprompt.network.routing import build_network_graph


# ---------------------------------------------------------------------------
# Geometry predicates
# ---------------------------------------------------------------------------


class TestGeometryTouches:
    def test_touching_polygons(self):
        """Two polygons sharing an edge should touch."""
        a = normalize_geometry({"type": "Polygon", "coordinates": [(0, 0), (1, 0), (1, 1), (0, 1)]})
        b = normalize_geometry({"type": "Polygon", "coordinates": [(1, 0), (2, 0), (2, 1), (1, 1)]})
        assert geometry_touches(a, b) is True

    def test_overlapping_polygons_do_not_touch(self):
        a = normalize_geometry({"type": "Polygon", "coordinates": [(0, 0), (2, 0), (2, 2), (0, 2)]})
        b = normalize_geometry({"type": "Polygon", "coordinates": [(1, 1), (3, 1), (3, 3), (1, 3)]})
        assert geometry_touches(a, b) is False

    def test_disjoint_polygons_do_not_touch(self):
        a = normalize_geometry({"type": "Polygon", "coordinates": [(0, 0), (1, 0), (1, 1), (0, 1)]})
        b = normalize_geometry({"type": "Polygon", "coordinates": [(5, 5), (6, 5), (6, 6), (5, 6)]})
        assert geometry_touches(a, b) is False

    def test_point_on_polygon_boundary(self):
        pt = normalize_geometry({"type": "Point", "coordinates": (1, 0.5)})
        poly = normalize_geometry({"type": "Polygon", "coordinates": [(1, 0), (2, 0), (2, 1), (1, 1)]})
        assert geometry_touches(pt, poly) is True


class TestGeometryCrosses:
    def test_crossing_lines(self):
        a = normalize_geometry({"type": "LineString", "coordinates": [(0, 0), (2, 2)]})
        b = normalize_geometry({"type": "LineString", "coordinates": [(0, 2), (2, 0)]})
        assert geometry_crosses(a, b) is True

    def test_parallel_lines(self):
        a = normalize_geometry({"type": "LineString", "coordinates": [(0, 0), (2, 0)]})
        b = normalize_geometry({"type": "LineString", "coordinates": [(0, 1), (2, 1)]})
        assert geometry_crosses(a, b) is False

    def test_line_crosses_polygon(self):
        line = normalize_geometry({"type": "LineString", "coordinates": [(-1, 0.5), (0.5, 0.5)]})
        poly = normalize_geometry({"type": "Polygon", "coordinates": [(0, 0), (1, 0), (1, 1), (0, 1)]})
        assert geometry_crosses(line, poly) is True


class TestGeometryOverlaps:
    def test_overlapping_polygons(self):
        a = normalize_geometry({"type": "Polygon", "coordinates": [(0, 0), (2, 0), (2, 2), (0, 2)]})
        b = normalize_geometry({"type": "Polygon", "coordinates": [(1, 1), (3, 1), (3, 3), (1, 3)]})
        assert geometry_overlaps(a, b) is True

    def test_contained_polygon_does_not_overlap(self):
        outer = normalize_geometry({"type": "Polygon", "coordinates": [(0, 0), (10, 0), (10, 10), (0, 10)]})
        inner = normalize_geometry({"type": "Polygon", "coordinates": [(1, 1), (2, 1), (2, 2), (1, 2)]})
        assert geometry_overlaps(outer, inner) is False

    def test_disjoint_polygons(self):
        a = normalize_geometry({"type": "Polygon", "coordinates": [(0, 0), (1, 0), (1, 1), (0, 1)]})
        b = normalize_geometry({"type": "Polygon", "coordinates": [(5, 5), (6, 5), (6, 6), (5, 6)]})
        assert geometry_overlaps(a, b) is False


# ---------------------------------------------------------------------------
# Frame ergonomics
# ---------------------------------------------------------------------------


def _sample_frame():
    return GeoPromptFrame(
        [
            {"site_id": "A", "region": "north", "value": 10, "geometry": {"type": "Point", "coordinates": (0, 0)}},
            {"site_id": "B", "region": "north", "value": 20, "geometry": {"type": "Point", "coordinates": (1, 1)}},
            {"site_id": "C", "region": "south", "value": 30, "geometry": {"type": "Point", "coordinates": (2, 2)}},
        ],
        geometry_column="geometry",
    )


class TestFrameSetResetIndex:
    def test_set_index(self):
        frame = _sample_frame().set_index("site_id")
        assert hasattr(frame, "_index_column")
        assert frame._index_column == "site_id"

    def test_reset_index_adds_column(self):
        frame = _sample_frame().set_index("site_id").reset_index()
        assert frame._index_column is None
        records = frame.to_records()
        assert "index" in records[0]

    def test_reset_index_drop(self):
        frame = _sample_frame().reset_index(drop=True)
        records = frame.to_records()
        assert "index" not in records[0]


class TestFrameCrosstab:
    def test_crosstab_counts(self):
        frame = GeoPromptFrame(
            [
                {"cat": "a", "color": "red", "val": 1, "geometry": {"type": "Point", "coordinates": (0, 0)}},
                {"cat": "a", "color": "blue", "val": 2, "geometry": {"type": "Point", "coordinates": (1, 0)}},
                {"cat": "b", "color": "red", "val": 3, "geometry": {"type": "Point", "coordinates": (0, 1)}},
                {"cat": "b", "color": "red", "val": 4, "geometry": {"type": "Point", "coordinates": (1, 1)}},
            ],
            geometry_column="geometry",
        )
        table = frame.crosstab("cat", "color")
        records = table.to_records()
        a_row = next(r for r in records if r["cat"] == "a")
        b_row = next(r for r in records if r["cat"] == "b")
        assert a_row["red"] == 1
        assert a_row["blue"] == 1
        assert b_row["red"] == 2

    def test_crosstab_sum(self):
        frame = GeoPromptFrame(
            [
                {"cat": "x", "group": "g1", "val": 10, "geometry": {"type": "Point", "coordinates": (0, 0)}},
                {"cat": "x", "group": "g1", "val": 5, "geometry": {"type": "Point", "coordinates": (1, 0)}},
                {"cat": "x", "group": "g2", "val": 20, "geometry": {"type": "Point", "coordinates": (0, 1)}},
            ],
            geometry_column="geometry",
        )
        table = frame.crosstab("cat", "group", values="val", agg="sum")
        records = table.to_records()
        assert records[0]["g1"] == 15.0
        assert records[0]["g2"] == 20.0


# ---------------------------------------------------------------------------
# Table: crosstab + conditional HTML
# ---------------------------------------------------------------------------


class TestTableCrosstab:
    def test_crosstab(self):
        t = PromptTable([
            {"a": "x", "b": "p", "v": 1},
            {"a": "x", "b": "q", "v": 2},
            {"a": "y", "b": "p", "v": 3},
        ])
        ct = t.crosstab("a", "b")
        recs = ct.to_records()
        assert len(recs) == 2


class TestConditionalHTML:
    def test_conditional_coloring(self):
        t = PromptTable([
            {"metric": "A", "delta": 5.0},
            {"metric": "B", "delta": -3.0},
        ])
        with tempfile.TemporaryDirectory() as td:
            path = t.to_html(
                Path(td) / "out.html",
                conditional={"column": "delta", "threshold": 0, "high_color": "#86efac", "low_color": "#fca5a5"},
            )
            html = Path(path).read_text()
            assert "#86efac" in html
            assert "#fca5a5" in html


# ---------------------------------------------------------------------------
# IO: schema validation
# ---------------------------------------------------------------------------


class TestSchemaReport:
    def test_basic_report(self):
        records = [{"name": "a", "value": 1}, {"name": "b", "value": 2}]
        report = schema_report(records)
        assert report["row_count"] == 2
        names = {c["name"] for c in report["columns"]}
        assert "name" in names
        assert "value" in names

    def test_validate_schema_passes(self):
        records = [{"name": "a", "value": 1}]
        result = validate_schema(records, {"name": "str", "value": "int"})
        assert result["valid"] is True

    def test_validate_schema_missing_column(self):
        records = [{"name": "a"}]
        result = validate_schema(records, {"name": "str", "value": "int"})
        assert result["valid"] is False
        assert any("missing" in v for v in result["violations"])

    def test_validate_schema_strict_extra(self):
        records = [{"name": "a", "extra": 1}]
        result = validate_schema(records, {"name": "str"}, strict=True)
        assert result["valid"] is False
        assert any("unexpected" in v for v in result["violations"])


# ---------------------------------------------------------------------------
# Network: facility siting and capital planning
# ---------------------------------------------------------------------------


def _simple_graph():
    edges = [
        {"from_node": "A", "to_node": "B", "cost": 1, "edge_id": "e1"},
        {"from_node": "B", "to_node": "C", "cost": 1, "edge_id": "e2"},
        {"from_node": "C", "to_node": "D", "cost": 1, "edge_id": "e3"},
        {"from_node": "B", "to_node": "D", "cost": 3, "edge_id": "e4"},
    ]
    return build_network_graph(edges, directed=False)


class TestFacilitySiting:
    def test_scores_sorted_best_first(self):
        g = _simple_graph()
        results = facility_siting_score(g, ["A", "B", "D"], ["A", "B", "C", "D"])
        assert len(results) == 3
        # B is central and should reach all 4 nodes cheaply
        assert results[0]["node"] == "B"
        assert results[0]["reachable_demand_nodes"] == 4


class TestCapitalPlanning:
    def test_ranking_by_bcr(self):
        projects = [
            {"name": "P1", "benefit": 100, "cost": 50},
            {"name": "P2", "benefit": 200, "cost": 50},
            {"name": "P3", "benefit": 50, "cost": 25},
        ]
        ranked = capital_planning_prioritization(projects)
        assert ranked[0]["name"] == "P2"
        assert ranked[0]["bcr"] > ranked[1]["bcr"]

    def test_budget_selection(self):
        projects = [
            {"name": "P1", "benefit": 100, "cost": 30},
            {"name": "P2", "benefit": 200, "cost": 40},
            {"name": "P3", "benefit": 50, "cost": 25},
        ]
        ranked = capital_planning_prioritization(projects, budget=60)
        selected = [p for p in ranked if p["selected"]]
        assert len(selected) >= 1
        total_cost = sum(p["cost"] for p in selected)
        assert total_cost <= 60


class TestPressureScenarioSweep:
    def test_multiple_pressures(self):
        g = _simple_graph()
        results = pressure_scenario_sweep(g, "A", [50.0, 80.0, 100.0])
        assert len(results) == 3
        # Higher inlet should give higher mean
        assert results[2]["mean_pressure"] >= results[0]["mean_pressure"]


class TestDependencyGraphOverlay:
    def test_shared_nodes(self):
        g1 = _simple_graph()
        g2 = build_network_graph(
            [{"from_node": "B", "to_node": "E", "cost": 1, "edge_id": "e1"}],
            directed=False,
        )
        results = dependency_graph_overlay({"water": g1, "electric": g2}, ["B", "X"])
        b_result = next(r for r in results if r["node"] == "B")
        assert b_result["is_cross_dependency"] is True
        x_result = next(r for r in results if r["node"] == "X")
        assert x_result["is_cross_dependency"] is False


# ---------------------------------------------------------------------------
# Viz: resilience style presets (import check)
# ---------------------------------------------------------------------------


class TestVizPresets:
    def test_resilience_style_map(self):
        from geoprompt.viz import resilience_style_map

        col, smap = resilience_style_map()
        assert col == "risk_tier"
        assert "critical" in smap
        assert "low" in smap

    def test_plot_restoration_timeline(self):
        pytest.importorskip("matplotlib", exc_type=ImportError)
        from geoprompt.viz import plot_restoration_timeline

        events = [
            {"node_id": "N1", "restored_at": 2},
            {"node_id": "N2", "restored_at": 5},
            {"node_id": "N3", "restored_at": 1},
        ]
        fig = plot_restoration_timeline(events)
        assert fig is not None
