"""Comprehensive tests for the new parity features added to GeoPromptFrame,
viz, stats, db, and io modules.
"""
from __future__ import annotations

import pytest

from geoprompt.frame import GeoPromptFrame
from geoprompt.table import PromptTable


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": [x, y]}


def _polygon(coords: list) -> dict:
    return {"type": "Polygon", "coordinates": coords}


def _sample_frame() -> GeoPromptFrame:
    return GeoPromptFrame([
        {"site_id": "A", "region": "north", "value": 10.0, "category": "high", "geometry": _point(0, 0)},
        {"site_id": "B", "region": "north", "value": 20.0, "category": "low", "geometry": _point(1, 1)},
        {"site_id": "C", "region": "south", "value": 30.0, "category": "high", "geometry": _point(2, 2)},
        {"site_id": "D", "region": "south", "value": None, "category": "low", "geometry": _point(3, 3)},
        {"site_id": "E", "region": "south", "value": 50.0, "category": "high", "geometry": _point(4, 4)},
    ])


def _polygon_frame() -> GeoPromptFrame:
    return GeoPromptFrame([
        {"site_id": "P1", "value": 100, "geometry": _polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])},
        {"site_id": "P2", "value": 200, "geometry": _polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])},
    ])


# ===========================================================================
# GeoPromptFrame ergonomics
# ===========================================================================

class TestGroupby:
    def test_groupby_agg_sum(self):
        frame = _sample_frame()
        result = frame.groupby("region").agg({"value": "sum"})
        assert isinstance(result, GeoPromptFrame)
        records = result.to_records()
        north = next(r for r in records if r["region"] == "north")
        south = next(r for r in records if r["region"] == "south")
        assert north["value_sum"] == 30.0
        assert south["value_sum"] == 80.0  # 30 + 50, None excluded

    def test_groupby_agg_count(self):
        frame = _sample_frame()
        result = frame.groupby("region").agg({"site_id": "count"})
        records = result.to_records()
        north = next(r for r in records if r["region"] == "north")
        assert north["site_id_count"] == 2
        assert north["row_count"] == 2

    def test_groupby_preserves_geometry(self):
        frame = _sample_frame()
        result = frame.groupby("region").agg({"value": "mean"})
        assert result.geometry_column == "geometry"
        for row in result:
            assert "geometry" in row

    def test_groupby_multi_column(self):
        frame = _sample_frame()
        result = frame.groupby(["region", "category"]).agg({"value": "sum"})
        assert len(result) >= 3  # north-high, north-low, south-high, south-low

    def test_groupby_missing_column_raises(self):
        frame = _sample_frame()
        with pytest.raises(KeyError):
            frame.groupby("nonexistent")


class TestPivot:
    def test_pivot_basic(self):
        # Use frame without None values since pivot sum can't handle None
        frame = GeoPromptFrame([
            {"site_id": "A", "region": "north", "value": 10.0, "category": "high", "geometry": _point(0, 0)},
            {"site_id": "B", "region": "north", "value": 20.0, "category": "low", "geometry": _point(1, 1)},
            {"site_id": "C", "region": "south", "value": 30.0, "category": "high", "geometry": _point(2, 2)},
            {"site_id": "D", "region": "south", "value": 40.0, "category": "low", "geometry": _point(3, 3)},
        ])
        result = frame.pivot(index="region", columns="category", values="value")
        assert isinstance(result, PromptTable)
        records = result.to_records()
        assert len(records) == 2  # north, south

    def test_pivot_agg_count(self):
        frame = _sample_frame()
        result = frame.pivot(index="region", columns="category", values="value", agg="count")
        records = result.to_records()
        north = next(r for r in records if r["region"] == "north")
        assert north["high"] == 1
        assert north["low"] == 1


class TestFillna:
    def test_fillna_scalar(self):
        frame = _sample_frame()
        result = frame.fillna(0.0, column="value")
        for row in result:
            assert row["value"] is not None
        d_row = next(r for r in result if r["site_id"] == "D")
        assert d_row["value"] == 0.0

    def test_fillna_dict(self):
        frame = _sample_frame()
        result = frame.fillna({"value": -1.0})
        d_row = next(r for r in result if r["site_id"] == "D")
        assert d_row["value"] == -1.0

    def test_fillna_ffill(self):
        frame = _sample_frame()
        result = frame.fillna(method="ffill")
        d_row = next(r for r in result if r["site_id"] == "D")
        assert d_row["value"] == 30.0  # forward fill from C

    def test_fillna_bfill(self):
        frame = _sample_frame()
        result = frame.fillna(method="bfill")
        d_row = next(r for r in result if r["site_id"] == "D")
        assert d_row["value"] == 50.0  # backward fill from E


class TestDropna:
    def test_dropna_any(self):
        frame = _sample_frame()
        result = frame.dropna()
        assert len(result) == 4  # row D has None value

    def test_dropna_all(self):
        frame = _sample_frame()
        result = frame.dropna(how="all")
        assert len(result) == 5  # no row is all-null

    def test_dropna_subset(self):
        frame = _sample_frame()
        result = frame.dropna(subset=["site_id"])
        assert len(result) == 5  # no nulls in site_id


class TestRenameColumns:
    def test_rename(self):
        frame = _sample_frame()
        result = frame.rename_columns({"value": "metric", "region": "area"})
        assert "metric" in result.columns
        assert "area" in result.columns
        assert "value" not in result.columns

    def test_rename_geometry_column(self):
        frame = _sample_frame()
        result = frame.rename_columns({"geometry": "geom"})
        assert result.geometry_column == "geom"


class TestDropColumns:
    def test_drop(self):
        frame = _sample_frame()
        result = frame.drop_columns(["category"])
        assert "category" not in result.columns
        assert "value" in result.columns

    def test_drop_geometry_raises(self):
        frame = _sample_frame()
        with pytest.raises(ValueError):
            frame.drop_columns(["geometry"])


class TestReorderColumns:
    def test_reorder(self):
        frame = _sample_frame()
        result = frame.reorder_columns(["value", "site_id"])
        cols = result.columns
        assert cols[0] == "value"
        assert cols[1] == "site_id"


class TestAstype:
    def test_astype_str(self):
        frame = _sample_frame()
        result = frame.astype("value", "str")
        a_row = next(r for r in result if r["site_id"] == "A")
        assert isinstance(a_row["value"], str)

    def test_astype_invalid(self):
        frame = _sample_frame()
        with pytest.raises(ValueError):
            frame.astype("value", "invalid_type")


class TestReplace:
    def test_replace_values(self):
        frame = _sample_frame()
        result = frame.replace("category", {"high": "H", "low": "L"})
        a_row = next(r for r in result if r["site_id"] == "A")
        assert a_row["category"] == "H"


class TestMerge:
    def test_merge_inner(self):
        frame = _sample_frame()
        lookup = GeoPromptFrame([
            {"site_id": "A", "extra": "x1", "geometry": _point(0, 0)},
            {"site_id": "C", "extra": "x2", "geometry": _point(2, 2)},
        ])
        result = frame.merge(lookup, on="site_id", how="inner")
        assert len(result) == 2

    def test_merge_left(self):
        frame = _sample_frame()
        lookup = GeoPromptFrame([
            {"site_id": "A", "extra": "x1", "geometry": _point(0, 0)},
        ])
        result = frame.merge(lookup, on="site_id", how="left")
        assert len(result) == 5


class TestConcat:
    def test_concat_two_frames(self):
        f1 = GeoPromptFrame([{"site_id": "A", "geometry": _point(0, 0)}])
        f2 = GeoPromptFrame([{"site_id": "B", "geometry": _point(1, 1)}])
        result = GeoPromptFrame.concat([f1, f2])
        assert len(result) == 2

    def test_concat_empty(self):
        result = GeoPromptFrame.concat([])
        assert len(result) == 0


class TestValueCounts:
    def test_value_counts(self):
        frame = _sample_frame()
        result = frame.value_counts("category")
        assert isinstance(result, PromptTable)
        records = result.to_records()
        assert len(records) == 2  # high, low


class TestDescribe:
    def test_describe(self):
        frame = _sample_frame()
        result = frame.describe(["value"])
        records = result.to_records()
        assert len(records) == 1
        assert records[0]["column"] == "value"
        assert records[0]["dtype"] == "numeric"


class TestSample:
    def test_sample_returns_correct_count(self):
        frame = _sample_frame()
        result = frame.sample(3, seed=42)
        assert len(result) == 3

    def test_sample_more_than_available(self):
        frame = _sample_frame()
        result = frame.sample(100, seed=42)
        assert len(result) == 5


class TestUnique:
    def test_unique(self):
        frame = _sample_frame()
        result = frame.unique("region")
        assert set(result) == {"north", "south"}


class TestSetGeometry:
    def test_set_geometry(self):
        frame = _sample_frame()
        frame2 = frame.with_column("geom2", [_point(10, 10)] * 5)
        result = frame2.set_geometry("geom2")
        assert result.geometry_column == "geom2"


class TestToPromptTable:
    def test_to_prompt_table(self):
        frame = _sample_frame()
        table = frame.to_prompt_table()
        assert isinstance(table, PromptTable)
        for row in table:
            assert "geometry" not in row


# ===========================================================================
# Geometry convenience methods
# ===========================================================================

class TestEnvelope:
    def test_envelope_creates_polygon(self):
        frame = GeoPromptFrame([
            {"site_id": "L1", "geometry": {"type": "LineString", "coordinates": [(0, 0), (3, 4)]}},
        ])
        result = frame.envelope()
        assert len(result) == 1
        geom = result[0]["geometry"]
        assert geom["type"] == "Polygon"


# Simplify and convex_hull require Shapely — mark them so they skip gracefully.

class TestSimplify:
    def test_simplify_runs(self):
        try:
            import shapely  # noqa: F401
        except ImportError:
            pytest.skip("shapely not installed")

        frame = _polygon_frame()
        result = frame.simplify(tolerance=0.5)
        assert len(result) > 0

class TestConvexHull:
    def test_convex_hull_runs(self):
        try:
            import shapely  # noqa: F401
        except ImportError:
            pytest.skip("shapely not installed")

        frame = _polygon_frame()
        result = frame.convex_hull()
        assert len(result) > 0


# ===========================================================================
# Overlay convenience methods
# ===========================================================================

class TestDifference:
    def test_difference_runs(self):
        try:
            import shapely  # noqa: F401
        except ImportError:
            pytest.skip("shapely not installed")

        frame = _polygon_frame()
        mask_frame = GeoPromptFrame([
            {"site_id": "M", "geometry": _polygon([(1, 1), (4, 1), (4, 4), (1, 4), (1, 1)])},
        ])
        result = frame.difference(mask_frame)
        assert isinstance(result, GeoPromptFrame)


class TestSymmetricDifference:
    def test_sym_diff_runs(self):
        try:
            import shapely  # noqa: F401
        except ImportError:
            pytest.skip("shapely not installed")

        f1 = GeoPromptFrame([{"geometry": _polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])}])
        f2 = GeoPromptFrame([{"geometry": _polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])}])
        result = f1.symmetric_difference(f2)
        assert isinstance(result, GeoPromptFrame)


class TestUnionOverlay:
    def test_union_runs(self):
        try:
            import shapely  # noqa: F401
        except ImportError:
            pytest.skip("shapely not installed")

        f1 = GeoPromptFrame([{"geometry": _polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])}])
        f2 = GeoPromptFrame([{"geometry": _polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])}])
        result = f1.union(f2)
        assert isinstance(result, GeoPromptFrame)


# ===========================================================================
# Stats module
# ===========================================================================

class TestMoransI:
    def test_morans_i(self):
        from geoprompt.stats import morans_i
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        centroids = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        result = morans_i(values, centroids, bandwidth=2.0)
        assert "morans_i" in result
        assert "z_score" in result

    def test_morans_i_k_neighbors(self):
        from geoprompt.stats import morans_i
        values = [10.0, 20.0, 30.0, 40.0]
        centroids = [(0, 0), (1, 0), (0, 1), (1, 1)]
        result = morans_i(values, centroids, k=2)
        assert "morans_i" in result


class TestGearysC:
    def test_gearys_c(self):
        from geoprompt.stats import gearys_c
        values = [1.0, 1.5, 2.0, 2.5, 3.0]
        centroids = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        result = gearys_c(values, centroids, bandwidth=2.0)
        assert "gearys_c" in result
        assert result["gearys_c"] >= 0


class TestLocalMoransI:
    def test_local_morans(self):
        from geoprompt.stats import local_morans_i
        values = [1.0, 10.0, 1.0, 10.0, 1.0]
        centroids = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        result = local_morans_i(values, centroids, bandwidth=2.0)
        assert len(result) == 5
        assert all("cluster_type" in r for r in result)


class TestGetisOrdG:
    def test_hotspot(self):
        from geoprompt.stats import getis_ord_g
        values = [100.0, 100.0, 100.0, 1.0, 1.0]
        centroids = [(0, 0), (1, 0), (2, 0), (10, 0), (11, 0)]
        result = getis_ord_g(values, centroids, bandwidth=3.0)
        assert len(result) == 5
        # First three should be hotter than last two
        assert result[0]["z_score"] > result[3]["z_score"]


class TestKernelDensity:
    def test_kernel_density(self):
        from geoprompt.stats import kernel_density
        centroids = [(0, 0), (1, 1), (2, 2)]
        result = kernel_density(centroids, bandwidth=1.0, cell_size=0.5)
        assert "grid" in result
        assert result["rows"] > 0
        assert result["cols"] > 0


class TestIDWInterpolation:
    def test_idw(self):
        from geoprompt.stats import idw_interpolation
        known = [(0, 0), (10, 0), (0, 10)]
        vals = [100.0, 200.0, 300.0]
        query = [(5, 5)]
        result = idw_interpolation(known, vals, query)
        assert len(result) == 1
        assert result[0] is not None
        assert 100 < result[0] < 300

    def test_idw_exact_match(self):
        from geoprompt.stats import idw_interpolation
        known = [(0, 0)]
        vals = [42.0]
        result = idw_interpolation(known, vals, [(0, 0)])
        assert result[0] == 42.0


class TestSpatialOutliers:
    def test_outliers(self):
        from geoprompt.stats import spatial_outliers
        values = [10.0, 10.0, 10.0, 10.0, 5000.0]
        centroids = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]
        result = spatial_outliers(values, centroids, bandwidth=2.0, threshold=1.5)
        assert len(result) == 5
        # The 5000.0 value should be flagged
        assert result[4]["is_outlier"] is True


# ===========================================================================
# IO helpers
# ===========================================================================

class TestGeometryWKT:
    def test_point_wkt(self):
        from geoprompt.io import _geometry_to_wkt
        wkt = _geometry_to_wkt({"type": "Point", "coordinates": [1.0, 2.0]})
        assert wkt == "POINT (1.0 2.0)"

    def test_linestring_wkt(self):
        from geoprompt.io import _geometry_to_wkt
        wkt = _geometry_to_wkt({"type": "LineString", "coordinates": [(0, 0), (1, 1)]})
        assert "LINESTRING" in wkt


# ===========================================================================
# Viz module (no folium needed — just test the helpers)
# ===========================================================================

class TestVizHelpers:
    def test_geojson_coords_point(self):
        from geoprompt.viz import _geojson_coords
        result = _geojson_coords({"type": "Point", "coordinates": (1.0, 2.0)})
        assert result["type"] == "Point"
        assert result["coordinates"] == [1.0, 2.0]

    def test_geojson_coords_polygon(self):
        from geoprompt.viz import _geojson_coords
        result = _geojson_coords({
            "type": "Polygon",
            "coordinates": ((0, 0), (1, 0), (1, 1), (0, 0)),
        })
        assert result["type"] == "Polygon"

    def test_frame_bounds(self):
        from geoprompt.viz import _frame_bounds
        rows = [
            {"geometry": _point(0, 0)},
            {"geometry": _point(10, 20)},
        ]
        bounds = _frame_bounds(rows, "geometry")
        assert bounds == (0.0, 0.0, 10.0, 20.0)


# ===========================================================================
# DB module (test internal parsers only — no real DB)
# ===========================================================================

class TestDBParsers:
    def test_parse_wkt_point(self):
        from geoprompt.db import _parse_wkt
        geom = _parse_wkt("POINT (1.5 2.5)")
        assert geom["type"] == "Point"
        assert geom["coordinates"] == (1.5, 2.5)

    def test_parse_wkt_linestring(self):
        from geoprompt.db import _parse_wkt
        geom = _parse_wkt("LINESTRING (0 0, 1 1, 2 0)")
        assert geom["type"] == "LineString"

    def test_parse_wkt_or_geojson_dict(self):
        from geoprompt.db import _parse_wkt_or_geojson
        geom = _parse_wkt_or_geojson({"type": "Point", "coordinates": [3, 4]})
        assert geom["type"] == "Point"

    def test_parse_wkt_or_geojson_json_string(self):
        from geoprompt.db import _parse_wkt_or_geojson
        geom = _parse_wkt_or_geojson('{"type": "Point", "coordinates": [5, 6]}')
        assert geom["type"] == "Point"
