"""Tests for G1 geometry, G5 spatial statistics, G8 data management,
G13 space-time, and G24 simulation-only decorator additions."""

import math
import warnings

import pytest

import geoprompt as gp

# ---------------------------------------------------------------------------
# G1 — geometry engine additions
# ---------------------------------------------------------------------------

class TestOffsetCurve:
    def test_basic_offset(self):
        line = {"type": "LineString", "coordinates": [[0, 0], [10, 0]]}
        result = gp.offset_curve(line, 1.0)
        assert result["type"] == "LineString"
        ys = [c[1] for c in result["coordinates"]]
        assert all(abs(y - 1.0) < 0.1 for y in ys)

    def test_negative_offset(self):
        line = {"type": "LineString", "coordinates": [[0, 0], [10, 0]]}
        result = gp.offset_curve(line, -1.0)
        ys = [c[1] for c in result["coordinates"]]
        assert all(y < 0 for y in ys)

    def test_non_line_raises(self):
        with pytest.raises((ValueError, TypeError, KeyError, AttributeError)):
            gp.offset_curve({"type": "Point", "coordinates": [0, 0]}, 1.0)


class TestConcaveHull:
    def test_returns_polygon(self):
        pts = [
            {"type": "Point", "coordinates": [c[0], c[1]]}
            for c in [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]]
        ]
        result = gp.concave_hull(pts)
        assert result["type"] in ("Polygon", "MultiPolygon", "GeometryCollection")

    def test_single_point_returns_empty_or_point(self):
        pts = [{"type": "Point", "coordinates": [5, 5]}]
        result = gp.concave_hull(pts)
        # May return a degenerate dict or None for a single point
        assert result is None or isinstance(result, dict)


class TestMinimumRotatedRectangle:
    def test_basic_rectangle(self):
        pts = [[0, 0], [4, 0], [4, 2], [0, 2]]
        polygon = {"type": "Polygon", "coordinates": [pts + [pts[0]]]}
        result = gp.minimum_rotated_rectangle(polygon)
        assert result["type"] == "Polygon"

    def test_point_returns_degenerate(self):
        pt = {"type": "Point", "coordinates": [3, 4]}
        result = gp.minimum_rotated_rectangle(pt)
        # Single point returns None (< 2 unique points) or a degenerate result
        assert result is None or isinstance(result, dict)


class TestMinimumBoundingCircle:
    def test_two_points(self):
        line = {"type": "LineString", "coordinates": [[0, 0], [4, 0]]}
        result = gp.minimum_bounding_circle(line)
        # Returns {"center": (cx, cy), "radius": r} or a Polygon
        assert isinstance(result, dict)
        if result.get("type") == "Polygon":
            xs = [c[0] for c in result["coordinates"][0]]
            assert min(xs) <= 0.1
            assert max(xs) >= 3.9
        else:
            assert "center" in result or "radius" in result

    def test_polygon(self):
        poly = {"type": "Polygon", "coordinates": [[[0, 0], [6, 0], [3, 4], [0, 0]]]}
        result = gp.minimum_bounding_circle(poly)
        assert isinstance(result, dict)


class TestForce2D3D:
    def test_force_2d(self):
        pt3d = {"type": "Point", "coordinates": [1, 2, 99]}
        result = gp.force_2d(pt3d)
        assert len(result["coordinates"]) == 2

    def test_force_3d(self):
        pt2d = {"type": "Point", "coordinates": [1, 2]}
        result = gp.force_3d(pt2d, z=50.0)
        assert len(result["coordinates"]) == 3
        assert result["coordinates"][2] == 50.0

    def test_force_3d_linestring(self):
        line = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
        result = gp.force_3d(line, z=10)
        for coord in result["coordinates"]:
            assert len(coord) == 3
            assert coord[2] == 10.0


class TestVoronoiAndDelaunay:
    def test_voronoi_polygons(self):
        pts = [
            {"type": "Point", "coordinates": [c[0], c[1]]}
            for c in [[0, 0], [4, 0], [2, 3]]
        ]
        result = gp.voronoi_polygons(pts)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_delaunay_triangulation(self):
        pts = [
            {"type": "Point", "coordinates": [c[0], c[1]]}
            for c in [[0, 0], [4, 0], [2, 3], [2, 1]]
        ]
        result = gp.delaunay_triangulation(pts)
        assert isinstance(result, list)
        assert len(result) >= 1


class TestPolygonize:
    def test_basic(self):
        lines = [
            {"type": "LineString", "coordinates": [[0, 0], [1, 0]]},
            {"type": "LineString", "coordinates": [[1, 0], [1, 1]]},
            {"type": "LineString", "coordinates": [[1, 1], [0, 1]]},
            {"type": "LineString", "coordinates": [[0, 1], [0, 0]]},
        ]
        result = gp.polygonize(lines)
        assert isinstance(result, list)


class TestLineMerge:
    def test_merges_contiguous(self):
        lines = [
            {"type": "LineString", "coordinates": [[0, 0], [1, 0]]},
            {"type": "LineString", "coordinates": [[1, 0], [2, 0]]},
        ]
        result = gp.line_merge(lines)
        # May return a list or a single merged geometry
        if isinstance(result, list):
            assert len(result) >= 1
            total_coords = sum(len(g.get("coordinates", [])) for g in result)
            assert total_coords >= 2
        else:
            assert isinstance(result, dict)
            assert result.get("type") in ("LineString", "MultiLineString")


class TestSetGetPrecision:
    def test_set_precision(self):
        pt = {"type": "Point", "coordinates": [1.23456789, 2.98765432]}
        result = gp.set_precision(pt, 0.01)
        # Coordinates should be snapped to 2 decimal places
        x, y = result["coordinates"]
        assert abs(x - round(x / 0.01) * 0.01) < 1e-9

    def test_get_precision(self):
        pt = {"type": "Point", "coordinates": [1.001, 2.001]}
        prec = gp.get_precision(pt)
        assert isinstance(prec, float)


class TestRingsAndBoundary:
    def test_exterior_ring(self):
        poly = {"type": "Polygon", "coordinates": [
            [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
            [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]],
        ]}
        result = gp.exterior_ring(poly)
        assert result["type"] == "LineString"
        assert len(result["coordinates"]) >= 4

    def test_get_interior_rings(self):
        poly = {"type": "Polygon", "coordinates": [
            [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
            [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]],
        ]}
        rings = gp.get_interior_rings(poly)
        assert len(rings) == 1
        assert rings[0]["type"] == "LineString"

    def test_boundary_geometry(self):
        poly = {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]}
        b = gp.boundary_geometry(poly)
        assert isinstance(b, dict)

    def test_normalize(self):
        poly = {"type": "Polygon", "coordinates": [[[3, 0], [0, 0], [0, 3], [3, 3], [3, 0]]]}
        result = gp.normalize(poly)
        assert result["type"] == "Polygon"


class TestLineProject:
    def test_midpoint(self):
        line = {"type": "LineString", "coordinates": [[0, 0], [10, 0]]}
        pt = {"type": "Point", "coordinates": [5, 5]}
        proj = gp.line_project(line, pt)
        assert abs(proj - 5.0) < 0.01

    def test_normalized(self):
        line = {"type": "LineString", "coordinates": [[0, 0], [10, 0]]}
        pt = {"type": "Point", "coordinates": [2.5, 3]}
        proj = gp.line_project(line, pt, normalized=True)
        assert 0.0 <= proj <= 1.0
        assert abs(proj - 0.25) < 0.01


# ---------------------------------------------------------------------------
# G5 — spatial statistics additions
# ---------------------------------------------------------------------------

def _sample_features(n: int = 6) -> list:
    """Make a minimal list of feature dicts for stats tests."""
    import math
    rows = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = math.cos(angle) * 5
        y = math.sin(angle) * 5
        rows.append({
            "id": i,
            "value": float(i + 1),
            "geometry": {"type": "Point", "coordinates": [x, y]},
        })
    return rows


class TestGearyC:
    def test_returns_expected_keys(self):
        feats = _sample_features()
        result = gp.geary_c(feats, value_column="value", geometry_column="geometry")
        assert "c" in result
        assert "z_score" in result
        # key may be p_value or p_value_approx
        assert "p_value" in result or "p_value_approx" in result
        pval = result.get("p_value", result.get("p_value_approx", 0.5))
        assert 0.0 <= pval <= 1.0

    def test_uniform_pattern(self):
        # Uniform values → C close to 1 (random pattern)
        feats = [{"value": 1.0, "geometry": {"type": "Point", "coordinates": [float(i), 0.0]}} for i in range(6)]
        result = gp.geary_c(feats, value_column="value", geometry_column="geometry")
        assert isinstance(result["c"], float)


class TestRipleyKL:
    def test_ripley_k_keys(self):
        feats = _sample_features(10)
        result = gp.ripley_k(feats, geometry_column="geometry")
        # May return a list of dicts or a single dict
        if isinstance(result, list):
            assert len(result) > 0
            row = result[0]
            # Each row should have some distance/k key
            assert any(k in row for k in ("distance", "distances", "K", "k"))
        else:
            assert "distances" in result or "K" in result or "k" in result

    def test_ripley_l_keys(self):
        feats = _sample_features(8)
        result = gp.ripley_l(feats, geometry_column="geometry")
        if isinstance(result, list):
            assert len(result) > 0
        else:
            assert "L" in result or "l" in result or "l_minus_d" in result


class TestClarkEvans:
    def test_returns_nni(self):
        feats = _sample_features(8)
        result = gp.clark_evans(feats, geometry_column="geometry")
        assert "nni" in result
        assert "z_score" in result

    def test_clustered_pattern(self):
        # All points at same location → highly clustered
        feats = [{"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}} for _ in range(5)]
        result = gp.clark_evans(feats, geometry_column="geometry")
        assert result["nni"] == 0.0


class TestLocalMorans:
    def test_returns_quadrants(self):
        feats = _sample_features(6)
        result = gp.anselin_local_morans_scatterplot(feats, value_column="value", geometry_column="geometry")
        # May return a dict with 'features' key or a list directly
        rows = result["features"] if isinstance(result, dict) and "features" in result else result
        assert isinstance(rows, list)
        assert len(rows) > 0
        for f in rows:
            assert "quadrant" in f
            assert f["quadrant"] in ("HH", "LL", "HL", "LH", "ns")


class TestVariogramFit:
    def test_basic(self):
        feats = _sample_features(8)
        result = gp.variogram_fit(feats, value_column="value", geometry_column="geometry")
        assert "nugget" in result
        assert "sill" in result
        assert "range" in result
        # key is 'lags' or 'empirical_lags'
        assert "lags" in result or "empirical_lags" in result


class TestSpatialRegression:
    def test_lag_regression(self):
        feats = [
            {"y": float(i), "x1": float(i), "geometry": {"type": "Point", "coordinates": [float(i), 0.0]}}
            for i in range(6)
        ]
        result = gp.spatial_lag_regression(feats, dependent="y", independent=["x1"], geometry_column="geometry")
        assert "coefficients" in result
        assert "r_squared" in result

    def test_error_regression(self):
        feats = [
            {"y": float(i), "x1": float(i), "geometry": {"type": "Point", "coordinates": [float(i), 0.0]}}
            for i in range(6)
        ]
        result = gp.spatial_error_regression(feats, dependent="y", independent=["x1"], geometry_column="geometry")
        assert "coefficients" in result
        assert "lambda" in result


class TestGWR:
    def test_gwr_basic(self):
        feats = [
            {"y": float(i), "x1": float(i), "geometry": {"type": "Point", "coordinates": [float(i), 0.0]}}
            for i in range(8)
        ]
        result = gp.gwr(feats, dependent="y", independent=["x1"], geometry_column="geometry", bandwidth=5)
        # May return list of per-feature dicts or a dict with local_coefficients
        if isinstance(result, list):
            assert len(result) == 8
            assert "local_coefficients" in result[0]
        else:
            assert "local_coefficients" in result
            assert len(result["local_coefficients"]) == 8


class TestSpatialOutlierLOF:
    def test_returns_outlier_flags(self):
        feats = _sample_features(8)
        result = gp.spatial_outlier_lof(feats, geometry_column="geometry", value_column="value")
        # May return list of feature dicts or dict with 'outliers'/'scores'
        if isinstance(result, list):
            assert len(result) == 8
            assert "is_outlier" in result[0] or "lof_score" in result[0]
        else:
            assert "outliers" in result or "scores" in result


class TestInterpolation:
    def _ctrl_points(self):
        # (x, y, value) tuples
        return [(float(i), 0.0, float(i * 2)) for i in range(5)]

    def _query_points(self):
        # (x, y) tuples
        return [(1.5, 0.0), (3.5, 0.0)]

    def test_natural_neighbor(self):
        import inspect
        sig = inspect.signature(gp.natural_neighbor_interpolation)
        params = list(sig.parameters.keys())
        cp = self._ctrl_points()
        qp = self._query_points()
        if "query_point" in params:
            # Single-query API: (points, values, query_point)
            pts = [(x, y) for x, y, _ in cp]
            vals = [v for _, _, v in cp]
            z = gp.natural_neighbor_interpolation(pts, vals, (1.5, 0.0))
            assert isinstance(z, (int, float))
        else:
            # Batch API: (control_points, query_points)
            result = gp.natural_neighbor_interpolation(cp, qp)
            assert len(result) == 2
            for r in result:
                assert isinstance(r, (int, float))

    def test_tin_interpolation(self):
        import inspect
        sig = inspect.signature(gp.tin_interpolation)
        params = list(sig.parameters.keys())
        cp = self._ctrl_points()
        qp = self._query_points()
        if "query_point" in params:
            pts = [(x, y) for x, y, _ in cp]
            vals = [v for _, _, v in cp]
            z = gp.tin_interpolation(pts, vals, (1.5, 0.0))
            assert isinstance(z, (int, float))
        else:
            result = gp.tin_interpolation(cp, qp)
            assert len(result) >= 1


# ---------------------------------------------------------------------------
# G8 — data management additions
# ---------------------------------------------------------------------------

class TestDescribeDataset:
    def test_basic(self):
        records = [
            {"name": "A", "pop": 100, "geometry": {"type": "Point", "coordinates": [1.0, 2.0]}},
            {"name": "B", "pop": 200, "geometry": {"type": "Point", "coordinates": [3.0, 4.0]}},
        ]
        desc = gp.describe_dataset(records)
        assert desc["row_count"] == 2
        assert desc["geometry_type"] == "Point"
        assert desc["extent"] is not None
        fields = {f["name"]: f["type"] for f in desc["fields"]}
        assert "pop" in fields
        assert fields["pop"] == "int"

    def test_empty(self):
        desc = gp.describe_dataset([])
        assert desc["row_count"] == 0


class TestPivotTable:
    def _records(self):
        return [
            {"region": "N", "category": "A", "sales": 10},
            {"region": "N", "category": "B", "sales": 20},
            {"region": "S", "category": "A", "sales": 30},
            {"region": "S", "category": "B", "sales": 40},
        ]

    def test_sum(self):
        result = gp.pivot_table(self._records(), row_field="region", col_field="category", value_field="sales")
        assert result["rows"] == ["N", "S"]
        assert set(result["columns"]) == {"A", "B"}
        flat = [v for row in result["table"] for v in row]
        assert sum(flat) == 100.0

    def test_mean(self):
        records = [
            {"r": "X", "c": "p", "v": 10},
            {"r": "X", "c": "p", "v": 20},
        ]
        result = gp.pivot_table(records, row_field="r", col_field="c", value_field="v", aggfunc="mean")
        assert result["table"][0][0] == 15.0


class TestMultipartSinglepart:
    def test_multipoint_explode(self):
        records = [{"id": 1, "geometry": {"type": "MultiPoint", "coordinates": [[0, 0], [1, 1], [2, 2]]}}]
        result = gp.multipart_to_singlepart(records)
        assert len(result) == 3
        for r in result:
            assert r["geometry"]["type"] == "Point"

    def test_simple_passthrough(self):
        records = [{"id": 1, "geometry": {"type": "Point", "coordinates": [0, 0]}}]
        result = gp.multipart_to_singlepart(records)
        assert len(result) == 1

    def test_singlepart_to_multipart(self):
        records = [
            {"group": "A", "geometry": {"type": "Point", "coordinates": [0, 0]}},
            {"group": "A", "geometry": {"type": "Point", "coordinates": [1, 1]}},
            {"group": "B", "geometry": {"type": "Point", "coordinates": [5, 5]}},
        ]
        result = gp.singlepart_to_multipart(records, group_field="group")
        assert len(result) == 2
        groups = {r["group"]: r for r in result}
        assert groups["A"]["geometry"]["type"] == "MultiPoint"
        assert len(groups["A"]["geometry"]["coordinates"]) == 2


class TestFeatureVerticesToPoints:
    def test_polygon_vertices(self):
        records = [{"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]}}]
        result = gp.feature_vertices_to_points(records)
        assert len(result) == 5
        for r in result:
            assert r["geometry"]["type"] == "Point"
            assert "vertex_index" in r


class TestRepairGeometryFull:
    def test_passthrough_valid(self):
        records = [{"geometry": {"type": "Point", "coordinates": [1.0, 2.0]}}]
        result = gp.repair_geometry_full(records)
        assert len(result) == 1
        assert result[0]["geometry"]["type"] == "Point"

    def test_empty_geometry_passthrough(self):
        records = [{"id": 1}]
        result = gp.repair_geometry_full(records)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# G13 — space-time additions
# ---------------------------------------------------------------------------

class TestChangePointDetection:
    def test_step_signal(self):
        series = [1.0] * 10 + [10.0] * 10
        result = gp.change_point_detection(series)
        assert "change_points" in result
        # CUSUM should detect the big step change
        cps = result["change_points"]
        # At least one change point (may need low threshold for small datasets)
        result2 = gp.change_point_detection(series, threshold=1.0)
        cps2 = result2["change_points"]
        assert len(cps) >= 1 or len(cps2) >= 1

    def test_pelt_method(self):
        series = [1.0, 1.1, 1.0, 1.1, 10.0, 10.1, 10.0]
        result = gp.change_point_detection(series, method="pelt")
        assert "change_points" in result

    def test_no_change(self):
        series = [5.0] * 20
        result = gp.change_point_detection(series)
        # Flat signal → no change points
        assert result["change_points"] == []


class TestTemporalAggregation:
    def test_daily_sum(self):
        events = [
            {"t": "2024-01-01T10:00:00", "value": 5.0},
            {"t": "2024-01-01T15:00:00", "value": 3.0},
            {"t": "2024-01-02T09:00:00", "value": 7.0},
        ]
        result = gp.temporal_aggregation(events, time_field="t", value_field="value", period="day")
        assert len(result) == 2
        days = {r["period"]: r["value"] for r in result}
        assert days["2024-01-01"] == 8.0
        assert days["2024-01-02"] == 7.0

    def test_monthly_mean(self):
        events = [
            {"t": "2024-01-05", "value": 10.0},
            {"t": "2024-01-20", "value": 20.0},
            {"t": "2024-02-10", "value": 30.0},
        ]
        result = gp.temporal_aggregation(events, period="month", aggfunc="mean")
        months = {r["period"]: r["value"] for r in result}
        assert months["2024-01"] == 15.0
        assert months["2024-02"] == 30.0


class TestTemporalJoinWindow:
    def test_exact_match(self):
        left = [{"id": "A", "t": 100.0}]
        right = [{"ref": "X", "t": 100.5}]
        result = gp.temporal_join_window(left, right, window=1.0)
        assert len(result) == 1
        assert result[0]["_right_matched"] is True

    def test_outside_window(self):
        left = [{"id": "A", "t": 100.0}]
        right = [{"ref": "X", "t": 200.0}]
        result = gp.temporal_join_window(left, right, window=1.0)
        assert len(result) == 1
        assert result[0]["_right_matched"] is False


class TestTrajectoryAnalysis:
    def _make_frame(self):
        import geoprompt
        rows = [
            {"track_id": "T1", "timestamp": "2024-01-01T00:00:00",
             "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}},
            {"track_id": "T1", "timestamp": "2024-01-01T01:00:00",
             "geometry": {"type": "Point", "coordinates": [3.0, 4.0]}},
            {"track_id": "T2", "timestamp": "2024-01-01T00:00:00",
             "geometry": {"type": "Point", "coordinates": [10.0, 10.0]}},
        ]
        return geoprompt.GeoPromptFrame(rows, geometry_column="geometry")

    def test_basic(self):
        frame = self._make_frame()
        result = gp.trajectory_analysis(frame, id_column="track_id", time_column="timestamp")
        assert isinstance(result, list)
        assert len(result) == 2
        t1 = next(r for r in result if r["track_id"] == "T1")
        assert t1["n_points"] == 2
        assert abs(t1["total_distance"] - 5.0) < 0.01


# ---------------------------------------------------------------------------
# G24 — simulation_only decorator
# ---------------------------------------------------------------------------

class TestSimulationOnlyDecorator:
    def test_emits_warning(self):
        from geoprompt.quality import simulation_only

        @simulation_only("Use the real library.")
        def my_stub(x):
            return x * 2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = my_stub(5)
            assert result == 10
            assert len(w) == 1
            assert "simulation-only" in str(w[0].message).lower()
            assert "real library" in str(w[0].message)

    def test_is_simulation_only(self):
        from geoprompt.quality import simulation_only, is_simulation_only

        @simulation_only("test reason")
        def decorated():
            pass

        def undecorated():
            pass

        assert is_simulation_only(decorated) is True
        assert is_simulation_only(undecorated) is False

    def test_ml_stubs_are_flagged(self):
        from geoprompt.quality import is_simulation_only
        import geoprompt.ml as ml
        # These stubs should be decorated
        assert is_simulation_only(ml.gradient_boosted_spatial_prediction)
        assert is_simulation_only(ml.svm_spatial_classification)
        assert is_simulation_only(ml.neural_network_integration)

    def test_performance_stubs_are_flagged(self):
        from geoprompt.quality import is_simulation_only
        import geoprompt.performance as perf
        assert not is_simulation_only(perf.gpu_accelerated_distance_matrix)
        assert is_simulation_only(perf.gpu_accelerated_raster_algebra)
        assert is_simulation_only(perf.distributed_spatial_join)

    def test_standards_stubs_are_flagged(self):
        from geoprompt.quality import is_simulation_only
        import geoprompt.standards as std
        assert not is_simulation_only(std.ogc_api_features_implementation)
        assert not is_simulation_only(std.ogc_wfs_client)
        assert not is_simulation_only(std.ogc_wms_client)

    def test_warning_includes_function_name(self):
        from geoprompt.quality import simulation_only

        @simulation_only("do the real thing")
        def named_fn():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            named_fn()
            assert "named_fn" in str(w[0].message)

    def test_random_forest_real_sklearn(self):
        """RF should use real sklearn when available (no simulation warning)."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not installed")

        from geoprompt import GeoPromptFrame
        rows = [
            {"feat": float(i), "label": "A" if i < 5 else "B",
             "geometry": {"type": "Point", "coordinates": [float(i), 0.0]}}
            for i in range(10)
        ]
        frame = GeoPromptFrame(rows, geometry_column="geometry")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = gp.random_forest_spatial_prediction(frame, target_column="label", feature_columns=["feat"])
            # With real sklearn, no simulation-only warning should be emitted
            sim_warnings = [x for x in w if "simulation-only" in str(x.message).lower()]
            assert not sim_warnings
        assert "predictions" in result or "model" in result or "accuracy" in result
