"""Tests for new modules: spatial_analysis, classification, cogo, geoprocessing, landscape."""
from __future__ import annotations

import math
import pytest


# ===================================================================
# spatial_analysis tests
# ===================================================================
class TestSpatialWeights:
    def test_distance_band_binary(self):
        from geoprompt.spatial_analysis import spatial_weights_distance_band
        pts = [(0, 0), (1, 0), (0, 1), (5, 5)]
        w = spatial_weights_distance_band(pts, 1.5)
        assert 1 in w[0]
        assert 2 in w[0]
        assert 3 not in w[0]

    def test_distance_band_non_binary(self):
        from geoprompt.spatial_analysis import spatial_weights_distance_band
        pts = [(0, 0), (1, 0)]
        w = spatial_weights_distance_band(pts, 2.0, binary=False)
        assert abs(w[0][1] - 1.0) < 1e-6

    def test_inverse_distance(self):
        from geoprompt.spatial_analysis import spatial_weights_inverse_distance
        pts = [(0, 0), (1, 0), (2, 0)]
        w = spatial_weights_inverse_distance(pts, power=2.0)
        assert w[0][1] > w[0][2]

    def test_kernel_gaussian(self):
        from geoprompt.spatial_analysis import spatial_weights_kernel
        pts = [(0, 0), (0.5, 0), (10, 10)]
        w = spatial_weights_kernel(pts, 5.0, kernel="gaussian")
        assert 1 in w[0]
        assert 2 not in w[0]

    def test_kernel_bisquare(self):
        from geoprompt.spatial_analysis import spatial_weights_kernel
        pts = [(0, 0), (1, 0)]
        w = spatial_weights_kernel(pts, 2.0, kernel="bisquare")
        assert w[0][1] > 0

    def test_kernel_triangular(self):
        from geoprompt.spatial_analysis import spatial_weights_kernel
        pts = [(0, 0), (1, 0)]
        w = spatial_weights_kernel(pts, 2.0, kernel="triangular")
        assert 0 < w[0][1] < 1

    def test_row_standardize(self):
        from geoprompt.spatial_analysis import spatial_weights_distance_band, row_standardize_weights
        pts = [(0, 0), (1, 0), (0, 1)]
        w = spatial_weights_distance_band(pts, 2.0)
        rs = row_standardize_weights(w)
        for i in rs:
            if rs[i]:
                assert abs(sum(rs[i].values()) - 1.0) < 1e-10


class TestGetisOrdGeneralG:
    def test_basic(self):
        from geoprompt.spatial_analysis import getis_ord_general_g, spatial_weights_distance_band
        pts = [(0, 0), (1, 0), (2, 0), (3, 0)]
        w = spatial_weights_distance_band(pts, 1.5)
        result = getis_ord_general_g([10, 20, 30, 40], w)
        assert "G" in result
        assert "z_score" in result


class TestRipleysK:
    def test_basic(self):
        from geoprompt.spatial_analysis import ripleys_k
        pts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        results = ripleys_k(pts, [0.5, 1.0, 2.0])
        assert len(results) == 3
        assert all("K" in r and "L" in r for r in results)


class TestPointDensity:
    def test_basic(self):
        from geoprompt.spatial_analysis import point_density
        pts = [(0.5, 0.5), (0.5, 0.5), (1.5, 1.5)]
        result = point_density(pts, 1.0, extent=(0, 0, 2, 2))
        assert result["rows"] == 2
        assert result["cols"] == 2
        assert sum(result["cells"]) == 3

    def test_no_points_raises(self):
        from geoprompt.spatial_analysis import point_density
        with pytest.raises(ValueError):
            point_density([], 1.0)


class TestLineDensity:
    def test_basic(self):
        from geoprompt.spatial_analysis import line_density
        lines = [[(0.5, 0.5), (0.5, 1.5)]]
        result = line_density(lines, 1.0, extent=(0, 0, 2, 2))
        assert sum(result["cells"]) > 0


class TestDirectionalDistribution:
    def test_basic(self):
        from geoprompt.spatial_analysis import directional_distribution
        pts = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        result = directional_distribution(pts)
        assert "center" in result
        assert "semi_major" in result
        assert result["semi_major"] >= result["semi_minor"]


class TestDBSCAN:
    def test_clusters(self):
        from geoprompt.spatial_analysis import dbscan_spatial
        pts = [(0, 0), (0.1, 0), (0, 0.1), (10, 10), (10.1, 10), (10, 10.1)]
        result = dbscan_spatial(pts, eps=0.5, min_samples=2)
        assert result["n_clusters"] == 2
        assert len(result["labels"]) == 6


class TestKMeans:
    def test_clusters(self):
        from geoprompt.spatial_analysis import kmeans_spatial
        pts = [(0, 0), (0.1, 0), (10, 10), (10.1, 10)]
        result = kmeans_spatial(pts, 2, seed=42)
        assert len(result["labels"]) == 4
        assert len(result["centroids"]) == 2

    def test_invalid_k(self):
        from geoprompt.spatial_analysis import kmeans_spatial
        with pytest.raises(ValueError):
            kmeans_spatial([(0, 0)], 5)


class TestOPTICS:
    def test_ordering(self):
        from geoprompt.spatial_analysis import optics_ordering
        pts = [(0, 0), (1, 0), (0, 1), (10, 10)]
        result = optics_ordering(pts, min_samples=2)
        assert len(result) == 4
        assert all("index" in r for r in result)


class TestAHP:
    def test_consistent_matrix(self):
        from geoprompt.spatial_analysis import analytic_hierarchy_process
        matrix = [[1, 3, 5], [1/3, 1, 2], [1/5, 1/2, 1]]
        result = analytic_hierarchy_process(matrix)
        assert len(result["weights"]) == 3
        assert abs(sum(result["weights"]) - 1.0) < 0.1
        assert result["consistency_ratio"] < 0.2


class TestTOPSIS:
    def test_basic(self):
        from geoprompt.spatial_analysis import topsis
        matrix = [[4, 7], [8, 4], [6, 6]]
        result = topsis(matrix, [0.5, 0.5])
        assert len(result["scores"]) == 3
        assert len(result["ranking"]) == 3


class TestWSMWPM:
    def test_wsm(self):
        from geoprompt.spatial_analysis import weighted_sum_model
        result = weighted_sum_model([[1, 2], [3, 4]], [0.5, 0.5])
        assert len(result) == 2
        assert result[0] == pytest.approx(1.5)

    def test_wpm(self):
        from geoprompt.spatial_analysis import weighted_product_model
        result = weighted_product_model([[2, 3], [4, 5]], [0.5, 0.5])
        assert len(result) == 2
        assert result[0] > 0


class TestFuzzyOverlay:
    def test_and(self):
        from geoprompt.spatial_analysis import fuzzy_overlay
        result = fuzzy_overlay([[0.8, 0.3], [0.6, 0.9]], method="and")
        assert result[0] == pytest.approx(0.6)
        assert result[1] == pytest.approx(0.3)

    def test_or(self):
        from geoprompt.spatial_analysis import fuzzy_overlay
        result = fuzzy_overlay([[0.8, 0.3], [0.6, 0.9]], method="or")
        assert result[0] == pytest.approx(0.8)

    def test_product(self):
        from geoprompt.spatial_analysis import fuzzy_overlay
        result = fuzzy_overlay([[0.5], [0.5]], method="product")
        assert result[0] == pytest.approx(0.25)


class TestBooleanOverlay:
    def test_and(self):
        from geoprompt.spatial_analysis import boolean_overlay
        result = boolean_overlay([[True, False], [True, True]], method="and")
        assert result == [True, False]

    def test_or(self):
        from geoprompt.spatial_analysis import boolean_overlay
        result = boolean_overlay([[True, False], [False, True]], method="or")
        assert result == [True, True]


class TestInterpolation:
    def test_natural_neighbor(self):
        from geoprompt.spatial_analysis import natural_neighbor_interpolation
        pts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        vals = [0, 1, 1, 2]
        result = natural_neighbor_interpolation(pts, vals, (0.5, 0.5))
        assert 0 < result < 2

    def test_local_polynomial(self):
        from geoprompt.spatial_analysis import local_polynomial_interpolation
        pts = [(0, 0), (1, 0), (0, 1), (1, 1)]
        vals = [0, 1, 1, 2]
        result = local_polynomial_interpolation(pts, vals, (0.5, 0.5))
        assert isinstance(result, float)

    def test_rbf(self):
        from geoprompt.spatial_analysis import radial_basis_function_interpolation
        pts = [(0, 0), (10, 0), (0, 10)]
        vals = [0.0, 10.0, 10.0]
        result = radial_basis_function_interpolation(pts, vals, (5, 5))
        assert isinstance(result, float)

    def test_cross_validation(self):
        from geoprompt.spatial_analysis import cross_validation_interpolation
        pts = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)]
        vals = [0, 1, 1, 2, 1]
        result = cross_validation_interpolation(pts, vals, "idw")
        assert "rmse" in result
        assert result["rmse"] >= 0


class TestServiceAreas:
    def test_buffer_service_area(self):
        from geoprompt.spatial_analysis import buffer_service_area
        result = buffer_service_area((0, 0), 1.0)
        assert result["type"] == "Polygon"
        assert len(result["coordinates"][0]) == 37

    def test_2sfca(self):
        from geoprompt.spatial_analysis import two_step_floating_catchment_area
        supply = [{"location": (0, 0), "capacity": 10}]
        demand = [{"location": (0, 0), "population": 5}, {"location": (100, 100), "population": 5}]
        scores = two_step_floating_catchment_area(supply, demand, 50)
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_e2sfca(self):
        from geoprompt.spatial_analysis import enhanced_two_step_fca
        supply = [{"location": (0, 0), "capacity": 10}]
        demand = [{"location": (0, 0), "population": 5}]
        scores = enhanced_two_step_fca(supply, demand, 50)
        assert len(scores) == 1
        assert scores[0] > 0


class TestHuffModel:
    def test_probabilities_sum_to_one(self):
        from geoprompt.spatial_analysis import huff_model
        stores = [{"location": (0, 0), "size": 10}, {"location": (10, 0), "size": 20}]
        probs = huff_model(stores, [(5, 0)])
        assert len(probs) == 1
        assert abs(sum(probs[0]) - 1.0) < 1e-10


class TestMarketPenetration:
    def test_bins(self):
        from geoprompt.spatial_analysis import market_penetration
        customers = [(1, 0), (2, 0), (6, 0), (20, 0)]
        result = market_penetration(customers, (0, 0), distance_bins=[3, 10])
        assert result["total_customers"] == 4
        assert result["bins"]["0-3"] == 2


class TestQuantileAndProbability:
    def test_quantile_map(self):
        from geoprompt.spatial_analysis import quantile_map
        result = quantile_map(list(range(20)), n_classes=4)
        assert len(result["breaks"]) == 3
        assert len(result["labels"]) == 20

    def test_probability_map(self):
        from geoprompt.spatial_analysis import probability_map
        probs = probability_map([1, 2, 3, 4, 5], 3)
        assert len(probs) == 5
        assert all(0 <= p <= 1 for p in probs)


# ===================================================================
# classification tests
# ===================================================================
class TestClassification:
    def test_natural_breaks(self):
        from geoprompt.classification import classify_natural_breaks
        vals = list(range(20))
        breaks = classify_natural_breaks(vals, 3)
        assert len(breaks) == 2
        assert breaks[0] < breaks[1]

    def test_equal_interval(self):
        from geoprompt.classification import classify_equal_interval
        breaks = classify_equal_interval([0, 10, 20, 30, 40], 4)
        assert len(breaks) == 3
        assert breaks[0] == pytest.approx(10.0)

    def test_quantile(self):
        from geoprompt.classification import classify_quantile
        breaks = classify_quantile(list(range(100)), 5)
        assert len(breaks) == 4

    def test_standard_deviation(self):
        from geoprompt.classification import classify_standard_deviation
        breaks = classify_standard_deviation(list(range(100)))
        assert len(breaks) == 5

    def test_manual_breaks(self):
        from geoprompt.classification import classify_manual_breaks
        labels = classify_manual_breaks([1, 5, 15, 25], [10, 20])
        assert labels == [0, 0, 1, 2]

    def test_geometric_interval(self):
        from geoprompt.classification import classify_geometric_interval
        breaks = classify_geometric_interval([1, 2, 4, 8, 16, 32], 3)
        assert len(breaks) == 2

    def test_defined_interval(self):
        from geoprompt.classification import classify_defined_interval
        breaks = classify_defined_interval(list(range(100)), 25)
        assert len(breaks) == 3

    def test_classify_values(self):
        from geoprompt.classification import classify_values
        labels = classify_values([5, 15, 25, 35], [10, 20, 30])
        assert labels == [0, 1, 2, 3]


class TestColorRamps:
    def test_sequential(self):
        from geoprompt.classification import sequential_color_ramp
        colors = sequential_color_ramp(5)
        assert len(colors) == 5
        assert all(c.startswith("#") for c in colors)

    def test_diverging(self):
        from geoprompt.classification import diverging_color_ramp
        colors = diverging_color_ramp(7)
        assert len(colors) == 7

    def test_qualitative(self):
        from geoprompt.classification import qualitative_color_palette
        colors = qualitative_color_palette(6)
        assert len(colors) == 6
        assert len(set(colors)) == 6

    def test_colorblind_safe(self):
        from geoprompt.classification import colorblind_safe_palette
        colors = colorblind_safe_palette(4)
        assert len(colors) == 4

    def test_custom_ramp(self):
        from geoprompt.classification import custom_color_ramp
        stops = [(0.0, (255, 0, 0)), (0.5, (0, 255, 0)), (1.0, (0, 0, 255))]
        colors = custom_color_ramp(stops, 5)
        assert len(colors) == 5

    def test_preset_ramp(self):
        from geoprompt.classification import get_color_ramp_preset
        colors = get_color_ramp_preset("viridis", 5)
        assert len(colors) == 5

    def test_invalid_preset(self):
        from geoprompt.classification import get_color_ramp_preset
        with pytest.raises(ValueError):
            get_color_ramp_preset("nonexistent", 5)


class TestReportGeneration:
    def test_markdown(self):
        from geoprompt.classification import generate_report_markdown
        md = generate_report_markdown("Test", [{"heading": "Section 1", "body": "Hello"}])
        assert "# Test" in md
        assert "## Section 1" in md

    def test_markdown_with_table(self):
        from geoprompt.classification import generate_report_markdown
        md = generate_report_markdown("T", [{"heading": "S", "table": [{"a": 1, "b": 2}]}])
        assert "| a | b |" in md

    def test_html(self):
        from geoprompt.classification import generate_report_html
        html = generate_report_html("Test", [{"heading": "S", "body": "Hi"}])
        assert "<h1>Test</h1>" in html

    def test_template(self):
        from geoprompt.classification import report_template
        result = report_template("Hello {{name}}!", {"name": "World"})
        assert result == "Hello World!"

    def test_geojson_preview(self):
        from geoprompt.classification import geojson_preview_text
        features = [{"geometry": {"type": "Point", "coordinates": [1.0, 2.0]}, "properties": {"id": 1}}]
        text = geojson_preview_text(features)
        assert "1 features" in text

    def test_wkt_preview(self):
        from geoprompt.classification import wkt_preview_text
        text = wkt_preview_text(["POINT (0 0)", "LINESTRING (0 0, 1 1)"])
        assert "2 geometries" in text

    def test_executive_summary(self):
        from geoprompt.classification import executive_summary
        summary = executive_summary({"total_features": 100, "crs": "EPSG:4326"})
        assert "100" in summary


# ===================================================================
# cogo tests
# ===================================================================
class TestBearingDistance:
    def test_bearing_north(self):
        from geoprompt.cogo import bearing_between_points
        assert bearing_between_points((0, 0), (0, 1)) == pytest.approx(0.0, abs=0.01)

    def test_bearing_east(self):
        from geoprompt.cogo import bearing_between_points
        assert bearing_between_points((0, 0), (1, 0)) == pytest.approx(90.0, abs=0.01)

    def test_distance(self):
        from geoprompt.cogo import distance_between_points
        assert distance_between_points((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_point_from_bearing(self):
        from geoprompt.cogo import point_from_bearing_distance
        pt = point_from_bearing_distance((0, 0), 90, 10)
        assert pt[0] == pytest.approx(10.0, abs=0.01)
        assert abs(pt[1]) < 0.01

    def test_bearing_distance_to_polygon(self):
        from geoprompt.cogo import bearing_distance_to_polygon
        legs = [(90, 10), (180, 10), (270, 10), (0, 10)]
        poly = bearing_distance_to_polygon((0, 0), legs)
        assert len(poly) == 6  # 4 legs + start + close
        assert poly[0] == poly[-1]


class TestTraverse:
    def test_closed_traverse(self):
        from geoprompt.cogo import compute_traverse
        obs = [
            {"bearing": 90, "distance": 100},
            {"bearing": 180, "distance": 100},
            {"bearing": 270, "distance": 100},
            {"bearing": 0, "distance": 100},
        ]
        result = compute_traverse((0, 0), obs)
        assert result["misclosure"] < 0.1
        assert result["precision_ratio"] > 1000

    def test_open_traverse(self):
        from geoprompt.cogo import compute_traverse
        obs = [{"bearing": 45, "distance": 100}]
        result = compute_traverse((0, 0), obs, adjust=False)
        assert len(result["coordinates"]) == 2


class TestCurveData:
    def test_radius_and_angle(self):
        from geoprompt.cogo import curve_data
        result = curve_data(radius=100, delta_angle=90)
        assert result["arc_length"] == pytest.approx(100 * math.pi / 2, abs=0.1)
        assert result["chord_length"] > 0
        assert result["tangent"] > 0

    def test_radius_and_arc(self):
        from geoprompt.cogo import curve_data
        result = curve_data(radius=100, arc_length=50)
        assert result["delta_angle"] > 0


class TestSurveyAdjustment:
    def test_translation_only(self):
        from geoprompt.cogo import adjust_survey_points
        observed = [(0, 0), (1, 0), (1, 1)]
        control = [(10, 10)]
        adjusted = adjust_survey_points(observed, control, [0])
        assert adjusted[0] == pytest.approx((10, 10), abs=0.01)

    def test_helmert(self):
        from geoprompt.cogo import adjust_survey_points
        observed = [(0, 0), (10, 0), (10, 10)]
        control = [(0, 0), (10, 0)]
        adjusted = adjust_survey_points(observed, control, [0, 1])
        assert adjusted[0][0] == pytest.approx(0, abs=0.5)


class TestMetesAndBounds:
    def test_parse(self):
        from geoprompt.cogo import parse_metes_and_bounds
        desc = "N 45°30' E 100.00 then S 30°0' W 200.50"
        legs = parse_metes_and_bounds(desc)
        assert len(legs) == 2
        assert legs[0]["distance"] == pytest.approx(100.0)
        assert legs[1]["distance"] == pytest.approx(200.5)


class TestSetbackFARCoverage:
    def test_setback_pass(self):
        from geoprompt.cogo import setback_check
        result = setback_check(
            [(5, 5), (5, 15), (15, 15), (15, 5)],
            [(0, 0), (0, 20), (20, 20), (20, 0)],
            required_setback=3.0,
        )
        assert result["passes"] is True

    def test_far(self):
        from geoprompt.cogo import floor_area_ratio
        assert floor_area_ratio(5000, 10000) == pytest.approx(0.5)

    def test_lot_coverage(self):
        from geoprompt.cogo import lot_coverage
        assert lot_coverage(2500, 10000) == pytest.approx(25.0)


class TestLeastSquares:
    def test_basic(self):
        from geoprompt.cogo import least_squares_adjustment
        obs = [
            {"from": (0, 0), "to": (10, 0), "bearing": 90, "distance": 10},
            {"from": (10, 0), "to": (10, 10), "bearing": 0, "distance": 10},
        ]
        result = least_squares_adjustment(obs)
        assert len(result["coordinates"]) == 3


# ===================================================================
# geoprocessing tests
# ===================================================================
class TestEnvironment:
    def test_get_set(self):
        from geoprompt.geoprocessing import get_environment, set_environment, reset_environment
        reset_environment()
        assert get_environment("workspace") == "."
        set_environment(workspace="/tmp")
        assert get_environment("workspace") == "/tmp"
        reset_environment()

    def test_list_environments(self):
        from geoprompt.geoprocessing import list_environments
        envs = list_environments()
        assert "workspace" in envs

    def test_invalid_key(self):
        from geoprompt.geoprocessing import set_environment
        with pytest.raises(KeyError):
            set_environment(nonexistent_key=True)


class TestToolResult:
    def test_basic(self):
        from geoprompt.geoprocessing import ToolResult
        r = ToolResult(tool_name="test", output="ok")
        assert r.succeeded()
        assert r.get_output() == "ok"
        assert r.run_id


class TestToolHistory:
    def test_log_and_retrieve(self):
        from geoprompt.geoprocessing import ToolResult, log_tool_execution, get_tool_history, clear_tool_history
        clear_tool_history()
        r = ToolResult(tool_name="test", output="result")
        log_tool_execution(r)
        hist = get_tool_history()
        assert len(hist) >= 1
        assert hist[-1]["tool_name"] == "test"
        clear_tool_history()


class TestProgressReporter:
    def test_update(self):
        from geoprompt.geoprocessing import ProgressReporter
        p = ProgressReporter(total=10)
        p.update(3, "step1")
        assert p.current == 3
        assert p.percentage() == pytest.approx(30.0)
        p.update(2)
        assert p.current == 5


class TestHooks:
    def test_register_fire(self):
        from geoprompt.geoprocessing import register_hook, fire_hooks, clear_hooks
        results = []
        def my_hook(**kwargs):
            results.append(kwargs)
        register_hook("pre_tool", my_hook)
        fire_hooks("pre_tool", tool_name="test")
        assert len(results) == 1
        clear_hooks()

    def test_unregister(self):
        from geoprompt.geoprocessing import register_hook, unregister_hook, fire_hooks, clear_hooks
        called = [False]
        def hook(**kw):
            called[0] = True
        register_hook("pre_tool", hook)
        unregister_hook("pre_tool", hook)
        fire_hooks("pre_tool")
        assert not called[0]
        clear_hooks()


class TestMiddleware:
    def test_pipeline(self):
        from geoprompt.geoprocessing import MiddlewarePipeline
        log = []
        def mw(ctx, next_fn):
            log.append("before")
            result = next_fn(ctx)
            log.append("after")
            return result
        def handler(ctx):
            return ctx.get("value", 0) * 2
        p = MiddlewarePipeline()
        p.add(mw)
        result = p.execute({"value": 5}, handler)
        assert result == 10
        assert log == ["before", "after"]


class TestToolChain:
    def test_chain_run(self):
        from geoprompt.geoprocessing import ToolChain, clear_hooks, clear_tool_history
        clear_hooks()
        clear_tool_history()
        chain = ToolChain()
        chain.add_step("double", lambda x: x * 2)
        chain.add_step("add_one", lambda x: x + 1)
        results = chain.run(5)
        assert len(results) == 2
        assert results[-1].output == 11

    def test_chain_error(self):
        from geoprompt.geoprocessing import ToolChain, clear_hooks, clear_tool_history
        clear_hooks()
        clear_tool_history()
        chain = ToolChain()
        chain.add_step("fail", lambda x: 1 / 0)
        results = chain.run(1)
        assert results[0].status == "failure"


class TestBatchProcess:
    def test_batch(self):
        from geoprompt.geoprocessing import batch_process
        def add(a, b):
            return a + b
        results = batch_process(add, [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert len(results) == 2
        assert results[0].output == 3
        assert results[1].output == 7


class TestDryRun:
    def test_valid(self):
        from geoprompt.geoprocessing import dry_run
        def my_func(a, b, c=10):
            pass
        result = dry_run(my_func, a=1, b=2)
        assert result["valid"]

    def test_missing_param(self):
        from geoprompt.geoprocessing import dry_run
        def my_func(a, b):
            pass
        result = dry_run(my_func, a=1)
        assert not result["valid"]


class TestTransaction:
    def test_commit(self):
        from geoprompt.geoprocessing import transaction_scope
        data = [1, 2, 3]
        with transaction_scope(data) as txn:
            data.append(4)
            txn.commit()
        assert len(data) == 4

    def test_rollback_on_error(self):
        from geoprompt.geoprocessing import transaction_scope
        data = [1, 2, 3]
        with pytest.raises(ValueError):
            with transaction_scope(data) as txn:
                raise ValueError("oops")


class TestProfiles:
    def test_load_profile(self):
        from geoprompt.geoprocessing import load_profile, get_environment, reset_environment
        reset_environment()
        load_profile("dev")
        assert get_environment("overwrite_output") is True
        reset_environment()

    def test_register_profile(self):
        from geoprompt.geoprocessing import register_profile, list_profiles
        register_profile("custom", {"overwrite_output": True})
        assert "custom" in list_profiles()

    def test_invalid_profile(self):
        from geoprompt.geoprocessing import load_profile
        with pytest.raises(ValueError):
            load_profile("nonexistent_profile")


class TestFeatureFlags:
    def test_set_get(self):
        from geoprompt.geoprocessing import set_feature_flag, get_feature_flag, clear_feature_flags
        set_feature_flag("new_algo", True)
        assert get_feature_flag("new_algo") is True
        assert get_feature_flag("unknown", False) is False
        clear_feature_flags()


class TestTestDataGenerators:
    def test_random_features_point(self):
        from geoprompt.geoprocessing import generate_random_features
        feats = generate_random_features(10, seed=42)
        assert len(feats) == 10
        assert feats[0]["geometry"]["type"] == "Point"

    def test_random_features_polygon(self):
        from geoprompt.geoprocessing import generate_random_features
        feats = generate_random_features(5, geometry_type="Polygon", seed=42)
        assert feats[0]["geometry"]["type"] == "Polygon"

    def test_random_network(self):
        from geoprompt.geoprocessing import generate_random_network
        net = generate_random_network(5, connectivity=0.5, seed=42)
        assert len(net["nodes"]) == 5
        assert len(net["edges"]) > 0


class TestSpatialError:
    def test_context(self):
        from geoprompt.geoprocessing import SpatialError
        err = SpatialError("bad geom", feature_id=42, location=(1.0, 2.0))
        assert "42" in str(err)
        assert "1.0" in str(err)


class TestInMemoryWorkspace:
    def test_create_and_retrieve(self):
        from geoprompt.geoprocessing import InMemoryWorkspace
        ws = InMemoryWorkspace()
        ws.create_feature_class("test", [{"geometry": {"type": "Point"}, "properties": {"id": 1}}])
        assert ws.exists("test")
        assert len(ws.get_feature_class("test")) == 1

    def test_describe(self):
        from geoprompt.geoprocessing import InMemoryWorkspace
        ws = InMemoryWorkspace()
        ws.create_table("t", [{"a": 1, "b": 2}])
        desc = ws.describe("t")
        assert desc["type"] == "Table"
        assert desc["count"] == 1

    def test_delete(self):
        from geoprompt.geoprocessing import InMemoryWorkspace
        ws = InMemoryWorkspace()
        ws.create_feature_class("x")
        ws.delete("x")
        assert not ws.exists("x")

    def test_list(self):
        from geoprompt.geoprocessing import InMemoryWorkspace
        ws = InMemoryWorkspace()
        ws.create_feature_class("a")
        ws.create_table("b")
        assert "a" in ws.list_feature_classes()
        assert "b" in ws.list_tables()


# ===================================================================
# landscape tests
# ===================================================================
class TestPatchAreas:
    def test_basic(self):
        from geoprompt.landscape import patch_areas
        grid = [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 3, 3]
        patches = patch_areas(grid, 3, 4)
        assert len(patches) >= 3
        total_cells = sum(p["n_cells"] for p in patches)
        assert total_cells == 12


class TestShapeIndex:
    def test_square_patch(self):
        from geoprompt.landscape import shape_index
        grid = [1, 1, 1, 1]
        cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        si = shape_index(cells, 2, 2, grid)
        assert si == pytest.approx(1.0, abs=0.01)


class TestFractalDimension:
    def test_basic(self):
        from geoprompt.landscape import fractal_dimension
        patches = [
            {"area": 1, "perimeter": 4},
            {"area": 4, "perimeter": 8},
            {"area": 16, "perimeter": 16},
        ]
        fd = fractal_dimension(patches)
        assert 1.0 <= fd <= 2.5


class TestEdgeDensity:
    def test_heterogeneous(self):
        from geoprompt.landscape import edge_density
        grid = [1, 2, 1, 2]
        ed = edge_density(grid, 2, 2)
        assert len(ed) > 0
        assert all(v > 0 for v in ed.values())


class TestCoreArea:
    def test_basic(self):
        from geoprompt.landscape import core_area
        grid = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ca = core_area(grid, 3, 3, edge_depth=1)
        assert 1 in ca
        assert ca[1] == 1.0  # only center cell


class TestContagion:
    def test_homogeneous(self):
        from geoprompt.landscape import contagion_index
        grid = [1] * 9
        ci = contagion_index(grid, 3, 3)
        assert ci == 100.0

    def test_heterogeneous(self):
        from geoprompt.landscape import contagion_index
        grid = [1, 2, 1, 2, 1, 2, 1, 2, 1]
        ci = contagion_index(grid, 3, 3)
        assert 0 < ci < 100


class TestDiversity:
    def test_shannon(self):
        from geoprompt.landscape import landscape_diversity_shannon
        grid = [1, 1, 2, 2, 3, 3]
        h = landscape_diversity_shannon(grid)
        assert h > 0

    def test_simpson(self):
        from geoprompt.landscape import landscape_diversity_simpson
        grid = [1, 1, 2, 2, 3, 3]
        d = landscape_diversity_simpson(grid)
        assert 0 < d < 1


class TestConnectivity:
    def test_index(self):
        from geoprompt.landscape import connectivity_index
        patches = [
            {"centroid": (0, 0), "area": 10},
            {"centroid": (1, 0), "area": 20},
            {"centroid": (100, 100), "area": 5},
        ]
        ci = connectivity_index(patches, 5)
        assert 0 <= ci <= 1

    def test_graph_metric(self):
        from geoprompt.landscape import graph_connectivity_metric
        adj = {0: [1, 2], 1: [0], 2: [0]}
        result = graph_connectivity_metric(adj)
        assert result["components"] == 1
        assert result["nodes"] == 3
        assert result["edges"] == 2


class TestCorridorAnalysis:
    def test_basic(self):
        from geoprompt.landscape import corridor_analysis
        grid = [1.0] * 9
        result = corridor_analysis(grid, 3, 3, [(0, 0)], [(2, 2)])
        assert result["min_cost"] > 0
        assert len(result["corridor"]) == 9
