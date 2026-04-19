"""Tests for new modules: coord_parsers, graph_algorithms, formats, query, utils."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pytest


# ── coord_parsers ──────────────────────────────────────────────────────────

class TestCoordParsers:
    def test_geohash_roundtrip(self):
        from geoprompt.coord_parsers import geohash_decode, geohash_encode
        gh = geohash_encode(-122.4194, 37.7749, precision=7)
        lon, lat = geohash_decode(gh)
        assert abs(lat - 37.7749) < 0.01
        assert abs(lon - (-122.4194)) < 0.01

    def test_geohash_neighbors(self):
        from geoprompt.coord_parsers import geohash_neighbors
        n = geohash_neighbors("9q8yy")
        assert isinstance(n, dict)
        assert "n" in n and "s" in n

    def test_hilbert_roundtrip(self):
        from geoprompt.coord_parsers import hilbert_d_to_xy, hilbert_xy_to_d
        n = 2**4  # 16x16 grid
        d = hilbert_xy_to_d(n, 3, 4)
        x, y = hilbert_d_to_xy(n, d)
        assert (x, y) == (3, 4)

    def test_hilbert_lonlat(self):
        from geoprompt.coord_parsers import hilbert_decode_to_lonlat, hilbert_encode_lonlat
        d = hilbert_encode_lonlat(-122.4, 37.7, level=16)
        lon, lat = hilbert_decode_to_lonlat(d, level=16)
        assert abs(lon - (-122.4)) < 0.01
        assert abs(lat - 37.7) < 0.01

    def test_kdtree_query(self):
        from geoprompt.coord_parsers import KDTree
        points = [(0, 0), (1, 1), (2, 2), (3, 3)]
        tree = KDTree(points)
        near = tree.query_nearest((0.1, 0.1), k=1)
        assert near[0][0] == 0  # index of nearest point

    def test_kdtree_radius(self):
        from geoprompt.coord_parsers import KDTree
        points = [(0, 0), (1, 1), (10, 10)]
        tree = KDTree(points)
        results = tree.query_radius((0.5, 0.5), radius=2.0)
        assert len(results) == 2  # indices for (0,0) and (1,1)

    def test_helmert_identity(self):
        from geoprompt.coord_parsers import helmert_transform
        out = helmert_transform(1.0, 2.0, 3.0)
        assert abs(out[0] - 1.0) < 1e-6

    def test_ecef_roundtrip(self):
        from geoprompt.coord_parsers import ecef_to_geodetic, geodetic_to_ecef
        x, y, z = geodetic_to_ecef(-122.4194, 37.7749, 0)
        lon, lat, alt = ecef_to_geodetic(x, y, z)
        assert abs(lat - 37.7749) < 0.001
        assert abs(lon - (-122.4194)) < 0.001

    def test_validate_crs_chain(self):
        from geoprompt.coord_parsers import validate_crs_chain
        result = validate_crs_chain("EPSG:4326", "EPSG:32610")
        assert isinstance(result, dict)
        assert result["source_crs"] == "EPSG:4326"

    def test_maidenhead_roundtrip(self):
        from geoprompt.coord_parsers import lonlat_to_maidenhead, maidenhead_to_lonlat
        loc = lonlat_to_maidenhead(-122.4194, 37.7749)
        lon, lat = maidenhead_to_lonlat(loc)
        assert abs(lon - (-122.4194)) < 2.0
        assert abs(lat - 37.7749) < 1.0

    def test_pluscode_roundtrip(self):
        from geoprompt.coord_parsers import lonlat_to_pluscode, pluscode_to_lonlat
        code = lonlat_to_pluscode(-122.4194, 37.7749)
        lon, lat = pluscode_to_lonlat(code)
        assert abs(lon - (-122.4194)) < 0.01
        assert abs(lat - 37.7749) < 0.01

    def test_georef_roundtrip(self):
        from geoprompt.coord_parsers import lonlat_to_georef, georef_to_lonlat
        code = lonlat_to_georef(-122.4194, 37.7749, precision=4)
        lon, lat = georef_to_lonlat(code)
        assert abs(lon - (-122.4194)) < 1.0
        assert abs(lat - 37.7749) < 1.0

    def test_gars_roundtrip(self):
        from geoprompt.coord_parsers import gars_to_lonlat, lonlat_to_gars
        code = lonlat_to_gars(-122.4194, 37.7749)
        lon, lat = gars_to_lonlat(code)
        assert abs(lon - (-122.4194)) < 1.0
        assert abs(lat - 37.7749) < 1.0

    def test_mgrs_roundtrip(self):
        from geoprompt.coord_parsers import lonlat_to_mgrs, mgrs_to_lonlat
        code = lonlat_to_mgrs(-122.4194, 37.7749)
        assert isinstance(code, str) and len(code) > 0
        lon, lat = mgrs_to_lonlat(code)
        assert abs(lon - (-122.4194)) < 1.0
        assert abs(lat - 37.7749) < 1.0

    def test_usng_roundtrip(self):
        from geoprompt.coord_parsers import lonlat_to_usng, usng_to_lonlat
        code = lonlat_to_usng(-122.4194, 37.7749)
        lon, lat = usng_to_lonlat(code)
        assert abs(lon - (-122.4194)) < 1.0
        assert abs(lat - 37.7749) < 1.0


# ── graph_algorithms ───────────────────────────────────────────────────────

class TestGraphAlgorithms:
    @pytest.fixture()
    def sample_graph(self):
        from geoprompt.network.core import NetworkGraph, Traversal
        g = NetworkGraph(directed=False, adjacency={}, edge_attributes={})
        # A--B--C, A--C
        for src, dst, eid, cost in [
            ("A", "B", "e1", 1.0),
            ("B", "C", "e2", 2.0),
            ("A", "C", "e3", 5.0),
        ]:
            t_fwd = Traversal(edge_id=eid, from_node=src, to_node=dst, cost=cost)
            t_rev = Traversal(edge_id=eid, from_node=dst, to_node=src, cost=cost)
            g.adjacency.setdefault(src, []).append(t_fwd)
            g.adjacency.setdefault(dst, []).append(t_rev)
            g.edge_attributes[eid] = {"from_node": src, "to_node": dst, "cost": cost, "length": cost}
        return g

    def test_astar(self, sample_graph):
        from geoprompt.network.graph_algorithms import astar_shortest_path
        result = astar_shortest_path(sample_graph, "A", "C")
        assert result["path"] == ["A", "B", "C"]
        assert result["total_cost"] == 3.0

    def test_bellman_ford(self, sample_graph):
        from geoprompt.network.graph_algorithms import bellman_ford_shortest_path
        result = bellman_ford_shortest_path(sample_graph, "A")
        assert result["distances"]["C"] == 3.0

    def test_floyd_warshall(self, sample_graph):
        from geoprompt.network.graph_algorithms import floyd_warshall
        dist = floyd_warshall(sample_graph)
        assert dist["A"]["C"] == 3.0

    def test_k_shortest(self, sample_graph):
        from geoprompt.network.graph_algorithms import k_shortest_paths
        paths = k_shortest_paths(sample_graph, "A", "C", k=2)
        assert len(paths) == 2
        assert paths[0]["total_cost"] <= paths[1]["total_cost"]

    def test_mst(self, sample_graph):
        from geoprompt.network.graph_algorithms import minimum_spanning_tree
        edges = minimum_spanning_tree(sample_graph)
        assert len(edges) == 2
        total = sum(e["cost"] for e in edges)
        assert total == 3.0

    def test_bridges(self, sample_graph):
        from geoprompt.network.graph_algorithms import find_bridges
        bridges = find_bridges(sample_graph)
        assert isinstance(bridges, list)

    def test_articulation_points(self, sample_graph):
        from geoprompt.network.graph_algorithms import find_articulation_points
        pts = find_articulation_points(sample_graph)
        assert isinstance(pts, list)

    def test_wcc(self, sample_graph):
        from geoprompt.network.graph_algorithms import weakly_connected_components
        comps = weakly_connected_components(sample_graph)
        assert len(comps) == 1
        assert len(comps[0]) == 3

    def test_scc(self, sample_graph):
        from geoprompt.network.graph_algorithms import strongly_connected_components
        comps = strongly_connected_components(sample_graph)
        assert len(comps) >= 1

    def test_trace_connected(self, sample_graph):
        from geoprompt.network.graph_algorithms import trace_connected
        result = trace_connected(sample_graph, "A")
        assert "A" in result["nodes"] and "B" in result["nodes"] and "C" in result["nodes"]

    def test_find_cycles(self, sample_graph):
        from geoprompt.network.graph_algorithms import find_cycles
        cycles = find_cycles(sample_graph, max_length=5)
        assert isinstance(cycles, list)

    def test_partition(self, sample_graph):
        from geoprompt.network.graph_algorithms import partition_network
        parts = partition_network(sample_graph, num_parts=2)
        assert len(parts) == 2

    def test_max_flow(self, sample_graph):
        from geoprompt.network.graph_algorithms import max_flow_min_cut
        result = max_flow_min_cut(sample_graph, "A", "C")
        assert result["max_flow"] > 0


# ── formats ────────────────────────────────────────────────────────────────

class TestFormats:
    @pytest.fixture()
    def sample_features(self):
        return [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-122.4, 37.7]}, "properties": {"name": "A"}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-122.5, 37.8]}, "properties": {"name": "B"}},
        ]

    def test_kml_roundtrip(self, sample_features, tmp_path):
        from geoprompt.formats import read_kml, write_kml
        p = str(tmp_path / "test.kml")
        write_kml(sample_features, p)
        loaded = read_kml(p)
        assert len(loaded) == 2

    def test_gpx_roundtrip(self, sample_features, tmp_path):
        from geoprompt.formats import read_gpx, write_gpx
        p = str(tmp_path / "test.gpx")
        write_gpx(sample_features, p)
        loaded = read_gpx(p)
        assert len(loaded) == 2

    def test_topojson_roundtrip(self, sample_features, tmp_path):
        from geoprompt.formats import read_topojson, write_topojson
        p = str(tmp_path / "test.topojson")
        write_topojson(sample_features, p)
        loaded = read_topojson(p)
        assert len(loaded) == 2

    def test_gml_roundtrip(self, sample_features, tmp_path):
        from geoprompt.formats import read_gml, write_gml
        p = str(tmp_path / "test.gml")
        write_gml(sample_features, p)
        loaded = read_gml(p)
        assert len(loaded) == 2

    def test_geojsonl_write(self, sample_features, tmp_path):
        from geoprompt.formats import write_geojsonl
        p = str(tmp_path / "test.geojsonl")
        write_geojsonl(sample_features, p)
        with open(p) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_iter_geojson_features(self, sample_features, tmp_path):
        from geoprompt.formats import iter_geojson_features, write_geojsonl
        p = str(tmp_path / "test.geojsonl")
        write_geojsonl(sample_features, p)
        loaded = list(iter_geojson_features(p))
        assert len(loaded) == 2

    def test_feature_count(self, sample_features, tmp_path):
        from geoprompt.formats import feature_count
        p = str(tmp_path / "test.geojson")
        with open(p, "w") as f:
            json.dump({"type": "FeatureCollection", "features": sample_features}, f)
        assert feature_count(p) == 2

    def test_quick_bounds(self, sample_features, tmp_path):
        from geoprompt.formats import quick_bounds
        p = str(tmp_path / "test.geojson")
        with open(p, "w") as f:
            json.dump({"type": "FeatureCollection", "features": sample_features}, f)
        bounds = quick_bounds(p)
        assert bounds[0] <= -122.5 and bounds[2] >= -122.4

    def test_round_coordinates(self):
        from geoprompt.formats import round_coordinates
        geom = {"type": "Point", "coordinates": [-122.41941234, 37.77491234]}
        out = round_coordinates(geom, precision=2)
        assert out["coordinates"] == [-122.42, 37.77]

    def test_geo_interface(self, sample_features):
        from geoprompt.formats import from_geo_interface, to_geo_interface
        iface = to_geo_interface(sample_features)
        assert iface["type"] == "FeatureCollection"
        features = from_geo_interface(iface)
        assert len(features) == 2

    def test_merge_files(self, sample_features, tmp_path):
        from geoprompt.formats import merge_geojson_files
        for i in range(2):
            p = str(tmp_path / f"test_{i}.geojson")
            with open(p, "w") as f:
                json.dump({"type": "FeatureCollection", "features": [sample_features[i]]}, f)
        out = str(tmp_path / "merged.geojson")
        merge_geojson_files([str(tmp_path / f"test_{i}.geojson") for i in range(2)], out)
        with open(out) as f:
            data = json.load(f)
        assert len(data["features"]) == 2

    def test_empty_layer(self, tmp_path):
        from geoprompt.formats import write_empty_layer
        p = str(tmp_path / "empty.geojson")
        write_empty_layer(p, schema={"name": "string"})
        with open(p) as f:
            data = json.load(f)
        assert data["features"] == []

    def test_copy_schema(self, sample_features):
        from geoprompt.formats import copy_schema
        schema = copy_schema(sample_features)
        assert "name" in schema

    def test_detect_encoding(self, tmp_path):
        from geoprompt.formats import detect_encoding
        p = str(tmp_path / "test.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("hello world")
        enc = detect_encoding(p)
        assert "utf" in enc.lower()

    def test_compressed_write(self, sample_features, tmp_path):
        from geoprompt.formats import write_compressed
        p = str(tmp_path / "test.geojson.gz")
        write_compressed(json.dumps({"type": "FeatureCollection", "features": sample_features}), p)
        assert Path(p).exists()

    def test_read_glob(self, sample_features, tmp_path):
        from geoprompt.formats import read_glob
        for i in range(2):
            p = str(tmp_path / f"test_{i}.geojson")
            with open(p, "w") as f:
                json.dump({"type": "FeatureCollection", "features": [sample_features[i]]}, f)
        loaded = read_glob(str(tmp_path / "*.geojson"))
        assert len(loaded) == 2


# ── query ──────────────────────────────────────────────────────────────────

class TestQuery:
    @pytest.fixture()
    def sample_rows(self):
        return [
            {"id": 1, "name": "Alice", "score": 90, "geometry": {"type": "Point", "coordinates": [0, 0]}},
            {"id": 2, "name": "Bob", "score": 75, "geometry": {"type": "Point", "coordinates": [1, 1]}},
            {"id": 3, "name": "Carol", "score": 85, "geometry": {"type": "Point", "coordinates": [5, 5]}},
        ]

    def test_parse_expression_eq(self, sample_rows):
        from geoprompt.query import parse_expression
        pred = parse_expression("name = 'Alice'")
        assert pred(sample_rows[0]) is True
        assert pred(sample_rows[1]) is False

    def test_parse_expression_gt(self, sample_rows):
        from geoprompt.query import parse_expression
        pred = parse_expression("score > 80")
        assert pred(sample_rows[0]) is True
        assert pred(sample_rows[1]) is False

    def test_parse_in(self, sample_rows):
        from geoprompt.query import parse_expression
        pred = parse_expression("name IN ('Alice', 'Carol')")
        assert pred(sample_rows[0]) is True
        assert pred(sample_rows[1]) is False
        assert pred(sample_rows[2]) is True

    def test_parse_like(self, sample_rows):
        from geoprompt.query import parse_expression
        pred = parse_expression("name LIKE 'A%'")
        assert pred(sample_rows[0]) is True
        assert pred(sample_rows[1]) is False

    def test_parse_is_null(self):
        from geoprompt.query import parse_expression
        pred = parse_expression("x IS NULL")
        assert pred({"x": None}) is True
        assert pred({"x": 1}) is False

    def test_parse_between(self, sample_rows):
        from geoprompt.query import parse_expression
        pred = parse_expression("score BETWEEN 80 AND 90")
        assert pred(sample_rows[0]) is True
        assert pred(sample_rows[1]) is False
        assert pred(sample_rows[2]) is True

    def test_where_clause_and(self, sample_rows):
        from geoprompt.query import where_clause
        pred = where_clause(["score > 80", "name = 'Alice'"])
        assert pred(sample_rows[0]) is True
        assert pred(sample_rows[2]) is False

    def test_where_clause_or(self, sample_rows):
        from geoprompt.query import where_clause
        pred = where_clause(["name = 'Alice'", "name = 'Bob'"], combine="OR")
        assert pred(sample_rows[0]) is True
        assert pred(sample_rows[1]) is True
        assert pred(sample_rows[2]) is False

    def test_search_cursor(self, sample_rows):
        from geoprompt.query import SearchCursor
        with SearchCursor(sample_rows, fields=["name", "score"], where="score > 80") as cur:
            results = list(cur)
        assert len(results) == 2
        assert all("name" in r for r in results)

    def test_search_cursor_order_by(self, sample_rows):
        from geoprompt.query import SearchCursor
        with SearchCursor(sample_rows, fields=["name"], order_by="-score") as cur:
            results = list(cur)
        assert results[0]["name"] == "Alice"

    def test_insert_cursor(self):
        from geoprompt.query import InsertCursor
        rows: list[dict] = []
        with InsertCursor(rows, fields=["a", "b"]) as cur:
            cur.insertRow((1, 2))
            cur.insertRow((3, 4))
        assert len(rows) == 2
        assert rows[0] == {"a": 1, "b": 2}

    def test_update_cursor(self, sample_rows):
        from geoprompt.query import UpdateCursor
        with UpdateCursor(sample_rows, where="name = 'Bob'") as cur:
            for row in cur:
                cur.updateRow({"score": 100})
        assert sample_rows[1]["score"] == 100

    def test_delete_rows(self, sample_rows):
        from geoprompt.query import delete_rows
        count = delete_rows(sample_rows, "score < 80")
        assert count == 1
        assert len(sample_rows) == 2

    def test_select_by_attributes(self, sample_rows):
        from geoprompt.query import select_by_attributes
        sel = select_by_attributes(sample_rows, "score > 80")
        assert sel == {0, 2}

    def test_select_by_location(self, sample_rows):
        from geoprompt.query import select_by_location
        sel = select_by_location(sample_rows, bbox=(-0.5, -0.5, 1.5, 1.5))
        assert sel == {0, 1}

    def test_switch_selection(self, sample_rows):
        from geoprompt.query import switch_selection
        sel = switch_selection(sample_rows, {0})
        assert sel == {1, 2}

    def test_calculate_field(self, sample_rows):
        from geoprompt.query import calculate_field
        calculate_field(sample_rows, "doubled", lambda r: r.get("score", 0) * 2)
        assert sample_rows[0]["doubled"] == 180

    def test_alter_field(self, sample_rows):
        from geoprompt.query import alter_field
        alter_field(sample_rows, "name", new_name="full_name")
        assert "full_name" in sample_rows[0]
        assert "name" not in sample_rows[0]

    def test_frequency_distribution(self):
        from geoprompt.query import frequency_distribution
        rows = [{"color": "red"}, {"color": "blue"}, {"color": "red"}]
        freq = frequency_distribution(rows, ["color"])
        assert freq[0]["color"] == "red"
        assert freq[0]["count"] == 2

    def test_extent(self, sample_rows):
        from geoprompt.query import Extent
        ext = Extent.from_features(sample_rows)
        assert ext.min_x == 0 and ext.max_x == 5
        assert ext.contains(3, 3)
        assert not ext.contains(10, 10)

    def test_spatial_diff(self):
        from geoprompt.query import spatial_diff
        old = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]
        new = [{"id": 2, "v": "c"}, {"id": 3, "v": "d"}]
        diff = spatial_diff(old, new)
        assert len(diff["added"]) == 1
        assert len(diff["removed"]) == 1
        assert len(diff["modified"]) == 1

    def test_bulk_update(self, sample_rows):
        from geoprompt.query import bulk_update
        count = bulk_update(sample_rows, {"flag": True}, where="score > 80")
        assert count == 2
        assert sample_rows[0]["flag"] is True

    def test_data_dictionary(self, sample_rows):
        from geoprompt.query import data_dictionary
        dd = data_dictionary(sample_rows)
        names = {d["name"] for d in dd}
        assert "name" in names and "score" in names

    def test_build_where_clause(self):
        from geoprompt.query import build_where_clause
        clause = build_where_clause(field="score", op=">", value=80)
        assert "score > 80" in clause


# ── utils ──────────────────────────────────────────────────────────────────

class TestUtils:
    def test_parallel_map(self):
        from geoprompt.utils import parallel_map
        results = parallel_map(lambda x: x * 2, [1, 2, 3])
        assert results == [2, 4, 6]

    def test_chunked(self):
        from geoprompt.utils import chunked
        result = chunked([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_config(self):
        from geoprompt.utils import get_config, reset_config, set_config
        reset_config()
        assert get_config("default_crs") == "EPSG:4326"
        set_config("default_crs", "EPSG:32610")
        assert get_config("default_crs") == "EPSG:32610"
        reset_config()
        assert get_config("default_crs") == "EPSG:4326"

    def test_disk_cache(self, tmp_path):
        from geoprompt.utils import DiskCache
        cache = DiskCache(cache_dir=tmp_path / "cache")
        assert cache.get("key1") is None
        cache.put("key1", {"result": 42})
        assert cache.get("key1") == {"result": 42}
        assert cache.clear() == 1

    def test_normalize_path(self, tmp_path):
        from geoprompt.utils import normalize_path
        p = normalize_path(str(tmp_path))
        assert p.is_absolute()

    def test_ensure_directory(self, tmp_path):
        from geoprompt.utils import ensure_directory
        d = ensure_directory(tmp_path / "sub" / "dir")
        assert d.is_dir()

    def test_file_hash(self, tmp_path):
        from geoprompt.utils import file_hash
        p = tmp_path / "test.txt"
        p.write_text("hello")
        h = file_hash(str(p))
        assert isinstance(h, str) and len(h) == 64

    def test_feature_hash(self):
        from geoprompt.utils import feature_hash
        h = feature_hash({"type": "Point", "coordinates": [0, 0]})
        assert isinstance(h, str)

    def test_atomic_write(self, tmp_path):
        from geoprompt.utils import atomic_write
        target = tmp_path / "output.txt"
        with atomic_write(target) as f:
            f.write("content")
        assert target.read_text() == "content"

    def test_temp_directory(self):
        from geoprompt.utils import temp_directory
        with temp_directory() as d:
            assert d.is_dir()
            (d / "file.txt").write_text("hi")
        assert not d.exists()

    def test_timeout(self):
        from geoprompt.utils import Timeout
        with Timeout(10.0) as t:
            assert not t.expired
        assert t.elapsed < 10.0

    def test_retry_success(self):
        from geoprompt.utils import retry
        calls = [0]
        def flaky():
            calls[0] += 1
            if calls[0] < 2:
                raise ValueError("fail")
            return "ok"
        result = retry(flaky, max_attempts=3, delay=0.01)
        assert result == "ok"

    def test_retry_failure(self):
        from geoprompt.utils import retry
        with pytest.raises(ZeroDivisionError):
            retry(lambda: 1 / 0, max_attempts=2, delay=0.01, exceptions=(ZeroDivisionError,))

    def test_get_logger(self):
        from geoprompt.utils import get_logger
        logger = get_logger("test_geoprompt")
        assert logger.name == "test_geoprompt"

    def test_memoize(self):
        from geoprompt.utils import memoize
        calls = [0]
        @memoize
        def expensive(x):
            calls[0] += 1
            return x * 2
        assert expensive(5) == 10
        assert expensive(5) == 10
        assert calls[0] == 1

    def test_load_save_config(self, tmp_path):
        from geoprompt.utils import load_config, reset_config, save_config
        reset_config()
        p = str(tmp_path / "config.json")
        save_config(p)
        reset_config()
        loaded = load_config(p)
        assert loaded["default_crs"] == "EPSG:4326"
        reset_config()
