from __future__ import annotations

from pathlib import Path

import pytest

from geoprompt.ai import (
    auto_benchmark,
    geometry_repair_advice,
    prompt_to_report,
    quality_control_scan,
    topology_validation_narrative,
)
from geoprompt.compare import export_benchmark_history
from geoprompt.frame import GeoPromptFrame
from geoprompt.geometry import geometry_erase, geometry_identity, geometry_wkb_read, geometry_wkb_write, update_overlay
from geoprompt.io import iter_data_with_preset, read_feather, write_feather
from geoprompt.topology import repair_suggestions, snap_points, validate_topology_rules


def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": (x, y)}


def _square(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"type": "Polygon", "coordinates": [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]}


def test_c2_spatial_overlay_and_topology_workflows() -> None:
    left = GeoPromptFrame(
        [
            {"id": "a", "group": "north", "value": 10, "geometry": _square(0, 0, 2, 2)},
            {"id": "b", "group": "north", "value": 20, "geometry": _square(3, 0, 5, 2)},
        ],
        geometry_column="geometry",
        crs="EPSG:4326",
    )
    right = GeoPromptFrame(
        [
            {"rid": "r1", "geometry": _square(1, 1, 4, 3)},
        ],
        geometry_column="geometry",
        crs="EPSG:4326",
    )

    joined = left.spatial_join(right, predicate="intersects", how="left")
    join_rows = joined.to_records()
    assert len(join_rows) == 2
    assert {row["rid"] for row in join_rows} == {"r1"}

    nearest = left.assign_nearest(
        GeoPromptFrame(
            [
                {"target": "t1", "geometry": _point(0.1, 0.1)},
                {"target": "t2", "geometry": _point(4.2, 1.0)},
            ],
            geometry_column="geometry",
            crs="EPSG:4326",
        ),
        origin_suffix="origin",
    )
    nearest_rows = nearest.to_records()
    assert len(nearest_rows) == 2
    assert {row["id"] for row in nearest_rows} == {"a", "b"}
    assert {row["nearest_rank_origin"] for row in nearest_rows} == {1}

    overlay = left.overlay_intersections(right)
    assert len(overlay) == 2

    dissolved = left.dissolve("group", aggregations={"value": "sum"}).to_records()
    assert dissolved[0]["value"] == 30.0

    identity_parts = geometry_identity(_square(0, 0, 2, 2), _square(1, 1, 3, 3))
    assert len(identity_parts) >= 1

    erased = geometry_erase(_square(0, 0, 2, 2), _square(1, 1, 3, 3))
    assert erased["type"] in {"Polygon", "MultiPolygon"}

    updated = update_overlay([{"id": "base", "geometry": _square(0, 0, 3, 3)}], [{"id": "patch", "geometry": _square(1, 1, 2, 2)}])
    assert len(updated) >= 1

    clustered = GeoPromptFrame(
        [
            {"id": 1, "geometry": _point(0.0, 0.0)},
            {"id": 2, "geometry": _point(0.03, 0.02)},
        ],
        geometry_column="geometry",
    )
    snapped = snap_points(clustered, tolerance=0.1)
    snap_rows = snapped.to_records()
    assert snap_rows[0]["geometry"]["coordinates"] == snap_rows[1]["geometry"]["coordinates"]

    overlap_report = validate_topology_rules(left, rule="must_not_overlap")
    assert overlap_report["valid"] is True

    broken = GeoPromptFrame(
        [
            {"id": "bad", "geometry": {"type": "Polygon", "coordinates": [(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]}},
        ],
        geometry_column="geometry",
    )
    repair = geometry_repair_advice(broken[0]["geometry"])
    assert repair["is_valid"] is False
    assert repair["suggested_fix"] is not None

    narrative = topology_validation_narrative([
        {"rule": "must_not_overlap", "feature_index": 0, "description": "overlap detected"},
    ])
    assert "must_not_overlap" in narrative

    suggestions = repair_suggestions(broken, rules=["must_not_self_intersect"])
    assert len(suggestions) >= 1


def test_c3_roundtrip_and_chunked_io_workflows(tmp_path: Path) -> None:
    frame = GeoPromptFrame(
        [
            {"id": "a", "score": 10, "geometry": _point(0, 0)},
            {"id": "b", "score": 20, "geometry": _point(1, 1)},
            {"id": "c", "score": 30, "geometry": _point(2, 2)},
        ],
        geometry_column="geometry",
        crs="EPSG:4326",
    )

    feather_path = tmp_path / "roundtrip.feather"
    written = write_feather(frame, feather_path)
    assert Path(written).exists()

    restored = read_feather(feather_path)
    assert restored.crs == "EPSG:4326"
    assert restored.geometry_column == "geometry"
    assert restored.to_records()[1]["score"] == 20

    blob = geometry_wkb_write(frame[0]["geometry"])
    parsed = geometry_wkb_read(blob)
    assert parsed["type"] == "Point"
    assert parsed["coordinates"] == (0.0, 0.0)

    geojson_path = tmp_path / "points.geojson"
    geojson_path.write_text(
        '{"type":"FeatureCollection","features":['
        '{"type":"Feature","properties":{"id":"a","score":10},"geometry":{"type":"Point","coordinates":[0,0]}},'
        '{"type":"Feature","properties":{"id":"b","score":20},"geometry":{"type":"Point","coordinates":[1,1]}},'
        '{"type":"Feature","properties":{"id":"c","score":30},"geometry":{"type":"Point","coordinates":[2,2]}}]}',
        encoding="utf-8",
    )
    chunks = list(iter_data_with_preset(geojson_path, workload="small", chunk_size=2))
    assert len(chunks) == 2
    assert sum(len(chunk) for chunk in chunks) == 3


def test_c4_c5_benchmark_reporting_and_analyst_outputs(tmp_path: Path) -> None:
    benchmark = auto_benchmark({"fast": lambda: sum(range(10)), "slow": lambda: sum(range(20))}, rounds=2)
    assert benchmark["fastest"] in {"fast", "slow"}
    assert set(benchmark["timings"].keys()) == {"fast", "slow"}

    qc = quality_control_scan([
        {"geometry": _point(0, 0), "score": 1.0},
        {"geometry": _point(1, 1), "score": 1000.0},
    ])
    assert "issues" in qc

    report_html = prompt_to_report({"sites": 12, "risk": "moderate"}, title="Decision Brief")
    assert "Decision Brief" in report_html
    assert "sites" in report_html

    (tmp_path / "report-a.json").write_text(
        '{"version":"0.2.0","summary":{"all_checks_passed":true},"datasets":[{"dataset":"demo","benchmarks":[{"operation":"demo.geoprompt.clip","median_seconds":0.5},{"operation":"demo.reference.clip","median_seconds":1.0}]}]}',
        encoding="utf-8",
    )
    exported = export_benchmark_history(tmp_path)
    assert Path(exported["html"]).exists()
    assert Path(exported["json"]).exists()
