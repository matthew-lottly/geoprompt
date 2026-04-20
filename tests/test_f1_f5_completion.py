from __future__ import annotations

from pathlib import Path

import geoprompt as gp


def test_f1_docs_define_scope_support_and_release_bar() -> None:
    when_to_use = Path("docs/when-to-use-geoprompt.md").read_text(encoding="utf-8")
    api_stability = Path("docs/api-stability.md").read_text(encoding="utf-8")

    assert "Top three flagship lanes" in when_to_use
    assert "Out of scope" in when_to_use
    assert "Support matrix" in api_stability
    assert "Alpha to Beta" in api_stability


def test_f2_frame_gets_string_datetime_query_and_ranking_helpers() -> None:
    frame = gp.geopromptframe.from_records(
        [
            {"site_id": "a", "name": " north pump ", "opened": "2024-01-15", "priority": 3, "status": "open", "geometry": {"type": "Point", "coordinates": [0, 0]}},
            {"site_id": "b", "name": "south valve", "opened": "2023-05-10", "priority": 1, "status": "closed", "geometry": {"type": "Point", "coordinates": [1, 1]}},
            {"site_id": "c", "name": "central tank", "opened": "2024-06-01", "priority": 2, "status": "open", "geometry": {"type": "Point", "coordinates": [2, 2]}},
        ],
        crs="EPSG:4326",
    )

    cleaned = frame.str.strip("name").str.upper("name", new_column="name_upper")
    enriched = cleaned.dt.year("opened", new_column="opened_year")
    filtered = enriched.query("priority >= 2 and status == 'open'")

    assert len(filtered) == 2
    assert filtered.head(1)[0]["name_upper"] == "NORTH PUMP"
    assert {row["opened_year"] for row in filtered.to_records()} == {2024}

    top = enriched.nlargest(2, "priority")
    bottom = enriched.nsmallest(1, "priority")
    assert [row["site_id"] for row in top.to_records()] == ["a", "c"]
    assert bottom.head(1)[0]["site_id"] == "b"


def test_f3_geometry_validation_and_repair_are_more_explanatory() -> None:
    polygon = {
        "type": "Polygon",
        "coordinates": [(0, 0, 5), (0.01, 0.001, 5), (2, 0, 5), (2, 2, 5), (0, 2, 5), (0, 0, 5)],
    }

    report = gp.validate_geometry(polygon)
    repaired = gp.repair_geometry(polygon)
    repaired_report = gp.validate_geometry(repaired)

    assert any(issue in report["issues"] for issue in ["clockwise_exterior", "mixed_dimension_coordinates", "narrow_spike"])
    assert repaired_report["is_valid"] is True


def test_f4_raster_window_alignment_and_nodata_handling_are_hardened() -> None:
    raster_a = {
        "data": [[1, 2, -9999], [4, 5, 6], [7, 8, 9]],
        "transform": (0.0, 3.0, 1.0, 1.0),
        "nodata": -9999,
        "width": 3,
        "height": 3,
    }
    raster_b = {
        "data": [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
        "transform": (0.0, 3.0, 1.0, 1.0),
        "nodata": -9999,
        "width": 3,
        "height": 3,
    }

    window = gp.raster_window(raster_a, row_start=1, row_end=3, col_start=0, col_end=2)
    alignment = gp.raster_alignment_report([raster_a, raster_b])
    algebra = gp.raster_lazy_algebra("a + b", {"a": raster_a, "b": raster_b})

    assert window["width"] == 2 and window["height"] == 2
    assert alignment["aligned"] is True
    assert algebra["data"][0][2] is None


def test_f5_enterprise_health_and_persistence_guidance_are_available(tmp_path: Path) -> None:
    records = [
        {"id": 1, "status": "open", "owner": "ops", "geometry": {"type": "Point", "coordinates": [0, 0]}},
        {"id": 2, "status": "closed", "owner": "ops", "geometry": {"type": "Point", "coordinates": [1, 1]}},
    ]

    catalog = gp.create_dataset_catalog(
        {"assets": records},
        workspace=tmp_path / "ops.gpkg",
    )
    index_plan = gp.index_planning_suggestions(records, candidate_fields=["id", "status", "owner"])
    matrix = gp.enterprise_persistence_matrix()

    assert catalog["dataset_count"] == 1
    assert any(item["field"] == "id" for item in index_plan["recommended_indexes"])
    assert "geopackage" in matrix
    assert matrix["geopackage"]["persistence"] == "real"
