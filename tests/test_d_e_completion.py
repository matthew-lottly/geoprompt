from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import geoprompt as gp


def test_d1_enterprise_editing_and_replica_workflows(tmp_path: Path):
    records = [{"id": 1, "name": "Valve A", "status": "open"}]

    catalog = gp.create_dataset_catalog({"assets": records}, workspace=tmp_path / "ops.gdb")
    assert catalog["dataset_count"] == 1
    assert set(catalog["datasets"]["assets"]["fields"]) >= {"id", "name", "status"}

    good_audit = gp.maintenance_audit_pipeline(
        records,
        domain_map={"status": gp.FieldDomain("status", values={"open": "Open", "closed": "Closed"})},
        required_fields=["id", "name", "status"],
        indexed_fields=["id"],
    )
    assert good_audit["valid"] is True
    assert good_audit["issue_count"] == 0

    bad_records = [{"id": 2, "status": "broken"}]
    bad_audit = gp.maintenance_audit_pipeline(
        bad_records,
        domain_map={"status": gp.FieldDomain("status", values={"open": "Open", "closed": "Closed"})},
        required_fields=["id", "name", "status"],
    )
    assert bad_audit["valid"] is False
    assert bad_audit["issue_count"] >= 1

    session = gp.start_edit_session(records, key_field="id", session_name="branch-a")
    session.update(1, {"status": "closed"})
    session.insert({"id": 2, "name": "Valve B", "status": "open"})
    preview = session.preview()
    assert preview["summary"]["pending_changes"] == 2
    assert len(preview["records"]) == 2

    rolled_back = session.rollback()
    assert len(rolled_back) == 1
    assert rolled_back[0]["status"] == "open"

    session = gp.start_edit_session(records, key_field="id", session_name="branch-a")
    session.update(1, {"status": "closed"})
    session.insert({"id": 2, "name": "Valve B", "status": "open"})
    committed = session.commit()
    assert len(committed["records"]) == 2
    assert committed["summary"]["pending_changes"] == 0
    assert len(committed["change_log"]) == 2

    posted = gp.versioning_reconcile_post(
        base_records=records,
        version_records=committed["records"],
        key_field="id",
        version_name="branch-a",
    )
    assert posted["conflict_count"] == 0
    assert posted["posted_count"] >= 1

    replica = gp.create_offline_replica_package(
        committed["records"],
        tmp_path / "field_bundle.json",
        attachments={2: ["photo.jpg"]},
    )
    assert Path(replica["path"]).exists()
    assert replica["record_count"] == 2
    assert replica["attachment_count"] == 1


def test_d2_and_d4_service_resilience_and_time_dependent_routing():
    auth = gp.AuthProfile(
        portal_url="https://example.test",
        username="analyst",
        token="secret-token",
        expiry=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    profile = gp.service_resilience_profile(
        "https://example.test/FeatureServer/0",
        auth_profile=auth,
        roles=["analyst", "editor"],
        retry_count=4,
        backoff_seconds=1.5,
        rate_limit_per_minute=30,
    )
    assert profile["headers"]["Authorization"] == "Bearer secret-token"
    assert profile["roles"] == ["analyst", "editor"]
    assert profile["circuit_breaker"]["failure_threshold"] == 4

    graph = gp.build_network_graph(
        [
            {"edge_id": "ab", "from_node": "A", "to_node": "B", "cost": 2, "time_costs": {8: 2, 18: 6}, "bidirectional": True},
            {"edge_id": "bc", "from_node": "B", "to_node": "C", "cost": 2, "time_costs": {8: 2, 18: 6}, "bidirectional": True},
            {"edge_id": "ac", "from_node": "A", "to_node": "C", "cost": 7, "time_costs": {8: 7, 18: 7}, "bidirectional": True},
        ]
    )
    morning = gp.time_dependent_shortest_path(graph, "A", "C", departure_hour=8)
    evening = gp.time_dependent_shortest_path(graph, "A", "C", departure_hour=18)

    assert morning["reachable"] is True
    assert morning["path_edges"] == ["ab", "bc"]
    assert evening["total_cost"] == 7
    assert evening["path_edges"] == ["ac"]


def test_d3_d5_d6_raster_cartography_and_3d_workflows(tmp_path: Path):
    raster = {
        "data": [[1, 2, 3], [2, 5, 2], [1, 2, 1]],
        "transform": (0.0, 3.0, 1.0, 1.0),
        "nodata": -9999.0,
        "width": 3,
        "height": 3,
    }

    processed = gp.raster_chunk_process(
        raster,
        lambda chunk: [[value * 10 for value in row] for row in chunk],
        chunk_rows=2,
        chunk_cols=2,
    )
    assert processed["data"][1][1] == 50

    report = gp.raster_report_card(raster)
    viewshed = gp.raster_viewshed(raster, (1.0, 2.0), observer_height=1.5)
    assert report["cell_count"] == 9
    assert viewshed["rows"] == 3

    features = [
        {"district": "North", "geometry": {"type": "Point", "coordinates": [0, 0]}},
        {"district": "South", "geometry": {"type": "Point", "coordinates": [1, 1]}},
    ]
    pages = gp.map_series(features, "district", title_template="District {value}")
    layout = gp.print_layout(
        title="Operations Atlas",
        legend_items=[{"label": "Asset", "color": "#1d4ed8"}],
        scale_bar=True,
    )
    dashboard = gp.interactive_dashboard([
        {"type": "metric", "title": "Status", "data": "Ready"},
        {"type": "table", "title": "Assets", "data": [{"name": "Valve A", "score": 95}]},
    ], title="Ops Dashboard")

    assert len(pages) == 2
    assert layout["scale_bar"] is True
    assert "Ops Dashboard" in dashboard

    block = gp.extrude_polygon_to_3d_block([(0, 0), (1, 0), (1, 1), (0, 1)], height=12)
    skyline = gp.skyline_analysis([{"name": "Tower", "x": 1, "y": 1, "height": 30}], observer=(0, 0, 2))
    los = gp.line_of_sight_analysis((0, 0, 10), (1, 1, 8), obstacles=[{"height": 3}])
    cut_fill = gp.surface_volume_cut_fill([[1, 1], [1, 1]], [[2, 0], [1, 3]])

    assert block["height"] == 12
    assert skyline[0]["prominence"] > 0
    assert los["visible"] is True
    assert cut_fill["fill_volume"] > 0
