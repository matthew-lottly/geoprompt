from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from geoprompt.network import (
    accessibility_equity_audit,
    build_network_graph,
    closest_facility_dispatch,
    multimodal_shortest_path,
    utility_monte_carlo_resilience,
    utility_stress_scenario_library,
)
from geoprompt.service import ServiceJobManager, service_benchmark_report


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="FastAPI not installed")
def test_f6_fastapi_service_health_smoke() -> None:
    from fastapi.testclient import TestClient

    from geoprompt.service import build_app

    client = TestClient(build_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_f6_service_job_manager_persists_jobs_and_results(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    manager = ServiceJobManager(path)
    manager.register_handler("echo", lambda payload: {"ok": True, "value": payload["value"] + 1})

    submitted = manager.submit("echo", {"value": 4}, user="ops", roles=["analyst"])
    snapshot = manager.get(submitted["job_id"])

    assert snapshot["status"] == "completed"
    assert snapshot["result"]["value"] == 5
    assert snapshot["roles"] == ["analyst"]

    reloaded = ServiceJobManager(path)
    persisted = reloaded.get(submitted["job_id"])
    assert persisted["result"]["value"] == 5

    benchmark = service_benchmark_report(manager, "echo", {"value": 1}, iterations=3)
    assert benchmark["success_count"] == 3
    assert benchmark["failure_count"] == 0


def test_f6_and_f7_docs_publish_runbook_and_flagship_track() -> None:
    deployment = Path("docs/deployment-guide.md").read_text(encoding="utf-8")
    network = Path("docs/network-scenario-recipes.md").read_text(encoding="utf-8")

    assert "Production operations runbook" in deployment
    assert "smoke checks" in deployment.lower()
    assert "top-tier network and utility analysis" in network.lower()
    assert "why this beats a generic dataframe gis workflow" in network.lower()


def test_f7_multimodal_routing_handles_transfer_penalties_and_turn_rules() -> None:
    graph = build_network_graph(
        [
            {"edge_id": "road_1", "from_node": "A", "to_node": "B", "cost": 1.0, "mode": "road", "bidirectional": False},
            {"edge_id": "rail_1", "from_node": "B", "to_node": "C", "cost": 1.0, "mode": "rail", "bidirectional": False},
            {"edge_id": "road_2", "from_node": "A", "to_node": "D", "cost": 2.0, "mode": "road", "bidirectional": False},
            {"edge_id": "road_3", "from_node": "D", "to_node": "C", "cost": 1.0, "mode": "road", "bidirectional": False},
        ],
        directed=True,
    )

    penalized = multimodal_shortest_path(graph, "A", "C", transfer_penalty=2.5)
    restricted = multimodal_shortest_path(graph, "A", "C", turn_restrictions=[("road_1", "rail_1")])

    assert penalized["path_edges"] == ["road_2", "road_3"]
    assert restricted["path_edges"] == ["road_2", "road_3"]
    assert penalized["mode_changes"] == 0


def test_f7_dispatch_equity_and_scenarios_support_decision_workflows() -> None:
    graph = build_network_graph(
        [
            {"edge_id": "e1", "from_node": "hub", "to_node": "n1", "cost": 1.0},
            {"edge_id": "e2", "from_node": "n1", "to_node": "n2", "cost": 1.0},
            {"edge_id": "e3", "from_node": "hub", "to_node": "n3", "cost": 4.0},
            {"edge_id": "e4", "from_node": "n2", "to_node": "n4", "cost": 1.0},
        ],
        directed=False,
    )

    dispatch = closest_facility_dispatch(
        graph,
        facility_nodes=["hub", "n2"],
        incident_nodes=["n1", "n3", "n4"],
        crew_count_by_facility={"hub": 1, "n2": 2},
    )
    equity = accessibility_equity_audit(
        graph,
        source_nodes=["hub"],
        demand_by_node={"n1": 5, "n2": 4, "n3": 9, "n4": 3},
        population_by_node={"n1": 50, "n2": 40, "n3": 100, "n4": 30},
        critical_nodes=["n3"],
    )
    scenarios = utility_stress_scenario_library(
        edge_groups={"fire_zone": ["e3"], "floodplain": ["e2", "e4"]},
        source_nodes=["hub"],
    )

    assert dispatch["assignment_count"] == 3
    assert dispatch["dispatches"][0]["assigned_facility"] in {"hub", "n2"}
    assert equity["gini"] >= 0.0
    assert any(row["node"] == "n3" for row in equity["underserved_nodes"])
    assert {"wildfire", "flood", "earthquake", "compound_event"}.issubset(scenarios)


def test_f7_monte_carlo_resilience_is_reproducible() -> None:
    graph = build_network_graph(
        [
            {"edge_id": "e1", "from_node": "A", "to_node": "B", "cost": 1.0},
            {"edge_id": "e2", "from_node": "B", "to_node": "C", "cost": 1.0},
            {"edge_id": "e3", "from_node": "A", "to_node": "C", "cost": 3.0},
        ],
        directed=False,
    )

    result = utility_monte_carlo_resilience(
        graph,
        source_nodes=["A"],
        candidate_failed_edges=["e1", "e2"],
        iterations=25,
        edge_failure_probability={"e1": 0.5, "e2": 0.25},
        demand_by_node={"B": 4, "C": 6},
        seed=42,
    )

    assert result["iterations"] == 25
    assert result["summary"]["average_deenergized_nodes"] >= 0.0
    assert result["summary"]["p95_deenergized_nodes"] >= result["summary"]["average_deenergized_nodes"]
