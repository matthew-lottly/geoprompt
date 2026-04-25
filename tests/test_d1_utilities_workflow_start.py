from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("utilities_api_workflow", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_d1_simple_and_complex_tracks_run_without_live_api() -> None:
    module_path = Path("examples/network/utilities_api_workflow.py")
    workflow = _load_module(module_path)

    simple = workflow.run_simple_track(allow_live_api=False)
    complex_payload = workflow.run_complex_track(allow_live_api=False, monte_carlo_runs=5)

    assert simple["track"] == "simple"
    assert simple["network_edge_count"] > 0
    assert isinstance(simple["nearest_facility_pairs"], list)
    assert simple["nearest_facility_match"] is True
    assert simple["service_area_match"] is True
    assert isinstance(simple["stress_index"], list)
    assert "api_sources" in simple

    assert complex_payload["track"] == "complex"
    assert complex_payload["monte_carlo_runs"] == 5
    assert "portfolio" in complex_payload
    assert "reliability_trend" in complex_payload
    assert "critical_nodes" in complex_payload
    assert "scenario_ranking" in complex_payload
    assert "equity_impact" in complex_payload


def test_d1_generates_full_artifact_bundle(tmp_path: Path) -> None:
    module_path = Path("examples/network/utilities_api_workflow.py")
    workflow = _load_module(module_path)

    simple = workflow.run_simple_track(allow_live_api=False)
    complex_payload = workflow.run_complex_track(allow_live_api=False, monte_carlo_runs=5)
    outputs = workflow.generate_d1_artifacts(simple, complex_payload, output_dir=tmp_path)

    expected_keys = {
        "simple_json",
        "complex_json",
        "comparison_json",
        "summary_csv",
        "critical_nodes_csv",
        "scenario_ranking_csv",
        "equity_impact_csv",
        "restoration_svg",
        "unmet_demand_svg",
        "reliability_svg",
        "service_area_geojson",
        "summary_html",
        "portfolio_html",
        "executive_html",
    }
    assert expected_keys.issubset(outputs)
    for key in expected_keys:
        assert Path(outputs[key]).exists(), key

    comparison = json.loads(Path(outputs["comparison_json"]).read_text(encoding="utf-8"))
    assert comparison["simple_track"]["nearest_facility_match"] is True
    assert comparison["simple_track"]["service_area_match"] is True

    geojson = json.loads(Path(outputs["service_area_geojson"]).read_text(encoding="utf-8"))
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) > 0

    executive_html = Path(outputs["executive_html"]).read_text(encoding="utf-8")
    assert "D1 Utilities Workflow Executive Report" in executive_html
    assert "Scenario Ranking" in executive_html
