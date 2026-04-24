from __future__ import annotations

import importlib.util
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

    assert complex_payload["track"] == "complex"
    assert complex_payload["monte_carlo_runs"] == 5
    assert "portfolio" in complex_payload
