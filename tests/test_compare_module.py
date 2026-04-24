from __future__ import annotations

import json
from pathlib import Path

from geoprompt.compare import (
    benchmark_summary_table,
    export_benchmark_dashboard_bundle,
    export_comparison_bundle,
)


def _report() -> dict[str, object]:
    return {
        "comparison": {"corpus": ["demo"], "engines": ["geoprompt", "reference"]},
        "summary": {"all_checks_passed": True},
        "datasets": [
            {
                "dataset": "demo",
                "feature_count": 3,
                "correctness": {
                    "bounds_match": True,
                    "nearest_neighbor_match": True,
                    "bounds_query_match": True,
                    "geometry_metrics_within_tolerance": True,
                    "projected_bounds_match": True,
                    "clip": None,
                    "dissolve": None,
                    "spatial_join": None,
                },
                "benchmarks": [
                    {"operation": "demo.geoprompt.nearest_neighbors", "median_seconds": 0.01},
                    {"operation": "demo.reference.nearest_neighbors", "median_seconds": 0.025},
                ],
            }
        ],
    }


def test_compare_module_summary_table_reports_relative_winner() -> None:
    rows = benchmark_summary_table(_report()).to_records()

    assert rows == [
        {
            "dataset": "demo",
            "operation": "nearest_neighbors",
            "geoprompt_median_seconds": 0.01,
            "reference_median_seconds": 0.025,
            "winner": "geoprompt",
            "speedup_ratio": 2.5,
            "relative_status": "2.50x faster",
        }
    ]


def test_compare_module_bundle_exports_json_markdown_and_html(tmp_path: Path) -> None:
    written = export_comparison_bundle(_report(), tmp_path)

    payload = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
    markdown = Path(written["markdown"]).read_text(encoding="utf-8")
    html = Path(written["html"]).read_text(encoding="utf-8")

    assert payload["comparison"]["corpus"] == ["demo"]
    assert payload["datasets"][0]["benchmarks"][0]["median_seconds"] == 0.01
    assert markdown.startswith("# GeoPrompt Comparison Summary\n")
    assert "| demo | nearest_neighbors | 0.01 | 0.025 | geoprompt | 2.5 | 2.50x faster |" in markdown
    assert "<h2>Benchmark Overview</h2>" in html
    assert "2.50x faster" in html


def test_benchmark_dashboard_bundle_exports_alerts_and_trends(tmp_path: Path) -> None:
    bundle = export_benchmark_dashboard_bundle(tmp_path, min_speedup_ratio=1.1)

    payload = json.loads(Path(bundle["json"]).read_text(encoding="utf-8"))
    markdown = Path(bundle["markdown"]).read_text(encoding="utf-8")
    html = Path(bundle["html"]).read_text(encoding="utf-8")

    assert payload["metadata"]["min_speedup_ratio"] == 1.1
    assert "trend_rows" in payload
    assert markdown.startswith("# GeoPrompt Benchmark Dashboard\n")
    assert "## Trend Table" in markdown
    assert "GeoPrompt Benchmark Dashboard" in html