from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from geoprompt.tools import build_resilience_summary_report, export_resilience_summary_report
from geoprompt.viz import audit_html_accessibility, audit_visual_quality, build_executive_briefing_pack


_GOLDEN_BRIEFING_PATH = Path("data/executive_briefing_golden.html")


def _sample_resilience_report() -> dict[str, object]:
    return build_resilience_summary_report(
        redundancy_rows=[
            {"node": "hospital", "single_source_dependency": True, "is_critical": True, "resilience_tier": "low"},
            {"node": "pump", "single_source_dependency": False, "is_critical": False, "resilience_tier": "high"},
        ],
        outage_report={
            "impacted_node_count": 2,
            "impacted_customer_count": 150,
            "estimated_cost": 800.0,
            "severity_tier": "high",
        },
        restoration_report={
            "total_steps": 2,
            "stages": [
                {"step": 1, "repair_edge_id": "e1", "cumulative_restored_demand": 50.0},
                {"step": 2, "repair_edge_id": "e2", "cumulative_restored_demand": 100.0},
            ],
        },
    )


def _sample_briefing_sections() -> list[dict[str, object]]:
    return [
        {
            "title": "Study Area",
            "type": "map",
            "content": [
                {"name": "hospital", "geometry": {"type": "Point", "coordinates": (-111.90, 40.72)}},
                {"name": "main feeder", "geometry": {"type": "LineString", "coordinates": [(-111.95, 40.70), (-111.84, 40.78)]}},
                {"name": "service zone", "geometry": {"type": "Polygon", "coordinates": [(-111.96, 40.68), (-111.82, 40.68), (-111.82, 40.79), (-111.96, 40.79), (-111.96, 40.68)]}},
            ],
        },
        {
            "title": "Risk Trend",
            "type": "chart",
            "content": [
                {"label": "Baseline", "value": 12.0},
                {"label": "Upgrade", "value": 4.0},
                {"label": "Target", "value": 2.0},
            ],
        },
        {
            "title": "Priority Actions",
            "type": "table",
            "content": [
                {"asset": "hospital", "action": "backup feed", "priority": "high"},
                {"asset": "pump", "action": "switch replacement", "priority": "medium"},
            ],
        },
    ]


def _normalize_markup(text: str) -> str:
    compact = re.sub(r">\s+<", "><", text)
    compact = re.sub(r"\s*([{}:;>,])\s*", r"\1", compact)
    return re.sub(r"\s+", " ", compact).strip()


def test_resilience_summary_html_output_contract_has_chart_and_table(tmp_path: Path) -> None:
    report = _sample_resilience_report()

    html_path = tmp_path / "resilience-summary.html"
    export_resilience_summary_report(report, html_path)
    html = html_path.read_text(encoding="utf-8")

    assert "data-output-contract=\"resilience-summary-v1\"" in html
    assert html.count("data-artifact-type=\"chart\"") == 2
    assert "data-artifact-type=\"table\"" in html
    assert "<main>" in html
    assert "<svg" in html
    assert "<table>" in html
    assert "placeholder" not in html.lower()
    assert "todo" not in html.lower()
    assert audit_html_accessibility(html)["passed"] is True
    assert audit_visual_quality(html)["passed"] is True


def test_resilience_summary_structured_formats_have_readable_contracts(tmp_path: Path) -> None:
    report = _sample_resilience_report()

    json_path = tmp_path / "resilience-summary.json"
    csv_path = tmp_path / "resilience-summary.csv"
    markdown_path = tmp_path / "resilience-summary.md"

    export_resilience_summary_report(report, json_path)
    export_resilience_summary_report(report, csv_path)
    export_resilience_summary_report(report, markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert sorted(payload.keys()) == ["metadata", "outage_report", "redundancy_rows", "restoration_report", "summary"]
    assert payload["summary"]["impacted_customer_count"] == 150

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == ["is_critical", "node", "resilience_tier", "single_source_dependency"]
    assert rows[0]["node"] == "hospital"
    assert rows[1]["resilience_tier"] == "high"

    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Resilience Summary Report\n")
    assert "| Node | Single Source | Critical | Tier |" in markdown
    assert "placeholder" not in markdown.lower()
    assert "todo" not in markdown.lower()


def test_executive_briefing_pack_supports_map_chart_table_artifacts(tmp_path: Path) -> None:
    html = build_executive_briefing_pack(
        _sample_briefing_sections(),
        title="Operations Briefing",
        organization="GeoPrompt",
        theme="utilities",
        output_path=tmp_path / "briefing.html",
    )

    assert "data-output-contract='executive-briefing-v1'" in html
    assert "data-artifact-type='map'" in html
    assert "data-artifact-type='chart'" in html
    assert "data-artifact-type='table'" in html
    assert html.count("<svg") >= 2
    assert "<table>" in html
    assert "map preview" in html.lower()
    assert "placeholder" not in html.lower()
    assert "todo" not in html.lower()
    assert audit_html_accessibility(html)["passed"] is True
    assert audit_visual_quality(html)["passed"] is True
    assert (tmp_path / "briefing.html").exists()


def test_executive_briefing_pack_requires_map_chart_table_triad() -> None:
    sections = [
        {
            "title": "Risk Trend",
            "type": "chart",
            "content": [{"label": "Baseline", "value": 12.0}],
        },
        {
            "title": "Priority Actions",
            "type": "table",
            "content": [{"asset": "hospital", "action": "backup feed", "priority": "high"}],
        },
    ]

    with pytest.raises(ValueError, match="requires map/chart/table"):
        build_executive_briefing_pack(sections)


def test_executive_briefing_pack_matches_golden_sample(tmp_path: Path) -> None:
    html = build_executive_briefing_pack(
        _sample_briefing_sections(),
        title="Operations Briefing",
        organization="GeoPrompt",
        theme="utilities",
        output_path=tmp_path / "briefing.html",
    )

    golden = _GOLDEN_BRIEFING_PATH.read_text(encoding="utf-8")
    assert _normalize_markup(html) == _normalize_markup(golden)