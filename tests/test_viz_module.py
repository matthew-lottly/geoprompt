from __future__ import annotations

from geoprompt.viz import MAP_STYLE_PACKS, audit_html_accessibility, build_executive_briefing_pack


def _sections() -> list[dict[str, object]]:
    return [
        {
            "title": "Study Area",
            "type": "map",
            "content": [
                {"name": "hospital", "geometry": {"type": "Point", "coordinates": (-111.90, 40.72)}},
                {"name": "main feeder", "geometry": {"type": "LineString", "coordinates": [(-111.95, 40.70), (-111.84, 40.78)]}},
            ],
        },
        {
            "title": "Risk Trend",
            "type": "chart",
            "content": [
                {"label": "Baseline", "value": 12.0},
                {"label": "Upgrade", "value": 4.0},
            ],
        },
        {
            "title": "Priority Actions",
            "type": "table",
            "content": [{"asset": "hospital", "action": "backup feed", "priority": "high"}],
        },
    ]


def test_viz_module_briefing_pack_uses_theme_and_contract_markers() -> None:
    html = build_executive_briefing_pack(
        _sections(),
        title="Operations Briefing",
        organization="GeoPrompt",
        theme="utilities",
    )

    assert "data-output-contract='executive-briefing-v1'" in html
    assert "data-theme='utilities'" in html
    assert f"background:{MAP_STYLE_PACKS['utilities']['background']}" in html
    assert html.count("data-artifact-type='") == 3


def test_viz_module_accessibility_audit_flags_missing_main_landmark() -> None:
    result = audit_html_accessibility("<html lang='en'><head><title>x</title></head><body><h1>Heading</h1></body></html>")

    assert result["passed"] is False
    assert "missing_main_landmark" in result["issues"]