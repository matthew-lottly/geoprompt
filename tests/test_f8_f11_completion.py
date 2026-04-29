from __future__ import annotations

from pathlib import Path

from geoprompt.cli import build_parser, shell_completion_script
from geoprompt.ecosystem import compatibility_matrix, extension_starter_template
from geoprompt.quality import release_readiness_report
from geoprompt.viz import MAP_STYLE_PACKS, audit_html_accessibility, build_executive_briefing_pack


def test_f8_briefing_pack_and_accessibility_audit_support_client_ready_output(tmp_path: Path) -> None:
    html = build_executive_briefing_pack(
        [
            {"title": "Resilience score", "content": "84 / 100", "type": "metric"},
            {"title": "Priority action", "content": "Stage crews near feeder B.", "type": "note"},
            {"title": "Service area map", "content": [], "type": "map"},
            {"title": "Outage trend", "content": [], "type": "chart"},
            {"title": "Asset summary", "content": [], "type": "table"},
        ],
        title="Utility Executive Briefing",
        organization="Geo Utility Lab",
        theme="utilities",
        output_path=tmp_path / "briefing.html",
    )

    assert "Geo Utility Lab" in html
    assert "Utility Executive Briefing" in html
    assert "utilities" in MAP_STYLE_PACKS

    audit = audit_html_accessibility(html)
    assert audit["passed"] is True
    assert audit["issue_count"] == 0
    assert (tmp_path / "briefing.html").exists()


def test_f9_docs_publish_migration_and_persona_guidance() -> None:
    gallery = Path("docs/notebook-gallery.md").read_text(encoding="utf-8")
    geopandas = Path("docs/migration-from-geopandas.md").read_text(encoding="utf-8")
    arcpy = Path("docs/migration-from-arcpy.md").read_text(encoding="utf-8")
    troubleshooting = Path("docs/troubleshooting.md").read_text(encoding="utf-8")

    assert "Analyst track" in gallery
    assert "Planner track" in gallery
    assert "Operations track" in gallery
    assert "GeoPandas to GeoPrompt cookbook" in geopandas
    assert "ArcPy to GeoPrompt side-by-side recipes" in arcpy
    assert "installation decision tree" in troubleshooting.lower()


def test_f10_release_readiness_and_beta_status_are_published() -> None:
    report = release_readiness_report(["src/geoprompt/cli.py", "src/geoprompt/ecosystem.py"])
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert report["release_stage"] in {"beta", "release-candidate"}
    assert "packaging_smoke_matrix" in report
    assert "api_stability" in report
    assert "Development Status :: 4 - Beta" in pyproject


def test_f11_extension_templates_compatibility_and_cli_discovery_are_available() -> None:
    plugin_template = extension_starter_template("plugin", name="demo_plugin")
    connector_template = extension_starter_template("connector", name="demo_connector")
    matrix = compatibility_matrix()
    parser = build_parser()
    parsed = parser.parse_args(["plugins"])
    completion = shell_completion_script()

    assert "def demo_plugin" in plugin_template
    assert "def demo_connector" in connector_template
    assert "profiles" in matrix and "platforms" in matrix
    assert parsed.command == "plugins"
    assert "geoprompt" in completion
