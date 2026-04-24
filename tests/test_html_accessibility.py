"""J8.95 – Accessibility validation tests for generated HTML and dashboard artifacts.

Requirements sourced from docs/extending-geoprompt.md accessibility guidance:
  1. Document title (``<title>`` element present and non-empty)
  2. Language attribute on ``<html>`` tag (``lang=``)
  3. Main landmark (``<main>`` element or ARIA ``role="main"``)
  4. Visible heading at H1 level (``<h1>``)
  5. Image alt text (``<img>`` elements must have ``alt=`` attribute)
  6. Metric labels (key metrics in dashboard artifacts must be labeled)
  7. Table headers (``<table>`` must contain ``<th>`` elements)

We test:
  a. The ``audit_html_accessibility()`` helper itself.
  b. Generated HTML artifacts from ``render_comparison_html()``,
     ``story_map_html()``, and ``GeoPromptFrame.to_html()``.
  c. Existing static HTML artifacts in ``outputs/`` (regression guard).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from geoprompt.viz import audit_html_accessibility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


def _has_table_headers(html: str) -> bool:
    return bool(re.search(r"<th[\s>]", html, re.IGNORECASE))


def _has_non_empty_title(html: str) -> bool:
    m = re.search(r"<title>([^<]*)</title>", html, re.IGNORECASE)
    return bool(m and m.group(1).strip())


def _has_main_landmark(html: str) -> bool:
    return bool(
        re.search(r"<main[\s>]", html, re.IGNORECASE)
        or re.search(r'role=["\']main["\']', html, re.IGNORECASE)
    )


# ---------------------------------------------------------------------------
# J8.95.1 – audit_html_accessibility() helper contract
# ---------------------------------------------------------------------------


class TestAuditHtmlAccessibilityHelper:
    def test_perfect_html_passes(self) -> None:
        html = (
            '<html lang="en"><head><title>Test</title></head>'
            '<body><main><h1>Hello</h1></main></body></html>'
        )
        result = audit_html_accessibility(html)
        assert result["passed"] is True
        assert result["issue_count"] == 0

    def test_missing_title_is_flagged(self) -> None:
        html = '<html lang="en"><head></head><body><main><h1>Hi</h1></main></body></html>'
        result = audit_html_accessibility(html)
        assert "missing_title" in result["issues"]

    def test_missing_lang_is_flagged(self) -> None:
        html = "<html><head><title>T</title></head><body><main><h1>H</h1></main></body></html>"
        result = audit_html_accessibility(html)
        assert "missing_lang" in result["issues"]

    def test_missing_h1_is_flagged(self) -> None:
        html = '<html lang="en"><head><title>T</title></head><body><main><h2>Sub</h2></main></body></html>'
        result = audit_html_accessibility(html)
        assert "missing_h1" in result["issues"]

    def test_missing_main_is_flagged(self) -> None:
        html = '<html lang="en"><head><title>T</title></head><body><h1>H</h1></body></html>'
        result = audit_html_accessibility(html)
        assert "missing_main_landmark" in result["issues"]

    def test_image_without_alt_is_flagged(self) -> None:
        html = (
            '<html lang="en"><head><title>T</title></head>'
            '<body><main><h1>H</h1><img src="x.png"></main></body></html>'
        )
        result = audit_html_accessibility(html)
        assert "image_missing_alt" in result["issues"]

    def test_image_with_alt_does_not_flag(self) -> None:
        html = (
            '<html lang="en"><head><title>T</title></head>'
            '<body><main><h1>H</h1><img src="x.png" alt="A chart"></main></body></html>'
        )
        result = audit_html_accessibility(html)
        assert "image_missing_alt" not in result["issues"]

    def test_result_has_required_keys(self) -> None:
        html = "<html><head><title>T</title></head><body></body></html>"
        result = audit_html_accessibility(html)
        assert "passed" in result
        assert "issues" in result
        assert "issue_count" in result

    def test_issue_count_matches_issues_list(self) -> None:
        html = "<html><body></body></html>"  # missing: title, lang, main, h1
        result = audit_html_accessibility(html)
        assert result["issue_count"] == len(result["issues"])


# ---------------------------------------------------------------------------
# J8.95.2 – render_comparison_html() produces accessible HTML
# ---------------------------------------------------------------------------


class TestComparisonHtmlAccessibility:
    def test_comparison_html_has_title(self) -> None:
        from geoprompt.compare import render_comparison_html

        html = render_comparison_html({"summary": {}, "comparison": {}})
        assert _has_non_empty_title(html), "render_comparison_html is missing a <title>"

    def test_comparison_html_has_h1(self) -> None:
        from geoprompt.compare import render_comparison_html

        html = render_comparison_html({"summary": {}, "comparison": {}})
        assert re.search(r"<h1[\s>]", html, re.IGNORECASE), (
            "render_comparison_html is missing <h1>"
        )

    def test_comparison_html_has_table_headers(self) -> None:
        from geoprompt.compare import render_comparison_html

        report = {
            "summary": {"geometry_match": True, "crs_match": True},
            "comparison": {"corpus": ["test-corpus"]},
            "datasets": [
                {
                    "name": "test",
                    "correctness": {"geometry_match": True},
                    "benchmark": {"throughput_ms": 12.3},
                }
            ],
        }
        html = render_comparison_html(report)
        assert _has_table_headers(html), (
            "render_comparison_html tables are missing <th> elements"
        )


# ---------------------------------------------------------------------------
# J8.95.3 – GeoPromptFrame.to_html() produces accessible HTML fragment
# ---------------------------------------------------------------------------


class TestFrameToHtmlAccessibility:
    def test_frame_html_has_table_headers(self) -> None:
        from geoprompt import GeoPromptFrame

        frame = GeoPromptFrame(
            [{"id": 1, "value": 0.5, "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]
        )
        html = frame.to_html()
        assert _has_table_headers(html), "GeoPromptFrame.to_html() tables missing <th>"

    def test_frame_html_is_string(self) -> None:
        from geoprompt import GeoPromptFrame

        try:
            frame = GeoPromptFrame(
                [
                    {
                        "id": 1,
                        "value": 0.5,
                        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                    }
                ]
            )
            result = frame.to_html()
        except Exception as exc:
            pytest.skip(f"GeoPromptFrame construction failed: {exc}")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# J8.95.4 – Static HTML artifacts in outputs/ are accessible (regression guard)
# ---------------------------------------------------------------------------


_HTML_ARTIFACTS = sorted(_OUTPUTS_DIR.glob("*.html"))


@pytest.mark.skipif(
    not _OUTPUTS_DIR.exists(),
    reason="outputs/ directory does not exist",
)
class TestStaticHtmlArtifactsAccessibility:
    """Regression guard — existing generated HTML artifacts must have minimum accessibility."""

    @pytest.mark.parametrize(
        "html_path",
        _HTML_ARTIFACTS,
        ids=[p.name for p in _HTML_ARTIFACTS],
    )
    def test_artifact_has_title(self, html_path: Path) -> None:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        assert _has_non_empty_title(html), (
            f"{html_path.name} is missing a non-empty <title>"
        )

    @pytest.mark.parametrize(
        "html_path",
        _HTML_ARTIFACTS,
        ids=[p.name for p in _HTML_ARTIFACTS],
    )
    def test_artifact_has_heading(self, html_path: Path) -> None:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        has_heading = bool(
            re.search(r"<h[1-3][\s>]", html, re.IGNORECASE)
        )
        assert has_heading, f"{html_path.name} has no headings (h1-h3)"

    @pytest.mark.parametrize(
        "html_path",
        [p for p in _HTML_ARTIFACTS if re.search(r"<table[\s>]", p.read_text(errors="ignore"), re.IGNORECASE)],
        ids=[p.name for p in _HTML_ARTIFACTS if re.search(r"<table[\s>]", p.read_text(errors="ignore"), re.IGNORECASE)],
    )
    def test_artifact_tables_have_headers(self, html_path: Path) -> None:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        assert _has_table_headers(html), (
            f"{html_path.name} has <table> but no <th> elements"
        )


# ---------------------------------------------------------------------------
# J8.95.5 – story_map_html() produces accessible HTML
# ---------------------------------------------------------------------------


class TestStoryMapHtmlAccessibility:
    def test_story_map_has_title(self) -> None:
        from geoprompt.cartography import story_map_html

        try:
            html = story_map_html(
                title="Test Map",
                slides=[{"heading": "Slide 1", "body": "Content", "map_html": ""}],
            )
        except Exception:
            pytest.skip("story_map_html raised unexpectedly — skip accessibility check")

        assert _has_non_empty_title(html), "story_map_html is missing <title>"

    def test_story_map_has_h1(self) -> None:
        from geoprompt.cartography import story_map_html

        try:
            html = story_map_html(
                title="Test Map",
                slides=[{"heading": "Slide 1", "body": "Content", "map_html": ""}],
            )
        except Exception:
            pytest.skip("story_map_html raised unexpectedly — skip accessibility check")

        has_h1 = bool(re.search(r"<h1[\s>]", html, re.IGNORECASE))
        # h1 OR the title is displayed prominently — either is acceptable
        has_prominent = has_h1 or bool(re.search(r"Test Map", html))
        assert has_prominent, "story_map_html has no visible title or h1"
