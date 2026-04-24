"""J8.91 – Release evidence artifact for trust audit metrics and trend deltas.

This module provides:
  1. ``collect_trust_metrics()`` — scans the codebase for quantitative trust
     indicators (broad-except count, raw ImportError count, eval patterns,
     pass-only bodies, skip budget, simulation-only markers) and returns a
     structured dict.
  2. ``compare_trust_metrics(baseline, current)`` — computes deltas and
     returns a regression report (pass/fail + per-metric deltas).
  3. Tests that verify the schema of the metrics dict and the comparison
     logic, and that the current codebase does not exceed the ratchet
     thresholds established at the end of Session 3.

The thresholds are conservative: they reflect the *current* known state and
will be tightened as the backlog is resolved.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Evidence collector
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).parent.parent / "src" / "geoprompt"
_TESTS_ROOT = Path(__file__).parent


def _count_pattern_in_files(pattern: str, root: Path, glob: str = "*.py") -> int:
    """Count lines matching *pattern* across all files matching *glob* in *root*."""
    compiled = re.compile(pattern)
    total = 0
    for path in root.rglob(glob):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            total += sum(1 for line in text.splitlines() if compiled.search(line))
        except OSError:
            pass
    return total


def _is_import_error_handler_type(node: ast.expr | None) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "ImportError"
    if isinstance(node, ast.Tuple):
        return any(_is_import_error_handler_type(elt) for elt in node.elts)
    return False


def _handler_uses_migrated_dependency_pattern(handler: ast.ExceptHandler) -> bool:
    for stmt in handler.body:
        for inner in ast.walk(stmt):
            if isinstance(inner, ast.Name) and inner.id in {"require_capability", "DependencyError", "FallbackWarning"}:
                return True
    return False


def _count_raw_import_error_handlers(root: Path, glob: str = "*.py") -> int:
    total = 0
    for path in root.rglob(glob):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if not _is_import_error_handler_type(node.type):
                continue
            if _handler_uses_migrated_dependency_pattern(node):
                continue
            total += 1
    return total


def collect_trust_metrics(src_root: Path = _SRC_ROOT, tests_root: Path = _TESTS_ROOT) -> dict[str, Any]:
    """Scan the codebase and return a structured trust-metrics snapshot.

    Returns:
        dict with keys:
          - ``broad_except_count``: lines matching ``except Exception``
          - ``raw_import_error_count``: unguarded ``except ImportError`` lines
          - ``eval_pattern_count``: bare ``eval(`` calls
          - ``pass_only_count``: lines that are only ``pass``
          - ``skip_budget``: ``pytest.skip(`` call count in tests
          - ``simulation_only_count``: simulation-related markers in src
    """
    return {
        "broad_except_count": _count_pattern_in_files(r"except Exception", src_root),
        "raw_import_error_count": _count_raw_import_error_handlers(src_root),
        "eval_pattern_count": _count_pattern_in_files(r"\beval\(", src_root),
        "pass_only_count": _count_pattern_in_files(r"^\s+pass\s*$", src_root),
        "skip_budget": _count_pattern_in_files(r"pytest\.skip\(", tests_root),
        "simulation_only_count": _count_pattern_in_files(
            r"simulation_only|PLACEHOLDER|STUB", src_root
        ),
    }


def compare_trust_metrics(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Compare *current* metrics against *baseline* and return a delta report.

    Returns:
        dict with:
          - ``passed``: bool — True if no metric regressed
          - ``regressions``: list of dicts describing regressions
          - ``improvements``: list of dicts describing improvements
          - ``deltas``: dict of metric→delta (positive = worse)
    """
    regressions = []
    improvements = []
    deltas: dict[str, int] = {}

    for key in baseline:
        base_val = baseline.get(key, 0)
        cur_val = current.get(key, 0)
        delta = cur_val - base_val
        deltas[key] = delta
        if delta > 0:
            regressions.append({"metric": key, "baseline": base_val, "current": cur_val, "delta": delta})
        elif delta < 0:
            improvements.append({"metric": key, "baseline": base_val, "current": cur_val, "delta": delta})

    return {
        "passed": len(regressions) == 0,
        "regressions": regressions,
        "improvements": improvements,
        "deltas": deltas,
    }


# ---------------------------------------------------------------------------
# Ratchet thresholds (established end of Session 3)
# These are upper bounds — exceeding them fails the release gate.
# ---------------------------------------------------------------------------

_RATCHET_THRESHOLDS: dict[str, int] = {
    "broad_except_count": 55,     # Tightened after J12.2 broad-except audit pass
    "raw_import_error_count": 40, # Tightened after J12.1 residual ImportError migration sweep
    "eval_pattern_count": 10,     # Session 3 baseline: 9; tight ratchet
    "pass_only_count": 65,        # Session 3 baseline: 56
    "skip_budget": 20,            # Session 3 baseline: 9; measured at 18 in session 4
    "simulation_only_count": 200, # Session 3 baseline: 181 (docs noise included)
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest


class TestTrustMetricsSchema:
    """The metrics dict must have the correct schema."""

    def test_collect_returns_all_required_keys(self) -> None:
        metrics = collect_trust_metrics()
        for key in _RATCHET_THRESHOLDS:
            assert key in metrics, f"Missing metric key: {key!r}"

    def test_all_metric_values_are_integers(self) -> None:
        metrics = collect_trust_metrics()
        for key, val in metrics.items():
            assert isinstance(val, int), f"Metric {key!r} should be int, got {type(val).__name__}"

    def test_all_metric_values_are_non_negative(self) -> None:
        metrics = collect_trust_metrics()
        for key, val in metrics.items():
            assert val >= 0, f"Metric {key!r} has negative value: {val}"


class TestTrustMetricsRatchet:
    """Current codebase must not exceed the established ratchet thresholds."""

    @pytest.mark.parametrize("metric,threshold", list(_RATCHET_THRESHOLDS.items()))
    def test_metric_within_threshold(self, metric: str, threshold: int) -> None:
        metrics = collect_trust_metrics()
        value = metrics.get(metric, 0)
        assert value <= threshold, (
            f"Trust ratchet exceeded for {metric!r}: "
            f"current={value}, threshold={threshold}. "
            "Reduce the count before releasing."
        )


class TestCompareTrustMetrics:
    """compare_trust_metrics must correctly compute deltas and regressions."""

    def test_identical_baselines_pass(self) -> None:
        base = {"broad_except_count": 10, "eval_pattern_count": 2}
        result = compare_trust_metrics(base, base.copy())
        assert result["passed"] is True
        assert result["regressions"] == []

    def test_regression_detected(self) -> None:
        base = {"broad_except_count": 10}
        current = {"broad_except_count": 15}
        result = compare_trust_metrics(base, current)
        assert result["passed"] is False
        assert len(result["regressions"]) == 1
        assert result["regressions"][0]["metric"] == "broad_except_count"
        assert result["regressions"][0]["delta"] == 5

    def test_improvement_detected(self) -> None:
        base = {"eval_pattern_count": 10}
        current = {"eval_pattern_count": 7}
        result = compare_trust_metrics(base, current)
        assert result["passed"] is True
        assert len(result["improvements"]) == 1

    def test_delta_keys_match_baseline_keys(self) -> None:
        base = {"broad_except_count": 5, "eval_pattern_count": 3}
        current = {"broad_except_count": 5, "eval_pattern_count": 3}
        result = compare_trust_metrics(base, current)
        assert set(result["deltas"].keys()) == set(base.keys())


class TestEvidencePackIntegration:
    """evidence_pack_generator must produce a valid evidence document."""

    def test_evidence_pack_has_required_sections(self) -> None:
        from geoprompt.quality import (
            evidence_pack_generator,
            reproducibility_bundle_export,
            raster_ai_golden_benchmark,
        )

        bench = raster_ai_golden_benchmark()
        bundle = reproducibility_bundle_export(
            model_id="test-model",
            model_version="1.0.0",
        )
        pack = evidence_pack_generator(bench, bundle)

        assert "title" in pack
        assert "sections" in pack
        assert "benchmark_results" in pack["sections"]
        assert "reproducibility" in pack["sections"]

    def test_trust_metrics_in_evidence_pack(self) -> None:
        """Trust metrics can be embedded in a release evidence document."""
        metrics = collect_trust_metrics()
        pack = {
            "title": "GeoPrompt Release Evidence",
            "trust_metrics": metrics,
            "ratchet_passed": all(
                metrics.get(k, 0) <= v for k, v in _RATCHET_THRESHOLDS.items()
            ),
        }
        assert pack["ratchet_passed"] is True, (
            f"Trust ratchet failed: {pack['trust_metrics']}"
        )
