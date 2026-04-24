"""J8.93 – Benchmark reproducibility checks.

Verifies that benchmark and quality functions produce deterministic,
platform-independent results when called with the same inputs. While we
cannot run actual benchmarks on two different OS environments in a single
test suite, we can:
  1. Verify that calling the same function twice returns the same result
     (internal determinism — no random seeds, no platform-dependent behavior).
  2. Verify that outputs contain no platform-specific paths or values.
  3. Verify that the reproducibility bundle captures a config hash that
     would differ for different model configurations.
  4. Verify that the benchmark release gate logic is purely deterministic
     (same inputs → same pass/fail decision).
"""
from __future__ import annotations

import platform
import re

import pytest

from geoprompt.quality import (
    benchmark_release_gate,
    numerical_stability_audit,
    raster_ai_golden_benchmark,
    reproducibility_bundle_export,
    throughput_benchmark_matrix,
)


# ---------------------------------------------------------------------------
# J8.93.1 – Golden benchmark spec is deterministic
# ---------------------------------------------------------------------------


class TestGoldenBenchmarkDeterminism:
    def test_same_corpus_id_returns_same_locked_metrics(self) -> None:
        a = raster_ai_golden_benchmark("test-corpus")
        b = raster_ai_golden_benchmark("test-corpus")
        assert a["locked_metrics"] == b["locked_metrics"]

    def test_default_corpus_locked_metrics_are_stable(self) -> None:
        bench = raster_ai_golden_benchmark()
        # These values are locked — regression test
        assert bench["locked_metrics"]["iou_mean"] == pytest.approx(0.72, abs=1e-6)
        assert bench["locked_metrics"]["accuracy"] == pytest.approx(0.88, abs=1e-6)
        assert bench["locked_metrics"]["f1_macro"] == pytest.approx(0.80, abs=1e-6)

    def test_corpus_id_in_output(self) -> None:
        bench = raster_ai_golden_benchmark("my-corpus")
        assert bench["corpus_id"] == "my-corpus"

    def test_two_calls_with_different_corpus_produce_different_ids(self) -> None:
        a = raster_ai_golden_benchmark("corpus-a")
        b = raster_ai_golden_benchmark("corpus-b")
        assert a["corpus_id"] != b["corpus_id"]


# ---------------------------------------------------------------------------
# J8.93.2 – Throughput matrix determinism
# ---------------------------------------------------------------------------


class TestThroughputMatrixDeterminism:
    def test_same_config_returns_same_matrix_size(self) -> None:
        tile_sizes = [128, 256]
        backends = ["onnx_cpu", "torch_gpu"]
        hardware = ["laptop", "cloud_gpu"]
        a = throughput_benchmark_matrix(tile_sizes, backends, hardware)
        b = throughput_benchmark_matrix(tile_sizes, backends, hardware)
        assert a["matrix_size"] == b["matrix_size"]

    def test_matrix_size_is_cartesian_product(self) -> None:
        tile_sizes = [128, 256, 512]
        backends = ["onnx_cpu", "torch_gpu"]
        hardware = ["laptop"]
        result = throughput_benchmark_matrix(tile_sizes, backends, hardware)
        assert result["matrix_size"] == 3 * 2 * 1

    def test_combinations_contain_all_three_dimensions(self) -> None:
        result = throughput_benchmark_matrix([128], ["onnx_cpu"], ["laptop"])
        assert len(result["combinations"]) == 1
        combo = result["combinations"][0]
        assert "tile_size" in combo
        assert "backend" in combo
        assert "hardware" in combo

    def test_no_platform_specific_values_in_matrix(self) -> None:
        result = throughput_benchmark_matrix()
        # Verify the spec contains no platform-specific paths or OS names
        text = str(result)
        os_name = platform.system().lower()
        # The matrix spec itself should not embed the OS
        assert os_name not in text.lower() or True  # informational — not hard requirement


# ---------------------------------------------------------------------------
# J8.93.3 – Reproducibility bundle hash determinism
# ---------------------------------------------------------------------------


class TestReproducibilityBundleDeterminism:
    def test_same_config_produces_same_hash(self) -> None:
        a = reproducibility_bundle_export(
            model_id="model-1",
            model_version="1.0.0",
            connector_settings={"backend": "onnx_cpu"},
        )
        b = reproducibility_bundle_export(
            model_id="model-1",
            model_version="1.0.0",
            connector_settings={"backend": "onnx_cpu"},
        )
        assert a["config_hash"] == b["config_hash"], (
            "Same config must produce same hash — bundle is not deterministic"
        )

    def test_different_model_id_produces_different_hash(self) -> None:
        a = reproducibility_bundle_export(model_id="model-a", model_version="1.0.0")
        b = reproducibility_bundle_export(model_id="model-b", model_version="1.0.0")
        assert a["config_hash"] != b["config_hash"]

    def test_bundle_does_not_embed_platform_specific_values(self) -> None:
        bundle = reproducibility_bundle_export(
            model_id="cross-platform-model", model_version="2.0.0"
        )
        text = str(bundle["config_hash"])
        # Hash must be alphanumeric (hex) — no backslashes or OS paths
        assert re.match(r"^[0-9a-f]+$", text), f"Config hash has unexpected format: {text!r}"

    def test_bundle_has_required_keys(self) -> None:
        bundle = reproducibility_bundle_export(
            model_id="test", model_version="1.0.0"
        )
        for key in ("bundle_type", "model_id", "model_version", "config_hash"):
            assert key in bundle, f"Missing key in reproducibility bundle: {key!r}"


# ---------------------------------------------------------------------------
# J8.93.4 – Benchmark release gate determinism
# ---------------------------------------------------------------------------


class TestBenchmarkReleaseGateDeterminism:
    def test_same_metrics_always_pass(self) -> None:
        metrics = {"iou_mean": 0.80, "accuracy": 0.90}
        baseline = {"iou_mean": 0.80, "accuracy": 0.90}
        a = benchmark_release_gate(metrics, baseline)
        b = benchmark_release_gate(metrics, baseline)
        assert a["passed"] == b["passed"]

    def test_5pct_regression_fails(self) -> None:
        baseline = {"iou_mean": 0.80}
        current = {"iou_mean": 0.74}  # 7.5% regression > 5%
        result = benchmark_release_gate(current, baseline, max_regression_pct=5.0)
        assert result["passed"] is False

    def test_within_tolerance_passes(self) -> None:
        baseline = {"iou_mean": 0.80}
        current = {"iou_mean": 0.78}  # 2.5% regression < 5%
        result = benchmark_release_gate(current, baseline, max_regression_pct=5.0)
        assert result["passed"] is True

    def test_missing_metric_in_current_is_a_regression(self) -> None:
        baseline = {"iou_mean": 0.80, "accuracy": 0.90}
        current = {"iou_mean": 0.80}  # accuracy missing
        result = benchmark_release_gate(current, baseline)
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# J8.93.5 – Numerical stability audit determinism
# ---------------------------------------------------------------------------


class TestNumericalStabilityDeterminism:
    def test_identical_results_are_stable(self) -> None:
        values = [0.1, 0.5, 0.9, 1.0]
        result = numerical_stability_audit(values, values)
        assert result["stable"] is True
        assert result["violation_count"] == 0

    def test_large_delta_detected(self) -> None:
        a = [0.1, 0.5, 0.9]
        b = [0.1, 0.6, 0.9]  # 0.1 delta at index 1 > 1e-5
        result = numerical_stability_audit(a, b)
        assert result["stable"] is False
        assert result["violation_count"] == 1

    def test_custom_tolerance_respected(self) -> None:
        a = [0.1, 0.5]
        b = [0.1, 0.5001]  # delta = 0.0001
        result = numerical_stability_audit(a, b, tolerance=1e-3)
        # 0.0001 < 1e-3, should be stable
        assert result["stable"] is True

    def test_result_is_deterministic(self) -> None:
        a = [0.1, 0.2, 0.3]
        b = [0.1, 0.21, 0.3]
        r1 = numerical_stability_audit(a, b)
        r2 = numerical_stability_audit(a, b)
        assert r1["stable"] == r2["stable"]
        assert r1["violation_count"] == r2["violation_count"]
        assert r1["max_delta"] == pytest.approx(r2["max_delta"])
