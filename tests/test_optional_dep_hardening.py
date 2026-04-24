"""Tests for J5 Optional Dependency and Degraded-Mode Hardening.

Covers:
  J5.1  – capability registry enumeration and classification
  J5.2  – degraded-mode behavior tests for optional-dependency paths
  J5.3  – explicit DependencyError (not bare ImportError) on missing dep
  J5.4  – deterministic fallback policy (no implicit fake outputs)
  J5.5  – per-feature capability checks in CLI (tested via capability helpers)
  J5.7  – runtime pip-extra hint included in DependencyError
  J5.12 – automatic chunk-size estimation with explicit-override guarantee
  J5.13 – chunking modes (fixed, adaptive, auto) with telemetry
  J5.14 – deterministic adaptive sizing
  J5.15 – chunk-decision telemetry (reasoning field)
  J5.16 – CI/environment matrix constants
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# J5.1 – capability registry enumeration and classification
# ---------------------------------------------------------------------------

class TestCapabilityRegistry:
    def test_registry_not_empty(self):
        from geoprompt._capabilities import CAPABILITY_REGISTRY, FailureMode
        assert len(CAPABILITY_REGISTRY) >= 10

    def test_all_entries_have_required_fields(self):
        from geoprompt._capabilities import CAPABILITY_REGISTRY
        for name, spec in CAPABILITY_REGISTRY.items():
            assert spec.name == name
            assert spec.import_name
            assert spec.pip_extra
            assert spec.failure_mode is not None

    def test_failure_modes_cover_all_three_tiers(self):
        from geoprompt._capabilities import CAPABILITY_REGISTRY, FailureMode
        modes = {spec.failure_mode for spec in CAPABILITY_REGISTRY.values()}
        assert FailureMode.HARD_FAIL in modes
        assert FailureMode.SOFT_FAIL in modes
        assert FailureMode.DEGRADED in modes

    def test_key_dependencies_are_registered(self):
        from geoprompt._capabilities import CAPABILITY_REGISTRY
        required = {"geopandas", "pandas", "pyarrow", "shapely", "pyproj", "rasterio",
                    "matplotlib", "sqlalchemy", "fastapi"}
        missing = required - CAPABILITY_REGISTRY.keys()
        assert not missing, f"Missing registry entries: {missing}"

    def test_capability_status_returns_all(self):
        from geoprompt._capabilities import CAPABILITY_REGISTRY, capability_status
        status = capability_status()
        assert set(status.keys()) == set(CAPABILITY_REGISTRY.keys())

    def test_capability_status_available_field_is_bool(self):
        from geoprompt._capabilities import capability_status
        for name, info in capability_status().items():
            assert isinstance(info["available"], bool), f"{name}.available must be bool"

    def test_capability_status_has_pip_hint(self):
        from geoprompt._capabilities import capability_status
        for name, info in capability_status().items():
            assert info["pip_extra"], f"{name} must have a pip_extra hint"


# ---------------------------------------------------------------------------
# J5.2 – degraded-mode behavior tests
# ---------------------------------------------------------------------------

class TestDegradedModeBehavior:
    """Verify that degraded-mode paths emit warnings, not silent fake output."""

    def test_warn_degraded_issues_user_warning(self):
        from geoprompt._capabilities import DegradedModePolicy, FailureMode
        policy = DegradedModePolicy.__new__(DegradedModePolicy)
        policy._capability = "matplotlib"
        policy._context = "test"
        policy.available = False
        # Inject a fake spec with DEGRADED mode
        from geoprompt._capabilities import CapabilitySpec
        policy._spec = CapabilitySpec(
            name="matplotlib",
            import_name="matplotlib",
            pip_extra="geoprompt[viz]",
            failure_mode=FailureMode.DEGRADED,
        )
        with pytest.warns(UserWarning, match="degraded mode"):
            policy.warn_degraded()

    def test_warn_degraded_no_warn_when_available(self):
        from geoprompt._capabilities import DegradedModePolicy, CapabilitySpec, FailureMode
        policy = DegradedModePolicy.__new__(DegradedModePolicy)
        policy._capability = "matplotlib"
        policy._context = ""
        policy.available = True
        policy._spec = CapabilitySpec(
            name="matplotlib",
            import_name="matplotlib",
            pip_extra="geoprompt[viz]",
            failure_mode=FailureMode.DEGRADED,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            policy.warn_degraded()  # should not raise

    def test_allow_degraded_returns_true_for_degraded_mode(self):
        from geoprompt._capabilities import DegradedModePolicy, CapabilitySpec, FailureMode
        policy = DegradedModePolicy.__new__(DegradedModePolicy)
        policy._capability = "matplotlib"
        policy._context = ""
        policy.available = False
        policy._spec = CapabilitySpec(
            name="matplotlib",
            import_name="matplotlib",
            pip_extra="geoprompt[viz]",
            failure_mode=FailureMode.DEGRADED,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = policy.allow_degraded_or_raise(allow=True)
        assert result is True

    def test_allow_degraded_raises_for_hard_fail(self):
        from geoprompt._capabilities import DegradedModePolicy, CapabilitySpec, FailureMode
        from geoprompt._exceptions import DependencyError
        policy = DegradedModePolicy.__new__(DegradedModePolicy)
        policy._capability = "geopandas"
        policy._context = ""
        policy.available = False
        policy._spec = CapabilitySpec(
            name="geopandas",
            import_name="geopandas",
            pip_extra="geoprompt[io]",
            failure_mode=FailureMode.HARD_FAIL,
        )
        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises(DependencyError):
                policy.allow_degraded_or_raise(allow=True)


# ---------------------------------------------------------------------------
# J5.3 – DependencyError on missing dependency (not bare ImportError)
# ---------------------------------------------------------------------------

class TestDependencyError:
    def test_require_capability_raises_dependency_error(self):
        from geoprompt._capabilities import require_capability
        from geoprompt._exceptions import DependencyError
        # Patch _is_importable to simulate missing dep
        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises(DependencyError):
                require_capability("geopandas")

    def test_require_capability_error_is_not_bare_import_error(self):
        from geoprompt._capabilities import require_capability
        from geoprompt._exceptions import DependencyError
        with patch("geoprompt._capabilities._is_importable", return_value=False):
            try:
                require_capability("geopandas")
            except DependencyError:
                pass  # correct
            except ImportError:
                pytest.fail("Should not raise bare ImportError, use DependencyError")

    def test_require_capability_raises_for_unknown_name(self):
        from geoprompt._capabilities import require_capability
        with pytest.raises(KeyError):
            require_capability("__nonexistent_dep__")

    def test_check_capability_returns_bool(self):
        from geoprompt._capabilities import check_capability
        result = check_capability("geopandas")
        assert isinstance(result, bool)

    def test_check_capability_false_for_unknown(self):
        from geoprompt._capabilities import check_capability
        assert check_capability("__nonexistent__") is False


# ---------------------------------------------------------------------------
# J5.4 – deterministic fallback policy
# ---------------------------------------------------------------------------

class TestFallbackPolicy:
    def test_enforce_raises_when_unavailable(self):
        from geoprompt._capabilities import DegradedModePolicy, CapabilitySpec, FailureMode
        from geoprompt._exceptions import DependencyError
        policy = DegradedModePolicy.__new__(DegradedModePolicy)
        policy._capability = "rasterio"
        policy._context = "test_fn"
        policy.available = False
        policy._spec = CapabilitySpec(
            name="rasterio",
            import_name="rasterio",
            pip_extra="geoprompt[raster]",
            failure_mode=FailureMode.HARD_FAIL,
        )
        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises(DependencyError, match="rasterio"):
                policy.enforce()

    def test_enforce_silent_when_available(self):
        from geoprompt._capabilities import DegradedModePolicy, CapabilitySpec, FailureMode
        policy = DegradedModePolicy.__new__(DegradedModePolicy)
        policy._capability = "rasterio"
        policy._context = ""
        policy.available = True
        policy._spec = CapabilitySpec(
            name="rasterio",
            import_name="rasterio",
            pip_extra="geoprompt[raster]",
            failure_mode=FailureMode.HARD_FAIL,
        )
        policy.enforce()  # must not raise


# ---------------------------------------------------------------------------
# J5.7 – runtime pip-extra hint in error message
# ---------------------------------------------------------------------------

class TestPipHintInError:
    def test_error_message_contains_pip_install(self):
        from geoprompt._capabilities import require_capability
        from geoprompt._exceptions import DependencyError
        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises(DependencyError) as exc_info:
                require_capability("geopandas")
        assert "pip install" in str(exc_info.value)

    def test_error_message_contains_extra_name(self):
        from geoprompt._capabilities import require_capability, CAPABILITY_REGISTRY
        from geoprompt._exceptions import DependencyError
        spec = CAPABILITY_REGISTRY["geopandas"]
        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises(DependencyError) as exc_info:
                require_capability("geopandas")
        assert spec.pip_extra in str(exc_info.value)

    def test_error_contains_context_when_provided(self):
        from geoprompt._capabilities import require_capability
        from geoprompt._exceptions import DependencyError
        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises(DependencyError) as exc_info:
                require_capability("geopandas", context="write_shapefile")
        assert "write_shapefile" in str(exc_info.value)


# ---------------------------------------------------------------------------
# J5.12 – explicit chunk_size override is always honoured
# ---------------------------------------------------------------------------

class TestChunkSizeExplicitOverride:
    def test_explicit_override_is_always_honoured(self):
        from geoprompt._capabilities import estimate_chunk_size, ChunkMode
        decision = estimate_chunk_size(explicit_chunk_size=1234)
        assert decision.chosen_size == 1234
        assert decision.explicit_override is True
        assert decision.mode == ChunkMode.FIXED

    def test_explicit_override_wins_over_adaptive_mode(self):
        from geoprompt._capabilities import estimate_chunk_size, ChunkMode
        decision = estimate_chunk_size(
            explicit_chunk_size=999,
            mode=ChunkMode.ADAPTIVE,
            sample_row={"a": "x" * 1000},  # large row – would produce small chunk
        )
        assert decision.chosen_size == 999

    def test_explicit_override_zero_raises(self):
        from geoprompt._capabilities import estimate_chunk_size
        with pytest.raises(ValueError):
            estimate_chunk_size(explicit_chunk_size=0)

    def test_explicit_override_negative_raises(self):
        from geoprompt._capabilities import estimate_chunk_size
        with pytest.raises(ValueError):
            estimate_chunk_size(explicit_chunk_size=-1)


# ---------------------------------------------------------------------------
# J5.13 – chunking modes and telemetry
# ---------------------------------------------------------------------------

class TestChunkingModes:
    def test_fixed_mode_returns_default_size(self):
        from geoprompt._capabilities import estimate_chunk_size, ChunkMode
        decision = estimate_chunk_size(mode=ChunkMode.FIXED)
        assert decision.mode == ChunkMode.FIXED
        assert decision.chosen_size == 50_000
        assert decision.explicit_override is False

    def test_adaptive_mode_returns_adaptive(self):
        from geoprompt._capabilities import estimate_chunk_size, ChunkMode
        decision = estimate_chunk_size(
            mode=ChunkMode.ADAPTIVE,
            columns=["id", "name", "value"],
            memory_budget_bytes=10 * 1024 * 1024,
        )
        assert decision.mode == ChunkMode.ADAPTIVE

    def test_auto_mode_without_explicit_is_adaptive(self):
        from geoprompt._capabilities import estimate_chunk_size, ChunkMode
        decision = estimate_chunk_size(mode=ChunkMode.AUTO)
        # AUTO with no explicit -> ADAPTIVE
        assert decision.mode == ChunkMode.ADAPTIVE

    def test_telemetry_has_reasoning(self):
        from geoprompt._capabilities import estimate_chunk_size, ChunkMode
        decision = estimate_chunk_size(mode=ChunkMode.ADAPTIVE, columns=["x", "y"])
        assert decision.reasoning, "reasoning field must be non-empty"

    def test_telemetry_explicit_override_reasoning(self):
        from geoprompt._capabilities import estimate_chunk_size
        decision = estimate_chunk_size(explicit_chunk_size=5000)
        assert "5000" in decision.reasoning

    def test_telemetry_fields_on_adaptive(self):
        from geoprompt._capabilities import estimate_chunk_size, ChunkMode
        decision = estimate_chunk_size(
            mode=ChunkMode.ADAPTIVE,
            sample_row={"a": 1, "b": 2.0, "c": "hello"},
            memory_budget_bytes=64 * 1024 * 1024,
        )
        assert decision.row_width_estimate_bytes is not None
        assert decision.memory_budget_bytes == 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# J5.14 – deterministic adaptive sizing (same inputs → same output)
# ---------------------------------------------------------------------------

class TestDeterministicAdaptiveSizing:
    def test_same_inputs_produce_same_result(self):
        from geoprompt._capabilities import estimate_chunk_size, ChunkMode
        kwargs = dict(
            mode=ChunkMode.ADAPTIVE,
            columns=["id", "name", "lat", "lon", "value"],
            memory_budget_bytes=50 * 1024 * 1024,
        )
        d1 = estimate_chunk_size(**kwargs)
        d2 = estimate_chunk_size(**kwargs)
        assert d1.chosen_size == d2.chosen_size
        assert d1.reasoning == d2.reasoning

    def test_larger_budget_produces_larger_chunk(self):
        from geoprompt._capabilities import estimate_chunk_size, ChunkMode
        cols = ["id", "name", "geometry"]
        d_small = estimate_chunk_size(
            mode=ChunkMode.ADAPTIVE, columns=cols, memory_budget_bytes=1 * 1024 * 1024
        )
        d_large = estimate_chunk_size(
            mode=ChunkMode.ADAPTIVE, columns=cols, memory_budget_bytes=100 * 1024 * 1024
        )
        assert d_large.chosen_size >= d_small.chosen_size

    def test_adaptive_clamps_to_min(self):
        from geoprompt._capabilities import estimate_chunk_size, ChunkMode, _ADAPTIVE_MIN_CHUNK
        # Very tight budget -> should clamp to min
        d = estimate_chunk_size(
            mode=ChunkMode.ADAPTIVE,
            columns=["x"] * 1000,
            memory_budget_bytes=1,  # impossibly tight
        )
        assert d.chosen_size >= _ADAPTIVE_MIN_CHUNK

    def test_adaptive_clamps_to_max(self):
        from geoprompt._capabilities import estimate_chunk_size, ChunkMode, _ADAPTIVE_MAX_CHUNK
        # Huge budget -> should clamp to max
        d = estimate_chunk_size(
            mode=ChunkMode.ADAPTIVE,
            columns=["id"],
            memory_budget_bytes=10 * 1024 * 1024 * 1024,  # 10 GiB
        )
        assert d.chosen_size <= _ADAPTIVE_MAX_CHUNK


# ---------------------------------------------------------------------------
# J5.16 – CI/environment matrix constants
# ---------------------------------------------------------------------------

class TestCIEnvironmentMatrix:
    def test_ci_profiles_defined(self):
        from geoprompt._capabilities import CI_EXTRAS_PROFILES
        assert "core-only" in CI_EXTRAS_PROFILES
        assert "all" in CI_EXTRAS_PROFILES
        assert isinstance(CI_EXTRAS_PROFILES["core-only"], list)

    def test_degraded_guarantees_defined(self):
        from geoprompt._capabilities import DEGRADED_MODE_GUARANTEES, CI_EXTRAS_PROFILES
        for profile in CI_EXTRAS_PROFILES:
            assert profile in DEGRADED_MODE_GUARANTEES, f"Missing degraded-mode guarantee for profile: {profile}"
            assert DEGRADED_MODE_GUARANTEES[profile].strip()

    def test_core_only_guarantee_mentions_json(self):
        from geoprompt._capabilities import DEGRADED_MODE_GUARANTEES
        assert "JSON" in DEGRADED_MODE_GUARANTEES["core-only"] or "json" in DEGRADED_MODE_GUARANTEES["core-only"].lower()

    def test_core_only_guarantee_mentions_dependency_error(self):
        from geoprompt._capabilities import DEGRADED_MODE_GUARANTEES
        text = DEGRADED_MODE_GUARANTEES["core-only"]
        assert "DependencyError" in text


# ---------------------------------------------------------------------------
# J5.5 – CLI capability check helper (unit-level)
# ---------------------------------------------------------------------------

class TestCLICapabilityCheck:
    def test_returns_none_when_available(self):
        from geoprompt.cli import _cli_require_capability
        with patch("geoprompt._capabilities._is_importable", return_value=True):
            result = _cli_require_capability("geopandas", "test-cmd")
        assert result is None

    def test_returns_1_when_unavailable(self, capsys):
        from geoprompt.cli import _cli_require_capability
        with patch("geoprompt._capabilities._is_importable", return_value=False):
            result = _cli_require_capability("fastapi", "serve")
        assert result == 1

    def test_prints_pip_hint_when_unavailable(self, capsys):
        from geoprompt.cli import _cli_require_capability
        with patch("geoprompt._capabilities._is_importable", return_value=False):
            _cli_require_capability("fastapi", "serve")
        out = capsys.readouterr().out
        assert "pip install" in out

    def test_serve_check_fires_before_import(self, capsys):
        """CLI serve command should exit early without importing uvicorn."""
        from geoprompt.cli import main
        with patch("geoprompt._capabilities._is_importable", return_value=False):
            rc = main(["serve"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "fastapi" in out.lower() or "pip" in out.lower()
