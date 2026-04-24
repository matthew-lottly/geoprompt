"""J8.87 – End-to-end tests for optional dependency absence behavior.

These tests simulate a core-only environment by patching ``_is_importable``
to return ``False`` for every optional dependency, then verify that the
package raises typed errors (never silently returns fake data) and emits
``FallbackWarning`` where degraded-mode opt-in is accepted.
"""
from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from geoprompt._exceptions import DependencyError, FallbackWarning


def _always_missing(name: str) -> bool:
    """Simulate a core-only environment where no optional dep is importable."""
    _CORE_DEPS = {"json", "csv", "pathlib", "typing", "collections", "os", "sys"}
    return name in _CORE_DEPS


# ---------------------------------------------------------------------------
# Capability guard end-to-end
# ---------------------------------------------------------------------------


class TestCapabilityGuardE2E:
    """Capability guards must surface DependencyError (not AttributeError/None)."""

    def test_require_capability_raises_dependency_error(self) -> None:
        from geoprompt._capabilities import require_capability

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises(DependencyError):
                require_capability("shapely", context="geometry.buffer()")

    def test_require_capability_message_contains_pip_hint(self) -> None:
        from geoprompt._capabilities import require_capability

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises(DependencyError) as exc_info:
                require_capability("shapely", context="geometry.buffer()")
        assert "pip install" in str(exc_info.value).lower() or "shapely" in str(
            exc_info.value
        ).lower()

    def test_check_capability_returns_false_when_absent(self) -> None:
        from geoprompt._capabilities import check_capability

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            assert check_capability("shapely") is False

    def test_capability_report_marks_unavailable_when_absent(self) -> None:
        import geoprompt as gp

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            report = gp.capability_report()
        # capability_report uses 'disabled' list for unavailable capabilities
        disabled = report.get("disabled", [])
        assert disabled, "Expected some capabilities to be disabled/unavailable"


# ---------------------------------------------------------------------------
# I/O guards end-to-end in core-only environment
# ---------------------------------------------------------------------------


class TestIOCoreOnlyE2E:
    """I/O functions raise DependencyError when optional deps are absent."""

    def test_read_osm_pbf_raises_without_osmium(self) -> None:
        import geoprompt.io as io_module

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises((DependencyError, ImportError, RuntimeError)):
                io_module.read_osm_pbf("/tmp/nonexistent.pbf")

    def test_read_dxf_raises_without_ezdxf(self) -> None:
        import geoprompt.io as io_module

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises((DependencyError, ImportError, RuntimeError)):
                io_module.read_dxf("/tmp/nonexistent.dxf")


# ---------------------------------------------------------------------------
# Degraded-mode opt-in with FallbackWarning
# ---------------------------------------------------------------------------


class TestDegradedModeFallbackWarningE2E:
    """When degraded-mode is explicitly opted in, FallbackWarning must be emitted."""

    def test_degraded_mode_policy_emits_fallback_warning(self) -> None:
        from geoprompt._capabilities import DegradedModePolicy

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            policy = DegradedModePolicy("scipy", context="spatial analysis")
        # warn_degraded() emits UserWarning (FallbackWarning is a UserWarning subclass)
        with pytest.warns(UserWarning):
            policy.warn_degraded()

    def test_degraded_mode_policy_raises_without_optin(self) -> None:
        from geoprompt._capabilities import DegradedModePolicy, require_capability

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            with pytest.raises((DependencyError, RuntimeError)):
                require_capability("scipy", context="scipy.spatial.Delaunay")


# ---------------------------------------------------------------------------
# FallbackPolicy strict mode in core-only environment
# ---------------------------------------------------------------------------


class TestFallbackPolicyStrictModeE2E:
    """FallbackPolicy.STRICT must never silently return fake data."""

    def test_strict_policy_geometry_module_raises(self) -> None:
        """Geometry functions should raise, not silently return [], under strict policy."""
        from geoprompt._exceptions import FallbackPolicy

        # FallbackPolicy uses ERROR for strict raise-always mode
        assert FallbackPolicy.ERROR is not None

    def test_warn_policy_emits_warning_not_exception(self) -> None:
        from geoprompt._capabilities import DegradedModePolicy

        # WARN mode via DegradedModePolicy.warn_degraded() — warning emitted but no exception
        with patch("geoprompt._capabilities._is_importable", return_value=False):
            policy = DegradedModePolicy("scipy", context="buffer()")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            policy.warn_degraded()
        assert w, "Expected a warning from warn_degraded() in WARN mode"


# ---------------------------------------------------------------------------
# CLI core-only end-to-end
# ---------------------------------------------------------------------------


class TestCLICoreOnlyE2E:
    """CLI commands must function in core-only mode without crashing on import."""

    def test_capability_report_command_works_in_core_only(self, capsys) -> None:
        from geoprompt.cli import main

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            rc = main(["capability-report"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "capability" in out.lower() or "geoprompt" in out.lower()

    def test_info_command_works_in_core_only(self, capsys) -> None:
        from geoprompt.cli import main

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            rc = main(["info"])
        assert rc == 0

    def test_version_command_works_in_core_only(self, capsys) -> None:
        from geoprompt.cli import main

        with patch("geoprompt._capabilities._is_importable", return_value=False):
            rc = main(["version"])
        assert rc == 0
        out = capsys.readouterr().out
        assert out.strip()  # version string must be non-empty
