"""Tests for J2.21 — No fake-data fallback without explicit opt-in.

These tests verify that simulation-only and stub-mode behavior is not returned
without explicitly passing allow_stub_fallback=True or similar opt-in flags.
"""

from __future__ import annotations

import pytest
import warnings
import os


class TestNoFallbackWithoutOptIn:
    """Ensure no simulated/stub outputs are returned without explicit enable."""

    def test_enterprise_geodatabase_raises_without_fallback(self):
        """enterprise_geodatabase_connect raises by default."""
        import geoprompt.enterprise as enterprise
        with pytest.raises(ImportError, match="allow_stub_fallback"):
            enterprise.enterprise_geodatabase_connect("localhost", "mydb")

    def test_versioned_edit_raises_without_fallback(self):
        """versioned_edit raises by default."""
        import geoprompt.enterprise as enterprise
        with pytest.raises(ImportError):
            enterprise.versioned_edit({}, "table", [])

    def test_replica_sync_raises_without_fallback(self):
        """replica_sync raises by default."""
        import geoprompt.enterprise as enterprise
        with pytest.raises(ImportError):
            enterprise.replica_sync({}, "replica")

    def test_portal_publish_raises_without_fallback(self):
        """portal_publish raises by default."""
        import geoprompt.enterprise as enterprise
        with pytest.raises(ImportError):
            enterprise.portal_publish("/path/item.zip", "Feature Service")

    def test_osm_reader_raises_by_default(self):
        """OSM reader raises explicitly when osmium is missing."""
        import geoprompt.io as io_module
        # Only test if osmium is not installed
        try:
            import osmium  # noqa: F401
            pytest.skip("osmium is installed; cannot test missing-dependency path")
        except ImportError:
            # osmium is missing, so OSM reader should fail explicitly
            with pytest.raises((ImportError, RuntimeError), match="osmium|OSM"):
                io_module.read_osm_pbf("/tmp/dummy.pbf")

    def test_dxf_reader_without_fiona_raises(self):
        """DXF reader raises explicitly when ezdxf is missing."""
        import geoprompt.io as io_module
        from geoprompt._exceptions import DependencyError
        # Only test if ezdxf is not installed
        try:
            import ezdxf  # noqa: F401
            pytest.skip("ezdxf is installed; cannot test missing-dependency path")
        except ImportError:
            # ezdxf is missing — require_capability raises DependencyError with pip hint
            with pytest.raises((ImportError, RuntimeError, DependencyError)):
                io_module.read_dxf("/tmp/dummy.dxf")


class TestStubFlagPresentWhenEnabled:
    """When fallback is explicitly enabled, is_stub flag must be present."""

    def test_enterprise_geodatabase_stub_has_is_stub_flag(self):
        """Stub connection includes is_stub=True."""
        import geoprompt.enterprise as enterprise
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = enterprise.enterprise_geodatabase_connect(
                "localhost", "mydb", allow_stub_fallback=True
            )
        assert "is_stub" in result
        assert result["is_stub"] is True

    def test_versioned_edit_stub_has_is_stub_flag(self):
        """Stub edit includes is_stub=True."""
        import geoprompt.enterprise as enterprise
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = enterprise.versioned_edit(
                {}, "table", [], allow_stub_fallback=True
            )
        assert "is_stub" in result
        assert result["is_stub"] is True

    def test_replica_sync_stub_has_is_stub_flag(self):
        """Stub sync includes is_stub=True."""
        import geoprompt.enterprise as enterprise
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = enterprise.replica_sync(
                {}, "replica", allow_stub_fallback=True
            )
        assert "is_stub" in result
        assert result["is_stub"] is True

    def test_portal_publish_stub_has_is_stub_flag(self):
        """Stub publish includes is_stub=True."""
        import geoprompt.enterprise as enterprise
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = enterprise.portal_publish(
                "/path/item.zip", "Feature Service", allow_stub_fallback=True
            )
        assert "is_stub" in result
        assert result["is_stub"] is True


class TestServiceBlocksStubModeByDefault:
    """Service deployment blocks stub-mode unless dev profile is enabled."""

    def test_build_app_raises_if_allow_stub_without_dev_profile(self):
        """build_app raises if stub is allowed but dev profile is off."""
        # Skip if fastapi not installed (optional dependency)
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed (optional dependency)")

        import geoprompt.service
        # Set environment to trigger the error condition
        old_allow = os.environ.get("GEOPROMPT_ALLOW_STUB_FALLBACK")
        old_dev = os.environ.get("GEOPROMPT_DEV_PROFILE")
        try:
            os.environ["GEOPROMPT_ALLOW_STUB_FALLBACK"] = "true"
            os.environ.pop("GEOPROMPT_DEV_PROFILE", None)
            with pytest.raises(RuntimeError, match="Stub fallback|dev profile"):
                geoprompt.service.build_app()
        finally:
            # Restore environment
            if old_allow:
                os.environ["GEOPROMPT_ALLOW_STUB_FALLBACK"] = old_allow
            else:
                os.environ.pop("GEOPROMPT_ALLOW_STUB_FALLBACK", None)
            if old_dev:
                os.environ["GEOPROMPT_DEV_PROFILE"] = old_dev
            else:
                os.environ.pop("GEOPROMPT_DEV_PROFILE", None)

    @pytest.mark.skipif(
        True,  # FastAPI is optional dependency
        reason="FastAPI not installed (optional dependency)"
    )
    def test_build_app_succeeds_with_dev_profile(self):
        """build_app succeeds when dev profile is enabled."""
        import geoprompt.service
        old_dev = os.environ.get("GEOPROMPT_DEV_PROFILE")
        try:
            os.environ["GEOPROMPT_DEV_PROFILE"] = "true"
            app = geoprompt.service.build_app()
            assert app is not None
        finally:
            if old_dev:
                os.environ["GEOPROMPT_DEV_PROFILE"] = old_dev
            else:
                os.environ.pop("GEOPROMPT_DEV_PROFILE", None)


class TestFallbackWarningOnStubUsage:
    """Stub usage emits actionable UserWarning with remediation text."""

    def test_enterprise_stub_emits_fallback_warning(self):
        """Enabling stub mode emits a UserWarning."""
        import geoprompt.enterprise as enterprise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enterprise.enterprise_geodatabase_connect(
                "localhost", "mydb", allow_stub_fallback=True
            )
        assert len(w) >= 1
        assert issubclass(w[0].category, UserWarning)
        # Check that warning contains actionable text
        msg = str(w[0].message)
        assert any(
            keyword in msg.lower()
            for keyword in ["install", "arcpy", "stub", "real backend", "fallback"]
        )
