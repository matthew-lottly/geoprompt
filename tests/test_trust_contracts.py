"""Contract tests for high-risk public API signatures (J4.48).

These tests assert stable call signatures for I/O, enterprise, service,
and security entry points.  Any change to a public API's required
parameters will cause these tests to fail, providing an early-warning gate.
"""

from __future__ import annotations

import inspect
import pytest


# ---------------------------------------------------------------------------
# I/O contract signatures
# ---------------------------------------------------------------------------

class TestIOContractSignatures:
    def _io(self):
        import geoprompt.io as io
        return io

    def test_read_geojson_signature(self):
        io = self._io()
        sig = inspect.signature(io.read_geojson)
        params = set(sig.parameters)
        assert "path" in params

    def test_write_geojson_signature(self):
        io = self._io()
        sig = inspect.signature(io.write_geojson)
        params = set(sig.parameters)
        # write_geojson accepts a GeoPromptFrame or sequence via 'frame' parameter
        assert "path" in params

    def test_read_cloud_json_signature(self):
        io = self._io()
        sig = inspect.signature(io.read_cloud_json)
        params = set(sig.parameters)
        assert "url" in params

    def test_validate_remote_url_blocks_non_http(self):
        """URL validation helper must reject file:// and ftp:// by default."""
        from geoprompt.io import _validate_remote_url
        with pytest.raises(ValueError, match="scheme"):
            _validate_remote_url("ftp://example.com/data.json")
        with pytest.raises(ValueError, match="scheme"):
            _validate_remote_url("file:///etc/passwd")

    def test_validate_remote_url_accepts_https(self):
        from geoprompt.io import _validate_remote_url
        # should not raise
        _validate_remote_url("https://example.com/data.geojson")

    def test_read_shapefile_signature(self):
        io = self._io()
        sig = inspect.signature(io.read_shapefile)
        params = set(sig.parameters)
        assert "path" in params


# ---------------------------------------------------------------------------
# Enterprise contract signatures
# ---------------------------------------------------------------------------

class TestEnterpriseContractSignatures:
    def _enterprise(self):
        import geoprompt.enterprise as e
        return e

    def test_enterprise_geodatabase_connect_raises_without_fallback(self):
        """Default call must raise ImportError when no real backend is present."""
        enterprise = self._enterprise()
        with pytest.raises(ImportError, match="allow_stub_fallback"):
            enterprise.enterprise_geodatabase_connect("localhost", "mydb")

    def test_enterprise_geodatabase_connect_stub_returns_is_stub(self):
        import warnings
        enterprise = self._enterprise()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = enterprise.enterprise_geodatabase_connect(
                "localhost", "mydb", allow_stub_fallback=True
            )
        assert result["is_stub"] is True

    def test_versioned_edit_raises_without_fallback(self):
        enterprise = self._enterprise()
        with pytest.raises(ImportError):
            enterprise.versioned_edit({}, "my_table", [])

    def test_replica_sync_raises_without_fallback(self):
        enterprise = self._enterprise()
        with pytest.raises(ImportError):
            enterprise.replica_sync({}, "my_replica")

    def test_portal_publish_raises_without_fallback(self):
        enterprise = self._enterprise()
        with pytest.raises(ImportError):
            enterprise.portal_publish("/tmp/item.zip", "Feature Service")

    def test_stub_payloads_carry_is_stub_flag(self):
        import warnings
        enterprise = self._enterprise()
        for fn, kwargs in [
            (enterprise.versioned_edit, {"connection": {}, "table": "t", "edits": []}),
            (enterprise.replica_sync, {"connection": {}, "replica_name": "r"}),
            (enterprise.portal_publish, {"item_path": "/p", "item_type": "T"}),
        ]:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = fn(**kwargs, allow_stub_fallback=True)
            assert result.get("is_stub") is True, f"{fn.__name__} missing is_stub flag"


# ---------------------------------------------------------------------------
# Database SQL injection prevention
# ---------------------------------------------------------------------------

class TestDBSQLInjectionPrevention:
    def test_sql_identifier_rejects_injection(self):
        from geoprompt.db import _validate_sql_identifier
        with pytest.raises(ValueError, match="unsafe"):
            _validate_sql_identifier("my_table; DROP TABLE users; --")

    def test_sql_identifier_rejects_backtick(self):
        from geoprompt.db import _validate_sql_identifier
        with pytest.raises(ValueError, match="unsafe"):
            _validate_sql_identifier("`my_table`")

    def test_sql_identifier_rejects_spaces(self):
        from geoprompt.db import _validate_sql_identifier
        with pytest.raises(ValueError, match="unsafe"):
            _validate_sql_identifier("my table")

    def test_sql_identifier_accepts_valid_name(self):
        from geoprompt.db import _validate_sql_identifier
        _validate_sql_identifier("my_table_1")  # should not raise

    def test_sql_identifier_rejects_empty(self):
        from geoprompt.db import _validate_sql_identifier
        with pytest.raises(ValueError, match="empty"):
            _validate_sql_identifier("")


# ---------------------------------------------------------------------------
# Security module contract
# ---------------------------------------------------------------------------

class TestSecurityContractSignatures:
    def _security(self):
        import geoprompt.security as s
        return s

    def test_security_module_importable(self):
        security = self._security()
        # security module must be importable and have the key callables available
        assert security is not None

    def test_validate_geometry_safe_callable(self):
        security = self._security()
        assert callable(getattr(security, "validate_geometry_safe", None))

    def test_rate_limit_check_callable(self):
        security = self._security()
        assert callable(getattr(security, "rate_limit_check", None))


# ---------------------------------------------------------------------------
# FallbackPolicy contract
# ---------------------------------------------------------------------------

class TestFallbackPolicyContract:
    def test_error_mode_raises_import_error(self):
        from geoprompt._exceptions import FallbackPolicy
        policy = FallbackPolicy(FallbackPolicy.ERROR)
        with pytest.raises(ImportError, match="real backend"):
            policy.enforce("my_func", "Install the library.")

    def test_warn_mode_issues_user_warning(self):
        import warnings
        from geoprompt._exceptions import FallbackPolicy
        policy = FallbackPolicy(FallbackPolicy.WARN)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            policy.enforce("my_func", "Install the library.", stacklevel=1)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "my_func" in str(w[0].message)

    def test_allow_mode_is_silent(self):
        import warnings
        from geoprompt._exceptions import FallbackPolicy
        policy = FallbackPolicy(FallbackPolicy.ALLOW)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            policy.enforce("my_func", "Install the library.", stacklevel=1)
        assert len(w) == 0

    def test_invalid_mode_raises(self):
        from geoprompt._exceptions import FallbackPolicy
        with pytest.raises(ValueError, match="Unknown"):
            FallbackPolicy("undefined_mode")

    def test_for_environment_defaults_to_error(self):
        import os
        from geoprompt._exceptions import FallbackPolicy
        # Remove env var if set
        os.environ.pop("GEOPROMPT_FALLBACK_POLICY", None)
        policy = FallbackPolicy.for_environment()
        assert policy.mode == FallbackPolicy.ERROR

    def test_for_environment_reads_env_var(self):
        import os
        from geoprompt._exceptions import FallbackPolicy
        os.environ["GEOPROMPT_FALLBACK_POLICY"] = "warn"
        try:
            policy = FallbackPolicy.for_environment()
            assert policy.mode == FallbackPolicy.WARN
        finally:
            os.environ.pop("GEOPROMPT_FALLBACK_POLICY", None)
