"""Tests for J6 Data I/O, Database Safety, and Boundary Validation.

Covers:
  J6.63 – injection-resistance for SQL / expression-like filters
  J6.66 – network failure classification (timeout/auth/transport/parse/remote-service)
  J6.67 – checksum and integrity verification for remote artifacts
  J6.68 – path traversal checks for write/export utilities
  J6.69 – atomic-write guarantees and rollback semantics
  J6.70 – GeoJSON-like schema validation before write
  J6.71 – CRS and geometry validation gate before persistence
  J6.72 – secure-default mode blocking insecure remote fetches
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# J6.63 – SQL injection-resistance
# ---------------------------------------------------------------------------

class TestSQLInjectionResistance:
    """Tests for _validate_sql_identifier used in db.py."""

    def test_valid_identifier_passes(self):
        from geoprompt.db import _validate_sql_identifier
        _validate_sql_identifier("parks", label="table name")  # no raise

    def test_identifier_with_spaces_raises(self):
        from geoprompt.db import _validate_sql_identifier
        with pytest.raises(ValueError, match="unsafe characters"):
            _validate_sql_identifier("my table", label="table name")

    def test_identifier_with_semicolon_raises(self):
        from geoprompt.db import _validate_sql_identifier
        with pytest.raises(ValueError):
            _validate_sql_identifier("parks; DROP TABLE parks--")

    def test_identifier_with_quote_raises(self):
        from geoprompt.db import _validate_sql_identifier
        with pytest.raises(ValueError):
            _validate_sql_identifier("parks'")

    def test_identifier_with_double_dash_raises(self):
        from geoprompt.db import _validate_sql_identifier
        with pytest.raises(ValueError):
            _validate_sql_identifier("parks--comment")

    def test_identifier_too_long_raises(self):
        from geoprompt.db import _validate_sql_identifier
        with pytest.raises(ValueError):
            _validate_sql_identifier("a" * 130, label="column")

    def test_empty_identifier_raises(self):
        from geoprompt.db import _validate_sql_identifier
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_sql_identifier("", label="table name")

    def test_numeric_start_raises(self):
        from geoprompt.db import _validate_sql_identifier
        with pytest.raises(ValueError):
            _validate_sql_identifier("1badname")

    def test_union_injection_attempt_raises(self):
        from geoprompt.db import _validate_sql_identifier
        with pytest.raises(ValueError):
            _validate_sql_identifier("x UNION SELECT * FROM users")

    def test_dollar_sign_allowed(self):
        from geoprompt.db import _validate_sql_identifier
        _validate_sql_identifier("col$1", label="column")  # no raise


class TestExpressionInjectionResistance:
    """Tests that SQL-like expression filters reject injection attempts."""

    def test_valid_comparison_parses(self):
        from geoprompt.query import parse_expression
        pred = parse_expression("name = 'parks'")
        assert callable(pred)
        row = {"name": "parks"}
        assert pred(row)

    def test_expression_with_semicolon_raises(self):
        from geoprompt.query import parse_expression
        with pytest.raises(ValueError):
            parse_expression("name = 'foo'; DROP TABLE--")

    def test_expression_with_comments_raises(self):
        from geoprompt.query import parse_expression
        with pytest.raises(ValueError):
            parse_expression("id = 1 -- comment")

    def test_expression_like_with_wildcard_only_parses(self):
        from geoprompt.query import parse_expression
        pred = parse_expression("name LIKE '%park%'")
        assert callable(pred)

    def test_expression_in_clause_parses(self):
        from geoprompt.query import parse_expression
        pred = parse_expression("type IN (road, trail)")
        assert callable(pred)

    def test_empty_expression_raises(self):
        from geoprompt.query import parse_expression
        with pytest.raises(ValueError):
            parse_expression("")

    def test_expression_union_injection_raises(self):
        from geoprompt.query import parse_expression
        with pytest.raises(ValueError):
            parse_expression("id=1 UNION SELECT password FROM users")


# ---------------------------------------------------------------------------
# J6.66 – network failure classification
# ---------------------------------------------------------------------------

class TestNetworkFailureClassification:
    def test_timeout_classification(self):
        from geoprompt.io import classify_network_error, NETWORK_ERROR_TIMEOUT
        exc = TimeoutError("timed out")
        classified = classify_network_error(exc)
        assert classified.category == NETWORK_ERROR_TIMEOUT

    def test_urlerror_with_timed_out_reason_is_timeout(self):
        from geoprompt.io import classify_network_error, NETWORK_ERROR_TIMEOUT
        exc = urllib.error.URLError(reason="timed out")
        classified = classify_network_error(exc)
        assert classified.category == NETWORK_ERROR_TIMEOUT

    def test_http_401_is_auth(self):
        from geoprompt.io import classify_network_error, NETWORK_ERROR_AUTH
        exc = urllib.error.HTTPError(url="http://x", code=401, msg="Unauthorized",
                                      hdrs=None, fp=None)  # type: ignore[arg-type]
        classified = classify_network_error(exc)
        assert classified.category == NETWORK_ERROR_AUTH

    def test_http_403_is_auth(self):
        from geoprompt.io import classify_network_error, NETWORK_ERROR_AUTH
        exc = urllib.error.HTTPError(url="http://x", code=403, msg="Forbidden",
                                      hdrs=None, fp=None)  # type: ignore[arg-type]
        classified = classify_network_error(exc)
        assert classified.category == NETWORK_ERROR_AUTH

    def test_http_500_is_remote_service(self):
        from geoprompt.io import classify_network_error, NETWORK_ERROR_REMOTE_SERVICE
        exc = urllib.error.HTTPError(url="http://x", code=500, msg="Server Error",
                                      hdrs=None, fp=None)  # type: ignore[arg-type]
        classified = classify_network_error(exc)
        assert classified.category == NETWORK_ERROR_REMOTE_SERVICE

    def test_json_decode_error_is_parse(self):
        from geoprompt.io import classify_network_error, NETWORK_ERROR_PARSE
        exc = json.JSONDecodeError("bad json", "", 0)
        classified = classify_network_error(exc)
        assert classified.category == NETWORK_ERROR_PARSE

    def test_urlerror_connection_refused_is_transport(self):
        from geoprompt.io import classify_network_error, NETWORK_ERROR_TRANSPORT
        exc = urllib.error.URLError(reason="Connection refused")
        classified = classify_network_error(exc)
        assert classified.category == NETWORK_ERROR_TRANSPORT

    def test_classified_error_is_runtime_error(self):
        from geoprompt.io import classify_network_error, ClassifiedNetworkError
        exc = TimeoutError("x")
        classified = classify_network_error(exc)
        assert isinstance(classified, RuntimeError)
        assert isinstance(classified, ClassifiedNetworkError)

    def test_original_is_preserved(self):
        from geoprompt.io import classify_network_error
        original = TimeoutError("original")
        classified = classify_network_error(original)
        assert classified.original is original


# ---------------------------------------------------------------------------
# J6.67 – checksum and integrity verification
# ---------------------------------------------------------------------------

class TestChecksumVerification:
    def test_correct_hash_passes(self):
        from geoprompt.io import verify_artifact_checksum
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        verify_artifact_checksum(data, expected_hash=expected)  # no raise

    def test_wrong_hash_raises(self):
        from geoprompt.io import verify_artifact_checksum
        with pytest.raises(ValueError, match="integrity check failed"):
            verify_artifact_checksum(b"hello world", expected_hash="deadbeef00")

    def test_sha512_algorithm(self):
        from geoprompt.io import verify_artifact_checksum
        data = b"test data"
        expected = hashlib.sha512(data).hexdigest()
        verify_artifact_checksum(data, expected_hash=expected, algorithm="sha512")

    def test_case_insensitive_hash(self):
        from geoprompt.io import verify_artifact_checksum
        data = b"abc"
        expected = hashlib.sha256(data).hexdigest().upper()
        verify_artifact_checksum(data, expected_hash=expected)  # no raise

    def test_empty_data_verified(self):
        from geoprompt.io import verify_artifact_checksum
        data = b""
        expected = hashlib.sha256(data).hexdigest()
        verify_artifact_checksum(data, expected_hash=expected)


# ---------------------------------------------------------------------------
# J6.68 – path traversal protection
# ---------------------------------------------------------------------------

class TestPathTraversalProtection:
    def test_valid_path_within_root_passes(self, tmp_path):
        from geoprompt.io import safe_output_path
        result = safe_output_path(tmp_path / "output.geojson", root=tmp_path)
        assert str(tmp_path) in str(result)

    def test_path_traversal_raises(self, tmp_path):
        from geoprompt.io import safe_output_path
        evil_path = tmp_path / ".." / "etc" / "passwd"
        with pytest.raises(ValueError, match="traversal"):
            safe_output_path(evil_path, root=tmp_path)

    def test_absolute_path_outside_root_raises(self, tmp_path):
        from geoprompt.io import safe_output_path
        other_dir = tmp_path.parent / "other"
        with pytest.raises(ValueError, match="traversal"):
            safe_output_path(other_dir / "file.json", root=tmp_path)

    def test_no_root_passes_any_path(self, tmp_path):
        from geoprompt.io import safe_output_path
        result = safe_output_path(tmp_path / "output.geojson")
        assert result.name == "output.geojson"

    def test_disallowed_suffix_raises(self, tmp_path):
        from geoprompt.io import safe_output_path
        with pytest.raises(ValueError, match="extension"):
            safe_output_path(
                tmp_path / "output.exe",
                allowed_suffixes=frozenset({".geojson", ".json"}),
            )

    def test_allowed_suffix_passes(self, tmp_path):
        from geoprompt.io import safe_output_path
        result = safe_output_path(
            tmp_path / "output.geojson",
            allowed_suffixes=frozenset({".geojson", ".json"}),
        )
        assert result.suffix == ".geojson"

    def test_double_dot_component_raises(self, tmp_path):
        from geoprompt.io import safe_output_path
        evil = str(tmp_path) + "/sub/../../etc/shadow"
        with pytest.raises(ValueError, match="traversal"):
            safe_output_path(evil, root=tmp_path)


# ---------------------------------------------------------------------------
# J6.69 – atomic-write guarantees
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_text_write_creates_file(self, tmp_path):
        from geoprompt.io import atomic_write_text
        dest = tmp_path / "output.txt"
        result = atomic_write_text(dest, "hello world")
        assert result == dest
        assert dest.read_text() == "hello world"

    def test_text_write_no_tmp_file_left_behind(self, tmp_path):
        from geoprompt.io import atomic_write_text
        dest = tmp_path / "output.txt"
        atomic_write_text(dest, "data")
        tmp_files = list(tmp_path.glob(".geoprompt_atomic_*"))
        assert not tmp_files, f"Stale temp files found: {tmp_files}"

    def test_bytes_write_creates_file(self, tmp_path):
        from geoprompt.io import atomic_write_bytes
        dest = tmp_path / "binary.bin"
        result = atomic_write_bytes(dest, b"\x00\x01\x02")
        assert result == dest
        assert dest.read_bytes() == b"\x00\x01\x02"

    def test_bytes_write_no_tmp_file_left_behind(self, tmp_path):
        from geoprompt.io import atomic_write_bytes
        dest = tmp_path / "binary.bin"
        atomic_write_bytes(dest, b"data")
        tmp_files = list(tmp_path.glob(".geoprompt_atomic_*"))
        assert not tmp_files

    def test_text_write_overwrites_existing(self, tmp_path):
        from geoprompt.io import atomic_write_text
        dest = tmp_path / "output.txt"
        dest.write_text("old content")
        atomic_write_text(dest, "new content")
        assert dest.read_text() == "new content"

    def test_write_creates_parent_directories(self, tmp_path):
        from geoprompt.io import atomic_write_text
        dest = tmp_path / "sub" / "dir" / "output.txt"
        atomic_write_text(dest, "hello")
        assert dest.exists()

    def test_rollback_on_write_error(self, tmp_path):
        """When the write fails mid-stream, no corrupted file is left."""
        from geoprompt.io import atomic_write_bytes
        dest = tmp_path / "output.bin"
        dest.write_bytes(b"original")

        # Simulate an error by patching os.replace to raise
        with patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                atomic_write_bytes(dest, b"new data")

        # The original content must still be intact
        assert dest.read_bytes() == b"original"


# ---------------------------------------------------------------------------
# J6.70 – GeoJSON schema validation
# ---------------------------------------------------------------------------

class TestGeoJSONSchemaValidation:
    def test_valid_feature_collection_passes(self):
        from geoprompt.io import validate_geojson_schema
        validate_geojson_schema({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {}}
            ],
        })

    def test_not_a_dict_raises(self):
        from geoprompt.io import validate_geojson_schema
        with pytest.raises(ValueError, match="must be a dict"):
            validate_geojson_schema([1, 2, 3])

    def test_invalid_type_raises(self):
        from geoprompt.io import validate_geojson_schema
        with pytest.raises(ValueError, match="Invalid GeoJSON type"):
            validate_geojson_schema({"type": "NotAType"})

    def test_feature_collection_without_features_list_raises(self):
        from geoprompt.io import validate_geojson_schema
        with pytest.raises(ValueError, match="features.*must be a list"):
            validate_geojson_schema({"type": "FeatureCollection", "features": "bad"})

    def test_feature_missing_geometry_raises(self):
        from geoprompt.io import validate_geojson_schema
        with pytest.raises(ValueError, match="missing required keys"):
            validate_geojson_schema({
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}}],  # no geometry
            })

    def test_bare_feature_passes(self):
        from geoprompt.io import validate_geojson_schema
        validate_geojson_schema({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [1, 2]},
            "properties": {"name": "x"},
        })

    def test_bare_feature_missing_properties_raises(self):
        from geoprompt.io import validate_geojson_schema
        with pytest.raises(ValueError, match="missing required keys"):
            validate_geojson_schema({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1, 2]},
                # properties missing
            })

    def test_plain_geometry_type_passes(self):
        from geoprompt.io import validate_geojson_schema
        validate_geojson_schema({"type": "Point"})


# ---------------------------------------------------------------------------
# J6.71 – CRS and geometry validation
# ---------------------------------------------------------------------------

class TestCRSValidation:
    def test_valid_epsg_passes(self):
        from geoprompt.io import validate_crs_before_persist
        validate_crs_before_persist("EPSG:4326")  # no raise

    def test_valid_epsg_27700_passes(self):
        from geoprompt.io import validate_crs_before_persist
        validate_crs_before_persist("EPSG:27700")

    def test_valid_proj4_passes(self):
        from geoprompt.io import validate_crs_before_persist
        validate_crs_before_persist("+proj=latlong +datum=WGS84")

    def test_garbage_crs_raises(self):
        from geoprompt.io import validate_crs_before_persist
        with pytest.raises(ValueError, match="does not match"):
            validate_crs_before_persist("not-a-crs")

    def test_none_crs_issues_warning(self):
        from geoprompt.io import validate_crs_before_persist
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_crs_before_persist(None)
        assert any("CRS" in str(warning.message) for warning in w)

    def test_empty_crs_issues_warning(self):
        from geoprompt.io import validate_crs_before_persist
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_crs_before_persist("")
        assert any("CRS" in str(warning.message) for warning in w)

    def test_urn_crs_passes(self):
        from geoprompt.io import validate_crs_before_persist
        validate_crs_before_persist("urn:ogc:def:crs:EPSG::4326")


class TestGeometryValidation:
    def test_valid_point_passes(self):
        from geoprompt.io import validate_geometry_before_persist
        validate_geometry_before_persist({"type": "Point", "coordinates": [0.0, 1.0]})

    def test_none_without_allow_raises(self):
        from geoprompt.io import validate_geometry_before_persist
        with pytest.raises(ValueError, match="must not be None"):
            validate_geometry_before_persist(None)

    def test_none_with_allow_passes(self):
        from geoprompt.io import validate_geometry_before_persist
        validate_geometry_before_persist(None, allow_none=True)

    def test_not_dict_raises(self):
        from geoprompt.io import validate_geometry_before_persist
        with pytest.raises(ValueError, match="must be a GeoJSON dict"):
            validate_geometry_before_persist("POINT(0 0)")

    def test_invalid_geometry_type_raises(self):
        from geoprompt.io import validate_geometry_before_persist
        with pytest.raises(ValueError, match="not a valid GeoJSON geometry type"):
            validate_geometry_before_persist({"type": "FeatureCollection"})

    def test_missing_coordinates_raises(self):
        from geoprompt.io import validate_geometry_before_persist
        with pytest.raises(ValueError, match="missing 'coordinates'"):
            validate_geometry_before_persist({"type": "Point"})

    def test_valid_polygon_passes(self):
        from geoprompt.io import validate_geometry_before_persist
        validate_geometry_before_persist({
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        })

    def test_geometry_collection_without_coordinates_passes(self):
        from geoprompt.io import validate_geometry_before_persist
        validate_geometry_before_persist({"type": "GeometryCollection", "geometries": []})


# ---------------------------------------------------------------------------
# J6.72 – secure-default mode for remote fetches
# ---------------------------------------------------------------------------

class TestSecureRemoteFetchDefault:
    def test_https_always_allowed(self):
        from geoprompt.io import _check_remote_fetch_allowed
        _check_remote_fetch_allowed("https://example.com/data.json")  # no raise

    def test_http_blocked_by_default(self):
        from geoprompt.io import _check_remote_fetch_allowed
        with pytest.raises(ValueError, match="Insecure fetch blocked"):
            _check_remote_fetch_allowed("http://example.com/data.json")

    def test_http_allowed_with_explicit_opt_in(self):
        from geoprompt.io import _check_remote_fetch_allowed
        _check_remote_fetch_allowed("http://example.com/data.json", allow_unsafe=True)

    def test_http_blocked_message_contains_suggestion(self):
        from geoprompt.io import _check_remote_fetch_allowed
        with pytest.raises(ValueError) as exc_info:
            _check_remote_fetch_allowed("http://example.com/data.json")
        msg = str(exc_info.value)
        assert "https://" in msg or "allow_unsafe_remote_fetch" in msg

    def test_validate_remote_url_still_requires_scheme(self):
        from geoprompt.io import _validate_remote_url
        with pytest.raises(ValueError, match="no scheme"):
            _validate_remote_url("example.com/data.json")

    def test_validate_remote_url_rejects_ftp(self):
        from geoprompt.io import _validate_remote_url
        with pytest.raises(ValueError, match="not allowed"):
            _validate_remote_url("ftp://example.com/data.json")

    def test_fetch_remote_artifact_blocks_http(self):
        from geoprompt.io import fetch_remote_artifact
        with pytest.raises(ValueError, match="Insecure fetch blocked"):
            fetch_remote_artifact("http://example.com/data.json")

    def test_fetch_remote_artifact_verifies_checksum(self):
        """A mocked successful fetch with a wrong checksum raises ValueError."""
        from geoprompt.io import fetch_remote_artifact
        from unittest.mock import MagicMock
        import io

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = b"real data"

        with patch("geoprompt.io.urlopen", return_value=mock_response):
            with pytest.raises(ValueError, match="integrity check failed"):
                fetch_remote_artifact(
                    "https://example.com/data.json",
                    expected_hash="deadbeef00",
                )

    def test_fetch_remote_artifact_passes_on_correct_checksum(self):
        from geoprompt.io import fetch_remote_artifact
        from unittest.mock import MagicMock

        data = b"real data"
        expected = hashlib.sha256(data).hexdigest()
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = data

        with patch("geoprompt.io.urlopen", return_value=mock_response):
            result = fetch_remote_artifact(
                "https://example.com/data.json",
                expected_hash=expected,
            )
        assert result == data
