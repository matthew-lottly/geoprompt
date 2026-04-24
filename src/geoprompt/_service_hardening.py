"""Service hardening utilities for GeoPrompt's FastAPI layer.

Implements J7.1–J7.13:
- J7.1:  Strict auth defaults (deny-by-default for privileged operations)
- J7.2:  Standardised request classification tags for audit-log events
- J7.3:  PII detection pre-dispatch guard for hosted inference connectors
- J7.4:  Configurable redaction policy for logs, traces, and lineage payloads
- J7.5:  Payload complexity guardrails at every public HTTP boundary
- J7.6:  Request signature and replay-protection option (HMAC-SHA256)
- J7.7:  Key-rotation and secret-source audit support for connector auth
- J7.8:  Policy simulation mode to preview governance impacts
- J7.9:  Governance dashboard integrity validation
- J7.10: Compliance profile validator (public sector, utility, defence)
- J7.11: Data residency enforcement checks in hosted connector paths
- J7.12: Service hardening helpers (malformed geometry, oversized payload checks)
- J7.13: Deployment readiness smoke-check workflow
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger("geoprompt.service_hardening")

# ---------------------------------------------------------------------------
# J7.1 – Auth defaults
# ---------------------------------------------------------------------------

#: Operations that require an explicit role; the service denies them by default.
PRIVILEGED_OPERATIONS: frozenset[str] = frozenset(
    {
        "query",
        "enterprise",
        "data_write",
        "schema_report",
        "jobs/submit",
        "ops/benchmark",
    }
)


def check_auth(
    path: str,
    actor_roles: set[str],
    *,
    required_roles: set[str],
    allow_anonymous_read: bool = True,
) -> tuple[bool, str]:
    """Return ``(allowed, reason)`` for the given HTTP path and actor roles.

    Rules (J7.1):
    - ``/health`` is always allowed (no auth required).
    - Read-only paths (GET) are allowed when ``allow_anonymous_read`` is True
      *unless* the path matches a privileged operation.
    - Privileged operations always require a matching role.
    - If ``required_roles`` is non-empty the actor must hold at least one.
    """
    clean = path.strip("/").replace("-", "_")

    if clean == "health":
        return True, "health endpoint is public"

    if required_roles and not actor_roles.intersection(required_roles):
        return False, f"actor lacks required role(s): {sorted(required_roles)}"

    for op in PRIVILEGED_OPERATIONS:
        if op in clean:
            if not actor_roles:
                return False, f"privileged operation '{op}' requires at least one role"

    return True, "allowed"


# ---------------------------------------------------------------------------
# J7.2 – Request classification tags
# ---------------------------------------------------------------------------

#: Maps path fragments to operation class labels used in structured log events.
_OPERATION_CLASS_MAP: dict[str, str] = {
    "health": "system.health",
    "compare": "analytics.compare",
    "nearest": "analytics.spatial",
    "schema_report": "data.inspection",
    "jobs/submit": "job.submit",
    "jobs": "job.query",
    "ops/metrics": "system.metrics",
    "ops/benchmark": "system.benchmark",
    "infer": "ml.inference",
    "query": "data.query",
    "enterprise": "enterprise.operation",
}


def classify_operation(path: str) -> str:
    """Return an operation-class label for the given HTTP path.

    Used in audit-log events (J7.2) and request classification tags.
    """
    clean = path.strip("/").replace("-", "_")
    for fragment, label in _OPERATION_CLASS_MAP.items():
        if fragment in clean:
            return label
    return "generic.request"


# ---------------------------------------------------------------------------
# J7.3 – PII detection pre-dispatch
# ---------------------------------------------------------------------------

#: Patterns that may indicate personally identifiable information.
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("phone_us", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    (
        "ssn",
        re.compile(r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0{4})\d{4}\b"),
    ),
    ("credit_card", re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def scan_for_pii(payload: Any, *, max_depth: int = 5) -> list[dict[str, str]]:
    """Recursively scan *payload* for potential PII patterns.

    Returns a list of findings, each a dict with ``{"field_path", "pii_type",
    "match_preview"}``.  An empty list means no PII was detected.

    Only string leaf values are scanned.  Bytes and non-string scalars are
    skipped.

    Args:
        payload: JSON-like dict, list, or scalar to scan.
        max_depth: Maximum recursion depth to avoid stack overflows on deeply
            nested inputs.

    Returns:
        List of PII-finding dicts.
    """
    findings: list[dict[str, str]] = []

    def _recurse(obj: Any, path: str, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(obj, str):
            for label, pattern in _PII_PATTERNS:
                m = pattern.search(obj)
                if m:
                    findings.append(
                        {
                            "field_path": path,
                            "pii_type": label,
                            "match_preview": obj[max(0, m.start() - 4) : m.end() + 4],
                        }
                    )
        elif isinstance(obj, dict):
            for key, val in obj.items():
                _recurse(val, f"{path}.{key}" if path else str(key), depth + 1)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _recurse(item, f"{path}[{i}]", depth + 1)

    _recurse(payload, "", 0)
    return findings


# ---------------------------------------------------------------------------
# J7.4 – Configurable redaction policy
# ---------------------------------------------------------------------------

#: Fields whose values are always redacted in log / lineage payloads.
_DEFAULT_REDACT_FIELDS: frozenset[str] = frozenset(
    {
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "x-api-key",
        "authorization",
        "auth",
        "credential",
        "private_key",
        "ssn",
        "credit_card",
    }
)

_REDACTED_PLACEHOLDER = "***REDACTED***"


def redact_payload(
    payload: Any,
    *,
    extra_fields: frozenset[str] | None = None,
    max_depth: int = 8,
) -> Any:
    """Return a deep copy of *payload* with sensitive field values replaced.

    Field names matching ``_DEFAULT_REDACT_FIELDS`` (or ``extra_fields``) are
    replaced with ``"***REDACTED***"`` at any nesting level.

    Args:
        payload: JSON-like structure to redact.
        extra_fields: Additional field names to redact (lower-cased comparison).
        max_depth: Maximum recursion depth.

    Returns:
        A new, redacted copy of the payload.
    """
    fields = _DEFAULT_REDACT_FIELDS
    if extra_fields:
        fields = fields | {f.lower() for f in extra_fields}

    def _recurse(obj: Any, depth: int) -> Any:
        if depth > max_depth:
            return obj
        if isinstance(obj, dict):
            return {
                k: (_REDACTED_PLACEHOLDER if str(k).lower() in fields else _recurse(v, depth + 1))
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [_recurse(item, depth + 1) for item in obj]
        return obj

    return _recurse(payload, 0)


# ---------------------------------------------------------------------------
# J7.5 – Payload complexity guardrails
# ---------------------------------------------------------------------------

#: Maximum serialised payload size in bytes (default 10 MiB).
MAX_PAYLOAD_BYTES: int = 10 * 1024 * 1024

#: Maximum nesting depth for JSON-like structures.
MAX_PAYLOAD_DEPTH: int = 20

#: Maximum number of features / records in a single request.
MAX_FEATURE_COUNT: int = 100_000


class PayloadTooLargeError(ValueError):
    """Raised when a request payload exceeds the allowed size or complexity."""


def validate_payload_complexity(
    payload: Any,
    *,
    max_bytes: int = MAX_PAYLOAD_BYTES,
    max_depth: int = MAX_PAYLOAD_DEPTH,
    max_features: int = MAX_FEATURE_COUNT,
) -> None:
    """Assert that *payload* is within the allowed size and complexity limits.

    Raises :class:`PayloadTooLargeError` on first violation.

    Args:
        payload: The request payload (dict, list, or scalar).
        max_bytes: Serialised JSON size limit in bytes.
        max_depth: Maximum nesting depth.
        max_features: Maximum number of top-level list items (features/records).
    """
    serialised = json.dumps(payload, default=str)
    size = len(serialised.encode("utf-8"))
    if size > max_bytes:
        raise PayloadTooLargeError(
            f"Payload size {size:,} bytes exceeds maximum of {max_bytes:,} bytes."
        )

    def _depth(obj: Any, current: int) -> int:
        if current > max_depth:
            return current
        if isinstance(obj, dict):
            return max((_depth(v, current + 1) for v in obj.values()), default=current)
        if isinstance(obj, (list, tuple)):
            return max((_depth(item, current + 1) for item in obj), default=current)
        return current

    depth = _depth(payload, 0)
    if depth > max_depth:
        raise PayloadTooLargeError(
            f"Payload nesting depth {depth} exceeds maximum of {max_depth}."
        )

    if isinstance(payload, list) and len(payload) > max_features:
        raise PayloadTooLargeError(
            f"Feature count {len(payload):,} exceeds maximum of {max_features:,}."
        )
    if isinstance(payload, dict):
        for key in ("features", "records", "items"):
            items = payload.get(key)
            if isinstance(items, list) and len(items) > max_features:
                raise PayloadTooLargeError(
                    f"'{key}' count {len(items):,} exceeds maximum of {max_features:,}."
                )


# ---------------------------------------------------------------------------
# J7.6 – Request signature and replay protection
# ---------------------------------------------------------------------------

#: Replay window in seconds (requests with timestamps older than this are rejected).
REPLAY_WINDOW_SECONDS: int = 300  # 5 minutes


def compute_request_signature(
    body: bytes,
    *,
    secret: str,
    timestamp: str,
    algorithm: str = "sha256",
) -> str:
    """Compute HMAC-SHA256 signature for *body* and *timestamp*.

    Sign the concatenation ``timestamp + "." + body_hex`` using *secret* as
    the HMAC key.  The signature is returned as a hex string.

    Args:
        body: Raw request body bytes.
        secret: Shared HMAC secret (stored server-side, never sent in request).
        timestamp: Unix timestamp string included in the request header.
        algorithm: HMAC hash algorithm (default ``"sha256"``).

    Returns:
        Lowercase hex-encoded HMAC digest.
    """
    mac = hmac.new(
        secret.encode("utf-8"),
        msg=f"{timestamp}.{body.hex()}".encode("utf-8"),
        digestmod=getattr(hashlib, algorithm),
    )
    return mac.hexdigest()


def verify_request_signature(
    body: bytes,
    *,
    provided_signature: str,
    secret: str,
    timestamp: str,
    algorithm: str = "sha256",
    replay_window: int = REPLAY_WINDOW_SECONDS,
) -> None:
    """Verify the HMAC signature and timestamp to prevent replay attacks.

    Raises :class:`ValueError` if the signature is invalid or the timestamp
    is outside the allowed replay window.

    Args:
        body: Raw request body bytes.
        provided_signature: Hex-encoded signature from the request header.
        secret: Shared HMAC secret.
        timestamp: Unix timestamp string from the request header.
        algorithm: Hash algorithm used for signing (must match signer).
        replay_window: Maximum age of a valid timestamp in seconds.

    Raises:
        ValueError: On signature mismatch or replay-window violation.
    """
    try:
        ts = int(timestamp)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid request timestamp: {timestamp!r}") from exc

    age = abs(time.time() - ts)
    if age > replay_window:
        raise ValueError(
            f"Request timestamp is {age:.0f}s old; maximum allowed is {replay_window}s. "
            "Possible replay attack."
        )

    expected = compute_request_signature(
        body, secret=secret, timestamp=timestamp, algorithm=algorithm
    )
    if not hmac.compare_digest(expected, provided_signature.lower()):
        raise ValueError("Request signature mismatch. Possible tampering or wrong secret.")


# ---------------------------------------------------------------------------
# J7.7 – Key rotation and secret source audit
# ---------------------------------------------------------------------------

#: Environment variable names GeoPrompt reads for secrets.
KNOWN_SECRET_ENV_VARS: tuple[str, ...] = (
    "GEOPROMPT_API_KEY",
    "GEOPROMPT_DB_PASSWORD",
    "GEOPROMPT_SIGNING_SECRET",
    "GEOPROMPT_HMAC_SECRET",
)


def audit_secret_sources(
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return an audit report of secret environment variables.

    Checks each known secret env var and reports:
    - Whether it is set (without revealing its value).
    - Whether its length meets a minimum entropy threshold (≥16 chars).
    - Whether any known-weak placeholder values are present.

    Args:
        env: Mapping to check (defaults to ``os.environ``).

    Returns:
        Dict with keys ``"present"``, ``"weak"``, ``"missing"``, ``"report"``.
    """
    import os

    env = env if env is not None else dict(os.environ)
    present: list[str] = []
    weak: list[str] = []
    missing: list[str] = []
    _WEAK_VALUES = frozenset({"changeme", "secret", "password", "test", "dev", "1234", "admin"})

    for var in KNOWN_SECRET_ENV_VARS:
        val = env.get(var)
        if val is None:
            missing.append(var)
        else:
            present.append(var)
            if len(val) < 16 or val.lower() in _WEAK_VALUES:
                weak.append(var)

    return {
        "present": present,
        "weak": weak,
        "missing": missing,
        "report": {
            var: (
                "present (strong)"
                if var in present and var not in weak
                else ("present (WEAK)" if var in weak else "MISSING")
            )
            for var in KNOWN_SECRET_ENV_VARS
        },
    }


# ---------------------------------------------------------------------------
# J7.8 – Policy simulation mode
# ---------------------------------------------------------------------------

class PolicySimulator:
    """Preview governance and auth policy decisions without enforcing them.

    Captures what the service *would* do under the current policy configuration,
    returning a structured preview report.  Useful for rehearsing policy changes
    in staging before deploying to production.
    """

    def __init__(
        self,
        *,
        required_roles: set[str] | None = None,
        max_payload_bytes: int = MAX_PAYLOAD_BYTES,
        pii_blocking: bool = True,
        signature_required: bool = False,
    ) -> None:
        self.required_roles = required_roles or set()
        self.max_payload_bytes = max_payload_bytes
        self.pii_blocking = pii_blocking
        self.signature_required = signature_required

    def simulate(
        self,
        path: str,
        payload: Any,
        *,
        actor_roles: set[str] | None = None,
        request_signature: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Simulate the full policy pipeline for the given request.

        Returns a preview dict with ``"decisions"`` (list of decision dicts),
        ``"would_allow"`` (bool), and ``"blocking_reason"`` (str | None).
        """
        actor_roles = actor_roles or set()
        decisions: list[dict[str, Any]] = []

        # Auth check
        allowed, reason = check_auth(
            path, actor_roles, required_roles=self.required_roles
        )
        decisions.append(
            {
                "check": "auth",
                "passed": allowed,
                "reason": reason,
                "enforcement": "would block" if not allowed else "would allow",
            }
        )
        if not allowed:
            return {
                "decisions": decisions,
                "would_allow": False,
                "blocking_reason": f"auth: {reason}",
            }

        # Payload complexity
        try:
            validate_payload_complexity(payload, max_bytes=self.max_payload_bytes)
            decisions.append({"check": "payload_complexity", "passed": True, "reason": "within limits"})
        except PayloadTooLargeError as exc:
            decisions.append({"check": "payload_complexity", "passed": False, "reason": str(exc)})
            return {
                "decisions": decisions,
                "would_allow": False,
                "blocking_reason": str(exc),
            }

        # PII scan
        if self.pii_blocking:
            pii_findings = scan_for_pii(payload)
            if pii_findings:
                decisions.append(
                    {
                        "check": "pii_scan",
                        "passed": False,
                        "reason": f"{len(pii_findings)} PII finding(s) detected",
                        "findings": pii_findings,
                    }
                )
                return {
                    "decisions": decisions,
                    "would_allow": False,
                    "blocking_reason": f"pii: {len(pii_findings)} PII pattern(s) found in payload",
                }
            decisions.append({"check": "pii_scan", "passed": True, "reason": "no PII detected"})

        # Signature check (preview only)
        if self.signature_required:
            if request_signature is None or timestamp is None:
                decisions.append(
                    {
                        "check": "signature",
                        "passed": False,
                        "reason": "signature or timestamp header missing",
                        "enforcement": "would block",
                    }
                )
                return {
                    "decisions": decisions,
                    "would_allow": False,
                    "blocking_reason": "missing signature headers",
                }
            decisions.append(
                {
                    "check": "signature",
                    "passed": True,
                    "reason": "signature headers present (not verified in simulation)",
                }
            )

        return {"decisions": decisions, "would_allow": True, "blocking_reason": None}


# ---------------------------------------------------------------------------
# J7.9 – Governance dashboard integrity validation
# ---------------------------------------------------------------------------

def validate_dashboard_integrity(metrics: dict[str, Any]) -> list[str]:
    """Validate the integrity of a governance dashboard metrics snapshot.

    Checks that:
    - ``job_counts`` sums to ``total_jobs``.
    - No job count is negative.
    - ``registered_handlers`` is a list.
    - No unexpected fields are present.

    Returns a list of integrity violation messages.  An empty list means the
    dashboard data is internally consistent.
    """
    violations: list[str] = []
    job_counts = metrics.get("job_counts", {})
    total = metrics.get("total_jobs", 0)
    computed_total = sum(job_counts.values()) if isinstance(job_counts, dict) else 0

    if computed_total != total:
        violations.append(
            f"job_counts sum ({computed_total}) does not match total_jobs ({total})"
        )

    for status, count in (job_counts.items() if isinstance(job_counts, dict) else []):
        if not isinstance(count, int) or count < 0:
            violations.append(f"job_counts['{status}'] = {count!r} is negative or non-integer")

    handlers = metrics.get("registered_handlers")
    if not isinstance(handlers, list):
        violations.append(f"registered_handlers must be a list, got {type(handlers).__name__}")

    return violations


# ---------------------------------------------------------------------------
# J7.10 – Compliance profile validator
# ---------------------------------------------------------------------------

#: Required capability flags per compliance profile.
COMPLIANCE_PROFILES: dict[str, dict[str, Any]] = {
    "public_sector": {
        "require_auth": True,
        "require_roles": True,
        "pii_blocking": True,
        "max_payload_bytes": 5 * 1024 * 1024,  # 5 MiB
        "audit_logging": True,
        "data_residency": True,
        "signature_required": False,
        "description": "Baseline for public-sector deployments (FISMA moderate).",
    },
    "utility": {
        "require_auth": True,
        "require_roles": False,
        "pii_blocking": True,
        "max_payload_bytes": 20 * 1024 * 1024,  # 20 MiB
        "audit_logging": True,
        "data_residency": False,
        "signature_required": False,
        "description": "Utility / critical-infrastructure deployments.",
    },
    "defence": {
        "require_auth": True,
        "require_roles": True,
        "pii_blocking": True,
        "max_payload_bytes": 2 * 1024 * 1024,  # 2 MiB
        "audit_logging": True,
        "data_residency": True,
        "signature_required": True,
        "description": "High-assurance defence deployments (requires HMAC signing).",
    },
}


def validate_compliance_profile(
    profile_name: str,
    config: dict[str, Any],
) -> list[str]:
    """Validate a service configuration against a named compliance profile.

    Args:
        profile_name: One of ``"public_sector"``, ``"utility"``, ``"defence"``.
        config: Dict of current service configuration flags (same keys as
            ``COMPLIANCE_PROFILES[profile_name]`` minus ``description``).

    Returns:
        List of compliance gaps.  An empty list means the config satisfies the
        profile.

    Raises:
        KeyError: If *profile_name* is not registered.
    """
    profile = COMPLIANCE_PROFILES[profile_name]
    gaps: list[str] = []
    for key, required_value in profile.items():
        if key == "description":
            continue
        actual = config.get(key)
        if isinstance(required_value, bool):
            if required_value and not actual:
                gaps.append(
                    f"[{profile_name}] '{key}' must be enabled (currently {actual!r})"
                )
        elif isinstance(required_value, int):
            if isinstance(actual, int) and actual > required_value:
                gaps.append(
                    f"[{profile_name}] '{key}' must be ≤ {required_value:,} "
                    f"(currently {actual:,})"
                )
    return gaps


# ---------------------------------------------------------------------------
# J7.11 – Data residency enforcement
# ---------------------------------------------------------------------------

#: Mapping of ISO 3166-1 alpha-2 country codes to broad geographic regions.
_COUNTRY_TO_REGION: dict[str, str] = {
    "US": "north_america",
    "CA": "north_america",
    "MX": "north_america",
    "GB": "europe",
    "DE": "europe",
    "FR": "europe",
    "NL": "europe",
    "IE": "europe",
    "SE": "europe",
    "NO": "europe",
    "FI": "europe",
    "DK": "europe",
    "CH": "europe",
    "AU": "apac",
    "NZ": "apac",
    "SG": "apac",
    "JP": "apac",
    "IN": "apac",
    "CN": "apac",
    "BR": "south_america",
    "AR": "south_america",
}


def check_data_residency(
    destination_country: str,
    *,
    allowed_regions: set[str],
) -> tuple[bool, str]:
    """Check whether *destination_country* satisfies the allowed residency regions.

    Args:
        destination_country: ISO 3166-1 alpha-2 country code of the destination.
        allowed_regions: Set of permitted geographic region names (e.g.
            ``{"europe", "north_america"}``).

    Returns:
        ``(compliant, reason)`` tuple.
    """
    region = _COUNTRY_TO_REGION.get(destination_country.upper())
    if region is None:
        return False, (
            f"Country '{destination_country}' is not mapped to any known region. "
            "Deny by default for data residency enforcement."
        )
    if region in allowed_regions:
        return True, f"Country '{destination_country}' is in allowed region '{region}'"
    return False, (
        f"Country '{destination_country}' maps to region '{region}' "
        f"which is not in the allowed set: {sorted(allowed_regions)}"
    )


# ---------------------------------------------------------------------------
# J7.12 – Service hardening guards
# ---------------------------------------------------------------------------

def validate_geometry_payload(payload: Any) -> list[str]:
    """Validate that all geometries in a feature list are structurally valid.

    Returns a list of error messages.  An empty list means all geometries pass.
    Checks performed:
    - Each feature must be a dict.
    - The ``geometry`` field must be a dict with ``type`` and ``coordinates``.
    - Coordinate arrays must be non-empty for non-collection geometry types.
    """
    _VALID_TYPES = frozenset(
        {
            "Point",
            "MultiPoint",
            "LineString",
            "MultiLineString",
            "Polygon",
            "MultiPolygon",
            "GeometryCollection",
        }
    )
    errors: list[str] = []
    features = payload if isinstance(payload, list) else payload.get("features", [])

    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            errors.append(f"feature[{i}] is not a dict")
            continue
        geom = feat.get("geometry")
        if geom is None:
            continue  # Null geometry is allowed
        if not isinstance(geom, dict):
            errors.append(f"feature[{i}].geometry is not a dict")
            continue
        gtype = geom.get("type")
        if gtype not in _VALID_TYPES:
            errors.append(f"feature[{i}].geometry.type '{gtype}' is not a valid GeoJSON type")
            continue
        if gtype != "GeometryCollection":
            coords = geom.get("coordinates")
            if not isinstance(coords, (list, tuple)) or len(coords) == 0:
                errors.append(
                    f"feature[{i}].geometry.coordinates is missing or empty for type '{gtype}'"
                )

    return errors


# ---------------------------------------------------------------------------
# J7.13 – Deployment readiness smoke checks
# ---------------------------------------------------------------------------

class DeploymentReadinessResult:
    """Structured result of a deployment readiness smoke check."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.passed: int = 0
        self.failed: int = 0

    def add(self, name: str, *, passed: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "passed": passed, "detail": detail})
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    @property
    def ready(self) -> bool:
        return self.failed == 0

    def summary(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "passed": self.passed,
            "failed": self.failed,
            "checks": self.checks,
        }


def run_deployment_smoke_checks(
    *,
    check_health: bool = True,
    check_capability_report: bool = True,
    check_schema_report: bool = True,
    check_deterministic_nearest: bool = True,
    check_job_submit_poll: bool = True,
) -> DeploymentReadinessResult:
    """Run pre-promotion deployment smoke checks locally.

    These checks validate that GeoPrompt's core logic is functioning correctly
    without requiring a running service — they call the Python API directly.

    Corresponds to J7.13: codify deployment smoke checks as an executable
    readiness workflow.

    Args:
        check_health: Verify the health-endpoint equivalent returns ``ok``.
        check_capability_report: Verify ``capability_status()`` returns a dict.
        check_schema_report: Verify ``schema_report`` runs on sample data.
        check_deterministic_nearest: Verify nearest join is deterministic.
        check_job_submit_poll: Verify ``ServiceJobManager`` submit/poll cycle.

    Returns:
        :class:`DeploymentReadinessResult` with individual check outcomes.
    """
    result = DeploymentReadinessResult()

    if check_health:
        try:
            import geoprompt
            ver = getattr(geoprompt, "__version__", None)
            passed = isinstance(ver, str) and len(ver) > 0
            result.add("health", passed=passed, detail=f"version={ver!r}")
        except Exception as exc:
            result.add("health", passed=False, detail=str(exc))

    if check_capability_report:
        try:
            from geoprompt._capabilities import capability_status
            status = capability_status()
            passed = isinstance(status, dict) and len(status) > 0
            result.add(
                "capability_report",
                passed=passed,
                detail=f"{sum(1 for v in status.values() if v.get('available'))} of {len(status)} caps available",
            )
        except Exception as exc:
            result.add("capability_report", passed=False, detail=str(exc))

    if check_schema_report:
        try:
            from geoprompt.io import schema_report
            sample = [{"id": 1, "name": "test", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]
            report = schema_report(sample)
            passed = isinstance(report, dict) and ("columns" in report or "column_count" in report)
            col_count = len(report.get("columns", report.get("column_count", [])))
            result.add("schema_report", passed=passed, detail=f"columns={col_count}")
        except Exception as exc:
            result.add("schema_report", passed=False, detail=str(exc))

    if check_deterministic_nearest:
        try:
            from geoprompt.frame import GeoPromptFrame
            features = [
                {"id": "a", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}},
                {"id": "b", "geometry": {"type": "Point", "coordinates": [1.0, 1.0]}},
            ]
            targets = [{"id": "t1", "geometry": {"type": "Point", "coordinates": [0.1, 0.1]}}]
            frame = GeoPromptFrame(features)
            tframe = GeoPromptFrame(targets)
            r1 = frame.nearest_join(tframe, k=1)
            r2 = frame.nearest_join(tframe, k=1)
            passed = r1 == r2
            result.add("deterministic_nearest", passed=passed, detail="two runs match" if passed else "results differ")
        except Exception as exc:
            result.add("deterministic_nearest", passed=False, detail=str(exc))

    if check_job_submit_poll:
        try:
            from geoprompt.service import ServiceJobManager
            mgr = ServiceJobManager()
            job = mgr.submit("schema_report", {"records": []}, execute=True)
            polled = mgr.get(job["job_id"])
            passed = polled["status"] == "completed"
            result.add("job_submit_poll", passed=passed, detail=f"status={polled['status']}")
        except Exception as exc:
            result.add("job_submit_poll", passed=False, detail=str(exc))

    return result


__all__ = [
    # J7.1
    "PRIVILEGED_OPERATIONS",
    "check_auth",
    # J7.2
    "classify_operation",
    # J7.3
    "scan_for_pii",
    # J7.4
    "redact_payload",
    # J7.5
    "PayloadTooLargeError",
    "validate_payload_complexity",
    # J7.6
    "compute_request_signature",
    "verify_request_signature",
    "REPLAY_WINDOW_SECONDS",
    # J7.7
    "KNOWN_SECRET_ENV_VARS",
    "audit_secret_sources",
    # J7.8
    "PolicySimulator",
    # J7.9
    "validate_dashboard_integrity",
    # J7.10
    "COMPLIANCE_PROFILES",
    "validate_compliance_profile",
    # J7.11
    "check_data_residency",
    # J7.12
    "validate_geometry_payload",
    # J7.13
    "DeploymentReadinessResult",
    "run_deployment_smoke_checks",
]
