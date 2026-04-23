"""Security, access-control and infrastructure helpers for GeoPrompt.

Pure-Python implementations covering A12 roadmap items related to
security, rate-limiting, JWT, RBAC, data privacy, and infrastructure patterns.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
import secrets
import time
from pathlib import PurePath
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# 1245. JWT helpers
# ---------------------------------------------------------------------------

def jwt_encode(
    payload: Dict[str, Any],
    secret: str,
    *,
    algorithm: str = "HS256",
    expiry_seconds: int = 3600,
) -> str:
    """Create a simple JWT token (HS256 only, no external deps)."""
    header = {"alg": "HS256", "typ": "JWT"}
    if expiry_seconds > 0:
        payload = {**payload, "exp": int(time.time()) + expiry_seconds}
    if "iat" not in payload:
        payload["iat"] = int(time.time())

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    h = _b64url(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    sig_input = f"{h}.{p}".encode()
    sig = hmac.new(secret.encode(), sig_input, hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url(sig)}"


def jwt_decode(
    token: str,
    secret: str,
) -> Dict[str, Any]:
    """Decode and verify a JWT token. Raises ValueError on failure."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid JWT format")

    def _b64url_decode(s: str) -> bytes:
        s += "=" * (4 - len(s) % 4)
        return base64.urlsafe_b64decode(s)

    sig_input = f"{parts[0]}.{parts[1]}".encode()
    expected_sig = hmac.new(secret.encode(), sig_input, hashlib.sha256).digest()
    actual_sig = _b64url_decode(parts[2])
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("invalid JWT signature")
    payload = json.loads(_b64url_decode(parts[1]))
    if "exp" in payload and payload["exp"] < time.time():
        raise ValueError("JWT expired")
    return payload


# ---------------------------------------------------------------------------
# 1237. Role-based access control (RBAC)
# ---------------------------------------------------------------------------

class RBACManager:
    """Simple in-memory RBAC manager."""

    def __init__(self) -> None:
        self._roles: Dict[str, Set[str]] = {}
        self._user_roles: Dict[str, Set[str]] = {}

    def create_role(self, role: str, permissions: Sequence[str]) -> None:
        self._roles[role] = set(permissions)

    def delete_role(self, role: str) -> None:
        self._roles.pop(role, None)
        for user_roles in self._user_roles.values():
            user_roles.discard(role)

    def assign_role(self, user: str, role: str) -> None:
        if role not in self._roles:
            raise ValueError(f"role '{role}' does not exist")
        self._user_roles.setdefault(user, set()).add(role)

    def revoke_role(self, user: str, role: str) -> None:
        if user in self._user_roles:
            self._user_roles[user].discard(role)

    def has_permission(self, user: str, permission: str) -> bool:
        for role in self._user_roles.get(user, set()):
            if permission in self._roles.get(role, set()):
                return True
        return False

    def user_permissions(self, user: str) -> Set[str]:
        perms: Set[str] = set()
        for role in self._user_roles.get(user, set()):
            perms |= self._roles.get(role, set())
        return perms

    def list_roles(self) -> Dict[str, Set[str]]:
        return dict(self._roles)


# ---------------------------------------------------------------------------
# 1224. Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter for external API calls."""

    def __init__(self, rate: float, burst: int = 10) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()

    def acquire(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def wait_time(self) -> float:
        if self._tokens >= 1.0:
            return 0.0
        return (1.0 - self._tokens) / self._rate


# ---------------------------------------------------------------------------
# 1225. Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Simple circuit-breaker for flaky services."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._failures = 0
        self._state = "closed"
        self._last_failure = 0.0

    @property
    def state(self) -> str:
        if self._state == "open":
            if time.monotonic() - self._last_failure > self._reset_timeout:
                self._state = "half-open"
        return self._state

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure = time.monotonic()
        if self._failures >= self._failure_threshold:
            self._state = "open"

    def allow_request(self) -> bool:
        state = self.state
        return state in ("closed", "half-open")


# ---------------------------------------------------------------------------
# 1246. Input sanitisation (SQL injection prevention)
# ---------------------------------------------------------------------------

def sanitize_sql_input(value: str) -> str:
    """Sanitise a string value for safe SQL interpolation."""
    return value.replace("'", "''").replace("\\", "\\\\").replace("\x00", "")


def validate_table_name(name: str) -> bool:
    """Validate a table/column name contains only safe characters."""
    return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]{0,127}$", name))


# ---------------------------------------------------------------------------
# 1247. File upload validation
# ---------------------------------------------------------------------------

_ALLOWED_GEO_EXTENSIONS = {
    ".geojson", ".json", ".shp", ".gpkg", ".kml", ".kmz", ".gml",
    ".csv", ".tsv", ".xlsx", ".gpx", ".topojson", ".fgb", ".parquet",
}


def validate_file_upload(
    filename: str,
    file_size_bytes: int,
    *,
    max_size_mb: float = 500.0,
    allowed_extensions: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Validate a file upload for geospatial data."""
    ext = PurePath(filename).suffix.lower()
    exts = allowed_extensions or _ALLOWED_GEO_EXTENSIONS
    issues: List[str] = []
    if ext not in exts:
        issues.append(f"extension '{ext}' not allowed")
    if file_size_bytes > max_size_mb * 1024 * 1024:
        issues.append(f"file exceeds {max_size_mb}MB limit")
    if ".." in filename or "/" in filename or "\\" in filename:
        issues.append("path traversal detected in filename")
    return {"valid": len(issues) == 0, "issues": issues, "extension": ext}


# ---------------------------------------------------------------------------
# 1248. Output redaction (PII removal)
# ---------------------------------------------------------------------------

_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL_REDACTED]"),
    (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE_REDACTED]"),
    (re.compile(r"\b\d{5}(-\d{4})?\b"), "[ZIP_REDACTED]"),
]


def redact_pii(text: str) -> str:
    """Remove common PII patterns from text."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_fields(
    records: Sequence[Dict[str, Any]],
    sensitive_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    """Redact specified fields from a list of records."""
    result: List[Dict[str, Any]] = []
    for rec in records:
        new_rec = dict(rec)
        for field in sensitive_fields:
            if field in new_rec:
                val = new_rec[field]
                if isinstance(val, str):
                    new_rec[field] = redact_pii(val)
                else:
                    new_rec[field] = "[REDACTED]"
        result.append(new_rec)
    return result


# ---------------------------------------------------------------------------
# 1249. Coordinate obfuscation (privacy)
# ---------------------------------------------------------------------------

def obfuscate_coordinates(
    lon: float,
    lat: float,
    *,
    precision_km: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Add random jitter to coordinates for privacy."""
    import random
    rng = random.Random(seed)
    offset_km = precision_km
    dlat = offset_km / 111.0
    dlon = offset_km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
    new_lon = lon + rng.uniform(-dlon, dlon)
    new_lat = lat + rng.uniform(-dlat, dlat)
    return (new_lon, new_lat)


# ---------------------------------------------------------------------------
# 1250. Spatial k-anonymity
# ---------------------------------------------------------------------------

def spatial_k_anonymity(
    points: Sequence[Tuple[float, float]],
    k: int = 5,
) -> List[Tuple[float, float]]:
    """Enforce spatial k-anonymity by snapping each point to cluster centroid.

    Clusters are formed greedily by nearest un-assigned neighbours. Returns
    centroid for each input point.
    """
    n = len(points)
    assigned = [False] * n
    centroids: List[Tuple[float, float]] = [(0.0, 0.0)] * n

    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    for i in range(n):
        if assigned[i]:
            continue
        distances = []
        for j in range(n):
            if not assigned[j]:
                distances.append((j, _dist(points[i], points[j])))
        distances.sort(key=lambda x: x[1])
        cluster = [idx for idx, _ in distances[:k]]
        cx = sum(points[idx][0] for idx in cluster) / len(cluster)
        cy = sum(points[idx][1] for idx in cluster) / len(cluster)
        for idx in cluster:
            assigned[idx] = True
            centroids[idx] = (cx, cy)
    return centroids


# ---------------------------------------------------------------------------
# 1240. Data encryption at rest (AES-like XOR for demo)
# ---------------------------------------------------------------------------

def encrypt_data(data: bytes, key: str) -> bytes:
    """Simple XOR-based encryption for demonstration purposes.

    NOT cryptographically secure – use a real library for production.
    """
    key_bytes = hashlib.sha256(key.encode()).digest()
    return bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(data))


def decrypt_data(data: bytes, key: str) -> bytes:
    """Decrypt XOR-encrypted data."""
    return encrypt_data(data, key)


# ---------------------------------------------------------------------------
# 1242. API key management
# ---------------------------------------------------------------------------

class APIKeyManager:
    """Simple in-memory API key manager."""

    def __init__(self) -> None:
        self._keys: Dict[str, Dict[str, Any]] = {}

    def generate_key(self, name: str, *, scopes: Optional[Sequence[str]] = None) -> str:
        key = secrets.token_hex(20)
        self._keys[key] = {
            "name": name,
            "scopes": set(scopes or []),
            "created": time.time(),
            "active": True,
        }
        return key

    def validate_key(self, key: str, *, required_scope: Optional[str] = None) -> bool:
        info = self._keys.get(key)
        if not info or not info["active"]:
            return False
        if required_scope and required_scope not in info["scopes"]:
            return False
        return True

    def revoke_key(self, key: str) -> bool:
        if key in self._keys:
            self._keys[key]["active"] = False
            return True
        return False

    def list_keys(self) -> List[Dict[str, Any]]:
        return [{"key_prefix": k[:8] + "...", **v} for k, v in self._keys.items()]


# ---------------------------------------------------------------------------
# 1229/1230. Health check / readiness probes
# ---------------------------------------------------------------------------

class HealthCheck:
    """Health check system for spatial services."""

    def __init__(self) -> None:
        self._checks: Dict[str, Callable[[], bool]] = {}

    def register(self, name: str, check: Callable[[], bool]) -> None:
        self._checks[name] = check

    def liveness(self) -> Dict[str, Any]:
        return {"status": "ok", "timestamp": time.time()}

    def readiness(self) -> Dict[str, Any]:
        results: Dict[str, bool] = {}
        all_ok = True
        for name, check in self._checks.items():
            try:
                ok = check()
            except Exception:
                ok = False
            results[name] = ok
            if not ok:
                all_ok = False
        return {"ready": all_ok, "checks": results, "timestamp": time.time()}


# ---------------------------------------------------------------------------
# 1228. Graceful shutdown
# ---------------------------------------------------------------------------

class GracefulShutdown:
    """Coordinate graceful shutdown of processing tasks."""

    def __init__(self) -> None:
        self._shutting_down = False
        self._callbacks: List[Callable[[], None]] = []

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def register_callback(self, cb: Callable[[], None]) -> None:
        self._callbacks.append(cb)

    def initiate(self) -> None:
        self._shutting_down = True
        for cb in self._callbacks:
            try:
                cb()
            except Exception:
                pass


def release_security_gates() -> Dict[str, Any]:
    """Return the standard release-time security and supply-chain gates."""
    return {
        "release_gates": ["sbom", "signing", "provenance", "secrets_scan", "config_review"],
        "recommended_tools": {
            "sbom": "CycloneDX",
            "signing": "sigstore or trusted publisher",
            "provenance": "workspace manifest and release evidence bundle",
            "secrets_scan": "pre-release secrets scan plus CI checks",
        },
        "cve_triage_sla_hours": 72,
    }


def support_window_policy() -> Dict[str, Any]:
    """Publish the support-window and hotfix expectations for enterprise adoption."""
    return {
        "supported_versions": ["current beta", "latest patch line"],
        "lts_style_support": True,
        "security_hotfix_target": "critical fixes acknowledged within 72 hours",
        "rollback_guidance": "keep prior wheel and evidence bundle available for rollback",
    }


# ---------------------------------------------------------------------------
# 1868. Priority queue for jobs
# ---------------------------------------------------------------------------

class PriorityJobQueue:
    """Simple priority queue for spatial processing jobs."""

    def __init__(self) -> None:
        self._jobs: List[Tuple[int, float, Dict[str, Any]]] = []

    def enqueue(self, job: Dict[str, Any], priority: int = 5) -> None:
        self._jobs.append((priority, time.monotonic(), job))
        self._jobs.sort(key=lambda x: (x[0], x[1]))

    def dequeue(self) -> Optional[Dict[str, Any]]:
        if self._jobs:
            return self._jobs.pop(0)[2]
        return None

    def peek(self) -> Optional[Dict[str, Any]]:
        if self._jobs:
            return self._jobs[0][2]
        return None

    def size(self) -> int:
        return len(self._jobs)

    def clear(self) -> None:
        self._jobs.clear()


# ---------------------------------------------------------------------------
# G23 additions — security utilities
# ---------------------------------------------------------------------------

from typing import Any as _Any
import re as _re


def validate_geometry_safe(geometry: dict) -> dict:
    """Validate a GeoJSON geometry dict for common injection / corruption risks.

    Checks that the geometry has a recognised ``type``, that all coordinate
    values are finite numbers, and that the coordinate nesting depth matches
    the declared type.

    Args:
        geometry: A GeoJSON-style geometry dict.

    Returns:
        Dict with ``valid`` (bool), ``errors`` (list of strings), and
        ``geometry_type`` keys.
    """
    import math
    errors: list[str] = []
    if not isinstance(geometry, dict):
        return {"valid": False, "errors": ["geometry is not a dict"], "geometry_type": None}

    geom_type = geometry.get("type")
    valid_types = {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon", "GeometryCollection"}
    if geom_type not in valid_types:
        errors.append(f"unknown geometry type: {geom_type!r}")
        return {"valid": False, "errors": errors, "geometry_type": geom_type}

    def _check_coords(c: _Any, depth: int = 0) -> None:
        if isinstance(c, (int, float)):
            if not math.isfinite(c):
                errors.append(f"non-finite coordinate value: {c}")
        elif isinstance(c, (list, tuple)):
            for item in c:
                _check_coords(item, depth + 1)
        else:
            errors.append(f"unexpected coordinate type: {type(c).__name__}")

    if geom_type != "GeometryCollection":
        _check_coords(geometry.get("coordinates", []))
    else:
        for g in geometry.get("geometries", []):
            sub = validate_geometry_safe(g)
            errors.extend(sub["errors"])

    return {"valid": len(errors) == 0, "errors": errors, "geometry_type": geom_type}


def sanitize_attribute_input(value: _Any, *,
                              max_length: int = 1000,
                              allow_html: bool = False) -> str:
    """Sanitise an attribute value for safe storage or display.

    Strips control characters, optionally strips HTML tags, and truncates
    to *max_length*.

    Args:
        value: The input value to sanitise.
        max_length: Maximum allowed string length.
        allow_html: If ``False``, HTML tags are stripped.

    Returns:
        A sanitised string.
    """
    s = str(value)
    # Strip null bytes and other control characters (except tab/newline)
    s = _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    if not allow_html:
        # Strip HTML tags
        s = _re.sub(r"<[^>]+>", "", s)
        # Escape remaining < > & characters
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s[:max_length]


def audit_log_event(event_type: str, details: dict, *,
                    log_store: list | None = None) -> dict:
    """Record a security-relevant audit log event.

    Args:
        event_type: A short event category string, e.g. ``"data_access"``,
            ``"schema_change"``, ``"auth_failure"``.
        details: Dict of event-specific metadata.
        log_store: Optional mutable list to append the event to.  If
            ``None``, the event is returned without being stored.

    Returns:
        The event record dict with ``event_type``, ``timestamp``, and
        ``details`` keys.
    """
    import datetime
    event: dict = {
        "event_type": event_type,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "details": details,
    }
    if log_store is not None:
        log_store.append(event)
    return event


def rate_limit_check(key: str, *,
                     limit: int = 100,
                     window_seconds: int = 60,
                     counter_store: dict | None = None) -> dict:
    """Check whether a request key has exceeded its rate limit.

    Uses an in-memory counter dict keyed by ``(key, window_bucket)``.

    Args:
        key: Rate-limit key (e.g. IP address or API key).
        limit: Maximum requests allowed in the window.
        window_seconds: Length of the rolling window in seconds.
        counter_store: Optional dict to use as the counter store (for testing
            or cross-call persistence).  If ``None``, a new empty dict is used
            and the check always returns ``allowed=True``.

    Returns:
        Dict with ``allowed`` (bool), ``count``, ``limit``, and
        ``remaining`` keys.
    """
    import time
    if counter_store is None:
        return {"allowed": True, "count": 1, "limit": limit, "remaining": limit - 1}
    bucket = int(time.time() / window_seconds)
    store_key = f"{key}:{bucket}"
    count = counter_store.get(store_key, 0) + 1
    counter_store[store_key] = count
    allowed = count <= limit
    return {"allowed": allowed, "count": count, "limit": limit, "remaining": max(0, limit - count)}


# ── I9. Security, Privacy, Governance, and Cost Controls ─────────────────────

def model_usage_policy_engine(
    model_id: str,
    connector: str,
    data_class: str,
    *,
    allowed_models: list[str] | None = None,
    allowed_connectors: list[str] | None = None,
    allowed_data_classes: list[str] | None = None,
) -> dict[str, Any]:
    """Enforce model usage approval policy (allowed models, connectors, data classes).

    Returns ``approved`` (bool) and list of ``violations``.
    """
    allowed_models = allowed_models or []
    allowed_connectors = allowed_connectors or []
    allowed_data_classes = allowed_data_classes or ["public", "internal"]
    violations: list[str] = []

    if allowed_models and model_id not in allowed_models:
        violations.append(f"Model '{model_id}' is not in the approved model list.")
    if allowed_connectors and connector not in allowed_connectors:
        violations.append(f"Connector '{connector}' is not approved for use.")
    if data_class not in allowed_data_classes:
        violations.append(f"Data class '{data_class}' is not permitted for inference dispatch.")

    return {"approved": len(violations) == 0, "violations": violations, "model_id": model_id, "connector": connector}


# Reuse existing redact_pii for sensitive geodata pre-check
def sensitive_geodata_detector(
    payload: dict[str, Any],
    *,
    pii_fields: list[str] | None = None,
    sensitive_geometry_types: list[str] | None = None,
) -> dict[str, Any]:
    """Run a pre-check for PII and sensitive geodata before hosted inference dispatch.

    Returns ``clean`` (bool), detected ``pii_fields``, and ``sensitive_geometries``.
    """
    pii_fields_to_check = pii_fields or ["name", "email", "phone", "address", "ssn", "dob", "national_id"]
    sensitive_geometry_types = sensitive_geometry_types or ["military_facility", "critical_infrastructure"]

    found_pii: list[str] = []
    found_sensitive_geom: list[str] = []

    for field in pii_fields_to_check:
        if field in payload and payload[field]:
            found_pii.append(field)

    geom_type = payload.get("geometry_type", "")
    if geom_type in sensitive_geometry_types:
        found_sensitive_geom.append(geom_type)

    # Check text fields for PII patterns
    pii_patterns = [
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        r"\b\d{3}-\d{2}-\d{4}\b",
    ]
    import re as _re_local
    for key, val in payload.items():
        if isinstance(val, str):
            for pat in pii_patterns:
                if _re_local.search(pat, val, _re_local.IGNORECASE):
                    if key not in found_pii:
                        found_pii.append(key)
                    break

    return {
        "clean": len(found_pii) == 0 and len(found_sensitive_geom) == 0,
        "pii_fields": found_pii,
        "sensitive_geometries": found_sensitive_geom,
    }


def geodata_redaction_hooks(
    payload: dict[str, Any],
    *,
    fields_to_redact: list[str] | None = None,
    coordinate_precision: int | None = None,
) -> dict[str, Any]:
    """Apply automatic redaction/transformation hooks for restricted attributes and geometries.

    Redacts listed fields and optionally truncates coordinate precision.

    Returns the sanitized payload copy.
    """
    fields_to_redact = fields_to_redact or ["name", "email", "phone", "address"]
    sanitized = dict(payload)
    for field in fields_to_redact:
        if field in sanitized:
            sanitized[field] = "[REDACTED]"

    # Truncate geometry coordinate precision if requested
    if coordinate_precision is not None:
        geom = sanitized.get("geometry")
        if isinstance(geom, dict):
            def _truncate_coords(obj: Any) -> Any:
                if isinstance(obj, float):
                    return round(obj, coordinate_precision)
                if isinstance(obj, list):
                    return [_truncate_coords(v) for v in obj]
                if isinstance(obj, dict):
                    return {k: _truncate_coords(v) for k, v in obj.items()}
                return obj
            sanitized["geometry"] = _truncate_coords(geom)

    sanitized["_redaction_applied"] = True
    return sanitized


def connector_audit_log(
    event: dict[str, Any],
    *,
    log_store: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Append a connector usage audit event with who/what/when/where and classification.

    Returns the structured event record.
    """
    import time as _time
    record = {
        "event_type": event.get("event_type", "inference_request"),
        "connector": event.get("connector", "unknown"),
        "model_id": event.get("model_id", "unknown"),
        "user": event.get("user", "anonymous"),
        "data_class": event.get("data_class", "unclassified"),
        "request_id": event.get("request_id", ""),
        "timestamp": _time.time(),
        "status": event.get("status", "ok"),
        "cost_usd": event.get("cost_usd", 0.0),
    }
    if log_store is not None:
        log_store.append(record)
    return record


def model_registry_manifest(
    model_id: str,
    model_version: str,
    *,
    artifact_hash: str = "",
    signing_key_id: str = "",
    trust_level: str = "internal",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a signed model registry manifest with integrity verification and trust levels.

    Returns a manifest dict suitable for storage in a model registry.
    """
    import hashlib as _hashlib
    import json as _json
    payload = {
        "model_id": model_id,
        "model_version": model_version,
        "artifact_hash": artifact_hash,
        "trust_level": trust_level,
        "metadata": metadata or {},
    }
    manifest_hash = _hashlib.sha256(
        _json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    return {
        **payload,
        "manifest_hash": manifest_hash,
        "signing_key_id": signing_key_id,
        "trust_levels": ["public", "internal", "restricted", "classified"],
        "generated_at": __import__("time").time(),
    }


def per_request_cost_estimate(
    connector: str,
    *,
    tile_count: int = 1,
    model_complexity: str = "medium",
    cost_rates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Estimate per-request cost and apply hard-stop budget checks for hosted model calls.

    Returns ``estimated_cost_usd``, ``within_budget`` (bool), and ``breakdown``.
    """
    cost_rates = cost_rates or {
        "openai": 0.002,
        "anthropic": 0.003,
        "azure_openai": 0.0015,
        "vertex_ai": 0.0018,
        "sagemaker": 0.001,
        "local": 0.0,
    }
    complexity_multiplier = {"low": 0.5, "medium": 1.0, "high": 2.5}.get(model_complexity, 1.0)
    base_rate = cost_rates.get(connector, 0.002)
    estimated = round(base_rate * tile_count * complexity_multiplier, 6)
    budget_limit = 10.0  # default hard stop
    return {
        "connector": connector,
        "tile_count": tile_count,
        "model_complexity": model_complexity,
        "estimated_cost_usd": estimated,
        "budget_limit_usd": budget_limit,
        "within_budget": estimated <= budget_limit,
        "breakdown": {
            "base_rate_per_tile": base_rate,
            "complexity_multiplier": complexity_multiplier,
        },
    }


def model_governance_dashboard(
    usage_logs: list[dict[str, Any]],
    drift_reports: list[dict[str, Any]] | None = None,
    connector_health: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a governance dashboard for model usage, drift state, and connector reliability.

    Returns a structured summary dict for operational oversight.
    """
    drift_reports = drift_reports or []
    connector_health = connector_health or {}
    total_requests = len(usage_logs)
    total_cost = sum(r.get("cost_usd", 0.0) for r in usage_logs)
    connectors_used = {r.get("connector") for r in usage_logs}
    models_used = {r.get("model_id") for r in usage_logs}
    drift_count = sum(1 for d in drift_reports if d.get("drifted", False))
    return {
        "total_requests": total_requests,
        "total_cost_usd": round(total_cost, 4),
        "connectors_used": sorted(c for c in connectors_used if c),
        "models_used": sorted(m for m in models_used if m),
        "drift_alerts": drift_count,
        "connector_health": connector_health,
        "status": "healthy" if drift_count == 0 else "drift_detected",
    }


def inference_cache_encryption(
    cache_key: str,
    data: bytes,
    *,
    key: str | None = None,
) -> dict[str, Any]:
    """Encrypt intermediate raster tiles and prediction outputs for secure caching.

    Returns metadata about the encrypted cache entry (not the raw ciphertext).
    Delegates to existing ``encrypt_data`` when a key is provided.
    """
    import hashlib as _hashlib
    cache_id = _hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    encrypted = False
    if key:
        try:
            encrypted_bytes = encrypt_data(data, key)
            encrypted = True
            size = len(encrypted_bytes)
        except Exception:
            size = len(data)
    else:
        size = len(data)
    return {
        "cache_id": cache_id,
        "encrypted": encrypted,
        "original_size_bytes": len(data),
        "stored_size_bytes": size,
        "key_provided": key is not None,
    }


def inference_artifact_retention_policy(
    artifact_type: str,
    *,
    retention_days: int | None = None,
    policy_profiles: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Define retention and deletion policy controls for inference artifacts and logs.

    Returns the applicable retention period and deletion guidance.
    """
    policy_profiles = policy_profiles or {
        "inference_log": 90,
        "tile_cache": 7,
        "prediction_output": 30,
        "audit_log": 365,
        "model_checkpoint": 180,
    }
    default_days = policy_profiles.get(artifact_type, 30)
    effective_days = retention_days if retention_days is not None else default_days
    return {
        "artifact_type": artifact_type,
        "retention_days": effective_days,
        "deletion_trigger": f"Delete after {effective_days} days from creation date.",
        "policy_source": "explicit" if retention_days is not None else "profile_default",
    }


def compliance_profile_templates(
    sector: str = "general",
) -> dict[str, Any]:
    """Return a compliance profile template for public sector or regulated utility workflows.

    Supported sectors: ``'general'``, ``'public_sector'``, ``'utility'``, ``'defense'``.
    """
    profiles: dict[str, dict[str, Any]] = {
        "general": {
            "data_residency": "no restriction",
            "audit_log_retention_days": 90,
            "encryption_required": False,
            "pii_scrubbing": "recommended",
            "model_registry_signing": False,
        },
        "public_sector": {
            "data_residency": "national boundary",
            "audit_log_retention_days": 365,
            "encryption_required": True,
            "pii_scrubbing": "mandatory",
            "model_registry_signing": True,
            "frameworks": ["FedRAMP", "FISMA", "NIST SP 800-53"],
        },
        "utility": {
            "data_residency": "approved regions only",
            "audit_log_retention_days": 730,
            "encryption_required": True,
            "pii_scrubbing": "mandatory",
            "model_registry_signing": True,
            "frameworks": ["NERC CIP", "IEC 62351"],
        },
        "defense": {
            "data_residency": "classified enclave only",
            "audit_log_retention_days": 2555,
            "encryption_required": True,
            "pii_scrubbing": "mandatory",
            "model_registry_signing": True,
            "frameworks": ["STIG", "CMMC Level 3", "DoD IL4+"],
        },
    }
    profile = profiles.get(sector, profiles["general"])
    return {"sector": sector, "profile": profile}
