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
import os
import re
import time
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
    ext = os.path.splitext(filename)[1].lower()
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
        key = hashlib.sha256(os.urandom(32)).hexdigest()[:40]
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
