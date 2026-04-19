"""Enterprise services, sharing, and deployment utilities for GeoPrompt.

Provides feature-service CRUD helpers, auth/token management, pagination,
publishing, sharing, audit logging, scheduled jobs, and deployment templates.
All network operations require ``requests`` (optional dependency).
"""
from __future__ import annotations

import datetime
import importlib
import json as _json
import time
from pathlib import Path
from typing import Any, Sequence


Record = dict[str, Any]


def _get_requests() -> Any:
    try:
        return importlib.import_module("requests")
    except ImportError as exc:
        raise RuntimeError(
            "Install requests ('pip install requests') for enterprise features."
        ) from exc


# ---------------------------------------------------------------------------
# A. Feature Service CRUD
# ---------------------------------------------------------------------------


def feature_service_query(
    url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    token: str | None = None,
    max_records: int = 1000,
) -> list[Record]:
    """Query a hosted feature service and return records."""
    requests = _get_requests()
    params: dict[str, Any] = {
        "where": where,
        "outFields": out_fields,
        "f": "json",
        "resultRecordCount": max_records,
    }
    if token:
        params["token"] = token

    records: list[Record] = []
    offset = 0
    while True:
        params["resultOffset"] = offset
        resp = requests.get(f"{url}/query", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        if not features:
            break
        for f in features:
            rec = dict(f.get("attributes", {}))
            if f.get("geometry"):
                rec["_geometry"] = f["geometry"]
            records.append(rec)
        if not data.get("exceededTransferLimit"):
            break
        offset += len(features)
    return records


def feature_service_add(
    url: str,
    records: Sequence[Record],
    *,
    token: str | None = None,
    geometry_field: str = "_geometry",
) -> dict[str, Any]:
    """Add features to a hosted feature service."""
    requests = _get_requests()
    features = []
    for rec in records:
        geom = rec.get(geometry_field)
        attrs = {k: v for k, v in rec.items() if k != geometry_field}
        feat: dict[str, Any] = {"attributes": attrs}
        if geom:
            feat["geometry"] = geom
        features.append(feat)

    payload: dict[str, Any] = {"features": _json.dumps(features), "f": "json"}
    if token:
        payload["token"] = token

    resp = requests.post(f"{url}/addFeatures", data=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def feature_service_update(
    url: str,
    records: Sequence[Record],
    *,
    token: str | None = None,
    geometry_field: str = "_geometry",
) -> dict[str, Any]:
    """Update existing features in a hosted feature service."""
    requests = _get_requests()
    features = []
    for rec in records:
        geom = rec.get(geometry_field)
        attrs = {k: v for k, v in rec.items() if k != geometry_field}
        feat: dict[str, Any] = {"attributes": attrs}
        if geom:
            feat["geometry"] = geom
        features.append(feat)

    payload: dict[str, Any] = {"features": _json.dumps(features), "f": "json"}
    if token:
        payload["token"] = token

    resp = requests.post(f"{url}/updateFeatures", data=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def feature_service_delete(
    url: str,
    object_ids: Sequence[int],
    *,
    token: str | None = None,
) -> dict[str, Any]:
    """Delete features from a hosted feature service by object IDs."""
    requests = _get_requests()
    payload: dict[str, Any] = {
        "objectIds": ",".join(str(i) for i in object_ids),
        "f": "json",
    }
    if token:
        payload["token"] = token

    resp = requests.post(f"{url}/deleteFeatures", data=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def feature_service_sync(
    url: str,
    local_records: Sequence[Record],
    *,
    key_field: str = "OBJECTID",
    token: str | None = None,
) -> dict[str, Any]:
    """Synchronize local records with a feature service.

    Adds new records, updates existing ones (matched by *key_field*).
    """
    remote = feature_service_query(url, token=token)
    remote_keys = {r.get(key_field) for r in remote if r.get(key_field) is not None}

    to_add = [r for r in local_records if r.get(key_field) not in remote_keys]
    to_update = [r for r in local_records if r.get(key_field) in remote_keys]

    add_result = feature_service_add(url, to_add, token=token) if to_add else {}
    update_result = feature_service_update(url, to_update, token=token) if to_update else {}

    return {"added": len(to_add), "updated": len(to_update),
            "add_result": add_result, "update_result": update_result}


# ---------------------------------------------------------------------------
# B. Token / Auth Management
# ---------------------------------------------------------------------------


class AuthProfile:
    """Simple token management for ArcGIS Online / Enterprise portals."""

    def __init__(
        self,
        *,
        portal_url: str = "https://www.arcgis.com",
        username: str = "",
        client_id: str | None = None,
        token: str | None = None,
        expiry: datetime.datetime | None = None,
    ) -> None:
        self.portal_url = portal_url.rstrip("/")
        self.username = username
        self.client_id = client_id
        self._token = token
        self._expiry = expiry

    def get_token(self, *, password: str | None = None, refresh_token: str | None = None) -> str:
        """Obtain or refresh an access token."""
        if self._token and self._expiry and datetime.datetime.now() < self._expiry:
            return self._token

        requests = _get_requests()
        token_url = f"{self.portal_url}/sharing/rest/generateToken"
        params: dict[str, Any] = {"f": "json", "referer": self.portal_url}

        if password:
            params.update({"username": self.username, "password": password, "client": "referer"})
        elif self.client_id and refresh_token:
            params.update({"client_id": self.client_id, "refresh_token": refresh_token,
                          "grant_type": "refresh_token"})
        else:
            raise ValueError("provide password or (client_id + refresh_token)")

        resp = requests.post(token_url, data=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "token" not in data:
            raise RuntimeError(f"token generation failed: {data}")
        self._token = data["token"]
        self._expiry = datetime.datetime.now() + datetime.timedelta(
            milliseconds=data.get("expires", 7200000) - int(time.time() * 1000)
        )
        return self._token

    @property
    def is_valid(self) -> bool:
        return bool(self._token and self._expiry and datetime.datetime.now() < self._expiry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "portal_url": self.portal_url,
            "username": self.username,
            "client_id": self.client_id,
            "has_token": self._token is not None,
        }


# ---------------------------------------------------------------------------
# C. Pagination / Retry Diagnostics
# ---------------------------------------------------------------------------


def paginated_request(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    token: str | None = None,
    page_size: int = 1000,
    max_pages: int = 100,
    retry_count: int = 3,
    retry_delay: float = 1.0,
) -> list[Record]:
    """Make paginated GET requests with retry logic.

    Returns combined results across all pages.
    """
    requests = _get_requests()
    all_records: list[Record] = []
    p = dict(params or {})
    p.setdefault("f", "json")
    p.setdefault("where", "1=1")
    p["resultRecordCount"] = page_size
    if token:
        p["token"] = token

    for page in range(max_pages):
        p["resultOffset"] = page * page_size
        last_error: Exception | None = None
        for attempt in range(retry_count):
            try:
                resp = requests.get(url, params=p, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                features = data.get("features", [])
                for f in features:
                    all_records.append(dict(f.get("attributes", {})))
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < retry_count - 1:
                    time.sleep(retry_delay * (attempt + 1))  # noqa: ASYNC251
        if last_error:
            raise last_error
        if not data.get("exceededTransferLimit"):
            break

    return all_records


# ---------------------------------------------------------------------------
# D. Publish / Package Generation
# ---------------------------------------------------------------------------


def publish_package(
    data: Sequence[Record],
    output_path: str | Path,
    *,
    title: str = "GeoPrompt Package",
    description: str = "",
    tags: Sequence[str] = (),
    format: str = "geojson",
) -> str:
    """Create a publish-ready data package.

    Bundles data with metadata suitable for upload to a portal.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    package = {
        "title": title,
        "description": description,
        "tags": list(tags),
        "format": format,
        "item_count": len(data),
        "created": datetime.datetime.now().isoformat(),
        "data": list(data),
    }
    p.write_text(_json.dumps(package, indent=2, default=str))
    return str(p)


# ---------------------------------------------------------------------------
# E. Sharing Helpers
# ---------------------------------------------------------------------------


def share_item(
    item_id: str,
    *,
    portal_url: str = "https://www.arcgis.com",
    token: str,
    groups: Sequence[str] = (),
    everyone: bool = False,
    org: bool = False,
) -> dict[str, Any]:
    """Share a portal item with groups, org, or everyone."""
    requests = _get_requests()
    payload: dict[str, Any] = {
        "f": "json",
        "token": token,
        "everyone": str(everyone).lower(),
        "org": str(org).lower(),
        "groups": ",".join(groups),
    }
    resp = requests.post(
        f"{portal_url}/sharing/rest/content/items/{item_id}/share",
        data=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# F. Audit Log
# ---------------------------------------------------------------------------


class AuditLog:
    """Record who ran what tool, when, and with which inputs."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._entries: list[dict[str, Any]] = []
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            self._entries = _json.loads(self._path.read_text())

    def log(
        self,
        tool: str,
        user: str = "unknown",
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "tool": tool,
            "user": user,
            "inputs": inputs or {},
            "outputs": outputs or {},
        }
        self._entries.append(entry)
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(_json.dumps(self._entries, indent=2, default=str))

    def query(
        self,
        *,
        tool: str | None = None,
        user: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        results = self._entries
        if tool:
            results = [e for e in results if e["tool"] == tool]
        if user:
            results = [e for e in results if e["user"] == user]
        if since:
            results = [e for e in results if e["timestamp"] >= since]
        return results

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)


# ---------------------------------------------------------------------------
# G. Scheduled Job Helpers
# ---------------------------------------------------------------------------


class ScheduledJob:
    """Define and track a scheduled processing job."""

    def __init__(
        self,
        name: str,
        func: Any,
        *,
        schedule: str = "daily",
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.func = func
        self.schedule = schedule
        self.args = args
        self.kwargs = kwargs or {}
        self.last_run: str | None = None
        self.last_result: Any = None
        self.run_count = 0

    def run(self) -> Any:
        self.last_run = datetime.datetime.now().isoformat()
        self.last_result = self.func(*self.args, **self.kwargs)
        self.run_count += 1
        return self.last_result

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schedule": self.schedule,
            "last_run": self.last_run,
            "run_count": self.run_count,
        }


def run_job_batch(jobs: Sequence[ScheduledJob]) -> list[dict[str, Any]]:
    """Run multiple scheduled jobs and return status reports."""
    results: list[dict[str, Any]] = []
    for job in jobs:
        try:
            job.run()
            results.append({**job.status(), "success": True})
        except Exception as e:
            results.append({**job.status(), "success": False, "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# H. Deployment Templates
# ---------------------------------------------------------------------------


def deployment_template(
    platform: str = "docker",
    *,
    app_name: str = "geoprompt-app",
    python_version: str = "3.11",
    port: int = 8000,
    extras: Sequence[str] = (),
) -> str:
    """Generate a deployment configuration template.

    Platforms: ``"docker"``, ``"azure"``, ``"aws"``, ``"gcp"``.
    """
    if platform == "docker":
        lines = [
            f"FROM python:{python_version}-slim",
            "WORKDIR /app",
            "COPY requirements.txt .",
            "RUN pip install --no-cache-dir -r requirements.txt",
            "COPY . .",
            f'EXPOSE {port}',
            f'CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]',
        ]
        return "\n".join(lines)

    if platform == "azure":
        return _json.dumps({
            "name": app_name,
            "properties": {
                "siteConfig": {
                    "linuxFxVersion": f"PYTHON|{python_version}",
                    "appCommandLine": f"gunicorn -w 2 -b :{port} main:app",
                },
            },
        }, indent=2)

    if platform == "aws":
        return _json.dumps({
            "AWSTemplateFormatVersion": "2010-09-09",
            "Resources": {
                "Function": {
                    "Type": "AWS::Lambda::Function",
                    "Properties": {
                        "FunctionName": app_name,
                        "Runtime": f"python{python_version}",
                        "Handler": "main.handler",
                    },
                },
            },
        }, indent=2)

    if platform == "gcp":
        return _json.dumps({
            "runtime": f"python{python_version.replace('.', '')}",
            "entrypoint": f"gunicorn -b :{port} main:app",
            "env_variables": {"PORT": str(port)},
        }, indent=2)

    raise ValueError(f"unknown platform: {platform}")


# ---------------------------------------------------------------------------
# I. Project Health Dashboard
# ---------------------------------------------------------------------------


def project_health_dashboard(
    *,
    workspace_path: str | Path | None = None,
    environment_info: dict[str, Any] | None = None,
    data_sources: Sequence[dict[str, Any]] | None = None,
    runtime_warnings: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate a project health dashboard report.

    Checks environment, dependencies, data source connectivity, and
    runtime warnings.
    """
    import sys

    health: dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "workspace": str(workspace_path) if workspace_path else None,
    }

    # Check optional dependencies
    deps = ["numpy", "scipy", "shapely", "rasterio", "geopandas", "matplotlib", "folium", "requests"]
    dep_status: dict[str, str] = {}
    for dep in deps:
        try:
            mod = importlib.import_module(dep)
            dep_status[dep] = getattr(mod, "__version__", "installed")
        except ImportError:
            dep_status[dep] = "not installed"
    health["dependencies"] = dep_status

    health["environment"] = environment_info or {}
    health["data_sources"] = list(data_sources or [])
    health["warnings"] = list(runtime_warnings or [])

    # Overall score
    installed_count = sum(1 for v in dep_status.values() if v != "not installed")
    health["dependency_score"] = f"{installed_count}/{len(deps)}"
    health["status"] = "healthy" if installed_count >= 3 else "degraded"

    return health


__all__ = [
    "AuditLog",
    "AuthProfile",
    "ScheduledJob",
    "deployment_template",
    "feature_service_add",
    "feature_service_delete",
    "feature_service_query",
    "feature_service_sync",
    "feature_service_update",
    "paginated_request",
    "project_health_dashboard",
    "publish_package",
    "run_job_batch",
    "share_item",
]
