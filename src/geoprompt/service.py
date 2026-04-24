"""Lightweight FastAPI service template for running GeoPrompt scenarios remotely.

Usage::

    pip install geoprompt[all] fastapi uvicorn
    uvicorn geoprompt.service:app --reload

The service exposes a handful of endpoints for running common workflows
via HTTP. Teams can extend this template for their own deployments.
"""

import importlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Union

from ._capabilities import require_capability

logger = logging.getLogger("geoprompt.service")

_FASTAPI_MISSING = (
    "Install FastAPI and Uvicorn to use the service module:\n"
    "  pip install fastapi uvicorn"
)


def _load_fastapi() -> Any:
    require_capability("fastapi", context="service module")
    try:
        return importlib.import_module("fastapi")
    except ImportError as exc:  # pragma: no cover - guarded by require_capability
        raise AssertionError("Capability guard failed for fastapi") from exc


class ServiceJobManager:
    """Persistent in-process job tracker for service workflows and polling."""

    def __init__(self, storage_path: Optional[Union[str, Path]] = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._jobs: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, Any] = {}
        if self.storage_path and self.storage_path.exists():
            try:
                raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
                self._jobs = {str(item["job_id"]): dict(item) for item in raw}
            except Exception:
                self._jobs = {}

    def _persist(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [self._jobs[key] for key in sorted(self._jobs)]
        self.storage_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def register_handler(self, job_type: str, func: Any) -> None:
        if not callable(func):
            raise TypeError("func must be callable")
        self._handlers[job_type] = func

    def _resolve_handler(self, job_type: str) -> Any:
        if job_type in self._handlers:
            return self._handlers[job_type]
        if job_type == "compare_scenarios":
            from .tools import compare_scenarios
            return lambda payload: compare_scenarios(
                payload.get("baseline", {}),
                payload.get("candidate", {}),
                higher_is_better=payload.get("higher_is_better"),
            )
        if job_type == "schema_report":
            from .io import schema_report
            return lambda payload: schema_report(payload.get("records", []))
        if job_type == "nearest":
            from .frame import GeoPromptFrame

            def _nearest(payload: dict[str, Any]) -> list[dict[str, Any]]:
                geometry_column = str(payload.get("geometry_column", "geometry"))
                frame = GeoPromptFrame(payload.get("features", []), geometry_column=geometry_column)
                target_frame = GeoPromptFrame(payload.get("targets", []), geometry_column=geometry_column)
                return frame.nearest_join(target_frame, k=int(payload.get("k", 1)))

            return _nearest
        raise KeyError(f"unknown job type: {job_type}")

    def submit(
        self,
        job_type: str,
        payload: dict[str, Any],
        *,
        user: str = "unknown",
        roles: Optional[list[str]] = None,
        execute: bool = True,
    ) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        job = {
            "job_id": job_id,
            "job_type": job_type,
            "user": user,
            "roles": list(roles or []),
            "payload": dict(payload),
            "status": "queued",
            "attempts": 0,
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }
        self._jobs[job_id] = job
        self._persist()
        return self.run(job_id) if execute else dict(job)

    def run(self, job_id: str) -> dict[str, Any]:
        if job_id not in self._jobs:
            raise KeyError(job_id)
        job = self._jobs[job_id]
        handler = self._resolve_handler(str(job["job_type"]))
        job["status"] = "running"
        job["attempts"] = int(job.get("attempts", 0)) + 1
        job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._persist()
        try:
            job["result"] = handler(dict(job.get("payload", {})))
            job["error"] = None
            job["status"] = "completed"
        except Exception as exc:
            job["result"] = None
            job["error"] = str(exc)
            job["status"] = "failed"
        job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._persist()
        return dict(job)

    def resume(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] == "completed":
            return job
        return self.run(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        if job_id not in self._jobs:
            raise KeyError(job_id)
        return dict(self._jobs[job_id])

    def list(self, *, status: Optional[str] = None) -> list[dict[str, Any]]:
        jobs = [dict(job) for job in self._jobs.values()]
        if status is not None:
            jobs = [job for job in jobs if str(job.get("status")) == status]
        jobs.sort(key=lambda item: (str(item.get("created_at")), str(item.get("job_id"))))
        return jobs

    def metrics(self) -> dict[str, Any]:
        counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
        for job in self._jobs.values():
            status = str(job.get("status", "queued"))
            counts[status] = counts.get(status, 0) + 1
        return {
            "job_counts": counts,
            "total_jobs": len(self._jobs),
            "registered_handlers": sorted(self._handlers),
        }


def service_benchmark_report(
    manager: ServiceJobManager,
    job_type: str,
    payload: dict[str, Any],
    *,
    iterations: int = 5,
    user: str = "benchmark",
    roles: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Measure local service job throughput and failure rate for release rehearsal."""
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")

    started = time.perf_counter()
    results = [manager.submit(job_type, payload, user=user, roles=roles, execute=True) for _ in range(iterations)]
    duration = max(time.perf_counter() - started, 1e-9)
    success_count = sum(1 for result in results if result["status"] == "completed")
    return {
        "job_type": job_type,
        "iterations": iterations,
        "success_count": success_count,
        "failure_count": iterations - success_count,
        "total_duration_seconds": duration,
        "jobs_per_second": success_count / duration,
    }


def build_app() -> Any:
    """Create and return a FastAPI application instance.

    By default, stub-mode behavior is disabled in service deployments.
    Set GEOPROMPT_DEV_PROFILE=true to enable stubs (development only).
    """
    # J2.20 — Block stub-mode behavior in service unless development profile is active
    is_dev_profile = os.getenv("GEOPROMPT_DEV_PROFILE", "").lower() in {"true", "1", "yes"}
    allow_stub_fallback = os.getenv("GEOPROMPT_ALLOW_STUB_FALLBACK", "").lower() in {"true", "1", "yes"}

    if not is_dev_profile and allow_stub_fallback:
        raise RuntimeError(
            "Stub fallback is enabled but GEOPROMPT_DEV_PROFILE is not set. "
            "Stub-mode behavior is not allowed in production service deployments. "
            "Set GEOPROMPT_DEV_PROFILE=true only for development/testing environments."
        )

    if not is_dev_profile:
        # Enforce production-safe fallback policy
        from ._exceptions import FallbackPolicy
        FallbackPolicy.for_environment()  # Validates and logs fallback policy from env
        logger.info("Service started in production mode (no stub fallbacks allowed). "
                   "Set GEOPROMPT_DEV_PROFILE=true to enable stubs for testing.")

    fastapi = _load_fastapi()
    from pydantic import BaseModel

    app = fastapi.FastAPI(
        title="GeoPrompt Scenario Service",
        version="0.1.0",
        description="Lightweight API for GeoPrompt spatial analysis workflows.",
    )
    # J7 – load service-hardening utilities
    from ._service_hardening import (
        PayloadTooLargeError,
        PolicySimulator,
        audit_secret_sources,
        check_auth,
        check_data_residency,
        classify_operation,
        redact_payload,
        run_deployment_smoke_checks,
        scan_for_pii,
        validate_compliance_profile,
        validate_dashboard_integrity,
        validate_geometry_payload,
        validate_payload_complexity,
        verify_request_signature,
    )

    expected_api_key = os.getenv("GEOPROMPT_API_KEY")
    required_roles = {role.strip().lower() for role in os.getenv("GEOPROMPT_REQUIRED_ROLES", "").split(",") if role.strip()}
    rate_limit_per_minute = int(os.getenv("GEOPROMPT_RATE_LIMIT_PER_MINUTE", "0") or "0")
    # J7.4 – redact sensitive fields from audit logs
    _redact_fields: frozenset[str] = frozenset(os.getenv("GEOPROMPT_REDACT_FIELDS", "").split(",")) - {""}
    # J7.3 – enable PII-blocking middleware (default off; set GEOPROMPT_PII_BLOCKING=true)
    _pii_blocking: bool = os.getenv("GEOPROMPT_PII_BLOCKING", "false").lower() in {"1", "true", "yes"}
    # J7.5 – payload complexity limit (default 10 MiB)
    _max_payload_bytes: int = int(os.getenv("GEOPROMPT_MAX_PAYLOAD_BYTES", str(10 * 1024 * 1024)))
    # J7.6 – optional HMAC request signing
    _signing_secret: str | None = os.getenv("GEOPROMPT_HMAC_SECRET") or None
    request_windows: dict[str, list[float]] = {}
    job_manager = ServiceJobManager(os.getenv("GEOPROMPT_JOB_STORE"))

    # J7.7 – audit secret sources on startup (logs weak/missing secrets)
    _secret_audit = audit_secret_sources()
    if _secret_audit["weak"]:
        logger.warning("J7.7 secret-source audit: weak secrets detected: %s", _secret_audit["weak"])
    if _secret_audit["missing"]:
        logger.info("J7.7 secret-source audit: missing (optional) secrets: %s", _secret_audit["missing"])

    @app.middleware("http")
    async def audit_requests(request, call_next):
        start = time.perf_counter()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        actor = request.headers.get("x-api-key") or getattr(getattr(request, "client", None), "host", "anonymous")
        actor_roles = {role.strip().lower() for role in request.headers.get("x-roles", "").split(",") if role.strip()}

        # J7.2 – tag every request with its operation class for structured audit logging
        operation_class = classify_operation(request.url.path)

        if expected_api_key and request.url.path != "/health":
            provided = request.headers.get("x-api-key")
            if provided != expected_api_key:
                response = fastapi.responses.JSONResponse({"detail": "unauthorized", "request_id": request_id}, status_code=401)
                response.headers["x-request-id"] = request_id
                logger.warning(
                    "request_id=%s op=%s unauthorized path=%s",
                    request_id, operation_class, request.url.path,
                )
                return response
        allowed, auth_reason = check_auth(
            request.url.path,
            actor_roles,
            required_roles=required_roles,
        )
        if request.url.path not in {"/health", "/ops/metrics"} and not allowed:
            response = fastapi.responses.JSONResponse(
                {"detail": f"forbidden: {auth_reason}", "request_id": request_id},
                status_code=403,
            )
            response.headers["x-request-id"] = request_id
            return response
        if rate_limit_per_minute > 0 and request.url.path != "/health":
            now = time.time()
            recent = [ts for ts in request_windows.get(actor, []) if now - ts < 60.0]
            if len(recent) >= rate_limit_per_minute:
                response = fastapi.responses.JSONResponse({"detail": "rate limit exceeded", "request_id": request_id}, status_code=429)
                response.headers["x-request-id"] = request_id
                return response
            recent.append(now)
            request_windows[actor] = recent

        # J7.6 – optional HMAC signature verification
        if _signing_secret and request.url.path not in {"/health"}:
            sig = request.headers.get("x-geoprompt-signature")
            ts = request.headers.get("x-geoprompt-timestamp")
            if sig is None or ts is None:
                response = fastapi.responses.JSONResponse(
                    {"detail": "missing signature headers (x-geoprompt-signature, x-geoprompt-timestamp)", "request_id": request_id},
                    status_code=401,
                )
                response.headers["x-request-id"] = request_id
                return response
            body = await request.body()
            try:
                verify_request_signature(body, provided_signature=sig, secret=_signing_secret, timestamp=ts)
            except ValueError as exc:
                response = fastapi.responses.JSONResponse(
                    {"detail": str(exc), "request_id": request_id}, status_code=401
                )
                response.headers["x-request-id"] = request_id
                return response

        # J7.5 – payload complexity guardrails for mutating requests
        if request.method in {"POST", "PUT", "PATCH"} and request.url.path not in {"/health"}:
            try:
                raw = await request.body()
                if raw:
                    try:
                        parsed = json.loads(raw)
                        validate_payload_complexity(parsed, max_bytes=_max_payload_bytes)
                    except PayloadTooLargeError as exc:
                        response = fastapi.responses.JSONResponse(
                            {"detail": str(exc), "request_id": request_id}, status_code=413
                        )
                        response.headers["x-request-id"] = request_id
                        return response
                    except (json.JSONDecodeError, ValueError):
                        pass  # non-JSON bodies are not validated here
            except Exception:
                pass  # body already consumed by signature check above; skip silently

        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        response.headers["x-request-id"] = request_id
        logger.info(
            "request_id=%s op=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id, operation_class, request.method, request.url.path,
            response.status_code, duration_ms,
        )
        return response

    # --- Request / Response models ---

    class HealthResponse(BaseModel):
        status: str
        version: str

    class ScenarioRequest(BaseModel):
        baseline: dict[str, float]
        candidate: dict[str, float]
        higher_is_better: Optional[list[str]] = None

    class ScenarioResponse(BaseModel):
        comparison: dict[str, Any]

    class NearestRequest(BaseModel):
        features: list[dict[str, Any]]
        targets: list[dict[str, Any]]
        k: int = 1
        geometry_column: str = "geometry"

    class NearestResponse(BaseModel):
        results: list[dict[str, Any]]

    class JobRequest(BaseModel):
        job_type: str
        payload: dict[str, Any]
        user: str = "unknown"
        roles: Optional[list[str]] = None
        execute: bool = True

    class JobResponse(BaseModel):
        job_id: str
        job_type: str
        status: str
        user: str
        roles: list[str]
        attempts: int
        result: Any = None
        error: Optional[str] = None

    # --- Routes ---

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        from . import __version__  # type: ignore[attr-defined]

        ver = __version__ if isinstance(__version__, str) else "unknown"
        return HealthResponse(status="ok", version=ver)

    @app.post("/compare-scenarios", response_model=ScenarioResponse)
    def compare_scenarios_endpoint(req: ScenarioRequest) -> ScenarioResponse:
        from .tools import compare_scenarios

        result = compare_scenarios(
            req.baseline,
            req.candidate,
            higher_is_better=req.higher_is_better,
        )
        return ScenarioResponse(comparison=result)

    @app.post("/nearest", response_model=NearestResponse)
    def nearest_endpoint(req: NearestRequest) -> NearestResponse:
        from .frame import GeoPromptFrame

        frame = GeoPromptFrame(req.features, geometry_column=req.geometry_column)
        target_frame = GeoPromptFrame(req.targets, geometry_column=req.geometry_column)
        rows = frame.nearest_join(target_frame, k=req.k)
        return NearestResponse(results=rows)

    @app.post("/schema-report")
    def schema_report_endpoint(records: list[dict[str, Any]]) -> dict[str, Any]:
        from .io import schema_report

        return schema_report(records)

    @app.post("/jobs/submit", response_model=JobResponse)
    def submit_job(req: JobRequest) -> JobResponse:
        job = job_manager.submit(req.job_type, req.payload, user=req.user, roles=req.roles, execute=req.execute)
        return JobResponse(**job)

    @app.get("/jobs", response_model=list[JobResponse])
    def list_jobs(status: Optional[str] = None) -> list[JobResponse]:
        return [JobResponse(**job) for job in job_manager.list(status=status)]

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str) -> JobResponse:
        if job_id not in {job["job_id"] for job in job_manager.list()}:
            raise fastapi.HTTPException(status_code=404, detail="job not found")
        return JobResponse(**job_manager.get(job_id))

    @app.post("/jobs/{job_id}/resume", response_model=JobResponse)
    def resume_job(job_id: str) -> JobResponse:
        try:
            return JobResponse(**job_manager.resume(job_id))
        except KeyError as exc:
            raise fastapi.HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/ops/metrics")
    def operations_metrics() -> dict[str, Any]:
        return job_manager.metrics()

    @app.post("/ops/benchmark")
    def operations_benchmark(req: JobRequest) -> dict[str, Any]:
        return service_benchmark_report(
            job_manager,
            req.job_type,
            req.payload,
            iterations=3,
            user=req.user,
            roles=req.roles,
        )

    return app
    # J7.8 – Policy simulation endpoint (dry-run governance preview)
    class PolicySimulateRequest(BaseModel):
        path: str
        payload: Any = None
        actor_roles: list[str] = []
        pii_blocking: bool = True
        max_payload_bytes: int = 10 * 1024 * 1024
        signature_required: bool = False

    @app.post("/ops/policy-simulate")
    def policy_simulate(req: PolicySimulateRequest) -> dict[str, Any]:
        simulator = PolicySimulator(
            required_roles=set(required_roles),
            max_payload_bytes=req.max_payload_bytes,
            pii_blocking=req.pii_blocking,
            signature_required=req.signature_required,
        )
        return simulator.simulate(
            req.path,
            req.payload,
            actor_roles=set(req.actor_roles),
        )

    # J7.10 – Compliance profile validation endpoint
    class ComplianceCheckRequest(BaseModel):
        profile: str
        config: dict[str, Any]

    @app.post("/ops/compliance-check")
    def compliance_check(req: ComplianceCheckRequest) -> dict[str, Any]:
        try:
            gaps = validate_compliance_profile(req.profile, req.config)
        except KeyError:
            raise fastapi.HTTPException(status_code=400, detail=f"Unknown compliance profile: {req.profile!r}")
        return {"profile": req.profile, "compliant": len(gaps) == 0, "gaps": gaps}

    # J7.13 – Deployment readiness smoke-check endpoint
    @app.get("/ops/readiness")
    def readiness_check() -> dict[str, Any]:
        result = run_deployment_smoke_checks()
        status_code = 200 if result.ready else 503
        summary = result.summary()
        if not result.ready:
            raise fastapi.HTTPException(status_code=status_code, detail=summary)
        return summary

    # J7.7 – Secret source audit endpoint (returns redacted report)
    @app.get("/ops/secret-audit")
    def secret_audit() -> dict[str, Any]:
        report = audit_secret_sources()
        # Never return actual values — only the status report
        return {"report": report["report"], "weak_count": len(report["weak"]), "missing_count": len(report["missing"])}

    # J7.9 – Governance dashboard integrity endpoint
    @app.get("/ops/dashboard-integrity")
    def dashboard_integrity() -> dict[str, Any]:
        metrics = job_manager.metrics()
        violations = validate_dashboard_integrity(metrics)
        return {"metrics": metrics, "violations": violations, "integrity_ok": len(violations) == 0}

    return app


# Convenience: create the app at module level for ``uvicorn geoprompt.service:app``
try:
    app = build_app()
except RuntimeError:
    app = None  # FastAPI not installed â€” module can still be imported

__all__ = [
    "ServiceJobManager",
    "build_app",
    "service_benchmark_report",
]
