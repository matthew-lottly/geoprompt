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

logger = logging.getLogger("geoprompt.service")

_FASTAPI_MISSING = (
    "Install FastAPI and Uvicorn to use the service module:\n"
    "  pip install fastapi uvicorn"
)


def _load_fastapi() -> Any:
    try:
        return importlib.import_module("fastapi")
    except ImportError as exc:
        raise RuntimeError(_FASTAPI_MISSING) from exc


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
    """Create and return a FastAPI application instance."""
    fastapi = _load_fastapi()
    from pydantic import BaseModel

    app = fastapi.FastAPI(
        title="GeoPrompt Scenario Service",
        version="0.1.0",
        description="Lightweight API for GeoPrompt spatial analysis workflows.",
    )
    expected_api_key = os.getenv("GEOPROMPT_API_KEY")
    required_roles = {role.strip().lower() for role in os.getenv("GEOPROMPT_REQUIRED_ROLES", "").split(",") if role.strip()}
    rate_limit_per_minute = int(os.getenv("GEOPROMPT_RATE_LIMIT_PER_MINUTE", "0") or "0")
    request_windows: dict[str, list[float]] = {}
    job_manager = ServiceJobManager(os.getenv("GEOPROMPT_JOB_STORE"))

    @app.middleware("http")
    async def audit_requests(request, call_next):
        start = time.perf_counter()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        actor = request.headers.get("x-api-key") or getattr(getattr(request, "client", None), "host", "anonymous")
        actor_roles = {role.strip().lower() for role in request.headers.get("x-roles", "").split(",") if role.strip()}
        if expected_api_key and request.url.path != "/health":
            provided = request.headers.get("x-api-key")
            if provided != expected_api_key:
                response = fastapi.responses.JSONResponse({"detail": "unauthorized", "request_id": request_id}, status_code=401)
                response.headers["x-request-id"] = request_id
                logger.warning("request_id=%s unauthorized path=%s", request_id, request.url.path)
                return response
        if required_roles and request.url.path not in {"/health", "/ops/metrics"} and not actor_roles.intersection(required_roles):
            response = fastapi.responses.JSONResponse({"detail": "forbidden", "request_id": request_id}, status_code=403)
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
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        response.headers["x-request-id"] = request_id
        logger.info("request_id=%s method=%s path=%s status=%s duration_ms=%.2f", request_id, request.method, request.url.path, response.status_code, duration_ms)
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


# Convenience: create the app at module level for ``uvicorn geoprompt.service:app``
try:
    app = build_app()
except RuntimeError:
    app = None  # FastAPI not installed â€” module can still be imported

__all__ = ["ServiceJobManager", "build_app", "service_benchmark_report"]
