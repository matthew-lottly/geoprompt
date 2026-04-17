"""Lightweight FastAPI service template for running GeoPrompt scenarios remotely.

Usage::

    pip install geoprompt[all] fastapi uvicorn
    uvicorn geoprompt.service:app --reload

The service exposes a handful of endpoints for running common workflows
via HTTP. Teams can extend this template for their own deployments.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import time
from typing import Any

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

    @app.middleware("http")
    async def audit_requests(request, call_next):
        start = time.perf_counter()
        if expected_api_key and request.url.path != "/health":
            provided = request.headers.get("x-api-key")
            if provided != expected_api_key:
                response = fastapi.responses.JSONResponse({"detail": "unauthorized"}, status_code=401)
                logger.warning("unauthorized request path=%s", request.url.path)
                return response
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.info("request method=%s path=%s status=%s duration_ms=%.2f", request.method, request.url.path, response.status_code, duration_ms)
        return response

    # --- Request / Response models ---

    class HealthResponse(BaseModel):
        status: str
        version: str

    class ScenarioRequest(BaseModel):
        baseline: dict[str, float]
        candidate: dict[str, float]
        higher_is_better: list[str] | None = None

    class ScenarioResponse(BaseModel):
        comparison: dict[str, Any]

    class NearestRequest(BaseModel):
        features: list[dict[str, Any]]
        targets: list[dict[str, Any]]
        k: int = 1
        geometry_column: str = "geometry"

    class NearestResponse(BaseModel):
        results: list[dict[str, Any]]

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

    return app


# Convenience: create the app at module level for ``uvicorn geoprompt.service:app``
try:
    app = build_app()
except RuntimeError:
    app = None  # FastAPI not installed — module can still be imported
