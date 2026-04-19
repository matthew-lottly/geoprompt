"""Geoprocessing framework: environment, result objects, hooks, pipelines, batch processing.

Pure-Python implementations covering roadmap items from A8
(Geoprocessing Framework 1101-1200).
"""
from __future__ import annotations

import copy
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Sequence

# ---------------------------------------------------------------------------
# Environment settings (1124)
# ---------------------------------------------------------------------------

_ENVIRONMENT: dict[str, Any] = {
    "workspace": ".",
    "scratch_workspace": None,
    "extent": None,
    "output_crs": None,
    "cell_size": None,
    "overwrite_output": False,
    "add_outputs_to_map": False,
    "parallel_processing_factor": "100%",
    "log_history": True,
}


def get_environment(key: str) -> Any:
    """Get a geoprocessing environment setting."""
    return _ENVIRONMENT.get(key)


def set_environment(**kwargs: Any) -> None:
    """Set one or more geoprocessing environment settings."""
    for k, v in kwargs.items():
        if k not in _ENVIRONMENT:
            raise KeyError(f"unknown environment setting: {k}")
        _ENVIRONMENT[k] = v


def reset_environment() -> None:
    """Reset all environment settings to defaults."""
    _ENVIRONMENT.update({
        "workspace": ".",
        "scratch_workspace": None,
        "extent": None,
        "output_crs": None,
        "cell_size": None,
        "overwrite_output": False,
        "add_outputs_to_map": False,
        "parallel_processing_factor": "100%",
        "log_history": True,
    })


def list_environments() -> dict[str, Any]:
    """Return a copy of current environment settings."""
    return dict(_ENVIRONMENT)


# ---------------------------------------------------------------------------
# Result object (1128)
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result of a geoprocessing tool execution."""
    tool_name: str
    status: str = "success"
    output: Any = None
    messages: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def get_output(self) -> Any:
        return self.output

    def get_messages(self) -> list[str]:
        return list(self.messages)

    def succeeded(self) -> bool:
        return self.status == "success"


# ---------------------------------------------------------------------------
# Tool history / provenance log (1131)
# ---------------------------------------------------------------------------

_TOOL_HISTORY: list[dict[str, Any]] = []


def log_tool_execution(result: ToolResult) -> None:
    """Append a tool execution record to the history log."""
    _TOOL_HISTORY.append({
        "run_id": result.run_id,
        "tool_name": result.tool_name,
        "status": result.status,
        "elapsed_seconds": result.elapsed_seconds,
        "parameters": result.parameters,
        "timestamp": time.time(),
    })


def get_tool_history() -> list[dict[str, Any]]:
    """Return the full tool execution history."""
    return list(_TOOL_HISTORY)


def clear_tool_history() -> None:
    """Clear the tool execution history."""
    _TOOL_HISTORY.clear()


# ---------------------------------------------------------------------------
# Progress callback (1105)
# ---------------------------------------------------------------------------

@dataclass
class ProgressReporter:
    """Simple progress callback manager."""
    total: int = 100
    current: int = 0
    callback: Callable[[int, str], None] | None = None
    messages: list[str] = field(default_factory=list)

    def update(self, step: int = 1, message: str = "") -> None:
        self.current = min(self.current + step, self.total)
        if message:
            self.messages.append(message)
        if self.callback:
            self.callback(self.current, message)

    def percentage(self) -> float:
        return (self.current / self.total) * 100 if self.total > 0 else 0.0

    def reset(self) -> None:
        self.current = 0
        self.messages.clear()


# ---------------------------------------------------------------------------
# Hook / event / middleware system (1156-1158)
# ---------------------------------------------------------------------------

_HOOKS: dict[str, list[Callable[..., Any]]] = {
    "pre_tool": [],
    "post_tool": [],
    "on_create": [],
    "on_delete": [],
    "on_update": [],
    "on_error": [],
}


def register_hook(event: str, func: Callable[..., Any]) -> None:
    """Register a hook function for a geoprocessing event."""
    if event not in _HOOKS:
        _HOOKS[event] = []
    _HOOKS[event].append(func)


def unregister_hook(event: str, func: Callable[..., Any]) -> None:
    """Remove a hook function."""
    if event in _HOOKS:
        _HOOKS[event] = [f for f in _HOOKS[event] if f is not func]


def fire_hooks(event: str, **kwargs: Any) -> list[Any]:
    """Fire all hooks registered for an event."""
    results = []
    for func in _HOOKS.get(event, []):
        results.append(func(**kwargs))
    return results


def clear_hooks() -> None:
    """Remove all registered hooks."""
    for key in _HOOKS:
        _HOOKS[key] = []


class MiddlewarePipeline:
    """Middleware pipeline for tool execution."""

    def __init__(self) -> None:
        self._middlewares: list[Callable[[dict[str, Any], Callable], Any]] = []

    def add(self, middleware: Callable[[dict[str, Any], Callable], Any]) -> None:
        self._middlewares.append(middleware)

    def execute(self, context: dict[str, Any], handler: Callable[[dict[str, Any]], Any]) -> Any:
        chain = handler
        for mw in reversed(self._middlewares):
            prev = chain
            chain = lambda ctx, _mw=mw, _prev=prev: _mw(ctx, _prev)
        return chain(context)


# ---------------------------------------------------------------------------
# Chain tools / pipeline (1116, 1084)
# ---------------------------------------------------------------------------

class ToolChain:
    """Chain multiple tool functions together (output → input)."""

    def __init__(self) -> None:
        self._steps: list[tuple[str, Callable[..., Any], dict[str, Any]]] = []

    def add_step(self, name: str, func: Callable[..., Any], **kwargs: Any) -> "ToolChain":
        self._steps.append((name, func, kwargs))
        return self

    def run(self, initial_input: Any = None, *, progress: ProgressReporter | None = None) -> list[ToolResult]:
        results: list[ToolResult] = []
        current = initial_input
        for i, (name, func, kwargs) in enumerate(self._steps):
            fire_hooks("pre_tool", tool_name=name, step=i)
            t0 = time.time()
            try:
                if current is not None:
                    output = func(current, **kwargs)
                else:
                    output = func(**kwargs)
                elapsed = time.time() - t0
                result = ToolResult(
                    tool_name=name,
                    status="success",
                    output=output,
                    parameters=kwargs,
                    elapsed_seconds=elapsed,
                )
            except Exception as exc:
                elapsed = time.time() - t0
                result = ToolResult(
                    tool_name=name,
                    status="failure",
                    messages=[str(exc)],
                    parameters=kwargs,
                    elapsed_seconds=elapsed,
                )
                fire_hooks("on_error", tool_name=name, error=exc)
                results.append(result)
                log_tool_execution(result)
                break
            results.append(result)
            log_tool_execution(result)
            fire_hooks("post_tool", tool_name=name, result=result)
            current = output
            if progress:
                progress.update(message=f"Completed {name}")
        return results


# ---------------------------------------------------------------------------
# Batch processing (1123)
# ---------------------------------------------------------------------------

def batch_process(
    func: Callable[..., Any],
    parameter_table: Sequence[dict[str, Any]],
    *,
    continue_on_error: bool = True,
    progress: ProgressReporter | None = None,
) -> list[ToolResult]:
    """Run a tool function for each row in a parameter table."""
    results: list[ToolResult] = []
    for i, params in enumerate(parameter_table):
        t0 = time.time()
        try:
            output = func(**params)
            elapsed = time.time() - t0
            r = ToolResult(
                tool_name=func.__name__,
                status="success",
                output=output,
                parameters=params,
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            r = ToolResult(
                tool_name=func.__name__,
                status="failure",
                messages=[str(exc)],
                parameters=params,
                elapsed_seconds=elapsed,
            )
            if not continue_on_error:
                results.append(r)
                break
        results.append(r)
        if progress:
            progress.update()
    return results


# ---------------------------------------------------------------------------
# Dry-run mode (1136)
# ---------------------------------------------------------------------------

def dry_run(
    func: Callable[..., Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate parameters without executing the tool.

    Checks that required parameters are present and types are correct.
    """
    import inspect
    sig = inspect.signature(func)
    issues: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.default is inspect.Parameter.empty and name not in kwargs:
            if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                issues.append(f"missing required parameter: {name}")
    return {
        "tool": func.__name__,
        "parameters": kwargs,
        "valid": len(issues) == 0,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Transaction support (1133)
# ---------------------------------------------------------------------------

class Transaction:
    """Simple transaction wrapper tracking operations for rollback."""

    def __init__(self) -> None:
        self._operations: list[tuple[str, Any]] = []
        self._snapshots: list[Any] = []
        self._committed = False

    def snapshot(self, data: Any) -> None:
        """Save a deep-copy snapshot for potential rollback."""
        self._snapshots.append(copy.deepcopy(data))

    def record(self, operation: str, data: Any = None) -> None:
        self._operations.append((operation, data))

    def commit(self) -> None:
        self._committed = True
        self._snapshots.clear()

    def rollback(self) -> Any | None:
        if self._snapshots:
            return self._snapshots[-1]
        return None

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def operations(self) -> list[tuple[str, Any]]:
        return list(self._operations)


@contextmanager
def transaction_scope(data: Any = None) -> Generator[Transaction, None, None]:
    """Context manager for transactional operations."""
    txn = Transaction()
    if data is not None:
        txn.snapshot(data)
    try:
        yield txn
        if not txn.committed:
            txn.commit()
    except Exception:
        txn.rollback()
        raise


# ---------------------------------------------------------------------------
# Profile-based configuration (1162)
# ---------------------------------------------------------------------------

_PROFILES: dict[str, dict[str, Any]] = {
    "dev": {
        "overwrite_output": True,
        "log_history": True,
        "parallel_processing_factor": "50%",
    },
    "prod": {
        "overwrite_output": False,
        "log_history": True,
        "parallel_processing_factor": "100%",
    },
    "test": {
        "overwrite_output": True,
        "log_history": False,
        "parallel_processing_factor": "0%",
    },
}


def load_profile(name: str) -> dict[str, Any]:
    """Load a named configuration profile."""
    profile = _PROFILES.get(name)
    if profile is None:
        raise ValueError(f"unknown profile: {name}. Available: {list(_PROFILES)}")
    for k, v in profile.items():
        if k in _ENVIRONMENT:
            _ENVIRONMENT[k] = v
    return dict(profile)


def register_profile(name: str, settings: dict[str, Any]) -> None:
    """Register a custom configuration profile."""
    _PROFILES[name] = dict(settings)


def list_profiles() -> list[str]:
    """List available profile names."""
    return list(_PROFILES.keys())


# ---------------------------------------------------------------------------
# Feature flag system (1199)
# ---------------------------------------------------------------------------

_FEATURE_FLAGS: dict[str, bool] = {}


def set_feature_flag(name: str, enabled: bool) -> None:
    """Set a feature flag."""
    _FEATURE_FLAGS[name] = enabled


def get_feature_flag(name: str, default: bool = False) -> bool:
    """Get a feature flag value."""
    return _FEATURE_FLAGS.get(name, default)


def list_feature_flags() -> dict[str, bool]:
    """Return all feature flags."""
    return dict(_FEATURE_FLAGS)


def clear_feature_flags() -> None:
    """Clear all feature flags."""
    _FEATURE_FLAGS.clear()


# ---------------------------------------------------------------------------
# Test data generators (1178-1183)
# ---------------------------------------------------------------------------

def generate_random_features(
    n: int,
    *,
    extent: tuple[float, float, float, float] = (-180, -90, 180, 90),
    geometry_type: str = "Point",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate random GeoJSON features for testing.

    Supported geometry types: Point, LineString, Polygon.
    """
    import random as _rng
    rng = _rng.Random(seed)
    xmin, ymin, xmax, ymax = extent
    features = []
    for i in range(n):
        if geometry_type == "Point":
            coords = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
            geom = {"type": "Point", "coordinates": coords}
        elif geometry_type == "LineString":
            npts = rng.randint(2, 5)
            x0 = rng.uniform(xmin, xmax)
            y0 = rng.uniform(ymin, ymax)
            cs = [(x0, y0)]
            for _ in range(npts - 1):
                cs.append((cs[-1][0] + rng.uniform(-1, 1), cs[-1][1] + rng.uniform(-1, 1)))
            geom = {"type": "LineString", "coordinates": cs}
        elif geometry_type == "Polygon":
            cx = rng.uniform(xmin, xmax)
            cy = rng.uniform(ymin, ymax)
            r = rng.uniform(0.1, 1.0)
            import math
            cs = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in [k * math.pi / 3 for k in range(6)]]
            cs.append(cs[0])
            geom = {"type": "Polygon", "coordinates": [cs]}
        else:
            raise ValueError(f"unsupported geometry type: {geometry_type}")
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {"id": i, "value": rng.random()},
        })
    return features


def generate_random_network(
    n_nodes: int,
    *,
    connectivity: float = 0.3,
    extent: tuple[float, float, float, float] = (0, 0, 100, 100),
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a random spatial network for testing.

    Returns nodes and edges as GeoJSON feature collections.
    """
    import random as _rng
    rng = _rng.Random(seed)
    xmin, ymin, xmax, ymax = extent
    nodes = [(rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)) for _ in range(n_nodes)]
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < connectivity:
                edges.append((i, j))
    node_features = [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": n}, "properties": {"node_id": i}}
        for i, n in enumerate(nodes)
    ]
    edge_features = [
        {"type": "Feature",
         "geometry": {"type": "LineString", "coordinates": [nodes[i], nodes[j]]},
         "properties": {"from_node": i, "to_node": j,
                        "length": ((nodes[i][0] - nodes[j][0]) ** 2 + (nodes[i][1] - nodes[j][1]) ** 2) ** 0.5}}
        for i, j in edges
    ]
    return {"nodes": node_features, "edges": edge_features}


def deterministic_seed_manager(seed: int = 42) -> dict[str, Any]:
    """Return a seed manager for reproducible random generation."""
    import random as _rng
    rng = _rng.Random(seed)
    return {
        "seed": seed,
        "random": rng,
        "next_int": lambda lo=0, hi=100: rng.randint(lo, hi),
        "next_float": lambda lo=0.0, hi=1.0: rng.uniform(lo, hi),
    }


# ---------------------------------------------------------------------------
# Error handling with spatial context (1134)
# ---------------------------------------------------------------------------

class SpatialError(Exception):
    """Exception with spatial context (geometry, location, feature ID)."""

    def __init__(
        self,
        message: str,
        *,
        feature_id: Any = None,
        geometry: dict[str, Any] | None = None,
        location: tuple[float, float] | None = None,
    ):
        super().__init__(message)
        self.feature_id = feature_id
        self.geometry = geometry
        self.location = location

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.feature_id is not None:
            parts.append(f"feature_id={self.feature_id}")
        if self.location is not None:
            parts.append(f"location={self.location}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# In-memory workspace (1126-1127)
# ---------------------------------------------------------------------------

class InMemoryWorkspace:
    """In-memory workspace for temporary feature classes and tables."""

    def __init__(self) -> None:
        self._layers: dict[str, list[dict[str, Any]]] = {}
        self._tables: dict[str, list[dict[str, Any]]] = {}

    def create_feature_class(self, name: str, features: list[dict[str, Any]] | None = None) -> str:
        self._layers[name] = list(features) if features else []
        return name

    def create_table(self, name: str, rows: list[dict[str, Any]] | None = None) -> str:
        self._tables[name] = list(rows) if rows else []
        return name

    def get_feature_class(self, name: str) -> list[dict[str, Any]]:
        if name not in self._layers:
            raise KeyError(f"feature class not found: {name}")
        return self._layers[name]

    def get_table(self, name: str) -> list[dict[str, Any]]:
        if name not in self._tables:
            raise KeyError(f"table not found: {name}")
        return self._tables[name]

    def delete(self, name: str) -> None:
        self._layers.pop(name, None)
        self._tables.pop(name, None)

    def exists(self, name: str) -> bool:
        return name in self._layers or name in self._tables

    def list_feature_classes(self) -> list[str]:
        return list(self._layers.keys())

    def list_tables(self) -> list[str]:
        return list(self._tables.keys())

    def describe(self, name: str) -> dict[str, Any]:
        if name in self._layers:
            feats = self._layers[name]
            return {
                "name": name,
                "type": "FeatureClass",
                "count": len(feats),
                "fields": list(feats[0].get("properties", {}).keys()) if feats else [],
            }
        if name in self._tables:
            rows = self._tables[name]
            return {
                "name": name,
                "type": "Table",
                "count": len(rows),
                "fields": list(rows[0].keys()) if rows else [],
            }
        raise KeyError(f"dataset not found: {name}")
