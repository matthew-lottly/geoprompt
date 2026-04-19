"""AI copilot, workspace-aware optimization, and intelligent automation helpers.

Provides dataset profiling, preset selection, backend routing, parameter
recommendations, quality control, and LLM-integration hooks.  All heavy
dependencies (numpy, shapely, etc.) are lazily imported so the module stays
lightweight when only pure-Python paths are needed.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable, Sequence

# ---------------------------------------------------------------------------
# Lazy optional imports
# ---------------------------------------------------------------------------

def _try_import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


# ── 1. Natural-language tool runner ──────────────────────────────────────────

_TOOL_CATALOG: dict[str, Callable[..., Any]] = {}


def register_tool(name: str, func: Callable[..., Any], description: str = "") -> None:
    """Register a callable in the AI tool catalog."""
    _TOOL_CATALOG[name] = func


def list_registered_tools() -> list[dict[str, str]]:
    """Return names and docstrings of all registered tools."""
    return [
        {"name": n, "description": (f.__doc__ or "").strip().split("\n")[0]}
        for n, f in _TOOL_CATALOG.items()
    ]


def natural_language_tool_runner(intent: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert analyst intent text into a reproducible GeoPrompt pipeline plan.

    Returns a dict with ``tool``, ``parameters``, and ``explanation`` keys.
    When an LLM backend is configured the intent is dispatched; otherwise a
    keyword-matching heuristic selects the best tool from the catalog.
    """
    intent_lower = intent.lower()
    best_match: str | None = None
    best_score = 0
    for name in _TOOL_CATALOG:
        tokens = name.replace("_", " ").split()
        score = sum(1 for t in tokens if t in intent_lower)
        if score > best_score:
            best_score = score
            best_match = name
    return {
        "tool": best_match,
        "parameters": context or {},
        "explanation": f"Matched '{best_match}' from intent: {intent}" if best_match else "No matching tool found",
    }


# ── 2. AI dataset profiler ──────────────────────────────────────────────────

def profile_dataset(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Inspect a record list and return a profile summary.

    Returned keys: ``row_count``, ``columns``, ``geometry_types``,
    ``null_rates``, ``crs``, ``bbox``, ``recommended_preset``.
    """
    if not records:
        return {"row_count": 0, "columns": [], "geometry_types": set(), "null_rates": {}, "crs": None, "bbox": None, "recommended_preset": "small"}

    sample = records[:1000]
    columns = sorted({k for r in sample for k in r})
    null_rates: dict[str, float] = {}
    for col in columns:
        nulls = sum(1 for r in sample if r.get(col) is None)
        null_rates[col] = round(nulls / len(sample), 4)

    geom_types: set[str] = set()
    xs: list[float] = []
    ys: list[float] = []
    crs: str | None = None
    for r in sample:
        g = r.get("geometry")
        if isinstance(g, dict):
            geom_types.add(g.get("type", "Unknown"))
            coords = g.get("coordinates")
            if coords and isinstance(coords, (list, tuple)):
                flat = coords
                while flat and isinstance(flat[0], (list, tuple)):
                    flat = flat[0]
                if len(flat) >= 2:
                    xs.append(float(flat[0]))
                    ys.append(float(flat[1]))
        if "crs" in r:
            crs = str(r["crs"])

    bbox = None
    if xs and ys:
        bbox = [min(xs), min(ys), max(xs), max(ys)]

    n = len(records)
    preset = "small" if n < 10_000 else "medium" if n < 100_000 else "large" if n < 1_000_000 else "huge"

    return {
        "row_count": n,
        "columns": columns,
        "geometry_types": sorted(geom_types),
        "null_rates": null_rates,
        "crs": crs,
        "bbox": bbox,
        "recommended_preset": preset,
    }


# ── 3. Auto-preset selector ─────────────────────────────────────────────────

def auto_select_preset(row_count: int, *, memory_mb: int | None = None) -> str:
    """Choose small / medium / large / huge execution preset by data size."""
    if memory_mb is not None and memory_mb < 512:
        thresholds = (2_000, 20_000, 200_000)
    else:
        thresholds = (10_000, 100_000, 1_000_000)
    if row_count < thresholds[0]:
        return "small"
    if row_count < thresholds[1]:
        return "medium"
    if row_count < thresholds[2]:
        return "large"
    return "huge"


# ── 4. Adaptive chunk sizing ────────────────────────────────────────────────

def adaptive_chunk_size(
    row_count: int,
    *,
    available_memory_mb: int | None = None,
    avg_row_bytes: int = 512,
) -> int:
    """Return a recommended chunk size for iterating over large datasets."""
    if available_memory_mb is None:
        try:
            import psutil  # type: ignore[import-untyped]
            available_memory_mb = int(psutil.virtual_memory().available / (1024 * 1024))
        except Exception:
            available_memory_mb = 2048
    target_bytes = int(available_memory_mb * 1024 * 1024 * 0.25)
    chunk = max(100, target_bytes // max(avg_row_bytes, 1))
    return min(chunk, row_count)


# ── 5. Backend auto-selection ────────────────────────────────────────────────

_BACKENDS = ("shapely", "numpy", "geopandas", "rasterio")


def detect_available_backends() -> dict[str, bool]:
    """Check which optional backends are installed."""
    return {name: _try_import(name) is not None for name in _BACKENDS}


def auto_select_backend(operation: str = "geometry") -> str:
    """Pick the best backend for *operation* from what is installed.

    Returns one of ``'pure_python'``, ``'numpy'``, ``'shapely'``,
    ``'geopandas'``, or ``'rasterio'``.
    """
    avail = detect_available_backends()
    if operation in ("raster", "rasterize", "mosaic") and avail.get("rasterio"):
        return "rasterio"
    if operation in ("overlay", "union", "identity", "erase") and avail.get("shapely"):
        return "shapely"
    if operation in ("join", "sjoin", "spatial_join") and avail.get("geopandas"):
        return "geopandas"
    if avail.get("numpy"):
        return "numpy"
    if avail.get("shapely"):
        return "shapely"
    return "pure_python"


# ── 6. Cost-based join planner ───────────────────────────────────────────────

def plan_spatial_join(
    left_count: int,
    right_count: int,
    *,
    has_index: bool = False,
) -> dict[str, Any]:
    """Recommend a spatial join strategy for the given data shape.

    Returns ``strategy``, ``estimated_complexity``, ``recommendation``.
    """
    if has_index or right_count > 5_000:
        strategy = "indexed_rtree"
        complexity = f"O({left_count} * log({right_count}))"
        note = "Use indexed join with prebuilt spatial index on right side."
    elif left_count * right_count < 1_000_000:
        strategy = "nested_loop"
        complexity = f"O({left_count * right_count})"
        note = "Small enough for brute-force nested loop."
    else:
        strategy = "partition_sweep"
        complexity = "O(n*log(n))"
        note = "Partition both sides by bounding-box grid, then sweep."
    return {"strategy": strategy, "estimated_complexity": complexity, "recommendation": note}


# ── 7. Cache-aware execution planner ─────────────────────────────────────────

class ExecutionCache:
    """Simple in-memory cache for intermediate results and spatial indexes."""

    def __init__(self, max_entries: int = 64) -> None:
        self._store: dict[str, Any] = {}
        self._max = max_entries
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any:
        if key in self._store:
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if len(self._store) >= self._max:
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = value

    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0


_GLOBAL_CACHE = ExecutionCache()


def get_execution_cache() -> ExecutionCache:
    """Return the global execution cache singleton."""
    return _GLOBAL_CACHE


# ── 8. AI-generated performance warnings ─────────────────────────────────────

def performance_warnings(
    row_count: int,
    operation: str = "join",
    *,
    right_count: int = 0,
) -> list[str]:
    """Return human-readable warnings about potential performance issues."""
    warnings: list[str] = []
    if row_count > 500_000:
        warnings.append(f"Dataset has {row_count:,} rows — consider chunked processing or a 'large' preset.")
    if operation == "join" and right_count > 100_000 and row_count > 100_000:
        warnings.append("Both sides of the join are large; ensure a spatial index is built on the right side.")
    if operation in ("buffer", "overlay") and row_count > 50_000:
        warnings.append(f"'{operation}' on {row_count:,} features may be slow without Shapely installed.")
    backends = detect_available_backends()
    if not any(backends.values()):
        warnings.append("No optional backends detected — running in pure-Python mode. Install numpy or shapely for faster execution.")
    return warnings


# ── 9. Parameter recommendation engine ───────────────────────────────────────

def recommend_parameters(
    operation: str,
    *,
    bbox: Sequence[float] | None = None,
    row_count: int = 0,
    geometry_type: str = "Point",
) -> dict[str, Any]:
    """Suggest sensible defaults for common spatial operations.

    Covers ``buffer_distance``, ``search_radius``, ``cell_size``, ``k``.
    """
    extent = 1.0
    if bbox and len(bbox) >= 4:
        dx = abs(bbox[2] - bbox[0])
        dy = abs(bbox[3] - bbox[1])
        extent = max(dx, dy) or 1.0

    params: dict[str, Any] = {"operation": operation}
    if operation == "buffer":
        params["buffer_distance"] = round(extent * 0.01, 6)
    elif operation == "search":
        params["search_radius"] = round(extent * 0.05, 6)
    elif operation in ("density", "interpolation"):
        params["cell_size"] = round(extent / 100.0, 6)
    elif operation == "nearest":
        params["k"] = min(5, max(1, row_count // 100))
    else:
        params["note"] = "No specific recommendation; review data extent and density."
    return params


# ── 10. CRS recommendation helper ───────────────────────────────────────────

_UTM_ZONE_LOOKUP = {
    range(-180, -174): 1, range(-174, -168): 2, range(-168, -162): 3,
    range(-162, -156): 4, range(-156, -150): 5, range(-150, -144): 6,
    range(-144, -138): 7, range(-138, -132): 8, range(-132, -126): 9,
    range(-126, -120): 10, range(-120, -114): 11, range(-114, -108): 12,
    range(-108, -102): 13, range(-102, -96): 14, range(-96, -90): 15,
    range(-90, -84): 16, range(-84, -78): 17, range(-78, -72): 18,
    range(-72, -66): 19, range(-66, -60): 20, range(-60, -54): 21,
}


def recommend_crs(longitude: float, latitude: float) -> dict[str, Any]:
    """Suggest a projected CRS appropriate for the given location.

    Returns ``epsg``, ``name``, and ``reason``.
    """
    zone = int((longitude + 180) / 6) + 1
    hemisphere = "N" if latitude >= 0 else "S"
    epsg = 32600 + zone if hemisphere == "N" else 32700 + zone
    return {
        "epsg": f"EPSG:{epsg}",
        "name": f"WGS 84 / UTM zone {zone}{hemisphere}",
        "reason": f"Best local UTM zone for lon={longitude:.2f}, lat={latitude:.2f}",
    }


# ── 11. AI geometry-repair advisor ───────────────────────────────────────────

def geometry_repair_advice(geometry: dict[str, Any]) -> dict[str, Any]:
    """Explain why a geometry is invalid and suggest the next fix.

    Returns ``is_valid``, ``issue``, ``suggested_fix``.
    """
    from geoprompt.geometry import validate_geometry

    report = validate_geometry(geometry)
    is_valid = report.get("is_valid", False)
    if is_valid:
        return {"is_valid": True, "issue": None, "suggested_fix": None}

    reason = report.get("reason", "Unknown validity issue")
    fix_map = {
        "ring": "Use geometry_repair() to fix ring orientation, or buffer(0) to clean topology.",
        "self-intersection": "Apply buffer(0) or geometry_repair() to resolve self-intersections.",
        "duplicate": "Remove duplicate consecutive vertices with geometry_simplify(tolerance=0).",
        "unclosed": "Close the ring by appending the first coordinate as the last.",
        "empty": "Feature has empty geometry — remove or replace it.",
    }
    suggested = "Run geometry_repair() as a general first step."
    for keyword, fix in fix_map.items():
        if keyword in reason.lower():
            suggested = fix
            break
    return {"is_valid": False, "issue": reason, "suggested_fix": suggested}


# ── 12. Quality-control assistant ────────────────────────────────────────────

def quality_control_scan(
    records: Sequence[dict[str, Any]],
    *,
    expected_count: int | None = None,
    check_gaps: bool = True,
    check_overlaps: bool = False,
) -> dict[str, Any]:
    """Scan output records for suspicious counts, nulls, and outliers.

    Returns ``issues`` list and ``passed`` boolean.
    """
    issues: list[str] = []
    n = len(records)
    if expected_count is not None and n != expected_count:
        issues.append(f"Row count {n} differs from expected {expected_count}.")
    if n == 0:
        issues.append("Output is empty.")
        return {"passed": False, "issues": issues}

    null_geom = sum(1 for r in records if r.get("geometry") is None)
    if null_geom > 0:
        issues.append(f"{null_geom} records have null geometry ({100*null_geom/n:.1f}%).")

    numeric_cols: dict[str, list[float]] = {}
    for r in records[:5000]:
        for k, v in r.items():
            if isinstance(v, (int, float)) and k != "geometry":
                numeric_cols.setdefault(k, []).append(float(v))

    for col, vals in numeric_cols.items():
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        if std > 0:
            outliers = sum(1 for v in vals if abs(v - mean) > 3 * std)
            if outliers > 0:
                issues.append(f"Column '{col}': {outliers} values beyond 3σ.")

    return {"passed": len(issues) == 0, "issues": issues}


# ── 13. Auto-benchmark runner ────────────────────────────────────────────────

def auto_benchmark(
    implementations: dict[str, Callable[[], Any]],
    *,
    rounds: int = 3,
) -> dict[str, Any]:
    """Run multiple implementations and report the fastest.

    Returns per-implementation timing and the ``fastest`` name.
    """
    results: dict[str, float] = {}
    for name, fn in implementations.items():
        times: list[float] = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        results[name] = round(min(times), 6)
    fastest = min(results, key=results.get) if results else None  # type: ignore[arg-type]
    return {"timings": results, "fastest": fastest}


# ── 14. Notebook explainer ───────────────────────────────────────────────────

def explain_step(description: str, inputs: list[str], outputs: list[str]) -> str:
    """Write a human-readable step summary for notebooks and reviews."""
    parts = [f"**Step:** {description}"]
    if inputs:
        parts.append(f"  Inputs: {', '.join(inputs)}")
    if outputs:
        parts.append(f"  Outputs: {', '.join(outputs)}")
    return "\n".join(parts)


def explain_pipeline(steps: Sequence[dict[str, Any]]) -> str:
    """Generate a readable pipeline narrative from a list of step dicts."""
    lines = ["## Pipeline Summary", ""]
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. **{step.get('name', 'Unnamed')}** — {step.get('description', '')}")
        if step.get("inputs"):
            lines.append(f"   - Reads: {', '.join(step['inputs'])}")
        if step.get("outputs"):
            lines.append(f"   - Produces: {', '.join(step['outputs'])}")
    return "\n".join(lines)


# ── 15. Prompt-to-report generator ───────────────────────────────────────────

def prompt_to_report(
    scenario_results: dict[str, Any],
    *,
    title: str = "Scenario Report",
    audience: str = "executive",
) -> str:
    """Build an executive summary from scenario results.

    Returns an HTML string suitable for export.
    """
    rows = ""
    for k, v in scenario_results.items():
        rows += f"<tr><td>{k}</td><td>{v}</td></tr>\n"
    tone = "concise high-level" if audience == "executive" else "detailed technical"
    return f"""<!DOCTYPE html>
<html><head><title>{title}</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:8px;text-align:left}}th{{background:#f5f5f5}}</style></head>
<body><h1>{title}</h1><p>Audience: {tone}</p>
<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table></body></html>"""


# ── 16. AI field-mapping helper ──────────────────────────────────────────────

def suggest_field_mapping(
    source_columns: Sequence[str],
    target_columns: Sequence[str],
) -> dict[str, str | None]:
    """Suggest column mappings between messy source and clean target schemas.

    Uses normalized string matching with common GIS synonyms.
    """
    synonyms: dict[str, list[str]] = {
        "id": ["fid", "objectid", "oid", "gid", "feature_id"],
        "name": ["label", "title", "description", "desc"],
        "geometry": ["shape", "geom", "the_geom", "wkb_geometry"],
        "latitude": ["lat", "y", "ycoord", "y_coord"],
        "longitude": ["lon", "lng", "x", "xcoord", "x_coord"],
    }
    reverse: dict[str, str] = {}
    for canonical, alts in synonyms.items():
        for a in alts:
            reverse[a] = canonical
        reverse[canonical] = canonical

    mapping: dict[str, str | None] = {}
    target_lower = {t.lower().replace(" ", "_"): t for t in target_columns}
    for src in source_columns:
        src_norm = src.lower().replace(" ", "_").strip("_")
        canonical = reverse.get(src_norm, src_norm)
        match = target_lower.get(canonical) or target_lower.get(src_norm)
        mapping[src] = match
    return mapping


# ── 17. Auto-detection of join key fields ────────────────────────────────────

def detect_join_keys(
    left_columns: Sequence[str],
    right_columns: Sequence[str],
) -> list[tuple[str, str]]:
    """Find candidate key pairs between two column lists.

    Returns pairs ``(left_col, right_col)`` based on name similarity.
    """
    pairs: list[tuple[str, str]] = []
    for lc in left_columns:
        ln = lc.lower().replace(" ", "_")
        for rc in right_columns:
            rn = rc.lower().replace(" ", "_")
            if ln == rn or ln.endswith(f"_{rn}") or rn.endswith(f"_{ln}"):
                pairs.append((lc, rc))
    return pairs


# ── 18. Duplicate-resolution suggestions ─────────────────────────────────────

def suggest_duplicate_resolution(
    records: Sequence[dict[str, Any]],
    key_columns: Sequence[str],
) -> dict[str, Any]:
    """Identify duplicate groups and suggest resolution strategies."""
    groups: dict[tuple[Any, ...], list[int]] = {}
    for i, r in enumerate(records):
        key = tuple(r.get(c) for c in key_columns)
        groups.setdefault(key, []).append(i)
    duplicates = {str(k): idxs for k, idxs in groups.items() if len(idxs) > 1}
    return {
        "duplicate_groups": len(duplicates),
        "total_duplicates": sum(len(v) - 1 for v in duplicates.values()),
        "suggestion": "Keep first occurrence per group, or merge attributes where values differ.",
        "groups": duplicates,
    }


# ── 19. Semantic search across docs ─────────────────────────────────────────

def search_docs(
    query: str,
    *,
    docs_dir: str | Path = "docs",
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Simple keyword search across markdown docs in the project."""
    results: list[dict[str, Any]] = []
    docs_path = Path(docs_dir)
    if not docs_path.is_dir():
        return results
    tokens = query.lower().split()
    for md in sorted(docs_path.rglob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        text_lower = text.lower()
        score = sum(text_lower.count(t) for t in tokens)
        if score > 0:
            snippet_idx = min((text_lower.find(t) for t in tokens if t in text_lower), default=0)
            snippet = text[max(0, snippet_idx - 40): snippet_idx + 120].strip()
            results.append({"file": str(md), "score": score, "snippet": snippet})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:max_results]


# ── 20. Workspace memory ────────────────────────────────────────────────────

class WorkspaceMemory:
    """Persist preferred presets and settings across sessions."""

    def __init__(self, path: str | Path = ".geoprompt_memory.json") -> None:
        self._path = Path(path)
        self._data: dict[str, Any] = {}
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def all(self) -> dict[str, Any]:
        return dict(self._data)

    def clear(self) -> None:
        self._data.clear()
        if self._path.exists():
            self._path.unlink()


# ── 21. Auto-retry and backoff ───────────────────────────────────────────────

def with_retry(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    backoff_factor: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Call *func* with exponential backoff on failure."""
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = backoff_factor * (2 ** attempt)
                time.sleep(wait)
    raise RuntimeError(f"All {max_attempts} attempts failed") from last_exc


# ── 22. Hardware-aware execution tuning ──────────────────────────────────────

def detect_hardware_profile() -> dict[str, Any]:
    """Return a hardware profile dict for execution tuning."""
    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1
    try:
        import psutil  # type: ignore[import-untyped]
        mem_mb = int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        mem_mb = 4096  # conservative default
    return {
        "platform": platform.system(),
        "cpu_count": cpu_count,
        "memory_mb": mem_mb,
        "python_version": platform.python_version(),
        "tier": "workstation" if cpu_count >= 8 and mem_mb >= 16384 else "laptop" if mem_mb >= 4096 else "ci",
    }


# ── 23. Local model integration hook ────────────────────────────────────────

def query_local_model(
    prompt: str,
    *,
    model_path: str | None = None,
    endpoint: str | None = None,
) -> str:
    """Send a prompt to a local LLM for secure offline assistance.

    Supports a local HTTP endpoint or falls back to a stub response.
    """
    if endpoint:
        import urllib.request
        req = urllib.request.Request(
            endpoint,
            data=json.dumps({"prompt": prompt}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            return body.get("response", body.get("text", str(body)))
    return f"[local-model-stub] Received prompt of {len(prompt)} chars. Configure endpoint for real inference."


# ── 24. LLM-guided topology validation narratives ───────────────────────────

def topology_validation_narrative(violations: Sequence[dict[str, Any]]) -> str:
    """Generate a plain-English narrative for topology violations."""
    if not violations:
        return "All topology rules passed — no violations detected."
    lines = [f"Found {len(violations)} topology violation(s):\n"]
    for i, v in enumerate(violations, 1):
        rule = v.get("rule", "unknown")
        feat = v.get("feature_index", "?")
        desc = v.get("description", "")
        lines.append(f"  {i}. Feature #{feat} violates **{rule}**{': ' + desc if desc else ''}.")
    lines.append("\nRecommendation: Review flagged features and apply targeted repairs.")
    return "\n".join(lines)


# ── 25. Tool recommendation engine ──────────────────────────────────────────

_TOOL_SEQUENCES: dict[str, list[str]] = {
    "read_data": ["profile_dataset", "validate_schema"],
    "spatial_join": ["build_spatial_index", "quality_control_scan"],
    "buffer": ["dissolve", "overlay_intersections"],
    "overlay": ["quality_control_scan", "write_data"],
    "dissolve": ["write_data", "to_choropleth"],
    "write_data": ["export_provenance_bundle"],
}


def recommend_next_tool(last_tool: str) -> list[str]:
    """Suggest the next tool(s) in a typical GeoPrompt pipeline."""
    return _TOOL_SEQUENCES.get(last_tool, ["quality_control_scan", "write_data"])


# ---------------------------------------------------------------------------
__all__ = [
    "adaptive_chunk_size",
    "auto_benchmark",
    "auto_select_backend",
    "auto_select_preset",
    "detect_available_backends",
    "detect_hardware_profile",
    "detect_join_keys",
    "ExecutionCache",
    "explain_pipeline",
    "explain_step",
    "geometry_repair_advice",
    "get_execution_cache",
    "list_registered_tools",
    "natural_language_tool_runner",
    "performance_warnings",
    "plan_spatial_join",
    "profile_dataset",
    "prompt_to_report",
    "quality_control_scan",
    "query_local_model",
    "recommend_crs",
    "recommend_next_tool",
    "recommend_parameters",
    "register_tool",
    "search_docs",
    "suggest_duplicate_resolution",
    "suggest_field_mapping",
    "topology_validation_narrative",
    "with_retry",
    "WorkspaceMemory",
]
