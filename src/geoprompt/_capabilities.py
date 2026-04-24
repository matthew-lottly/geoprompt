"""Central optional-dependency capability registry for GeoPrompt.

Provides:
- A machine-readable catalogue of every optional dependency, classified by
  failure mode (hard-fail, soft-fail, or degraded).
- ``require_capability`` / ``check_capability`` helpers that raise
  :class:`~._exceptions.DependencyError` with an actionable ``pip install``
  hint instead of a bare ``ImportError``.
- ``DegradedModePolicy`` – a deterministic fallback guard that prevents
  implicit fake outputs when a dependency is missing.
- ``ChunkSizer`` – adaptive / fixed / auto chunk-size estimation with
  run-time telemetry so callers can verify deterministic sizing decisions.
- ``capability_status`` – a snapshot dict of all registered capabilities
  consumed by the CLI capability-report command and test suites.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from ._exceptions import DependencyError


# ---------------------------------------------------------------------------
# Optional dependency catalogue  (J5.1)
# ---------------------------------------------------------------------------

class FailureMode(str, Enum):
    """How the system behaves when a capability's dependency is absent."""

    HARD_FAIL = "hard-fail"
    """Raise immediately; no useful output can be produced."""

    SOFT_FAIL = "soft-fail"
    """Raise with a clear message; partial results may be available via a
    different code path that does not require the dependency."""

    DEGRADED = "degraded"
    """Continue with reduced functionality and emit a warning; the output
    remains valid but may be less accurate, slower, or format-limited."""


@dataclass(frozen=True)
class CapabilitySpec:
    """Describes one optional dependency and its operational contract."""

    name: str
    """Logical capability identifier (e.g. ``"geopandas"``)."""

    import_name: str
    """Python module name used with ``importlib.import_module``."""

    pip_extra: str
    """Pip extra or package to install (e.g. ``"geoprompt[io]"``)."""

    failure_mode: FailureMode
    """Declared failure mode when the dependency is absent."""

    description: str = ""
    """One-line human description of what this capability enables."""

    affected_functions: tuple[str, ...] = field(default_factory=tuple)
    """Public API names whose behaviour is affected by this capability."""


# The canonical registry.  Every optional import in the codebase should have
# a corresponding entry here so the failure-mode is documented and testable.
CAPABILITY_REGISTRY: dict[str, CapabilitySpec] = {
    spec.name: spec
    for spec in [
        CapabilitySpec(
            name="geopandas",
            import_name="geopandas",
            pip_extra="geoprompt[io]",
            failure_mode=FailureMode.HARD_FAIL,
            description="Enables GeoDataFrame round-trips and shapefile export.",
            affected_functions=(
                "to_geodataframe",
                "from_geodataframe",
                "write_shapefile",
                "write_geoparquet",
                "read_geoparquet",
                "read_shapefile",
            ),
        ),
        CapabilitySpec(
            name="pandas",
            import_name="pandas",
            pip_extra="geoprompt[io]",
            failure_mode=FailureMode.HARD_FAIL,
            description="Enables DataFrame conversion, Excel, and Feather I/O.",
            affected_functions=(
                "to_dataframe",
                "from_dataframe",
                "read_excel",
                "write_excel",
                "read_feather",
                "write_feather",
            ),
        ),
        CapabilitySpec(
            name="pyarrow",
            import_name="pyarrow",
            pip_extra="geoprompt[io]",
            failure_mode=FailureMode.HARD_FAIL,
            description="Enables Apache Arrow / Feather and GeoParquet serialisation.",
            affected_functions=("to_arrow", "from_arrow", "write_feather", "read_feather"),
        ),
        CapabilitySpec(
            name="shapely",
            import_name="shapely",
            pip_extra="geoprompt[overlay]",
            failure_mode=FailureMode.SOFT_FAIL,
            description="Enables advanced geometry operations and overlay analysis.",
            affected_functions=(
                "geometry_union",
                "geometry_intersection",
                "geometry_difference",
                "geometry_buffer",
                "clip_frame",
            ),
        ),
        CapabilitySpec(
            name="pyproj",
            import_name="pyproj",
            pip_extra="geoprompt[projection]",
            failure_mode=FailureMode.SOFT_FAIL,
            description="Enables CRS transformation and coordinate projection.",
            affected_functions=("reproject_frame", "transform_geometry", "crs_info"),
        ),
        CapabilitySpec(
            name="fiona",
            import_name="fiona",
            pip_extra="geoprompt[io]",
            failure_mode=FailureMode.SOFT_FAIL,
            description="Enables reading of ESRI Shapefile, GeoPackage, and other OGR formats.",
            affected_functions=("read_shapefile", "read_vector"),
        ),
        CapabilitySpec(
            name="pyogrio",
            import_name="pyogrio",
            pip_extra="geoprompt[io]",
            failure_mode=FailureMode.SOFT_FAIL,
            description="Fast OGR vector I/O backend; falls back to fiona when absent.",
            affected_functions=("read_shapefile", "read_vector"),
        ),
        CapabilitySpec(
            name="rasterio",
            import_name="rasterio",
            pip_extra="geoprompt[raster]",
            failure_mode=FailureMode.HARD_FAIL,
            description="Enables raster read/write and band algebra.",
            affected_functions=("read_raster", "write_raster", "raster_algebra"),
        ),
        CapabilitySpec(
            name="matplotlib",
            import_name="matplotlib",
            pip_extra="geoprompt[viz]",
            failure_mode=FailureMode.DEGRADED,
            description="Enables static map and chart rendering.",
            affected_functions=("plot_frame", "choropleth_map", "histogram"),
        ),
        CapabilitySpec(
            name="folium",
            import_name="folium",
            pip_extra="geoprompt[viz]",
            failure_mode=FailureMode.DEGRADED,
            description="Enables interactive web-map export.",
            affected_functions=("to_folium_map", "interactive_map"),
        ),
        CapabilitySpec(
            name="plotly",
            import_name="plotly",
            pip_extra="geoprompt[viz]",
            failure_mode=FailureMode.DEGRADED,
            description="Enables Plotly chart and dashboard rendering.",
            affected_functions=("scatter_map", "density_plot"),
        ),
        CapabilitySpec(
            name="openpyxl",
            import_name="openpyxl",
            pip_extra="geoprompt[excel]",
            failure_mode=FailureMode.HARD_FAIL,
            description="Enables reading and writing of .xlsx Excel files.",
            affected_functions=("read_excel", "write_excel"),
        ),
        CapabilitySpec(
            name="sqlalchemy",
            import_name="sqlalchemy",
            pip_extra="geoprompt[db]",
            failure_mode=FailureMode.HARD_FAIL,
            description="Enables PostGIS database read/write via SQLAlchemy.",
            affected_functions=("read_postgis", "write_postgis"),
        ),
        CapabilitySpec(
            name="duckdb",
            import_name="duckdb",
            pip_extra="pip install duckdb",
            failure_mode=FailureMode.HARD_FAIL,
            description="Enables DuckDB in-process analytical database with spatial support.",
            affected_functions=("read_duckdb", "write_duckdb"),
        ),
        CapabilitySpec(
            name="numpy",
            import_name="numpy",
            pip_extra="geoprompt[network]",
            failure_mode=FailureMode.DEGRADED,
            description="Enables fast numeric operations and network analysis.",
            affected_functions=("network_centrality", "raster_statistics"),
        ),
        CapabilitySpec(
            name="fastapi",
            import_name="fastapi",
            pip_extra="geoprompt[service]",
            failure_mode=FailureMode.HARD_FAIL,
            description="Enables the REST service layer.",
            affected_functions=("serve",),
        ),
        CapabilitySpec(
            name="uvicorn",
            import_name="uvicorn",
            pip_extra="geoprompt[service]",
            failure_mode=FailureMode.HARD_FAIL,
            description="ASGI server required by the REST service layer.",
            affected_functions=("serve",),
        ),
        CapabilitySpec(
            name="ezdxf",
            import_name="ezdxf",
            pip_extra="pip install ezdxf",
            failure_mode=FailureMode.HARD_FAIL,
            description="Enables reading of AutoCAD DXF files.",
            affected_functions=("read_dxf",),
        ),
        CapabilitySpec(
            name="polars",
            import_name="polars",
            pip_extra="pip install polars",
            failure_mode=FailureMode.SOFT_FAIL,
            description="Enables Polars DataFrame conversion.",
            affected_functions=("to_polars", "from_polars"),
        ),
        CapabilitySpec(
            name="fsspec",
            import_name="fsspec",
            pip_extra="pip install fsspec[s3,gcs,azure]",
            failure_mode=FailureMode.HARD_FAIL,
            description="Enables cloud object-store I/O (S3, GCS, Azure Blob).",
            affected_functions=(
                "read_cloud_json",
                "write_cloud_json",
                "list_remote_dataset_entries",
                "inspect_remote_dataset_metadata",
            ),
        ),
        CapabilitySpec(
            name="scipy",
            import_name="scipy",
            pip_extra="pip install scipy",
            failure_mode=FailureMode.SOFT_FAIL,
            description="Enables Voronoi diagrams, Delaunay triangulation, and spatial stats.",
            affected_functions=("geometry_voronoi", "geometry_delaunay"),
        ),
        CapabilitySpec(
            name="dask",
            import_name="dask",
            pip_extra="pip install dask",
            failure_mode=FailureMode.DEGRADED,
            description="Enables parallel task execution via Dask delayed computation.",
            affected_functions=("parallel_map",),
        ),
        CapabilitySpec(
            name="osmium",
            import_name="osmium",
            pip_extra="pip install osmium",
            failure_mode=FailureMode.HARD_FAIL,
            description="Enables reading OpenStreetMap PBF files.",
            affected_functions=("read_osm_pbf",),
        ),
        CapabilitySpec(
            name="pystac",
            import_name="pystac",
            pip_extra="pip install pystac pystac-client",
            failure_mode=FailureMode.SOFT_FAIL,
            description="Enables STAC catalog traversal and item resolution.",
            affected_functions=("read_stac_catalog",),
        ),
    ]
}


# ---------------------------------------------------------------------------
# Capability check helpers  (J5.3, J5.7)
# ---------------------------------------------------------------------------

def _is_importable(import_name: str) -> bool:
    """Return True if *import_name* can be imported in the current environment."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def check_capability(name: str) -> bool:
    """Return ``True`` if the capability *name* is available, ``False`` otherwise.

    Does not raise; intended for conditional branching.
    """
    spec = CAPABILITY_REGISTRY.get(name)
    if spec is None:
        return False
    return _is_importable(spec.import_name)


def require_capability(name: str, *, context: str = "") -> None:
    """Assert that the capability *name* is available.

    Raises :class:`~._exceptions.DependencyError` with an actionable
    ``pip install`` hint when the dependency is absent.

    Args:
        name: Logical capability identifier (must be in ``CAPABILITY_REGISTRY``).
        context: Optional caller description used in the error message.

    Raises:
        DependencyError: When the capability is not importable.
        KeyError: When *name* is not found in the registry.
    """
    spec = CAPABILITY_REGISTRY[name]
    if not _is_importable(spec.import_name):
        ctx_part = f" (needed by {context})" if context else ""
        raise DependencyError(
            f"Optional dependency '{spec.import_name}' is required{ctx_part} but is not installed.\n"
            f"  {spec.description}\n"
            f"  Install with: pip install {spec.pip_extra}"
        )


def capability_status() -> dict[str, Any]:
    """Return a snapshot of all registered capabilities and their availability.

    Returns a dict suitable for CLI capability-report and test assertions::

        {
            "geopandas":  {"available": True,  "failure_mode": "hard-fail", ...},
            "shapely":    {"available": False, "failure_mode": "soft-fail", ...},
            ...
        }
    """
    return {
        name: {
            "available": _is_importable(spec.import_name),
            "failure_mode": spec.failure_mode.value,
            "pip_extra": spec.pip_extra,
            "description": spec.description,
            "affected_functions": list(spec.affected_functions),
        }
        for name, spec in CAPABILITY_REGISTRY.items()
    }


# ---------------------------------------------------------------------------
# Deterministic fallback policy  (J5.4)
# ---------------------------------------------------------------------------

class DegradedModePolicy:
    """Enforce a deterministic fallback policy when optional deps are absent.

    Prevents silent fake-output paths: callers must explicitly opt in to
    degraded behaviour and receive a warning; they never get silently wrong
    results.

    Example::

        policy = DegradedModePolicy("geopandas", context="write_shapefile")
        if not policy.available:
            policy.enforce()          # raises DependencyError
            # or
            policy.warn_degraded()    # UserWarning + continues
    """

    def __init__(self, capability: str, *, context: str = "") -> None:
        self._capability = capability
        self._context = context
        self._spec = CAPABILITY_REGISTRY.get(capability)
        self.available: bool = check_capability(capability)

    @property
    def failure_mode(self) -> FailureMode | None:
        return self._spec.failure_mode if self._spec else None

    def enforce(self) -> None:
        """Raise :class:`DependencyError` if the capability is unavailable."""
        if self.available:
            return
        if self._spec is None:
            raise DependencyError(
                f"Unknown optional dependency capability: {self._capability!r}"
            )
        if not self.available:
            require_capability(self._capability, context=self._context)

    def warn_degraded(self) -> None:
        """Issue a ``UserWarning`` (not an error) when running in degraded mode."""
        if not self.available and self._spec is not None:
            import warnings
            warnings.warn(
                f"Running in degraded mode: '{self._capability}' is not installed. "
                f"Install with: pip install {self._spec.pip_extra}",
                UserWarning,
                stacklevel=2,
            )

    def allow_degraded_or_raise(self, *, allow: bool = True) -> bool:
        """Return True if degraded mode is acceptable; raise otherwise.

        Args:
            allow: When False, always raises on missing dependency regardless
                of the registered failure mode.

        Returns:
            True if execution should proceed in degraded mode.
        """
        if self.available:
            return True
        if allow and self._spec is not None and self._spec.failure_mode == FailureMode.DEGRADED:
            self.warn_degraded()
            return True
        self.enforce()
        return False  # unreachable, satisfies type checkers


# ---------------------------------------------------------------------------
# Chunk-size estimation  (J5.12, J5.13, J5.14, J5.15)
# ---------------------------------------------------------------------------

class ChunkMode(str, Enum):
    """Chunking strategy for iterating large datasets."""

    FIXED = "fixed"
    """Always use the caller-supplied or preset chunk size."""

    ADAPTIVE = "adaptive"
    """Adjust chunk size based on measured row width and available memory."""

    AUTO = "auto"
    """Use FIXED when an explicit chunk_size is provided, ADAPTIVE otherwise."""


@dataclass
class ChunkDecision:
    """Telemetry record explaining why a particular chunk size was chosen.

    Consumers can inspect this to verify deterministic sizing decisions and
    audit why a chunk size differs between environments.
    """

    mode: ChunkMode
    chosen_size: int
    explicit_override: bool
    """True when the caller supplied an explicit chunk_size."""

    row_width_estimate_bytes: int | None = None
    """Estimated bytes per row used in the adaptive calculation."""

    memory_budget_bytes: int | None = None
    """Memory budget used for the adaptive calculation."""

    reasoning: str = ""
    """Human-readable explanation of the sizing decision."""


# Default memory budget for adaptive sizing: 128 MiB
_DEFAULT_MEMORY_BUDGET_BYTES: int = 128 * 1024 * 1024

# Minimum and maximum chunk sizes for adaptive mode
_ADAPTIVE_MIN_CHUNK: int = 1_000
_ADAPTIVE_MAX_CHUNK: int = 500_000


def estimate_chunk_size(
    *,
    columns: Iterable[str] | None = None,
    sample_row: dict[str, Any] | None = None,
    explicit_chunk_size: int | None = None,
    memory_budget_bytes: int | None = None,
    mode: ChunkMode | str = ChunkMode.AUTO,
) -> ChunkDecision:
    """Estimate an appropriate chunk size for iter_data / read_data workloads.

    This function is deterministic: the same inputs always produce the same
    output regardless of platform (no OS memory queries are used unless
    ``memory_budget_bytes`` is None and ``psutil`` is available).

    Args:
        columns: Iterable of column names to estimate row width from.
        sample_row: An actual sample row dict; overrides *columns* for the
            width estimate when provided.
        explicit_chunk_size: When not None, this value is always honoured and
            the mode reverts to FIXED regardless of the *mode* argument.
        memory_budget_bytes: Byte budget for each chunk.  Defaults to 128 MiB
            or a ``psutil``-derived available-memory fraction if psutil is
            installed.
        mode: Desired chunking mode.

    Returns:
        :class:`ChunkDecision` with the chosen size and full telemetry.
    """
    mode = ChunkMode(mode)

    # Explicit override always wins and reverts to FIXED mode  (J5.12 guarantee)
    if explicit_chunk_size is not None:
        if explicit_chunk_size <= 0:
            raise ValueError("explicit_chunk_size must be >= 1")
        return ChunkDecision(
            mode=ChunkMode.FIXED,
            chosen_size=explicit_chunk_size,
            explicit_override=True,
            reasoning=f"Explicit override: caller supplied chunk_size={explicit_chunk_size}",
        )

    effective_mode = ChunkMode.ADAPTIVE if mode == ChunkMode.AUTO else mode

    if effective_mode == ChunkMode.FIXED:
        # Default fixed size matches the WORKLOAD_PRESETS "large" preset
        chosen = 50_000
        return ChunkDecision(
            mode=ChunkMode.FIXED,
            chosen_size=chosen,
            explicit_override=False,
            reasoning=f"Fixed mode: using default chunk_size={chosen}",
        )

    # ADAPTIVE: estimate row width then compute chunk count from budget
    row_bytes = _estimate_row_bytes(sample_row=sample_row, columns=columns)
    budget = memory_budget_bytes if memory_budget_bytes is not None else _resolve_memory_budget()

    if row_bytes <= 0:
        chosen = 50_000
        return ChunkDecision(
            mode=ChunkMode.ADAPTIVE,
            chosen_size=chosen,
            explicit_override=False,
            row_width_estimate_bytes=row_bytes,
            memory_budget_bytes=budget,
            reasoning=f"Adaptive mode: row width estimate is 0; defaulting to {chosen}",
        )

    raw = budget // row_bytes
    chosen = max(_ADAPTIVE_MIN_CHUNK, min(_ADAPTIVE_MAX_CHUNK, raw))
    return ChunkDecision(
        mode=ChunkMode.ADAPTIVE,
        chosen_size=chosen,
        explicit_override=False,
        row_width_estimate_bytes=row_bytes,
        memory_budget_bytes=budget,
        reasoning=(
            f"Adaptive mode: estimated {row_bytes} B/row, "
            f"budget {budget} B → raw={raw}, "
            f"clamped to [{_ADAPTIVE_MIN_CHUNK}, {_ADAPTIVE_MAX_CHUNK}] → {chosen}"
        ),
    )


def _estimate_row_bytes(
    *,
    sample_row: dict[str, Any] | None,
    columns: Iterable[str] | None,
) -> int:
    """Estimate the in-memory byte footprint of a single row."""
    if sample_row is not None:
        try:
            import sys as _sys
            return sum(_sys.getsizeof(v) for v in sample_row.values())
        except Exception:
            return 256  # safe fallback

    if columns is not None:
        col_list = list(columns)
        if col_list:
            # Assume 64 bytes per column value (int/float/short string estimate)
            return len(col_list) * 64

    return 0


def _resolve_memory_budget() -> int:
    """Return an appropriate memory budget in bytes.

    If ``psutil`` is available, use 25 % of available virtual memory.
    Otherwise return the static default of 128 MiB so behaviour is
    deterministic in CI where psutil may not be installed.
    """
    try:
        import psutil  # type: ignore[import]
        return max(_DEFAULT_MEMORY_BUDGET_BYTES, psutil.virtual_memory().available // 4)
    except ImportError:
        return _DEFAULT_MEMORY_BUDGET_BYTES


# ---------------------------------------------------------------------------
# CI / environment matrix  (J5.8, J5.16)
# ---------------------------------------------------------------------------

#: Canonical extras profiles used in the CI test-environment matrix.
CI_EXTRAS_PROFILES: dict[str, list[str]] = {
    "core-only": [],
    "common": ["io", "viz", "overlay"],
    "analyst": ["analyst"],
    "all": ["all"],
}

#: Per-profile degraded-mode guarantees for core-only installs.
DEGRADED_MODE_GUARANTEES: dict[str, str] = {
    "core-only": (
        "All geometry, query, expression, and CLI commands work without optional "
        "dependencies.  I/O is limited to JSON/GeoJSON and plain CSV.  "
        "Shapefile, GeoParquet, Excel, and DB connectors raise DependencyError "
        "with a pip-install hint."
    ),
    "common": (
        "Adds GeoDataFrame round-trips (geopandas), Arrow serialisation (pyarrow), "
        "and web-map export (folium/plotly).  Raster, database, and service "
        "features still require their respective extras."
    ),
    "analyst": (
        "Analyst profile includes io + viz + overlay + excel + projection. "
        "Raster, database, and service features still require explicit extras."
    ),
    "all": (
        "All optional features are available.  No degraded-mode paths are "
        "expected to be triggered except by deliberate capability-mismatch tests."
    ),
}


__all__ = [
    "CAPABILITY_REGISTRY",
    "DEGRADED_MODE_GUARANTEES",
    "CI_EXTRAS_PROFILES",
    "CapabilitySpec",
    "ChunkDecision",
    "ChunkMode",
    "DegradedModePolicy",
    "FailureMode",
    "capability_status",
    "check_capability",
    "estimate_chunk_size",
    "require_capability",
]
