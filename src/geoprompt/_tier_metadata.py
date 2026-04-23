"""API tier metadata system for GeoPrompt.

Defines stability tiers for all public functions:
- STABLE: Production-ready, 5+ releases, 95%+ coverage, committed to API stability
- BETA: Actively tested, 2+ releases, 80%+ coverage, may evolve with notice
- EXPERIMENTAL: Under development, <2 releases, 50%+ coverage, subject to change
- SIMULATION_ONLY: Stub implementations, 0% real functionality, for roadmap only

Usage:
    from ._tier_metadata import TIER_METADATA, warn_if_non_stable, get_tier

    # Check tier of a function
    tier = get_tier('natural_neighbor_interpolation')

    # Add tier-aware deprecation warning to non-stable functions
    warn_if_non_stable('experimental_function')
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

TIER_STABLE = "stable"
TIER_BETA = "beta"
TIER_EXPERIMENTAL = "experimental"
TIER_SIMULATION = "simulation_only"

TierLevel = Literal["stable", "beta", "experimental", "simulation_only"]

# Comprehensive tier metadata for GeoPrompt functions
# Format: "module.function": "tier_level"
# This is the source of truth for function maturity levels
TIER_METADATA: dict[str, TierLevel] = {
    # Frame operations (frame.py) - mostly stable
    "frame.GeoPromptFrame": TIER_STABLE,
    "frame.explore": TIER_STABLE,
    "frame.plot": TIER_STABLE,
    "frame.dissolve": TIER_STABLE,
    "frame.spatial_join": TIER_STABLE,
    "frame.nearest_join": TIER_STABLE,
    "frame.buffer": TIER_STABLE,
    "frame.clip": TIER_STABLE,
    
    # Geometry operations (geometry.py) - mostly stable with some experimental
    "geometry.hausdorff_distance": TIER_STABLE,
    "geometry.frechet_distance": TIER_STABLE,
    "geometry.offset_curve": TIER_STABLE,
    "geometry.clip_by_rect": TIER_STABLE,
    "geometry.concave_hull": TIER_BETA,  # Needs more real-world testing
    "geometry.minimum_rotated_rectangle": TIER_STABLE,
    "geometry.minimum_bounding_circle": TIER_STABLE,
    "geometry.representative_point": TIER_STABLE,
    "geometry.get_coordinates": TIER_STABLE,
    "geometry.affine_transform": TIER_BETA,  # M-coordinate support partial
    "geometry.buffer_3d_sphere": TIER_EXPERIMENTAL,
    
    # Statistics (stats.py) - mixed maturity
    "stats.moran_i": TIER_STABLE,
    "stats.geary_c": TIER_STABLE,
    "stats.ripley_k": TIER_STABLE,
    "stats.clark_evans": TIER_STABLE,
    "stats.natural_neighbor_interpolation": TIER_BETA,  # IDW fallback, not true Sibson
    "stats.variogram_fit": TIER_BETA,  # Simple heuristic, not robust model fitting
    "stats.kriging": TIER_STABLE,
    "stats.idw_interpolation": TIER_STABLE,
    "stats.rbf_interpolation": TIER_STABLE,
    "stats.gaussian_kde": TIER_STABLE,
    "stats.gwr": TIER_BETA,  # O(n^3) bandwidth computation
    
    # Network operations (network/) - mostly stable
    "network.vrp_solver": TIER_STABLE,
    "network.tsp_solver": TIER_STABLE,
    "network.location_allocation": TIER_STABLE,
    "network.closest_facility": TIER_STABLE,
    "network.multimodal_network": TIER_BETA,
    "network.network_partition": TIER_STABLE,
    "network.service_area": TIER_STABLE,
    "network.accessibility_analysis": TIER_BETA,
    
    # Raster operations (raster.py) - mixed maturity
    "raster.contour_generation": TIER_STABLE,
    "raster.flow_direction_raster": TIER_STABLE,
    "raster.raster_slope_aspect": TIER_STABLE,
    "raster.raster_hillshade": TIER_STABLE,
    "raster.raster_watershed": TIER_STABLE,
    "raster.zonal_summary": TIER_STABLE,
    "raster.raster_curvature": TIER_BETA,
    "raster.terrain_ruggedness_index": TIER_BETA,
    "raster.topographic_wetness_index": TIER_BETA,
    "raster.raster_cost_distance": TIER_STABLE,
    "raster.raster_least_cost_path": TIER_STABLE,
    "raster.neighborhood_operation": TIER_EXPERIMENTAL,  # New in recent release
    
    # I/O operations (io.py) - mostly stable
    "io.read_geojson": TIER_STABLE,
    "io.write_geojson": TIER_STABLE,
    "io.read_shapefile": TIER_STABLE,
    "io.write_shapefile": TIER_STABLE,
    "io.read_gpkg": TIER_STABLE,
    "io.write_gpkg": TIER_STABLE,
    "io.read_cloud_json": TIER_BETA,
    "io.write_cloud_json": TIER_BETA,
    "io.read_dxf": TIER_BETA,  # Falls back to stub without fiona
    "io.read_osm": TIER_BETA,  # Falls back to stub without osmnx
    
    # ML operations (ml.py) - mostly experimental
    "ml.gradient_boosted_spatial_prediction": TIER_SIMULATION,  # Stub
    "ml.svm_spatial_classification": TIER_SIMULATION,  # Stub
    "ml.spatial_cross_validation": TIER_BETA,
    "ml.kmeans_clustering": TIER_STABLE,
    "ml.dbscan_clustering": TIER_STABLE,
    
    # Geoprocessing (geoprocessing.py)
    "geoprocessing.ModelBuilder": TIER_STABLE,
    "geoprocessing.ToolChain": TIER_BETA,
    "geoprocessing.batch_process": TIER_BETA,
    "geoprocessing.get_environment": TIER_STABLE,
    "geoprocessing.set_environment": TIER_STABLE,
    
    # Standards (standards.py) - mostly simulation
    "standards.wms_capabilities_document": TIER_SIMULATION,
    "standards.wfs_capabilities_document": TIER_SIMULATION,
    "standards.gml_to_features": TIER_SIMULATION,
    "standards.wkt_service_query": TIER_SIMULATION,
    "standards.ogc_filter_encoding": TIER_SIMULATION,
    
    # Enterprise (enterprise.py)
    "enterprise.connect_geodatabase": TIER_SIMULATION,
    "enterprise.replicate_geodatabase": TIER_SIMULATION,
    "enterprise.validate_geodatabase_schema": TIER_SIMULATION,
    
    # Service (service.py)
    "service.start_service": TIER_STABLE,
    "service.create_feature_service": TIER_BETA,
    "service.enforce_rbac": TIER_BETA,
    
    # Security (security.py)
    "security.validate_geometry_payload": TIER_STABLE,
    "security.rate_limiter": TIER_STABLE,
    "security.check_injection_risk": TIER_STABLE,
    
    # Data management (data_management.py)
    "data_management.add_field": TIER_STABLE,
    "data_management.delete_field": TIER_STABLE,
    "data_management.field_calculate": TIER_STABLE,
    "data_management.near_table_multi": TIER_STABLE,
    "data_management.topology_validate": TIER_BETA,
    
    # AI/copilot (ai.py)
    "ai.natural_language_tool_runner": TIER_BETA,
    "ai.recommend_parameters": TIER_BETA,
    "ai.explain_pipeline": TIER_EXPERIMENTAL,
    "ai.auto_select_backend": TIER_BETA,
    
    # Visualization (visualization.py)
    "visualization.map_series": TIER_BETA,
    "visualization.time_slider_control": TIER_BETA,
    "visualization.interactive_dashboard": TIER_EXPERIMENTAL,
    
    # Performance (performance.py)
    "performance.benchmark_suite": TIER_BETA,
    "performance.profile_operation": TIER_STABLE,
    "performance.scale_analysis": TIER_EXPERIMENTAL,
    
    # Quality (quality.py)
    "quality.coverage_report": TIER_BETA,
    "quality.api_stability_check": TIER_STABLE,
}


def get_tier(function_name: str) -> TierLevel | None:
    """Get tier level for a function.
    
    Args:
        function_name: Name of function, optionally with module prefix (e.g., 'stats.moran_i' or 'moran_i')
    
    Returns:
        Tier level string, or None if not found
    """
    # Try with module prefix first
    if function_name in TIER_METADATA:
        return TIER_METADATA[function_name]
    
    # Try to find any match (search across modules)
    for key in TIER_METADATA:
        if key.endswith("." + function_name):
            return TIER_METADATA[key]
    
    return None


def warn_if_non_stable(function_name: str, *, stacklevel: int = 2) -> None:
    """Issue warning if function is not marked as stable.
    
    Args:
        function_name: Name of function being called
        stacklevel: Stack level for warning (caller's context)
    
    Returns:
        None (issues warning side effect)
    """
    tier = get_tier(function_name)
    
    if tier is None:
        # Function not in metadata, assume stable to avoid noise
        return
    
    if tier == TIER_STABLE:
        return
    elif tier == TIER_BETA:
        warnings.warn(
            f"{function_name} is BETA and may change in a future release. "
            f"Please report issues via GitHub.",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )
    elif tier == TIER_EXPERIMENTAL:
        warnings.warn(
            f"{function_name} is EXPERIMENTAL and subject to significant change. "
            f"Not recommended for production use.",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )
    elif tier == TIER_SIMULATION:
        warnings.warn(
            f"{function_name} is SIMULATION-ONLY (stub) and does not perform real operations. "
            f"See docs/roadmap for completion status.",
            UserWarning,
            stacklevel=stacklevel + 1,
        )


def tier_description(tier: TierLevel) -> str:
    """Get human-readable description of tier level."""
    descriptions = {
        TIER_STABLE: "Production-ready, stable API",
        TIER_BETA: "Actively tested, subject to change",
        TIER_EXPERIMENTAL: "Under development, may change significantly",
        TIER_SIMULATION: "Stub implementation, not functional",
    }
    return descriptions.get(tier, "Unknown tier")


def add_tier_info_to_docstring(docstring: str, tier: TierLevel) -> str:
    """Prepend tier info to docstring.
    
    Args:
        docstring: Original docstring
        tier: Tier level
    
    Returns:
        Updated docstring with tier info
    """
    tier_badge = f"[{tier.upper()}] {tier_description(tier)}"
    
    if docstring is None:
        return tier_badge
    
    # If docstring already has tier info, don't add again
    if "[STABLE]" in docstring or "[BETA]" in docstring or "[EXPERIMENTAL]" in docstring or "[SIMULATION_ONLY]" in docstring:
        return docstring
    
    return f"{tier_badge}\n\n{docstring}"


def decorator_tier(tier: TierLevel):
    """Decorator to mark a function with its tier level.
    
    Usage:
        @decorator_tier(TIER_BETA)
        def my_function():
            pass
    """
    def decorator(func):
        # Store tier in function attribute
        func._geoprompt_tier = tier
        
        # Add tier info to docstring
        original_docstring = func.__doc__ or ""
        func.__doc__ = add_tier_info_to_docstring(original_docstring, tier)
        
        return func
    
    return decorator
