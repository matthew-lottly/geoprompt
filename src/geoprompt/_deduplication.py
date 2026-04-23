"""Deduplication registry and helpers for GeoPrompt API.

Tracks duplicate/aliased functions and routes to canonical implementations.
This module helps users understand function equivalence and avoid confusion
when multiple functions do similar things.
"""

from __future__ import annotations

from typing import Any, Callable

# Duplication patterns found in audit (2026-04-22)
# These are documented here for migration and cleanup tracking
FUNCTION_DUPLICATES: dict[str, list[str]] = {
    # (canonical, [aliases])
    "natural_neighbor_interpolation": [
        # In stats.py: (control_points, query_points) signature, scipy-based
        # In spatial_analysis.py: (points, values, query_point) signature, pure-Python IDW
        # CANONICAL: stats.natural_neighbor_interpolation (vectorized, scipy)
        # ALIAS: spatial_analysis.natural_neighbor_interpolation (single-point, fallback-only)
        "spatial_analysis.natural_neighbor_interpolation",
    ],
    "read_mapinfo_tab": [
        # In io.py: returns GeoPromptFrame
        # In format_bridges.py: returns RecordList
        # CANONICAL: format_bridges.read_mapinfo_tab (lightweight parser)
        # ALIAS: io.read_mapinfo_tab (wrapper for consistency)
        "io.read_mapinfo_tab",
    ],
    "change_point_detection": [
        # In environmental.py: for environmental/time-series data
        # In spacetime.py: general temporal change-point detection
        # CANONICAL: spacetime.change_point_detection (general temporal)
        # ALIAS: environmental.change_point_detection (domain-specific wrapper)
        "environmental.change_point_detection",
    ],
    "feature_vertices_to_points": [
        # In data_management.py: records-based wrapper
        # In geometry.py: geometry-based core implementation
        # CANONICAL: geometry.feature_vertices_to_points (geometric primitives)
        # ALIAS: data_management.feature_vertices_to_points (data-mgmt wrapper)
        "data_management.feature_vertices_to_points",
    ],
}

# Functions that should be consolidated or have known signature mismatches
API_SIGNATURE_MISMATCHES: dict[str, dict[str, Any]] = {
    "natural_neighbor_interpolation": {
        "stats.py": {
            "signature": "(control_points, query_points) -> list[float]",
            "description": "Vectorized natural neighbor using scipy griddata",
            "canonical": True,
        },
        "spatial_analysis.py": {
            "signature": "(points, values, query_point) -> float",
            "description": "Single-point pure-Python IDW approximation",
            "canonical": False,
            "note": "Rename to idw_interpolation_single_point() or deprecate",
        },
    },
}


def get_canonical_function_name(function_name: str, module: str | None = None) -> str:
    """Return canonical function name if this is a duplicate/alias.
    
    Args:
        function_name: Name of function to look up
        module: Optional module to disambiguate (e.g., 'stats', 'spatial_analysis')
    
    Returns:
        Canonical function name (same as input if not a known duplicate)
    """
    for canonical, aliases in FUNCTION_DUPLICATES.items():
        if function_name == canonical:
            return canonical
        if function_name in aliases:
            return canonical
    
    return function_name


def get_duplicate_info(function_name: str) -> dict[str, list[str]] | None:
    """Get information about function duplicates.
    
    Args:
        function_name: Name to check
    
    Returns:
        Dict with 'canonical' and 'aliases', or None if not a duplicate
    """
    canonical = get_canonical_function_name(function_name)
    
    if canonical not in FUNCTION_DUPLICATES:
        return None
    
    return {
        "canonical": canonical,
        "aliases": FUNCTION_DUPLICATES[canonical],
    }


def list_duplicates() -> dict[str, list[str]]:
    """Return all known duplicate function groups.
    
    Returns:
        Dict mapping canonical names to lists of aliases
    """
    return dict(FUNCTION_DUPLICATES)


def recommend_consolidation_plan() -> list[dict[str, Any]]:
    """Generate a consolidation plan for all duplicates.
    
    Returns:
        List of consolidation recommendations with effort/impact
    """
    plan = []
    
    for canonical, aliases in FUNCTION_DUPLICATES.items():
        for alias in aliases:
            plan.append({
                "canonical": canonical,
                "alias_to_remove": alias,
                "effort": "LOW",  # Mostly documentation + deprecation
                "impact": "HIGH",  # Reduces confusion, simplifies API
                "action": f"Keep {canonical}, deprecate {alias} in v0.2.0, remove in v1.0.0",
            })
    
    return plan
