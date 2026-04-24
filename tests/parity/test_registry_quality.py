from __future__ import annotations

from .benchmark_registry import benchmark_assignment_details, benchmark_assignment_map
from geoprompt._tier_metadata import TIER_SIMULATION, get_tier


def test_non_canonical_assignments_have_module_provenance() -> None:
    details = benchmark_assignment_details()
    non_canonical = [
        item for item in details if item["benchmark_class"] != "canonical_non_geopandas_reference"
    ]

    assert non_canonical, "Expected non-canonical benchmark assignments"
    assert all(item["module"] != "unknown" for item in non_canonical)


def test_simulation_only_symbols_not_promoted_to_parity_classes() -> None:
    assignments = benchmark_assignment_map()
    promoted = [
        symbol
        for symbol, benchmark_class in assignments.items()
        if benchmark_class in {"direct_geopandas_parity", "geopandas_plus_shapely_parity"}
        and get_tier(symbol) == TIER_SIMULATION
    ]
    assert not promoted


def test_quality_gate_for_non_canonical_assignment_count() -> None:
    assignments = benchmark_assignment_map()
    non_canonical_count = sum(
        1 for klass in assignments.values() if klass != "canonical_non_geopandas_reference"
    )

    # Maintain broad comparator coverage for API-surface trust checks.
    assert non_canonical_count >= 300
