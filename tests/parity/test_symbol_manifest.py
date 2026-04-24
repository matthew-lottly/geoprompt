from __future__ import annotations

from collections import Counter

from .benchmark_registry import benchmark_assignment_map, public_symbols


ALLOWED_CLASSES = {
    "direct_geopandas_parity",
    "geopandas_plus_shapely_parity",
    "canonical_non_geopandas_reference",
}


def test_every_public_symbol_has_benchmark_assignment() -> None:
    symbols = public_symbols()
    assignments = benchmark_assignment_map()
    assert symbols, "No public symbols discovered from geoprompt.__all__"
    assert set(symbols) == set(assignments)


def test_benchmark_assignment_classes_are_valid() -> None:
    assignments = benchmark_assignment_map()
    invalid = {symbol: klass for symbol, klass in assignments.items() if klass not in ALLOWED_CLASSES}
    assert not invalid


def test_benchmark_assignment_distribution_is_not_catch_all() -> None:
    assignments = benchmark_assignment_map()
    counts = Counter(assignments.values())
    total = sum(counts.values())
    canonical_ratio = counts["canonical_non_geopandas_reference"] / max(total, 1)

    # Professional-grade benchmarking should have meaningful direct/geometry parity
    # coverage, not only a catch-all fallback class.
    assert counts["direct_geopandas_parity"] >= 200
    assert counts["geopandas_plus_shapely_parity"] >= 100
    assert canonical_ratio <= 0.80
