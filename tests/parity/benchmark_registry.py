from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import geoprompt as gp


BenchmarkClass = Literal[
    "direct_geopandas_parity",
    "geopandas_plus_shapely_parity",
    "canonical_non_geopandas_reference",
]


EQUATION_SYMBOLS: set[str] = {
    "accessibility_gini",
    "age_adjusted_failure_rate",
    "area_similarity",
    "composite_resilience_index",
    "coordinate_distance",
    "corridor_strength",
    "directional_alignment",
    "directional_bearing",
    "euclidean_distance",
    "expected_outage_impact",
    "exponential_decay",
    "gaussian_decay",
    "gravity_interaction",
    "haversine_distance",
    "logistic_service_probability",
    "prompt_decay",
    "prompt_influence",
    "prompt_interaction",
    "weighted_accessibility_score",
}


DIRECT_GEOPANDAS_PARITY_SYMBOLS: set[str] = {
    "to_geopandas",
    "from_geopandas",
    "read_geojson",
    "write_geojson",
    "read_geopackage",
    "write_geopackage",
    "read_shapefile",
    "write_shapefile",
    "read_flatgeobuf",
    "write_flatgeobuf",
    "read_mapinfo_tab",
}


GEOPANDAS_PLUS_SHAPELY_SYMBOLS: set[str] = {
    "geometry_area",
    "geometry_bounds",
    "geometry_centroid",
    "geometry_contains",
    "geometry_convex_hull",
    "geometry_crosses",
    "geometry_disjoint",
    "geometry_distance",
    "geometry_equals",
    "geometry_intersects",
    "geometry_length",
    "geometry_overlaps",
    "geometry_simplify",
    "geometry_snap",
    "geometry_split",
    "geometry_touches",
    "geometry_union",
    "geometry_within",
    "geometry_voronoi",
    "geometry_delaunay",
}


ALLOWED_BENCHMARK_CLASSES: set[str] = {
    "direct_geopandas_parity",
    "geopandas_plus_shapely_parity",
    "canonical_non_geopandas_reference",
}

EXPLICIT_MAPPING_PATH = Path("data") / "parity_baseline.json"


@dataclass(frozen=True)
class BenchmarkAssignment:
    symbol: str
    benchmark_class: BenchmarkClass
    module: str


def public_symbols() -> list[str]:
    return sorted({name for name in getattr(gp, "__all__", []) if hasattr(gp, name)})


def symbol_module(symbol: str) -> str:
    obj = getattr(gp, symbol, None)
    module = inspect.getmodule(obj)
    if module is not None:
        return module.__name__
    return "unknown"


def explicit_assignment_map() -> dict[str, BenchmarkClass]:
    payload = json.loads(EXPLICIT_MAPPING_PATH.read_text(encoding="utf-8"))
    symbols = payload.get("symbols", {})
    if not isinstance(symbols, dict):
        raise ValueError("Explicit parity baseline must include a 'symbols' dictionary")

    invalid = sorted({klass for klass in symbols.values() if klass not in ALLOWED_BENCHMARK_CLASSES})
    if invalid:
        raise ValueError(f"Invalid benchmark classes in explicit mapping: {invalid}")
    return symbols


def benchmark_assignments() -> list[BenchmarkAssignment]:
    explicit_map = explicit_assignment_map()
    records: list[BenchmarkAssignment] = []
    for symbol in public_symbols():
        if symbol not in explicit_map:
            raise KeyError(f"Missing explicit benchmark class mapping for symbol: {symbol}")
        module_name = symbol_module(symbol)
        records.append(
            BenchmarkAssignment(
                symbol=symbol,
                benchmark_class=explicit_map[symbol],
                module=module_name,
            )
        )
    return records


def benchmark_assignment_map() -> dict[str, BenchmarkClass]:
    return {item.symbol: item.benchmark_class for item in benchmark_assignments()}


def benchmark_assignment_details() -> list[dict[str, str]]:
    return [
        {
            "symbol": item.symbol,
            "benchmark_class": item.benchmark_class,
            "module": item.module,
        }
        for item in benchmark_assignments()
    ]
