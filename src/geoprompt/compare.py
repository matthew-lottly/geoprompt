from __future__ import annotations

import argparse
import html
import importlib
import json
import logging
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable

from ._capabilities import require_capability
from .frame import GeoPromptFrame
from .geometry import geometry_centroid
from .overlay import geometry_from_shapely, geometry_to_shapely
from .io import read_features, write_json
from .table import PromptTable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "sample_features.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_features.json"
DEFAULT_JOIN_PATH = PROJECT_ROOT / "data" / "benchmark_regions.json"

logger = logging.getLogger("geoprompt")


@dataclass(frozen=True)
class CorpusCase:
    name: str
    feature_path: Path | None
    crs: str
    query_bounds: tuple[float, float, float, float]
    join_path: Path | None = None
    feature_records: list[dict[str, Any]] | None = None
    join_records: list[dict[str, Any]] | None = None


_ZIP_SENTINEL = object()


def _zip_strict(*iterables: Any):
    for items in zip_longest(*iterables, fillvalue=_ZIP_SENTINEL):
        if _ZIP_SENTINEL in items:
            raise ValueError("zip() arguments must have equal length")
        yield items


DEFAULT_CORPUS = [
    CorpusCase(
        name="sample",
        feature_path=DEFAULT_INPUT_PATH,
        crs="EPSG:4326",
        query_bounds=(-111.97, 40.68, -111.84, 40.79),
    ),
    CorpusCase(
        name="benchmark",
        feature_path=DEFAULT_BENCHMARK_PATH,
        crs="EPSG:4326",
        query_bounds=(-112.01, 40.64, -111.84, 40.8),
        join_path=DEFAULT_JOIN_PATH,
    ),
    CorpusCase(
        name="stress",
        feature_path=None,
        crs="EPSG:4326",
        query_bounds=(-111.97, 40.69, -111.87, 40.77),
        feature_records=[],
        join_records=[],
    ),
]


def _stress_feature_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    min_x = -112.0
    min_y = 40.64
    point_step_x = 0.02
    point_step_y = 0.02

    def coord(value: float) -> float:
        return round(value, 6)

    for row in range(8):
        for column in range(8):
            x_value = coord(min_x + (column * point_step_x))
            y_value = coord(min_y + (row * point_step_y))
            records.append(
                {
                    "site_id": f"stress-point-{row:02d}-{column:02d}",
                    "name": f"Stress Point {row:02d}-{column:02d}",
                    "geometry": {"type": "Point", "coordinates": [x_value, y_value]},
                    "demand_index": round(0.55 + (row * 0.03) + (column * 0.01), 3),
                    "capacity_index": round(0.65 + (column * 0.02), 3),
                    "priority_index": round(0.8 + (row * 0.025), 3),
                }
            )

    for row in range(6):
        y_value = coord(40.655 + (row * 0.025))
        records.append(
            {
                "site_id": f"stress-horizontal-line-{row:02d}",
                "name": f"Stress Horizontal Line {row:02d}",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [-112.01, y_value],
                        [-111.92, coord(y_value + 0.003)],
                        [-111.83, y_value],
                    ],
                },
                "demand_index": round(0.7 + (row * 0.02), 3),
                "capacity_index": round(0.82 + (row * 0.015), 3),
                "priority_index": round(0.9 + (row * 0.018), 3),
            }
        )

    for column in range(6):
        x_value = coord(-111.995 + (column * 0.03))
        records.append(
            {
                "site_id": f"stress-vertical-line-{column:02d}",
                "name": f"Stress Vertical Line {column:02d}",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [x_value, 40.63],
                        [coord(x_value + 0.003), 40.72],
                        [x_value, 40.81],
                    ],
                },
                "demand_index": round(0.68 + (column * 0.025), 3),
                "capacity_index": round(0.79 + (column * 0.018), 3),
                "priority_index": round(0.88 + (column * 0.02), 3),
            }
        )

    for row in range(4):
        for column in range(4):
            min_polygon_x = coord(-112.0 + (column * 0.04))
            min_polygon_y = coord(40.645 + (row * 0.04))
            max_polygon_x = coord(min_polygon_x + 0.028)
            max_polygon_y = coord(min_polygon_y + 0.028)
            records.append(
                {
                    "site_id": f"stress-polygon-{row:02d}-{column:02d}",
                    "name": f"Stress Polygon {row:02d}-{column:02d}",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [min_polygon_x, min_polygon_y],
                            [max_polygon_x, min_polygon_y],
                            [max_polygon_x, max_polygon_y],
                            [min_polygon_x, max_polygon_y],
                        ]],
                    },
                    "demand_index": round(0.74 + (row * 0.03) + (column * 0.015), 3),
                    "capacity_index": round(0.77 + (column * 0.02), 3),
                    "priority_index": round(0.93 + (row * 0.025), 3),
                }
            )

    records.append(
        {
            "site_id": "stress-remote-point",
            "name": "Stress Remote Point",
            "geometry": {"type": "Point", "coordinates": [-111.72, 40.86]},
            "demand_index": 0.5,
            "capacity_index": 0.62,
            "priority_index": 0.71,
        }
    )

    return records


def _large_stress_feature_records(scale: int = 4) -> list[dict[str, Any]]:
    """Generate a larger stress corpus by scaling the base generator.

    ``scale=4`` produces ~372 features (4× the base 93).  The generator
    tiles across a wider geographic extent to avoid overlapping the base
    corpus.
    """
    records: list[dict[str, Any]] = []
    base_min_x = -112.5
    base_min_y = 40.3
    point_step_x = 0.015
    point_step_y = 0.015

    def coord(value: float) -> float:
        return round(value, 6)

    grid_side = 8 * scale
    for row in range(grid_side):
        for column in range(grid_side):
            if row % scale != 0 and column % scale != 0:
                continue
            x_value = coord(base_min_x + (column * point_step_x))
            y_value = coord(base_min_y + (row * point_step_y))
            records.append({
                "site_id": f"lg-point-{row:03d}-{column:03d}",
                "name": f"Large Point {row:03d}-{column:03d}",
                "geometry": {"type": "Point", "coordinates": [x_value, y_value]},
                "demand_index": round(0.5 + (row * 0.004) + (column * 0.002), 3),
                "capacity_index": round(0.6 + (column * 0.003), 3),
                "priority_index": round(0.7 + (row * 0.005), 3),
            })

    line_count = 6 * scale
    for row in range(line_count):
        y_value = coord(base_min_y + 0.01 + (row * 0.018))
        records.append({
            "site_id": f"lg-hline-{row:03d}",
            "name": f"Large Horizontal Line {row:03d}",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [base_min_x, y_value],
                    [coord(base_min_x + 0.15), coord(y_value + 0.002)],
                    [coord(base_min_x + 0.3), y_value],
                ],
            },
            "demand_index": round(0.65 + (row * 0.008), 3),
            "capacity_index": round(0.75 + (row * 0.005), 3),
            "priority_index": round(0.85 + (row * 0.006), 3),
        })

    for column in range(line_count):
        x_value = coord(base_min_x + 0.01 + (column * 0.02))
        records.append({
            "site_id": f"lg-vline-{column:03d}",
            "name": f"Large Vertical Line {column:03d}",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [x_value, base_min_y],
                    [coord(x_value + 0.002), coord(base_min_y + 0.15)],
                    [x_value, coord(base_min_y + 0.3)],
                ],
            },
            "demand_index": round(0.62 + (column * 0.007), 3),
            "capacity_index": round(0.72 + (column * 0.004), 3),
            "priority_index": round(0.82 + (column * 0.005), 3),
        })

    poly_side = 4 * scale
    for row in range(poly_side):
        for column in range(poly_side):
            if row % scale != 0 and column % scale != 0:
                continue
            min_polygon_x = coord(base_min_x + (column * 0.025))
            min_polygon_y = coord(base_min_y + (row * 0.025))
            max_polygon_x = coord(min_polygon_x + 0.018)
            max_polygon_y = coord(min_polygon_y + 0.018)
            records.append({
                "site_id": f"lg-polygon-{row:03d}-{column:03d}",
                "name": f"Large Polygon {row:03d}-{column:03d}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_polygon_x, min_polygon_y],
                        [max_polygon_x, min_polygon_y],
                        [max_polygon_x, max_polygon_y],
                        [min_polygon_x, max_polygon_y],
                    ]],
                },
                "demand_index": round(0.7 + (row * 0.005) + (column * 0.003), 3),
                "capacity_index": round(0.73 + (column * 0.004), 3),
                "priority_index": round(0.9 + (row * 0.006), 3),
            })

    # Multi-geometry entries
    for idx in range(4):
        records.append({
            "site_id": f"lg-multipoint-{idx:03d}",
            "name": f"Large MultiPoint {idx:03d}",
            "geometry": {
                "type": "MultiPoint",
                "coordinates": [
                    [coord(base_min_x + idx * 0.05), coord(base_min_y + idx * 0.04)],
                    [coord(base_min_x + idx * 0.05 + 0.01), coord(base_min_y + idx * 0.04 + 0.01)],
                ],
            },
            "demand_index": round(0.55 + idx * 0.03, 3),
            "capacity_index": round(0.65 + idx * 0.02, 3),
            "priority_index": round(0.75 + idx * 0.04, 3),
        })

    records.append({
        "site_id": "lg-remote-point",
        "name": "Large Remote Point",
        "geometry": {"type": "Point", "coordinates": [-111.5, 41.0]},
        "demand_index": 0.45,
        "capacity_index": 0.55,
        "priority_index": 0.65,
    })

    return records


def _large_stress_region_records(scale: int = 4) -> list[dict[str, Any]]:
    """Generate a larger region corpus for the stress benchmark."""
    records: list[dict[str, Any]] = []
    min_x = -112.55
    min_y = 40.25
    cell_width = 0.04
    cell_height = 0.04

    def coord(value: float) -> float:
        return round(value, 6)

    grid_side = 4 * scale
    for row in range(grid_side):
        for column in range(grid_side):
            if row % scale != 0 and column % scale != 0:
                continue
            region_min_x = coord(min_x + (column * cell_width))
            region_min_y = coord(min_y + (row * cell_height))
            region_max_x = coord(region_min_x + cell_width)
            region_max_y = coord(region_min_y + cell_height)
            records.append({
                "region_id": f"lg-sector-{row:03d}-{column:03d}",
                "region_name": f"Large Sector {row:03d}-{column:03d}",
                "region_band": "north" if row >= grid_side // 2 else "south",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [region_min_x, region_min_y],
                        [region_max_x, region_min_y],
                        [region_max_x, region_max_y],
                        [region_min_x, region_max_y],
                    ]],
                },
            })

    return records


def save_benchmark_snapshot(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    tolerance: float = 1e-7,
    include_large_corpus: bool = True,
) -> dict[str, Path]:
    """Run the full comparison and save a reproducible benchmark snapshot.

    Writes JSON, Markdown, and HTML reports. If ``include_large_corpus`` is True,
    also runs and saves results for the large stress corpus.
    """
    report = build_comparison_report(output_dir=output_dir, tolerance=tolerance)
    result = export_comparison_bundle(report, output_dir)
    paths: dict[str, Path] = {k: Path(v) for k, v in result.items()}

    if include_large_corpus:
        large_dir = Path(output_dir) / "large-stress"
        large_dir.mkdir(parents=True, exist_ok=True)
        large_features = _large_stress_feature_records()
        large_regions = _large_stress_region_records()
        snapshot = {
            "corpus": "large-stress",
            "feature_count": len(large_features),
            "region_count": len(large_regions),
            "geometry_types": sorted({
                str(r["geometry"]["type"]) for r in large_features if "geometry" in r
            }),
        }
        snapshot_path = large_dir / "large_stress_snapshot.json"
        from .io import write_json as _write_json
        _write_json(snapshot_path, snapshot)
        paths["large_stress_snapshot"] = snapshot_path

    return paths


def _stress_region_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    min_x = -112.02
    min_y = 40.62
    cell_width = 0.05
    cell_height = 0.05

    def coord(value: float) -> float:
        return round(value, 6)

    for row in range(4):
        for column in range(4):
            region_min_x = coord(min_x + (column * cell_width))
            region_min_y = coord(min_y + (row * cell_height))
            region_max_x = coord(region_min_x + cell_width)
            region_max_y = coord(region_min_y + cell_height)
            records.append(
                {
                    "region_id": f"stress-sector-{row:02d}-{column:02d}",
                    "region_name": f"Stress Sector {row:02d}-{column:02d}",
                    "region_band": "north" if row >= 2 else "south",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [region_min_x, region_min_y],
                            [region_max_x, region_min_y],
                            [region_max_x, region_max_y],
                            [region_min_x, region_max_y],
                        ]],
                    },
                }
            )

    return records


def _materialize_case(case: CorpusCase) -> CorpusCase:
    if case.name != "stress":
        return case
    return CorpusCase(
        name=case.name,
        feature_path=case.feature_path,
        crs=case.crs,
        query_bounds=case.query_bounds,
        join_path=case.join_path,
        feature_records=_stress_feature_records(),
        join_records=_stress_region_records(),
    )


def _frame_from_case(case: CorpusCase) -> GeoPromptFrame:
    if case.feature_records is not None:
        return GeoPromptFrame.from_records(case.feature_records, crs=case.crs)
    if case.feature_path is None:
        raise ValueError(f"case '{case.name}' is missing feature input")
    return read_features(case.feature_path, crs=case.crs)


def _join_frame_from_case(case: CorpusCase) -> GeoPromptFrame | None:
    if case.join_records is not None:
        return GeoPromptFrame.from_records(case.join_records, crs=case.crs)
    if case.join_path is None:
        return None
    return read_features(case.join_path, crs=case.crs)


def _load_compare_dependencies() -> tuple[Any, Callable[[dict[str, Any]], Any], Callable[..., Any]]:
    require_capability("geopandas", context="geoprompt-compare")
    require_capability("shapely", context="geoprompt-compare")
    try:
        geopandas = importlib.import_module("geopandas")
        shapely_geometry = importlib.import_module("shapely.geometry")
    except ImportError as exc:  # pragma: no cover - guarded by require_capability
        require_capability("geopandas", context="geoprompt-compare")  # re-raises with proper message
        raise AssertionError("Capability guard failed for comparison dependencies") from exc

    return geopandas, shapely_geometry.shape, shapely_geometry.box


def _benchmark(operation: str, func: Callable[[], Any], repeats: int = 20) -> tuple[dict[str, Any], Any]:
    timings: list[float] = []
    result: Any = None
    for _ in range(repeats):
        started_at = time.perf_counter()
        result = func()
        timings.append(time.perf_counter() - started_at)
    return (
        {
            "operation": operation,
            "median_seconds": statistics.median(timings),
            "min_seconds": min(timings),
            "max_seconds": max(timings),
            "repeats": repeats,
        },
        result,
    )


def _nearest_neighbors_reference(ids: list[str], centroids: list[Any]) -> list[dict[str, Any]]:
    neighbors: list[dict[str, Any]] = []
    for index, origin in enumerate(ids):
        candidates: list[dict[str, Any]] = []
        for other_index, destination in enumerate(ids):
            if other_index == index:
                continue
            candidates.append(
                {
                    "origin": origin,
                    "neighbor": destination,
                    "distance": float(centroids[index].distance(centroids[other_index])),
                }
            )
        candidates.sort(key=lambda item: (float(item["distance"]), str(item["neighbor"])))
        neighbors.append(candidates[0])
    return neighbors


def _as_geojson_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    geometry_type = str(geometry["type"])
    coordinates = geometry["coordinates"]
    if geometry_type == "Point":
        return {"type": "Point", "coordinates": list(coordinates)}
    if geometry_type == "LineString":
        return {"type": "LineString", "coordinates": [list(coord) for coord in coordinates]}
    if geometry_type == "Polygon":
        return {"type": "Polygon", "coordinates": [[list(coord) for coord in coordinates]]}
    raise TypeError(f"unsupported geometry type for comparison: {geometry_type}")


def _dataset_report(case: CorpusCase, tolerance: float) -> dict[str, Any]:
    case = _materialize_case(case)
    geopandas, shape, box = _load_compare_dependencies()
    projection_tolerance = max(tolerance, 1e-6)

    frame = _frame_from_case(case)
    records = frame.to_records()
    feature_ids = [str(record.get("site_id", index)) for index, record in enumerate(records)]
    geoprompt_lengths = frame.geometry_lengths()
    geoprompt_areas = frame.geometry_areas()
    geoprompt_centroids = [geometry_centroid(record["geometry"]) for record in records]

    shapely_geometries = [shape(_as_geojson_geometry(record["geometry"])) for record in records]
    geopandas_frame = geopandas.GeoDataFrame(
        [{key: value for key, value in record.items() if key != "geometry"} for record in records],
        geometry=shapely_geometries,
        crs=case.crs,
    )

    geometry_comparison: list[dict[str, Any]] = []
    for record, geoprompt_length, geoprompt_area, geoprompt_centroid, shapely_geometry in _zip_strict(
        records,
        geoprompt_lengths,
        geoprompt_areas,
        geoprompt_centroids,
        shapely_geometries,
    ):
        shapely_centroid = shapely_geometry.centroid
        geometry_comparison.append(
            {
                "site_id": str(record.get("site_id", "unknown")),
                "geometry_type": str(record["geometry"]["type"]),
                "length_delta": abs(float(geoprompt_length) - float(shapely_geometry.length)),
                "area_delta": abs(float(geoprompt_area) - float(shapely_geometry.area)),
                "centroid_x_delta": abs(float(geoprompt_centroid[0]) - float(shapely_centroid.x)),
                "centroid_y_delta": abs(float(geoprompt_centroid[1]) - float(shapely_centroid.y)),
            }
        )

    geoprompt_bounds = frame.bounds().__dict__
    geopandas_bounds = geopandas_frame.total_bounds.tolist()
    geoprompt_nearest = frame.nearest_neighbors(k=1)
    reference_nearest = _nearest_neighbors_reference(feature_ids, [geometry.centroid for geometry in shapely_geometries])

    min_x, min_y, max_x, max_y = case.query_bounds
    geoprompt_query = [str(record.get("site_id", "unknown")) for record in frame.query_bounds(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)]
    reference_query = [
        str(value)
        for value in geopandas_frame.loc[
            geopandas_frame.geometry.intersects(box(min_x, min_y, max_x, max_y)),
            "site_id",
        ].tolist()
    ]

    projected_frame = frame.to_crs("EPSG:3857")
    projected_reference = geopandas_frame.to_crs("EPSG:3857")
    projected_bounds = projected_frame.bounds().__dict__
    projected_reference_bounds = projected_reference.total_bounds.tolist()

    join_report: dict[str, Any] | None = None
    clip_report: dict[str, Any] | None = None
    dissolve_report: dict[str, Any] | None = None
    regions = _join_frame_from_case(case)
    if regions is not None:
        geoprompt_join = regions.spatial_join(frame, predicate="intersects")
        geoprompt_pairs = sorted(f"{row['region_id']}->{row['site_id']}" for row in geoprompt_join)

        region_geometries = [shape(_as_geojson_geometry(record["geometry"])) for record in regions.to_records()]
        region_ids = [str(record["region_id"]) for record in regions.to_records()]
        reference_pairs: list[str] = []
        for region_id, region_geometry in _zip_strict(region_ids, region_geometries):
            for feature_id, feature_geometry in _zip_strict(feature_ids, shapely_geometries):
                if region_geometry.intersects(feature_geometry):
                    reference_pairs.append(f"{region_id}->{feature_id}")
        join_report = {
            "pair_match": geoprompt_pairs == sorted(set(reference_pairs)),
            "geoprompt_pairs": geoprompt_pairs,
            "reference_pairs": sorted(set(reference_pairs)),
        }

        clipped = frame.clip(regions)
        shapely_ops = importlib.import_module("shapely.ops")
        mask_union = shapely_ops.unary_union([geometry_to_shapely(record["geometry"]) for record in regions.to_records()])
        reference_clip_geometries = [
            geometry
            for feature_geometry in shapely_geometries
            for geometry in geometry_from_shapely(feature_geometry.intersection(mask_union))
        ]
        clip_report = {
            "feature_count_match": len(clipped) == len(reference_clip_geometries),
            "geoprompt_feature_count": len(clipped),
            "reference_feature_count": len(reference_clip_geometries),
        }

        geoprompt_dissolved = regions.dissolve(by="region_band", aggregations={"region_name": "count"})
        geopandas_dissolved = geopandas.GeoDataFrame(
            [{key: value for key, value in record.items() if key != "geometry"} for record in regions.to_records()],
            geometry=region_geometries,
            crs=case.crs,
        ).dissolve(by="region_band", aggfunc={"region_name": "count"}).reset_index()
        dissolve_report = {
            "feature_count_match": len(geoprompt_dissolved) == len(geopandas_dissolved),
            "bands_match": sorted(str(record["region_band"]) for record in geoprompt_dissolved) == sorted(str(value) for value in geopandas_dissolved["region_band"].tolist()),
        }

    benchmarks: list[dict[str, Any]] = []
    for operation, func in [
        (f"{case.name}.geoprompt.geometry_metrics", lambda: (frame.geometry_lengths(), frame.geometry_areas(), frame.bounds())),
        (f"{case.name}.reference.geometry_metrics", lambda: ([geometry.length for geometry in shapely_geometries], [geometry.area for geometry in shapely_geometries], geopandas_frame.total_bounds)),
        (f"{case.name}.geoprompt.nearest_neighbors", lambda: frame.nearest_neighbors(k=1)),
        (f"{case.name}.reference.nearest_neighbors", lambda: _nearest_neighbors_reference(feature_ids, [geometry.centroid for geometry in shapely_geometries])),
        (f"{case.name}.geoprompt.query_bounds", lambda: frame.query_bounds(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)),
        (f"{case.name}.reference.query_bounds", lambda: geopandas_frame.loc[geopandas_frame.geometry.intersects(box(min_x, min_y, max_x, max_y))]),
        (f"{case.name}.geoprompt.to_crs", lambda: frame.to_crs("EPSG:3857")),
        (f"{case.name}.reference.to_crs", lambda: geopandas_frame.to_crs("EPSG:3857")),
    ]:
        benchmark, _ = _benchmark(operation, func)
        benchmarks.append(benchmark)

    if regions is not None:
        region_geometries = [shape(_as_geojson_geometry(record["geometry"])) for record in regions.to_records()]
        region_ids = [str(record["region_id"]) for record in regions.to_records()]
        benchmark, _ = _benchmark(
            f"{case.name}.geoprompt.spatial_join",
            lambda: regions.spatial_join(frame, predicate="intersects"),
        )
        benchmarks.append(benchmark)
        benchmark, _ = _benchmark(
            f"{case.name}.reference.spatial_join",
            lambda: [
                f"{region_id}->{feature_id}"
                for region_id, region_geometry in _zip_strict(region_ids, region_geometries)
                for feature_id, feature_geometry in _zip_strict(feature_ids, shapely_geometries)
                if region_geometry.intersects(feature_geometry)
            ],
        )
        benchmarks.append(benchmark)
        benchmark, _ = _benchmark(
            f"{case.name}.geoprompt.clip",
            lambda: frame.clip(regions),
        )
        benchmarks.append(benchmark)
        shapely_ops = importlib.import_module("shapely.ops")
        mask_union = shapely_ops.unary_union([geometry_to_shapely(record["geometry"]) for record in regions.to_records()])
        benchmark, _ = _benchmark(
            f"{case.name}.reference.clip",
            lambda: [
                geometry
                for feature_geometry in shapely_geometries
                for geometry in geometry_from_shapely(feature_geometry.intersection(mask_union))
            ],
        )
        benchmarks.append(benchmark)
        benchmark, _ = _benchmark(
            f"{case.name}.geoprompt.dissolve",
            lambda: regions.dissolve(by="region_band", aggregations={"region_name": "count"}),
        )
        benchmarks.append(benchmark)
        benchmark, _ = _benchmark(
            f"{case.name}.reference.dissolve",
            lambda: geopandas.GeoDataFrame(
                [{key: value for key, value in record.items() if key != "geometry"} for record in regions.to_records()],
                geometry=region_geometries,
                crs=case.crs,
            ).dissolve(by="region_band", aggfunc={"region_name": "count"}).reset_index(),
        )
        benchmarks.append(benchmark)

    return {
        "dataset": case.name,
        "input_path": str(case.feature_path) if case.feature_path is not None else "generated:stress",
        "join_path": str(case.join_path) if case.join_path is not None else ("generated:stress-regions" if case.join_records is not None else None),
        "crs": case.crs,
        "feature_count": len(records),
        "join_feature_count": len(regions) if regions is not None else 0,
        "correctness": {
            "bounds_match": all(
                abs(left - right) <= tolerance
                for left, right in _zip_strict(
                    [geoprompt_bounds["min_x"], geoprompt_bounds["min_y"], geoprompt_bounds["max_x"], geoprompt_bounds["max_y"]],
                    geopandas_bounds,
                )
            ),
            "nearest_neighbor_match": [item["neighbor"] for item in geoprompt_nearest] == [item["neighbor"] for item in reference_nearest],
            "bounds_query_match": sorted(geoprompt_query) == sorted(reference_query),
            "geometry_metrics_within_tolerance": all(
                item["length_delta"] <= tolerance
                and item["area_delta"] <= tolerance
                and item["centroid_x_delta"] <= tolerance
                and item["centroid_y_delta"] <= tolerance
                for item in geometry_comparison
            ),
            "projected_bounds_match": all(
                abs(left - right) <= projection_tolerance
                for left, right in _zip_strict(
                    [projected_bounds["min_x"], projected_bounds["min_y"], projected_bounds["max_x"], projected_bounds["max_y"]],
                    projected_reference_bounds,
                )
            ),
            "geometry_comparison": geometry_comparison,
            "clip": clip_report,
            "dissolve": dissolve_report,
            "spatial_join": join_report,
        },
        "benchmarks": benchmarks,
    }


def _parse_benchmark_operation(operation: str, dataset_name: str | None = None) -> tuple[str, str, str]:
    parts = str(operation).split(".")
    if len(parts) >= 3:
        return parts[0], parts[1], ".".join(parts[2:])
    if len(parts) == 2:
        return dataset_name or parts[0], "unknown", parts[1]
    return dataset_name or "unknown", "unknown", str(operation)


def benchmark_summary_table(report: dict[str, Any]) -> PromptTable:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in report.get("datasets", []):
        dataset_name = str(dataset.get("dataset", "unknown"))
        for benchmark in dataset.get("benchmarks", []):
            benchmark_dataset, engine, operation = _parse_benchmark_operation(
                str(benchmark.get("operation", "unknown")),
                dataset_name=dataset_name,
            )
            row = rows_by_key.setdefault(
                (benchmark_dataset, operation),
                {
                    "dataset": benchmark_dataset,
                    "operation": operation,
                    "geoprompt_median_seconds": None,
                    "reference_median_seconds": None,
                    "winner": "unknown",
                    "speedup_ratio": None,
                    "relative_status": "not compared",
                },
            )
            row[f"{engine}_median_seconds"] = float(benchmark.get("median_seconds", 0.0))

    summary_rows: list[dict[str, Any]] = []
    for _, row in sorted(rows_by_key.items(), key=lambda item: item[0]):
        geoprompt_seconds = row.get("geoprompt_median_seconds")
        reference_seconds = row.get("reference_median_seconds")
        if isinstance(geoprompt_seconds, float) and isinstance(reference_seconds, float) and geoprompt_seconds > 0 and reference_seconds > 0:
            speedup_ratio = reference_seconds / geoprompt_seconds
            row["speedup_ratio"] = speedup_ratio
            if abs(speedup_ratio - 1.0) <= 0.05:
                row["winner"] = "tie"
                row["relative_status"] = "roughly equal"
            elif speedup_ratio > 1.0:
                row["winner"] = "geoprompt"
                row["relative_status"] = f"{speedup_ratio:.2f}x faster"
            else:
                row["winner"] = "reference"
                row["relative_status"] = f"{(1.0 / speedup_ratio):.2f}x slower"
        summary_rows.append(row)

    return PromptTable(summary_rows)


def correctness_summary_table(report: dict[str, Any]) -> PromptTable:
    rows: list[dict[str, Any]] = []
    for dataset in report.get("datasets", []):
        correctness = dataset.get("correctness", {})
        checks = {
            "bounds_match": bool(correctness.get("bounds_match")),
            "nearest_neighbor_match": bool(correctness.get("nearest_neighbor_match")),
            "bounds_query_match": bool(correctness.get("bounds_query_match")),
            "geometry_metrics_match": bool(correctness.get("geometry_metrics_within_tolerance")),
            "projected_bounds_match": bool(correctness.get("projected_bounds_match")),
            "clip_match": correctness.get("clip") is None or bool(correctness.get("clip", {}).get("feature_count_match")),
            "dissolve_match": correctness.get("dissolve") is None or (
                bool(correctness.get("dissolve", {}).get("feature_count_match"))
                and bool(correctness.get("dissolve", {}).get("bands_match"))
            ),
            "spatial_join_match": correctness.get("spatial_join") is None or bool(correctness.get("spatial_join", {}).get("pair_match")),
        }
        rows.append(
            {
                "dataset": str(dataset.get("dataset", "unknown")),
                "feature_count": int(dataset.get("feature_count", 0)),
                **checks,
                "all_checks_passed": all(checks.values()),
            }
        )
    return PromptTable(rows)


def _html_table(table: PromptTable) -> str:
    rows = table.to_records()
    if not rows:
        return "<table></table>"
    columns = table.columns
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _ascii_sparkline(values: list[float]) -> str:
    if not values:
        return ""
    blocks = "._-:=+*#%@"
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-9:
        return blocks[4] * len(values)
    chars: list[str] = []
    for value in values:
        idx = int(round(((value - lo) / span) * (len(blocks) - 1)))
        idx = max(0, min(len(blocks) - 1, idx))
        chars.append(blocks[idx])
    return "".join(chars)


def render_comparison_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    benchmark_table = benchmark_summary_table(report)
    correctness_table = correctness_summary_table(report)
    speedups = [
        float(row.get("speedup_ratio"))
        for row in benchmark_table.to_records()
        if isinstance(row.get("speedup_ratio"), (int, float))
    ]
    speedup_trend = _ascii_sparkline(speedups)
    pass_count = sum(1 for value in summary.values() if bool(value))
    total_count = len(summary)
    lines = [
        "# GeoPrompt Comparison Summary",
        "",
        f"- Checks passed: {pass_count}/{total_count}",
        f"- Speedup trend: {speedup_trend}",
        f"- Corpus: {', '.join(str(item) for item in report.get('comparison', {}).get('corpus', []))}",
        f"- Engines: {', '.join(str(item) for item in report.get('comparison', {}).get('engines', []))}",
        "",
        "## Correctness Overview",
        "",
        correctness_table.to_markdown(),
        "## Benchmark Overview",
        "",
        benchmark_table.to_markdown(),
    ]
    return "\n".join(lines).strip() + "\n"


def render_comparison_html(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    pass_count = sum(1 for value in summary.values() if bool(value))
    total_count = len(summary)
    speedups = [
        float(row.get("speedup_ratio"))
        for row in benchmark_summary_table(report).to_records()
        if isinstance(row.get("speedup_ratio"), (int, float))
    ]
    speedup_trend = _ascii_sparkline(speedups)
    correctness_table = _html_table(correctness_summary_table(report))
    benchmark_table = _html_table(benchmark_summary_table(report))
    return (
        "<html><head><meta charset='utf-8'><title>GeoPrompt Comparison Summary</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1f2937;background:#f5f7fa;}"
        "main{max-width:1100px;margin:0 auto;}section{background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:12px 14px;margin:12px 0;}"
        "table{border-collapse:collapse;width:100%;margin:16px 0;background:#fff;}"
        "th,td{border:1px solid #d0d7de;padding:8px;text-align:left;}th{background:#eef2f7;}"
        ".pill{display:inline-block;padding:4px 10px;border-radius:999px;background:#e8efff;color:#1e3a8a;font-weight:600;}</style>"
        "</head><body>"
        "<main>"
        "<h1>GeoPrompt Comparison Summary</h1>"
        f"<p><span class='pill'>Checks passed: {pass_count}/{total_count}</span></p>"
        f"<p><strong>Speedup trend:</strong> {html.escape(speedup_trend)}</p>"
        f"<p><strong>Corpus:</strong> {html.escape(', '.join(str(item) for item in report.get('comparison', {}).get('corpus', [])))}</p>"
        "<section><h2>Correctness Overview</h2>"
        f"{correctness_table}"
        "</section><section><h2>Benchmark Overview</h2>"
        f"{benchmark_table}"
        "</section></main></body></html>"
    )


def export_comparison_bundle(report: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output_dir / "geoprompt_comparison_report.json", report)
    markdown_path = output_dir / "geoprompt_comparison_summary.md"
    html_path = output_dir / "geoprompt_comparison_summary.html"
    markdown_path.write_text(render_comparison_markdown(report), encoding="utf-8")
    html_path.write_text(render_comparison_html(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "html": str(html_path),
    }


def build_comparison_report(
    input_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tolerance: float = 1e-7,
    crs: str = "EPSG:4326",
    join_path: Path | None = None,
) -> dict[str, Any]:
    cases = DEFAULT_CORPUS if input_path is None else [
        CorpusCase(
            name=input_path.stem,
            feature_path=input_path,
            crs=crs,
            query_bounds=(-111.97, 40.68, -111.84, 40.79),
            join_path=join_path,
        )
    ]
    datasets = [_dataset_report(case, tolerance=tolerance) for case in cases]

    return {
        "package": "geoprompt",
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generation_script": "geoprompt.compare",
            "data_version": "repository-benchmark-corpus-v1",
            "refresh_cadence": "each release and benchmark refresh",
        },
        "comparison": {
            "output_dir": str(output_dir),
            "tolerance": tolerance,
            "engines": ["geoprompt", "shapely", "geopandas"],
            "corpus": [dataset["dataset"] for dataset in datasets],
            "notes": [
                "Planar metric comparisons use raw coordinate space so Geoprompt matches Shapely and GeoPandas defaults.",
                "Geographic haversine distance is a separate Geoprompt mode and is not directly compared to Shapely or GeoPandas here.",
                "Projection comparisons use EPSG:3857 as a common reference target.",
            ],
        },
        "summary": {
            "all_bounds_match": all(dataset["correctness"]["bounds_match"] for dataset in datasets),
            "all_nearest_neighbor_match": all(dataset["correctness"]["nearest_neighbor_match"] for dataset in datasets),
            "all_bounds_query_match": all(dataset["correctness"]["bounds_query_match"] for dataset in datasets),
            "all_geometry_metrics_within_tolerance": all(dataset["correctness"]["geometry_metrics_within_tolerance"] for dataset in datasets),
            "all_projected_bounds_match": all(dataset["correctness"]["projected_bounds_match"] for dataset in datasets),
            "all_clip_matches": all(
                dataset["correctness"]["clip"] is None or dataset["correctness"]["clip"]["feature_count_match"]
                for dataset in datasets
            ),
            "all_dissolve_matches": all(
                dataset["correctness"]["dissolve"] is None or (
                    dataset["correctness"]["dissolve"]["feature_count_match"] and dataset["correctness"]["dissolve"]["bands_match"]
                )
                for dataset in datasets
            ),
            "all_spatial_joins_match": all(
                dataset["correctness"]["spatial_join"] is None or dataset["correctness"]["spatial_join"]["pair_match"]
                for dataset in datasets
            ),
        },
        "datasets": datasets,
    }


def _load_benchmark_history_reports(history_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if not history_dir.exists():
        return reports
    for path in sorted(history_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "datasets" in payload:
            payload.setdefault("_source_path", str(path))
            reports.append(payload)
    return reports


def benchmark_history_table(history_dir: str | Path) -> PromptTable:
    """Build a per-release benchmark history summary table."""
    history_path = Path(history_dir)
    rows: list[dict[str, Any]] = []
    for report in _load_benchmark_history_reports(history_path):
        summary_table = benchmark_summary_table(report)
        summary_rows = summary_table.to_records()
        speedups = [float(r["speedup_ratio"]) for r in summary_rows if isinstance(r.get("speedup_ratio"), float)]
        wins = sum(1 for r in summary_rows if r.get("winner") == "geoprompt")
        losses = sum(1 for r in summary_rows if r.get("winner") == "reference")
        version = str(report.get("version") or Path(str(report.get("_source_path", "unknown"))).stem)
        rows.append({
            "version": version,
            "dataset_count": len(report.get("datasets", [])),
            "all_checks_passed": all(bool(v) for v in report.get("summary", {}).values()) if report.get("summary") else None,
            "benchmark_rows": len(summary_rows),
            "geoprompt_wins": wins,
            "reference_wins": losses,
            "mean_speedup_ratio": (sum(speedups) / len(speedups)) if speedups else None,
            "source": str(report.get("_source_path", "")),
        })
    return PromptTable(rows)


def render_benchmark_history_html(history_dir: str | Path) -> str:
    """Render an HTML summary of benchmark history snapshots."""
    table = benchmark_history_table(history_dir)
    return (
        "<html><head><meta charset='utf-8'><title>GeoPrompt Benchmark History</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;width:100%;margin:16px 0;}"
        "th,td{border:1px solid #d0d7de;padding:8px;text-align:left;}th{background:#f6f8fa;}</style>"
        "</head><body>"
        "<h1>GeoPrompt Benchmark History</h1>"
        f"{_html_table(table)}"
        "</body></html>"
    )


def _benchmark_dashboard_alerts(rows: list[dict[str, Any]], min_speedup_ratio: float = 1.05) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for row in rows:
        version = str(row.get("version", "unknown"))
        all_checks_passed = bool(row.get("all_checks_passed"))
        speedup = row.get("mean_speedup_ratio")
        if not all_checks_passed:
            alerts.append({
                "version": version,
                "severity": "high",
                "kind": "correctness",
                "message": "One or more correctness checks failed.",
            })
        if isinstance(speedup, (int, float)) and float(speedup) < min_speedup_ratio:
            alerts.append({
                "version": version,
                "severity": "medium",
                "kind": "performance",
                "message": f"Mean speedup ratio {float(speedup):.3f} below threshold {min_speedup_ratio:.3f}.",
            })
    return alerts


def render_benchmark_dashboard_markdown(
    history_dir: str | Path,
    *,
    min_speedup_ratio: float = 1.05,
) -> str:
    table = benchmark_history_table(history_dir)
    rows = table.to_records()
    alerts = _benchmark_dashboard_alerts(rows, min_speedup_ratio=min_speedup_ratio)
    lines = [
        "# GeoPrompt Benchmark Dashboard",
        "",
        f"- Releases tracked: {len(rows)}",
        f"- Alert threshold (mean speedup ratio): {min_speedup_ratio:.3f}",
        f"- Alerts: {len(alerts)}",
        "",
    ]
    if alerts:
        lines.extend([
            "## Alerts",
            "",
            "| Version | Severity | Type | Message |",
            "| --- | --- | --- | --- |",
        ])
        lines.extend(
            f"| {alert['version']} | {alert['severity']} | {alert['kind']} | {alert['message']} |"
            for alert in alerts
        )
        lines.append("")

    lines.extend([
        "## Trend Table",
        "",
        table.to_markdown(),
    ])
    return "\n".join(lines).strip() + "\n"


def render_benchmark_dashboard_html(
    history_dir: str | Path,
    *,
    min_speedup_ratio: float = 1.05,
) -> str:
    table = benchmark_history_table(history_dir)
    rows = table.to_records()
    alerts = _benchmark_dashboard_alerts(rows, min_speedup_ratio=min_speedup_ratio)
    alert_items = "".join(
        "<li>"
        f"<strong>{html.escape(str(item['version']))}</strong> "
        f"[{html.escape(str(item['severity']))}/{html.escape(str(item['kind']))}] "
        f"{html.escape(str(item['message']))}"
        "</li>"
        for item in alerts
    )
    alert_block = (
        "<section><h2>Alerts</h2><ul>" + alert_items + "</ul></section>"
        if alerts
        else "<section><h2>Alerts</h2><p>No threshold alerts.</p></section>"
    )
    return (
        "<html><head><meta charset='utf-8'><title>GeoPrompt Benchmark Dashboard</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1f2937;background:#f5f7fa;}"
        "main{max-width:1100px;margin:0 auto;}section{background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:12px 14px;margin:12px 0;}"
        "table{border-collapse:collapse;width:100%;margin:12px 0;}th,td{border:1px solid #d0d7de;padding:8px;text-align:left;}th{background:#eef2f7;}"
        "</style></head><body><main>"
        "<h1>GeoPrompt Benchmark Dashboard</h1>"
        f"<p><strong>Releases tracked:</strong> {len(rows)}<br>"
        f"<strong>Alert threshold:</strong> {min_speedup_ratio:.3f}</p>"
        f"{alert_block}"
        "<section><h2>Trend Table</h2>"
        f"{_html_table(table)}"
        "</section></main></body></html>"
    )


def export_benchmark_dashboard_bundle(
    history_dir: str | Path,
    *,
    min_speedup_ratio: float = 1.05,
) -> dict[str, str]:
    history_path = Path(history_dir)
    history_path.mkdir(parents=True, exist_ok=True)
    table = benchmark_history_table(history_path)
    rows = table.to_records()
    alerts = _benchmark_dashboard_alerts(rows, min_speedup_ratio=min_speedup_ratio)

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(history_path),
            "min_speedup_ratio": min_speedup_ratio,
        },
        "summary": {
            "release_count": len(rows),
            "alert_count": len(alerts),
            "high_alert_count": sum(1 for alert in alerts if alert.get("severity") == "high"),
            "medium_alert_count": sum(1 for alert in alerts if alert.get("severity") == "medium"),
        },
        "alerts": alerts,
        "trend_rows": rows,
    }

    json_path = history_path / "benchmark_dashboard.json"
    markdown_path = history_path / "benchmark_dashboard.md"
    html_path = history_path / "benchmark_dashboard.html"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(
        render_benchmark_dashboard_markdown(history_path, min_speedup_ratio=min_speedup_ratio),
        encoding="utf-8",
    )
    html_path.write_text(
        render_benchmark_dashboard_html(history_path, min_speedup_ratio=min_speedup_ratio),
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "html": str(html_path),
    }


def export_benchmark_history(history_dir: str | Path) -> dict[str, str]:
    """Export benchmark history summary artifacts to the given directory."""
    history_path = Path(history_dir)
    history_path.mkdir(parents=True, exist_ok=True)
    table = benchmark_history_table(history_path)
    html_path = history_path / "benchmark_history.html"
    json_path = history_path / "benchmark_history.json"
    html_path.write_text(render_benchmark_history_html(history_path), encoding="utf-8")
    json_path.write_text(json.dumps(table.to_records(), indent=2), encoding="utf-8")
    markdown_path = history_path / "benchmark_history.md"
    markdown_path.write_text(benchmark_history_table(history_path).to_markdown() + "\n", encoding="utf-8")
    return {"html": str(html_path), "json": str(json_path), "markdown": str(markdown_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Geoprompt results and runtime against Shapely and GeoPandas.")
    parser.add_argument("--input-path", type=Path, default=None, help="Optional single input fixture to compare. If omitted, the built-in corpus is used.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for the comparison report.")
    parser.add_argument("--tolerance", type=float, default=1e-7, help="Absolute tolerance for correctness checks.")
    parser.add_argument("--crs", type=str, default="EPSG:4326", help="Source CRS for a custom single input fixture.")
    parser.add_argument("--join-path", type=Path, default=None, help="Optional polygon region fixture for a custom single input comparison.")
    parser.add_argument("--export-history", action="store_true", help="Also export a benchmark history page from JSON reports in the output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_comparison_report(
        input_path=args.input_path,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
        crs=args.crs,
        join_path=args.join_path,
    )
    written = export_comparison_bundle(report, args.output_dir)
    logger.info("Wrote GeoPrompt comparison bundle")
    logger.info("JSON output: %s", written["json"])
    logger.info("Markdown output: %s", written["markdown"])
    logger.info("HTML output: %s", written["html"])
    if args.export_history:
        history = export_benchmark_history(args.output_dir)
        logger.info("Benchmark history HTML: %s", history["html"])
        logger.info("Benchmark history JSON: %s", history["json"])


__all__ = [
    "benchmark_history_table",
    "benchmark_summary_table",
    "build_comparison_report",
    "correctness_summary_table",
    "export_benchmark_history",
    "export_benchmark_dashboard_bundle",
    "export_comparison_bundle",
    "main",
    "render_benchmark_history_html",
    "render_benchmark_dashboard_html",
    "render_benchmark_dashboard_markdown",
    "render_comparison_html",
    "render_comparison_markdown",
    "save_benchmark_snapshot",
]


if __name__ == "__main__":
    main()