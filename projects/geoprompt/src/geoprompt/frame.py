from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from heapq import heappop, heappush, nsmallest
from typing import Any, Iterable, Literal, Sequence

from .equations import accessibility_index, area_similarity, coordinate_distance, corridor_strength, directional_alignment, gravity_model, prompt_decay, prompt_influence, prompt_interaction
from .geometry import Geometry, geometry_area, geometry_bounds, geometry_centroid, geometry_contains, geometry_convex_hull, geometry_distance, geometry_envelope, geometry_intersects, geometry_intersects_bounds, geometry_length, geometry_type, geometry_within, geometry_within_bounds, normalize_geometry, transform_geometry
from .overlay import buffer_geometries, clip_geometries, dissolve_geometries, overlay_intersections
from .spatial_index import SpatialIndex


Record = dict[str, Any]
BoundsQueryMode = Literal["intersects", "within", "centroid"]
SpatialJoinPredicate = Literal["intersects", "within", "contains"]
SpatialJoinMode = Literal["inner", "left"]
AggregationName = Literal["sum", "mean", "min", "max", "first", "count"]
OverlayNormalizeMode = Literal["left", "right", "both"]
ZoneGroupAggregation = Literal["max", "mean", "sum"]
CorridorDistanceMode = Literal["direct", "network"]
CorridorScoreMode = Literal["distance", "strength", "alignment", "combined"]
ClusterRecommendMetric = Literal["silhouette", "sse"]
CorridorPathAnchor = Literal["start", "end", "nearest"]
GridShape = Literal["fishnet", "hexbin"]


Coordinate = tuple[float, float]


def _row_sort_key(row: Record) -> str:
    return str(row.get("site_id", row.get("region_id", "")))


def _bounds_intersect(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return not (
        left[2] < right[0]
        or left[0] > right[2]
        or left[3] < right[1]
        or left[1] > right[3]
    )


def _bounds_within(candidate: tuple[float, float, float, float], container: tuple[float, float, float, float]) -> bool:
    return (
        candidate[0] >= container[0]
        and candidate[1] >= container[1]
        and candidate[2] <= container[2]
        and candidate[3] <= container[3]
    )


@dataclass(frozen=True)
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


class GeoPromptFrame:
    def __init__(self, rows: Sequence[Record], geometry_column: str = "geometry", crs: str | None = None) -> None:
        self.geometry_column = geometry_column
        self.crs = crs
        self._rows = [dict(row) for row in rows]
        for row in self._rows:
            row[self.geometry_column] = normalize_geometry(row[self.geometry_column])

    @classmethod
    def from_records(cls, records: Iterable[Record], geometry: str = "geometry", crs: str | None = None) -> "GeoPromptFrame":
        return cls(list(records), geometry_column=geometry, crs=crs)

    @classmethod
    def _from_internal_rows(
        cls,
        rows: Sequence[Record],
        geometry_column: str = "geometry",
        crs: str | None = None,
    ) -> "GeoPromptFrame":
        frame = cls.__new__(cls)
        frame.geometry_column = geometry_column
        frame.crs = crs
        frame._rows = [dict(row) for row in rows]
        return frame

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __repr__(self) -> str:
        geometry_col = self.geometry_column
        row_count = len(self._rows)
        col_count = len(self.columns)
        crs_label = self.crs or "None"
        return f"GeoPromptFrame({row_count} rows, {col_count} columns, crs={crs_label})"

    def __getitem__(self, name: str) -> list[Any]:
        return [row.get(name) for row in self._rows]

    @property
    def columns(self) -> list[str]:
        return list(self._rows[0].keys()) if self._rows else []

    def head(self, count: int = 5) -> list[Record]:
        return [dict(row) for row in self._rows[:count]]

    def to_records(self) -> list[Record]:
        return [dict(row) for row in self._rows]

    def bounds(self) -> Bounds:
        xs: list[float] = []
        ys: list[float] = []
        for row in self._rows:
            min_x, min_y, max_x, max_y = geometry_bounds(row[self.geometry_column])
            xs.extend([min_x, max_x])
            ys.extend([min_y, max_y])
        return Bounds(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))

    def centroid(self) -> Coordinate:
        centroids = [geometry_centroid(row[self.geometry_column]) for row in self._rows]
        xs = [coord[0] for coord in centroids]
        ys = [coord[1] for coord in centroids]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _centroids(self, rows: Sequence[Record] | None = None, geometry_column: str | None = None) -> list[Coordinate]:
        active_rows = rows if rows is not None else self._rows
        active_geometry_column = geometry_column or self.geometry_column
        return [geometry_centroid(row[active_geometry_column]) for row in active_rows]

    def spatial_index(self, mode: Literal["geometry", "centroid"] = "geometry", cell_size: float | None = None) -> SpatialIndex:
        if mode == "geometry":
            return SpatialIndex([geometry_bounds(row[self.geometry_column]) for row in self._rows], cell_size=cell_size)
        if mode == "centroid":
            return SpatialIndex.from_points(self._centroids(), cell_size=cell_size)
        raise ValueError("mode must be 'geometry' or 'centroid'")

    def _resolve_anchor_geometry(self, anchor: str | Geometry | Coordinate, id_column: str) -> tuple[Geometry, str | None]:
        if isinstance(anchor, str):
            self._require_column(id_column)
            anchor_row = next((row for row in self._rows if str(row[id_column]) == anchor), None)
            if anchor_row is None:
                raise KeyError(f"anchor '{anchor}' was not found in column '{id_column}'")
            return anchor_row[self.geometry_column], anchor
        return normalize_geometry(anchor), None

    def _aggregate_rows(
        self,
        rows: Sequence[Record],
        aggregations: dict[str, AggregationName] | None,
        suffix: str,
    ) -> dict[str, Any]:
        aggregate_values: dict[str, Any] = {}
        for column, operation in (aggregations or {}).items():
            values = [row[column] for row in rows if column in row and row[column] is not None]
            output_name = f"{column}_{operation}_{suffix}"
            if not values:
                aggregate_values[output_name] = None
                continue
            if operation == "sum":
                aggregate_values[output_name] = sum(float(value) for value in values)
            elif operation == "mean":
                aggregate_values[output_name] = sum(float(value) for value in values) / len(values)
            elif operation == "min":
                aggregate_values[output_name] = min(values)
            elif operation == "max":
                aggregate_values[output_name] = max(values)
            elif operation == "first":
                aggregate_values[output_name] = values[0]
            elif operation == "count":
                aggregate_values[output_name] = len(values)
            else:
                raise ValueError(f"unsupported aggregation: {operation}")
        return aggregate_values

    def distance_matrix(self, distance_method: str = "euclidean") -> list[list[float]]:
        return [
            [
                geometry_distance(origin=row[self.geometry_column], destination=other[self.geometry_column], method=distance_method)
                for other in self._rows
            ]
            for row in self._rows
        ]

    def geometry_types(self) -> list[str]:
        return [geometry_type(row[self.geometry_column]) for row in self._rows]

    def geometry_lengths(self) -> list[float]:
        return [geometry_length(row[self.geometry_column]) for row in self._rows]

    def geometry_areas(self) -> list[float]:
        return [geometry_area(row[self.geometry_column]) for row in self._rows]

    def nearest_neighbors(
        self,
        id_column: str = "site_id",
        k: int = 1,
        distance_method: str = "euclidean",
    ) -> list[Record]:
        self._require_column(id_column)
        if k <= 0:
            raise ValueError("k must be greater than zero")

        centroids = self._centroids()
        geometry_types = self.geometry_types()
        nearest: list[Record] = []
        for origin_index, origin in enumerate(self._rows):
            candidates = [
                (
                    destination,
                    geometry_types[destination_index],
                    coordinate_distance(centroids[origin_index], centroids[destination_index], method=distance_method),
                )
                for destination_index, destination in enumerate(self._rows)
                if destination_index != origin_index
            ]
            for rank, (destination, destination_geometry_type, distance_value) in enumerate(
                nsmallest(k, candidates, key=lambda item: (float(item[2]), _row_sort_key(item[0]))),
                start=1,
            ):
                nearest.append(
                    {
                        "origin": origin[id_column],
                        "neighbor": destination[id_column],
                        "distance": distance_value,
                        "origin_geometry_type": geometry_types[origin_index],
                        "neighbor_geometry_type": destination_geometry_type,
                        "rank": rank,
                        "distance_method": distance_method,
                    }
                )
        return nearest

    def _nearest_row_matches(
        self,
        origin_centroid: Coordinate,
        right_rows: Sequence[Record],
        right_centroids: Sequence[Coordinate],
        k: int,
        distance_method: str,
        max_distance: float | None = None,
        candidate_indexes: Sequence[int] | None = None,
    ) -> list[tuple[Record, float]]:
        candidates: list[tuple[Record, float]] = []
        indexes = candidate_indexes if candidate_indexes is not None else range(len(right_rows))
        for index in indexes:
            right_row = right_rows[index]
            right_centroid = right_centroids[index]
            if max_distance is not None and distance_method == "euclidean":
                if abs(origin_centroid[0] - right_centroid[0]) > max_distance or abs(origin_centroid[1] - right_centroid[1]) > max_distance:
                    continue
            distance_value = coordinate_distance(origin_centroid, right_centroid, method=distance_method)
            if max_distance is None or distance_value <= max_distance:
                candidates.append((right_row, distance_value))
        return nsmallest(k, candidates, key=lambda item: (float(item[1]), _row_sort_key(item[0])))

    def nearest_join(
        self,
        other: "GeoPromptFrame",
        k: int = 1,
        how: SpatialJoinMode = "inner",
        lsuffix: str = "left",
        rsuffix: str = "right",
        max_distance: float | None = None,
        distance_method: str = "euclidean",
    ) -> "GeoPromptFrame":
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if max_distance is not None and max_distance < 0:
            raise ValueError("max_distance must be zero or greater")
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before nearest joins")

        right_rows = list(other._rows)
        right_columns = [column for column in other.columns if column != other.geometry_column]
        left_centroids = self._centroids()
        right_centroids = other._centroids()
        right_index = other.spatial_index(mode="centroid", cell_size=max_distance) if distance_method == "euclidean" and max_distance is not None and right_rows else None
        joined_rows: list[Record] = []

        for left_row, left_centroid in zip(self._rows, left_centroids, strict=True):
            candidate_indexes = None
            if right_index is not None and max_distance is not None:
                candidate_indexes = right_index.query(
                    (
                        left_centroid[0] - max_distance,
                        left_centroid[1] - max_distance,
                        left_centroid[0] + max_distance,
                        left_centroid[1] + max_distance,
                    )
                )
            row_matches = self._nearest_row_matches(
                origin_centroid=left_centroid,
                right_rows=right_rows,
                right_centroids=right_centroids,
                k=k,
                distance_method=distance_method,
                max_distance=max_distance,
                candidate_indexes=candidate_indexes,
            )

            if not row_matches and how == "left":
                merged_row = dict(left_row)
                for column in right_columns:
                    target_name = column if column not in merged_row else f"{column}_{rsuffix}"
                    merged_row[target_name] = None
                merged_row[f"{other.geometry_column}_{rsuffix}"] = None
                merged_row[f"distance_{rsuffix}"] = None
                merged_row[f"distance_method_{rsuffix}"] = distance_method
                merged_row[f"nearest_rank_{rsuffix}"] = None
                joined_rows.append(merged_row)

            for rank, (right_row, distance_value) in enumerate(row_matches, start=1):
                merged_row = dict(left_row)
                for column in right_columns:
                    target_name = column if column not in merged_row else f"{column}_{rsuffix}"
                    merged_row[target_name] = right_row[column]
                merged_row[f"{other.geometry_column}_{rsuffix}"] = right_row[other.geometry_column]
                merged_row[f"distance_{rsuffix}"] = distance_value
                merged_row[f"distance_method_{rsuffix}"] = distance_method
                merged_row[f"nearest_rank_{rsuffix}"] = rank
                joined_rows.append(merged_row)

        return GeoPromptFrame._from_internal_rows(joined_rows, geometry_column=self.geometry_column, crs=self.crs or other.crs)

    def assign_nearest(
        self,
        targets: "GeoPromptFrame",
        how: SpatialJoinMode = "inner",
        max_distance: float | None = None,
        distance_method: str = "euclidean",
        origin_suffix: str = "origin",
    ) -> "GeoPromptFrame":
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if max_distance is not None and max_distance < 0:
            raise ValueError("max_distance must be zero or greater")
        return targets.nearest_join(
            self,
            k=1,
            how=how,
            rsuffix=origin_suffix,
            max_distance=max_distance,
            distance_method=distance_method,
        )

    def summarize_assignments(
        self,
        targets: "GeoPromptFrame",
        origin_id_column: str = "site_id",
        target_id_column: str = "site_id",
        aggregations: dict[str, AggregationName] | None = None,
        how: SpatialJoinMode = "left",
        max_distance: float | None = None,
        distance_method: str = "euclidean",
        assignment_suffix: str = "assigned",
    ) -> "GeoPromptFrame":
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if max_distance is not None and max_distance < 0:
            raise ValueError("max_distance must be zero or greater")
        if self.crs and targets.crs and self.crs != targets.crs:
            raise ValueError("frames must share the same CRS before assignment summaries")

        self._require_column(origin_id_column)
        targets._require_column(target_id_column)

        origin_centroids = self._centroids()
        target_rows = list(targets._rows)
        target_centroids = targets._centroids()
        origin_index = self.spatial_index(mode="centroid", cell_size=max_distance) if distance_method == "euclidean" and max_distance is not None and self._rows else None
        assignments: dict[str, list[tuple[Record, float]]] = {}

        for target_row, target_centroid in zip(target_rows, target_centroids, strict=True):
            candidate_indexes = None
            if origin_index is not None and max_distance is not None:
                candidate_indexes = origin_index.query(
                    (
                        target_centroid[0] - max_distance,
                        target_centroid[1] - max_distance,
                        target_centroid[0] + max_distance,
                        target_centroid[1] + max_distance,
                    )
                )
            row_matches = self._nearest_row_matches(
                origin_centroid=target_centroid,
                right_rows=self._rows,
                right_centroids=origin_centroids,
                k=1,
                distance_method=distance_method,
                max_distance=max_distance,
                candidate_indexes=candidate_indexes,
            )
            if not row_matches:
                continue
            origin_row, distance_value = row_matches[0]
            origin_key = str(origin_row[origin_id_column])
            assignments.setdefault(origin_key, []).append((target_row, distance_value))

        rows: list[Record] = []
        for origin_row in self._rows:
            origin_key = str(origin_row[origin_id_column])
            assigned_matches = assignments.get(origin_key, [])
            if not assigned_matches and how == "inner":
                continue

            assigned_rows = [row for row, _ in assigned_matches]
            assigned_distances = [distance for _, distance in assigned_matches]
            resolved_row = dict(origin_row)
            resolved_row[f"{target_id_column}s_{assignment_suffix}"] = [str(row[target_id_column]) for row in assigned_rows]
            resolved_row[f"count_{assignment_suffix}"] = len(assigned_rows)
            resolved_row[f"distance_method_{assignment_suffix}"] = distance_method
            resolved_row[f"distance_min_{assignment_suffix}"] = min(assigned_distances) if assigned_distances else None
            resolved_row[f"distance_max_{assignment_suffix}"] = max(assigned_distances) if assigned_distances else None
            resolved_row[f"distance_mean_{assignment_suffix}"] = (
                sum(assigned_distances) / len(assigned_distances) if assigned_distances else None
            )
            resolved_row.update(self._aggregate_rows(assigned_rows, aggregations=aggregations, suffix=assignment_suffix))
            rows.append(resolved_row)

        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs or targets.crs)

    def catchment_competition(
        self,
        targets: "GeoPromptFrame",
        max_distance: float,
        origin_id_column: str = "site_id",
        target_id_column: str = "site_id",
        aggregations: dict[str, AggregationName] | None = None,
        how: SpatialJoinMode = "left",
        distance_method: str = "euclidean",
        competition_suffix: str = "catchment",
    ) -> "GeoPromptFrame":
        if max_distance < 0:
            raise ValueError("max_distance must be zero or greater")
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if self.crs and targets.crs and self.crs != targets.crs:
            raise ValueError("frames must share the same CRS before catchment competition summaries")

        self._require_column(origin_id_column)
        targets._require_column(target_id_column)

        origin_centroids = self._centroids()
        target_rows = list(targets._rows)
        target_centroids = targets._centroids()
        origin_index = self.spatial_index(mode="centroid", cell_size=max_distance) if distance_method == "euclidean" and self._rows else None

        coverage_buckets: dict[str, dict[str, list[Record]]] = {
            str(origin_row[origin_id_column]): {
                "covered": [],
                "exclusive": [],
                "shared": [],
                "won": [],
            }
            for origin_row in self._rows
        }
        unserved_target_ids: list[str] = []

        for target_row, target_centroid in zip(target_rows, target_centroids, strict=True):
            candidate_indexes = None
            if origin_index is not None:
                candidate_indexes = origin_index.query(
                    (
                        target_centroid[0] - max_distance,
                        target_centroid[1] - max_distance,
                        target_centroid[0] + max_distance,
                        target_centroid[1] + max_distance,
                    )
                )
            row_matches = self._nearest_row_matches(
                origin_centroid=target_centroid,
                right_rows=self._rows,
                right_centroids=origin_centroids,
                k=len(self._rows),
                distance_method=distance_method,
                max_distance=max_distance,
                candidate_indexes=candidate_indexes,
            )
            if not row_matches:
                unserved_target_ids.append(str(target_row[target_id_column]))
                continue

            is_shared = len(row_matches) > 1
            for origin_row, _distance_value in row_matches:
                origin_key = str(origin_row[origin_id_column])
                coverage_buckets[origin_key]["covered"].append(target_row)
                coverage_buckets[origin_key]["shared" if is_shared else "exclusive"].append(target_row)

            winning_origin, _winning_distance = row_matches[0]
            coverage_buckets[str(winning_origin[origin_id_column])]["won"].append(target_row)

        rows: list[Record] = []
        for origin_row in self._rows:
            origin_key = str(origin_row[origin_id_column])
            bucket = coverage_buckets[origin_key]
            covered_rows = bucket["covered"]
            if not covered_rows and how == "inner":
                continue

            resolved_row = dict(origin_row)
            resolved_row[f"{target_id_column}s_{competition_suffix}"] = [str(row[target_id_column]) for row in covered_rows]
            resolved_row[f"count_{competition_suffix}"] = len(covered_rows)
            resolved_row[f"{target_id_column}s_exclusive_{competition_suffix}"] = [
                str(row[target_id_column]) for row in bucket["exclusive"]
            ]
            resolved_row[f"count_exclusive_{competition_suffix}"] = len(bucket["exclusive"])
            resolved_row[f"{target_id_column}s_shared_{competition_suffix}"] = [
                str(row[target_id_column]) for row in bucket["shared"]
            ]
            resolved_row[f"count_shared_{competition_suffix}"] = len(bucket["shared"])
            resolved_row[f"{target_id_column}s_won_{competition_suffix}"] = [
                str(row[target_id_column]) for row in bucket["won"]
            ]
            resolved_row[f"count_won_{competition_suffix}"] = len(bucket["won"])
            resolved_row[f"{target_id_column}s_unserved_{competition_suffix}"] = list(unserved_target_ids)
            resolved_row[f"count_unserved_{competition_suffix}"] = len(unserved_target_ids)
            resolved_row[f"distance_limit_{competition_suffix}"] = max_distance
            resolved_row[f"distance_method_{competition_suffix}"] = distance_method
            resolved_row.update(self._aggregate_rows(covered_rows, aggregations=aggregations, suffix=competition_suffix))
            rows.append(resolved_row)

        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs or targets.crs)

    def query_radius(
        self,
        anchor: str | Geometry | Coordinate,
        max_distance: float,
        id_column: str = "site_id",
        include_anchor: bool = False,
        distance_method: str = "euclidean",
    ) -> "GeoPromptFrame":
        if max_distance < 0:
            raise ValueError("max_distance must be zero or greater")

        anchor_geometry, anchor_id = self._resolve_anchor_geometry(anchor, id_column=id_column)
        anchor_centroid = geometry_centroid(anchor_geometry)
        centroid_index = self.spatial_index(mode="centroid", cell_size=max_distance) if distance_method == "euclidean" and self._rows else None
        candidate_indexes = (
            centroid_index.query(
                (
                    anchor_centroid[0] - max_distance,
                    anchor_centroid[1] - max_distance,
                    anchor_centroid[0] + max_distance,
                    anchor_centroid[1] + max_distance,
                )
            )
            if centroid_index is not None
            else list(range(len(self._rows)))
        )

        rows: list[Record] = []
        for index in candidate_indexes:
            row = self._rows[index]
            row_id = str(row.get(id_column)) if id_column in row else None
            distance_value = coordinate_distance(anchor_centroid, geometry_centroid(row[self.geometry_column]), method=distance_method)
            if anchor_id is not None and not include_anchor and row_id == anchor_id:
                continue
            if distance_value <= max_distance:
                resolved_row = dict(row)
                resolved_row["distance"] = distance_value
                resolved_row["distance_method"] = distance_method
                if anchor_id is not None:
                    resolved_row["anchor_id"] = anchor_id
                rows.append(resolved_row)

        rows.sort(key=lambda item: (float(item["distance"]), str(item.get(id_column, ""))))
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def within_distance(
        self,
        anchor: str | Geometry | Coordinate,
        max_distance: float,
        id_column: str = "site_id",
        include_anchor: bool = False,
        distance_method: str = "euclidean",
    ) -> list[bool]:
        if max_distance < 0:
            raise ValueError("max_distance must be zero or greater")

        anchor_geometry, anchor_id = self._resolve_anchor_geometry(anchor, id_column=id_column)
        anchor_centroid = geometry_centroid(anchor_geometry)
        centroids = self._centroids()
        centroid_index = self.spatial_index(mode="centroid", cell_size=max_distance) if distance_method == "euclidean" and self._rows else None
        candidate_indexes = set(
            centroid_index.query(
                (
                    anchor_centroid[0] - max_distance,
                    anchor_centroid[1] - max_distance,
                    anchor_centroid[0] + max_distance,
                    anchor_centroid[1] + max_distance,
                )
            )
        ) if centroid_index is not None else set(range(len(self._rows)))

        mask: list[bool] = []
        for index, (row, centroid) in enumerate(zip(self._rows, centroids, strict=True)):
            if anchor_id is not None and not include_anchor and id_column in row and str(row[id_column]) == anchor_id:
                mask.append(False)
                continue
            if index not in candidate_indexes:
                mask.append(False)
                continue
            mask.append(coordinate_distance(anchor_centroid, centroid, method=distance_method) <= max_distance)
        return mask

    def proximity_join(
        self,
        other: "GeoPromptFrame",
        max_distance: float,
        how: SpatialJoinMode = "inner",
        lsuffix: str = "left",
        rsuffix: str = "right",
        distance_method: str = "euclidean",
    ) -> "GeoPromptFrame":
        if max_distance < 0:
            raise ValueError("max_distance must be zero or greater")
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before proximity joins")

        right_rows = list(other._rows)
        right_columns = [column for column in other.columns if column != other.geometry_column]
        left_centroids = self._centroids()
        right_centroids = other._centroids()
        right_index = other.spatial_index(mode="centroid", cell_size=max_distance) if distance_method == "euclidean" and right_rows else None
        joined_rows: list[Record] = []

        for left_row, left_centroid in zip(self._rows, left_centroids, strict=True):
            row_matches: list[tuple[Record, float]] = []
            candidate_indexes = (
                right_index.query(
                    (
                        left_centroid[0] - max_distance,
                        left_centroid[1] - max_distance,
                        left_centroid[0] + max_distance,
                        left_centroid[1] + max_distance,
                    )
                )
                if right_index is not None
                else range(len(right_rows))
            )
            for index in candidate_indexes:
                right_row = right_rows[index]
                right_centroid = right_centroids[index]
                if distance_method == "euclidean":
                    if abs(left_centroid[0] - right_centroid[0]) > max_distance or abs(left_centroid[1] - right_centroid[1]) > max_distance:
                        continue
                distance_value = coordinate_distance(left_centroid, right_centroid, method=distance_method)
                if distance_value <= max_distance:
                    row_matches.append((right_row, distance_value))

            row_matches.sort(key=lambda item: (float(item[1]), str(item[0].get("site_id", item[0].get("region_id", "")))))

            if not row_matches and how == "left":
                merged_row = dict(left_row)
                for column in right_columns:
                    target_name = column if column not in merged_row else f"{column}_{rsuffix}"
                    merged_row[target_name] = None
                merged_row[f"{other.geometry_column}_{rsuffix}"] = None
                merged_row[f"distance_{rsuffix}"] = None
                merged_row[f"distance_method_{rsuffix}"] = distance_method
                joined_rows.append(merged_row)

            for right_row, distance_value in row_matches:
                merged_row = dict(left_row)
                for column in right_columns:
                    target_name = column if column not in merged_row else f"{column}_{rsuffix}"
                    merged_row[target_name] = right_row[column]
                merged_row[f"{other.geometry_column}_{rsuffix}"] = right_row[other.geometry_column]
                merged_row[f"distance_{rsuffix}"] = distance_value
                merged_row[f"distance_method_{rsuffix}"] = distance_method
                joined_rows.append(merged_row)

        return GeoPromptFrame._from_internal_rows(joined_rows, geometry_column=self.geometry_column, crs=self.crs or other.crs)

    def buffer(self, distance: float, resolution: int = 16) -> "GeoPromptFrame":
        buffered_groups = buffer_geometries(
            [row[self.geometry_column] for row in self._rows],
            distance=distance,
            resolution=resolution,
        )
        rows: list[Record] = []
        for row, buffered_geometries in zip(self._rows, buffered_groups, strict=True):
            for buffered_geometry in buffered_geometries:
                buffered_row = dict(row)
                buffered_row[self.geometry_column] = buffered_geometry
                rows.append(buffered_row)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def buffer_join(
        self,
        other: "GeoPromptFrame",
        distance: float,
        how: SpatialJoinMode = "inner",
        lsuffix: str = "left",
        rsuffix: str = "right",
        resolution: int = 16,
    ) -> "GeoPromptFrame":
        if distance < 0:
            raise ValueError("distance must be zero or greater")
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before buffer joins")

        right_rows = list(other._rows)
        right_columns = [column for column in other.columns if column != other.geometry_column]
        right_index = other.spatial_index(mode="geometry") if right_rows else None
        buffered_groups = buffer_geometries(
            [row[self.geometry_column] for row in self._rows],
            distance=distance,
            resolution=resolution,
        )

        joined_rows: list[Record] = []
        for left_row, buffered_geometries in zip(self._rows, buffered_groups, strict=True):
            row_matches: list[tuple[Record, Geometry]] = []
            for buffered_geometry in buffered_geometries:
                buffered_bounds = geometry_bounds(buffered_geometry)
                candidate_indexes = right_index.query(buffered_bounds) if right_index is not None else []
                for index in candidate_indexes:
                    right_row = right_rows[index]
                    right_bound = geometry_bounds(right_row[other.geometry_column])
                    if not _bounds_intersect(buffered_bounds, right_bound):
                        continue
                    if geometry_intersects(buffered_geometry, right_row[other.geometry_column]):
                        row_matches.append((right_row, buffered_geometry))

            if not row_matches and how == "left":
                merged_row = dict(left_row)
                merged_row[f"buffer_geometry_{lsuffix}"] = None
                merged_row[f"buffer_distance_{lsuffix}"] = distance
                for column in right_columns:
                    target_name = column if column not in merged_row else f"{column}_{rsuffix}"
                    merged_row[target_name] = None
                merged_row[f"{other.geometry_column}_{rsuffix}"] = None
                joined_rows.append(merged_row)

            for right_row, buffered_geometry in row_matches:
                merged_row = dict(left_row)
                merged_row[f"buffer_geometry_{lsuffix}"] = buffered_geometry
                merged_row[f"buffer_distance_{lsuffix}"] = distance
                for column in right_columns:
                    target_name = column if column not in merged_row else f"{column}_{rsuffix}"
                    merged_row[target_name] = right_row[column]
                merged_row[f"{other.geometry_column}_{rsuffix}"] = right_row[other.geometry_column]
                joined_rows.append(merged_row)

        return GeoPromptFrame._from_internal_rows(joined_rows, geometry_column=self.geometry_column, crs=self.crs or other.crs)

    def coverage_summary(
        self,
        targets: "GeoPromptFrame",
        predicate: SpatialJoinPredicate = "intersects",
        target_id_column: str = "site_id",
        aggregations: dict[str, AggregationName] | None = None,
        rsuffix: str = "covered",
    ) -> "GeoPromptFrame":
        if self.crs and targets.crs and self.crs != targets.crs:
            raise ValueError("frames must share the same CRS before coverage summaries")
        if predicate not in {"intersects", "within", "contains"}:
            raise ValueError(f"unsupported spatial join predicate: {predicate}")

        targets._require_column(target_id_column)
        target_rows = list(targets._rows)
        target_index = targets.spatial_index(mode="geometry") if target_rows else None
        predicate_bounds_filter = {
            "intersects": _bounds_intersect,
            "within": _bounds_within,
            "contains": lambda left, right: _bounds_within(right, left),
        }

        def matches(left_geometry: Geometry, right_geometry: Geometry) -> bool:
            if predicate == "intersects":
                return geometry_intersects(left_geometry, right_geometry)
            if predicate == "within":
                return geometry_within(left_geometry, right_geometry)
            return geometry_contains(left_geometry, right_geometry)

        rows: list[Record] = []
        for row in self._rows:
            left_geometry = row[self.geometry_column]
            left_bounds = geometry_bounds(left_geometry)
            matched_rows: list[Record] = []
            candidate_indexes = target_index.query(left_bounds) if target_index is not None else []
            for index in candidate_indexes:
                target_row = target_rows[index]
                target_bound = geometry_bounds(target_row[targets.geometry_column])
                if not predicate_bounds_filter[predicate](left_bounds, target_bound):
                    continue
                if matches(left_geometry, target_row[targets.geometry_column]):
                    matched_rows.append(target_row)

            resolved_row = dict(row)
            resolved_row[f"{target_id_column}s_{rsuffix}"] = [str(item[target_id_column]) for item in matched_rows]
            resolved_row[f"count_{rsuffix}"] = len(matched_rows)
            resolved_row[f"predicate_{rsuffix}"] = predicate
            resolved_row.update(self._aggregate_rows(matched_rows, aggregations=aggregations, suffix=rsuffix))
            rows.append(resolved_row)

        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs or targets.crs)

    def query_bounds(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        mode: BoundsQueryMode = "intersects",
    ) -> "GeoPromptFrame":
        if min_x > max_x or min_y > max_y:
            raise ValueError("query bounds must be ordered from minimum to maximum")

        rows: list[Record] = []
        active_index = self.spatial_index(mode="centroid") if mode == "centroid" else self.spatial_index(mode="geometry")
        query_mode = "within" if mode == "within" else "intersects"
        candidate_indexes = active_index.query((min_x, min_y, max_x, max_y), mode=query_mode)
        for index in candidate_indexes:
            row = self._rows[index]
            geometry = row[self.geometry_column]
            if mode == "intersects":
                include_row = geometry_intersects_bounds(geometry, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)
            elif mode == "within":
                include_row = geometry_within_bounds(geometry, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)
            elif mode == "centroid":
                centroid_x, centroid_y = geometry_centroid(geometry)
                include_row = min_x <= centroid_x <= max_x and min_y <= centroid_y <= max_y
            else:
                raise ValueError(f"unsupported bounds query mode: {mode}")

            if include_row:
                rows.append(dict(row))

        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def set_crs(self, crs: str, allow_override: bool = False) -> "GeoPromptFrame":
        if self.crs is not None and self.crs != crs and not allow_override:
            raise ValueError("frame already has a CRS; pass allow_override=True to replace it")
        return GeoPromptFrame(rows=self.to_records(), geometry_column=self.geometry_column, crs=crs)

    def to_crs(self, target_crs: str) -> "GeoPromptFrame":
        if self.crs is None:
            raise ValueError("frame CRS is not set; call set_crs before reprojecting")
        if self.crs == target_crs:
            return GeoPromptFrame(rows=self.to_records(), geometry_column=self.geometry_column, crs=self.crs)

        try:
            pyproj = importlib.import_module("pyproj")
        except ImportError as exc:
            raise RuntimeError("Install projection support with 'pip install -e .[projection]' before calling to_crs.") from exc

        transformer = pyproj.Transformer.from_crs(self.crs, target_crs, always_xy=True)

        def reproject_coordinate(coordinate: Coordinate) -> Coordinate:
            x_value, y_value = transformer.transform(coordinate[0], coordinate[1])
            return (float(x_value), float(y_value))

        rows = self.to_records()
        for row in rows:
            row[self.geometry_column] = transform_geometry(row[self.geometry_column], reproject_coordinate)
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=target_crs)

    def spatial_join(
        self,
        other: "GeoPromptFrame",
        predicate: SpatialJoinPredicate = "intersects",
        how: SpatialJoinMode = "inner",
        lsuffix: str = "left",
        rsuffix: str = "right",
        include_diagnostics: bool = False,
    ) -> "GeoPromptFrame":
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before spatial joins")

        right_columns = [column for column in other.columns if column != other.geometry_column]
        joined_rows: list[Record] = []
        right_rows = list(other._rows)
        right_index = other.spatial_index(mode="geometry") if right_rows else None

        predicate_bounds_filter = {
            "intersects": _bounds_intersect,
            "within": _bounds_within,
            "contains": lambda left, right: _bounds_within(right, left),
        }
        if predicate not in predicate_bounds_filter:
            raise ValueError(f"unsupported spatial join predicate: {predicate}")

        def matches(left_geometry: Geometry, right_geometry: Geometry) -> bool:
            if predicate == "intersects":
                return geometry_intersects(left_geometry, right_geometry)
            if predicate == "within":
                return geometry_within(left_geometry, right_geometry)
            if predicate == "contains":
                return geometry_contains(left_geometry, right_geometry)
            raise ValueError(f"unsupported spatial join predicate: {predicate}")

        for left_index, left_row in enumerate(self._rows):
            left_geometry = left_row[self.geometry_column]
            left_bounds = geometry_bounds(left_geometry)
            row_matches = []
            candidate_indexes = right_index.query(left_bounds) if right_index is not None else []
            for candidate_index in candidate_indexes:
                right_row = right_rows[candidate_index]
                right_bound = geometry_bounds(right_row[other.geometry_column])
                if not predicate_bounds_filter[predicate](left_bounds, right_bound):
                    continue
                if matches(left_geometry, right_row[other.geometry_column]):
                    row_matches.append(right_row)

            if not row_matches and how == "left":
                merged_row = dict(left_row)
                for column in right_columns:
                    target_name = column if column not in merged_row else f"{column}_{rsuffix}"
                    merged_row[target_name] = None
                merged_row[f"join_predicate_{rsuffix}"] = predicate
                if include_diagnostics:
                    merged_row[f"candidate_count_{rsuffix}"] = len(candidate_indexes)
                    merged_row[f"pruning_ratio_{rsuffix}"] = 1.0 - (len(candidate_indexes) / len(right_rows) if right_rows else 0.0)
                    merged_row[f"match_count_{rsuffix}"] = 0
                joined_rows.append(merged_row)

            for right_row in row_matches:
                merged_row = dict(left_row)
                for column in right_columns:
                    target_name = column if column not in merged_row else f"{column}_{rsuffix}"
                    merged_row[target_name] = right_row[column]
                merged_row[f"{other.geometry_column}_{rsuffix}"] = right_row[other.geometry_column]
                merged_row[f"join_predicate_{rsuffix}"] = predicate
                if include_diagnostics:
                    merged_row[f"candidate_count_{rsuffix}"] = len(candidate_indexes)
                    merged_row[f"pruning_ratio_{rsuffix}"] = 1.0 - (len(candidate_indexes) / len(right_rows) if right_rows else 0.0)
                    merged_row[f"match_count_{rsuffix}"] = len(row_matches)
                joined_rows.append(merged_row)

        return GeoPromptFrame._from_internal_rows(joined_rows, geometry_column=self.geometry_column, crs=self.crs or other.crs)

    def overlay_summary(
        self,
        other: "GeoPromptFrame",
        right_id_column: str = "region_id",
        aggregations: dict[str, AggregationName] | None = None,
        how: SpatialJoinMode = "left",
        group_by: str | None = None,
        normalize_by: OverlayNormalizeMode = "left",
        top_n_groups: int | None = None,
        summary_suffix: str = "overlay",
    ) -> "GeoPromptFrame":
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before overlay summaries")
        if normalize_by not in {"left", "right", "both"}:
            raise ValueError("normalize_by must be 'left', 'right', or 'both'")
        if top_n_groups is not None and top_n_groups <= 0:
            raise ValueError("top_n_groups must be greater than zero")

        other._require_column(right_id_column)
        if group_by is not None:
            other._require_column(group_by)
        other_rows = list(other._rows)
        grouped: dict[int, list[tuple[int, list[Geometry]]]] = {}
        for left_index, right_index, geometries in overlay_intersections(
            [row[self.geometry_column] for row in self._rows],
            [row[other.geometry_column] for row in other_rows],
        ):
            grouped.setdefault(left_index, []).append((right_index, geometries))

        rows: list[Record] = []
        for left_index, left_row in enumerate(self._rows):
            matches = grouped.get(left_index, [])
            if not matches and how == "inner":
                continue

            matched_rows = [other_rows[right_index] for right_index, _ in matches]
            overlap_area = sum(geometry_area(geometry) for _right_index, geometries in matches for geometry in geometries)
            overlap_length = sum(geometry_length(geometry) for _right_index, geometries in matches for geometry in geometries)
            intersection_count = sum(len(geometries) for _right_index, geometries in matches)
            right_area_total = sum(geometry_area(other_rows[right_index][other.geometry_column]) for right_index, _ in matches)
            right_length_total = sum(geometry_length(other_rows[right_index][other.geometry_column]) for right_index, _ in matches)

            left_geometry = left_row[self.geometry_column]
            left_area = geometry_area(left_geometry)
            left_length = geometry_length(left_geometry)

            resolved_row = dict(left_row)
            resolved_row[f"{right_id_column}s_{summary_suffix}"] = [str(row[right_id_column]) for row in matched_rows]
            resolved_row[f"count_{summary_suffix}"] = len(matched_rows)
            resolved_row[f"intersection_count_{summary_suffix}"] = intersection_count
            resolved_row[f"area_overlap_{summary_suffix}"] = overlap_area
            resolved_row[f"length_overlap_{summary_suffix}"] = overlap_length
            resolved_row[f"area_share_{summary_suffix}"] = (overlap_area / left_area) if left_area > 0 and normalize_by in {"left", "both"} else None
            resolved_row[f"length_share_{summary_suffix}"] = (overlap_length / left_length) if left_length > 0 and normalize_by in {"left", "both"} else None
            resolved_row[f"area_share_right_{summary_suffix}"] = (overlap_area / right_area_total) if right_area_total > 0 and normalize_by in {"right", "both"} else None
            resolved_row[f"length_share_right_{summary_suffix}"] = (overlap_length / right_length_total) if right_length_total > 0 and normalize_by in {"right", "both"} else None
            if group_by is not None:
                grouped_matches: dict[str, dict[str, Any]] = {}
                for right_index, geometries in matches:
                    group_value = str(other_rows[right_index][group_by])
                    bucket = grouped_matches.setdefault(
                        group_value,
                        {
                            "group": group_value,
                            "count": 0,
                            "intersection_count": 0,
                            "area_overlap": 0.0,
                            "length_overlap": 0.0,
                            f"{right_id_column}s": [],
                            "right_area_total": 0.0,
                            "right_length_total": 0.0,
                        },
                    )
                    bucket["count"] += 1
                    bucket["intersection_count"] += len(geometries)
                    bucket["area_overlap"] += sum(geometry_area(geometry) for geometry in geometries)
                    bucket["length_overlap"] += sum(geometry_length(geometry) for geometry in geometries)
                    bucket[f"{right_id_column}s"].append(str(other_rows[right_index][right_id_column]))
                    bucket["right_area_total"] += geometry_area(other_rows[right_index][other.geometry_column])
                    bucket["right_length_total"] += geometry_length(other_rows[right_index][other.geometry_column])

                group_summaries = []
                for group_summary in grouped_matches.values():
                    group_record = dict(group_summary)
                    group_record["area_share_left"] = (group_summary["area_overlap"] / left_area) if left_area > 0 and normalize_by in {"left", "both"} else None
                    group_record["length_share_left"] = (group_summary["length_overlap"] / left_length) if left_length > 0 and normalize_by in {"left", "both"} else None
                    group_record["area_share_right"] = (group_summary["area_overlap"] / group_summary["right_area_total"]) if group_summary["right_area_total"] > 0 and normalize_by in {"right", "both"} else None
                    group_record["length_share_right"] = (group_summary["length_overlap"] / group_summary["right_length_total"]) if group_summary["right_length_total"] > 0 and normalize_by in {"right", "both"} else None
                    group_record.pop("right_area_total")
                    group_record.pop("right_length_total")
                    group_summaries.append(group_record)

                group_summaries.sort(key=lambda item: (-float(item["area_overlap"]), -float(item["length_overlap"]), str(item["group"])))
                if top_n_groups is not None:
                    group_summaries = group_summaries[:top_n_groups]
                resolved_row[f"groups_{summary_suffix}"] = group_summaries
                resolved_row[f"best_group_{summary_suffix}"] = group_summaries[0]["group"] if group_summaries else None
            resolved_row.update(self._aggregate_rows(matched_rows, aggregations=aggregations, suffix=summary_suffix))
            rows.append(resolved_row)

        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs or other.crs)

    def clip(self, mask: "GeoPromptFrame") -> "GeoPromptFrame":
        if self.crs and mask.crs and self.crs != mask.crs:
            raise ValueError("frames must share the same CRS before clip operations")

        mask_rows = list(mask)
        clipped_groups = clip_geometries(
            [row[self.geometry_column] for row in self._rows],
            [row[mask.geometry_column] for row in mask_rows],
        )
        rows: list[Record] = []
        for row, clipped_geometries in zip(self._rows, clipped_groups, strict=True):
            for clipped_geometry in clipped_geometries:
                clipped_row = dict(row)
                clipped_row[self.geometry_column] = clipped_geometry
                rows.append(clipped_row)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs or mask.crs)

    def overlay_intersections(
        self,
        other: "GeoPromptFrame",
        lsuffix: str = "left",
        rsuffix: str = "right",
    ) -> "GeoPromptFrame":
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before overlay operations")

        intersections = overlay_intersections(
            [row[self.geometry_column] for row in self._rows],
            [row[other.geometry_column] for row in other],
        )
        right_columns = [column for column in other.columns if column != other.geometry_column]
        rows: list[Record] = []
        for left_index, right_index, geometries in intersections:
            left_row = self._rows[left_index]
            right_row = other.to_records()[right_index]
            for geometry in geometries:
                merged_row = dict(left_row)
                for column in right_columns:
                    target_name = column if column not in merged_row else f"{column}_{rsuffix}"
                    merged_row[target_name] = right_row[column]
                merged_row[f"{other.geometry_column}_{rsuffix}"] = right_row[other.geometry_column]
                merged_row[self.geometry_column] = geometry
                rows.append(merged_row)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs or other.crs)

    def dissolve(
        self,
        by: str,
        aggregations: dict[str, AggregationName] | None = None,
    ) -> "GeoPromptFrame":
        self._require_column(by)
        grouped_rows: dict[Any, list[Record]] = {}
        for row in self._rows:
            grouped_rows.setdefault(row[by], []).append(row)

        rows: list[Record] = []
        for group_value, group_rows in grouped_rows.items():
            dissolved_geometries = dissolve_geometries([row[self.geometry_column] for row in group_rows])
            aggregate_values: dict[str, Any] = {by: group_value}
            for column in self.columns:
                if column in {by, self.geometry_column}:
                    continue
                operation = (aggregations or {}).get(column)
                values = [row[column] for row in group_rows if column in row]
                if not values:
                    continue
                if operation is None:
                    aggregate_values[column] = values[0]
                elif operation == "sum":
                    aggregate_values[column] = sum(float(value) for value in values)
                elif operation == "mean":
                    aggregate_values[column] = sum(float(value) for value in values) / len(values)
                elif operation == "min":
                    aggregate_values[column] = min(values)
                elif operation == "max":
                    aggregate_values[column] = max(values)
                elif operation == "first":
                    aggregate_values[column] = values[0]
                elif operation == "count":
                    aggregate_values[column] = len(values)
                else:
                    raise ValueError(f"unsupported aggregation: {operation}")

            for geometry in dissolved_geometries:
                rows.append({**aggregate_values, self.geometry_column: geometry})

        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs)

    def with_column(self, name: str, values: Sequence[Any]) -> "GeoPromptFrame":
        if len(values) != len(self._rows):
            raise ValueError("column length must match the frame length")
        rows = self.to_records()
        for row, value in zip(rows, values, strict=True):
            row[name] = value
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs)

    def assign(self, **columns: Any) -> "GeoPromptFrame":
        frame = self
        for name, value in columns.items():
            if callable(value):
                resolved = value(frame)
            else:
                resolved = value

            if isinstance(resolved, Sequence) and not isinstance(resolved, (str, bytes)):
                values = list(resolved)
            else:
                values = [resolved for _ in frame._rows]

            frame = frame.with_column(name=name, values=values)
        return frame

    def _require_column(self, name: str) -> None:
        if name not in self.columns:
            raise KeyError(f"column '{name}' is not present")

    def neighborhood_pressure(
        self,
        weight_column: str,
        scale: float = 1.0,
        power: float = 2.0,
        include_self: bool = False,
        distance_method: str = "euclidean",
    ) -> list[float]:
        self._require_column(weight_column)
        pressures: list[float] = []
        for row in self._rows:
            total = 0.0
            for other in self._rows:
                if not include_self and row is other:
                    continue
                distance_value = geometry_distance(row[self.geometry_column], other[self.geometry_column], method=distance_method)
                total += prompt_influence(
                    weight=float(other[weight_column]),
                    distance_value=distance_value,
                    scale=scale,
                    power=power,
                )
            pressures.append(total)
        return pressures

    def anchor_influence(
        self,
        weight_column: str,
        anchor: str,
        id_column: str = "site_id",
        scale: float = 1.0,
        power: float = 2.0,
        distance_method: str = "euclidean",
    ) -> list[float]:
        self._require_column(weight_column)
        self._require_column(id_column)
        anchor_row = next((row for row in self._rows if row[id_column] == anchor), None)
        if anchor_row is None:
            raise KeyError(f"anchor '{anchor}' was not found in column '{id_column}'")

        return [
            prompt_influence(
                weight=float(row[weight_column]),
                distance_value=geometry_distance(anchor_row[self.geometry_column], row[self.geometry_column], method=distance_method),
                scale=scale,
                power=power,
            )
            for row in self._rows
        ]

    def corridor_accessibility(
        self,
        weight_column: str,
        anchor: str,
        id_column: str = "site_id",
        scale: float = 1.0,
        power: float = 2.0,
        distance_method: str = "euclidean",
    ) -> list[float]:
        self._require_column(weight_column)
        self._require_column(id_column)
        anchor_row = next((row for row in self._rows if row[id_column] == anchor), None)
        if anchor_row is None:
            raise KeyError(f"anchor '{anchor}' was not found in column '{id_column}'")

        return [
            corridor_strength(
                weight=float(row[weight_column]),
                corridor_length=geometry_length(row[self.geometry_column]),
                distance_value=geometry_distance(anchor_row[self.geometry_column], row[self.geometry_column], method=distance_method),
                scale=scale,
                power=power,
            )
            for row in self._rows
        ]

    def area_similarity_table(
        self,
        id_column: str = "site_id",
        scale: float = 1.0,
        power: float = 1.0,
        distance_method: str = "euclidean",
    ) -> list[Record]:
        self._require_column(id_column)
        interactions: list[Record] = []
        for origin in self._rows:
            for destination in self._rows:
                if origin is destination:
                    continue
                interactions.append(
                    {
                        "origin": origin[id_column],
                        "destination": destination[id_column],
                        "area_similarity": area_similarity(
                            origin_area=geometry_area(origin[self.geometry_column]),
                            destination_area=geometry_area(destination[self.geometry_column]),
                            distance_value=geometry_distance(origin[self.geometry_column], destination[self.geometry_column], method=distance_method),
                            scale=scale,
                            power=power,
                        ),
                        "distance_method": distance_method,
                    }
                )
        return interactions

    def interaction_table(
        self,
        origin_weight: str,
        destination_weight: str,
        id_column: str = "site_id",
        scale: float = 1.0,
        power: float = 2.0,
        preferred_bearing: float | None = None,
        distance_method: str = "euclidean",
    ) -> list[Record]:
        self._require_column(origin_weight)
        self._require_column(destination_weight)
        self._require_column(id_column)

        interactions: list[Record] = []
        for origin in self._rows:
            for destination in self._rows:
                if origin is destination:
                    continue
                distance_value = geometry_distance(origin[self.geometry_column], destination[self.geometry_column], method=distance_method)
                interaction = prompt_interaction(
                    origin_weight=float(origin[origin_weight]),
                    destination_weight=float(destination[destination_weight]),
                    distance_value=distance_value,
                    scale=scale,
                    power=power,
                )
                record: Record = {
                    "origin": origin[id_column],
                    "destination": destination[id_column],
                    "distance": distance_value,
                    "interaction": interaction,
                    "distance_method": distance_method,
                }
                if preferred_bearing is not None:
                    record["directional_alignment"] = directional_alignment(
                        origin=geometry_centroid(origin[self.geometry_column]),
                        destination=geometry_centroid(destination[self.geometry_column]),
                        preferred_bearing=preferred_bearing,
                    )
                interactions.append(record)
        return interactions

    def select(self, *columns: str) -> "GeoPromptFrame":
        missing = [col for col in columns if col not in self.columns]
        if missing:
            raise KeyError(f"columns not found: {missing}")
        selected_columns = list(columns)
        if self.geometry_column not in selected_columns:
            selected_columns.append(self.geometry_column)
        rows = [{col: row[col] for col in selected_columns if col in row} for row in self._rows]
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def rename_columns(self, mapping: dict[str, str]) -> "GeoPromptFrame":
        new_geometry_column = mapping.get(self.geometry_column, self.geometry_column)
        rows: list[Record] = []
        for row in self._rows:
            new_row = {mapping.get(key, key): value for key, value in row.items()}
            rows.append(new_row)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=new_geometry_column, crs=self.crs)

    def filter(self, predicate: Any) -> "GeoPromptFrame":
        if callable(predicate):
            rows = [row for row in self._rows if predicate(row)]
        elif isinstance(predicate, Sequence) and not isinstance(predicate, (str, bytes)) and all(isinstance(v, bool) for v in predicate):
            if len(predicate) != len(self._rows):
                raise ValueError("boolean mask length must match frame length")
            rows = [row for row, keep in zip(self._rows, predicate, strict=True) if keep]
        else:
            raise TypeError("predicate must be a callable or a boolean sequence")
        return GeoPromptFrame._from_internal_rows([dict(r) for r in rows], geometry_column=self.geometry_column, crs=self.crs)

    def sort(self, by: str, descending: bool = False) -> "GeoPromptFrame":
        self._require_column(by)
        non_null_rows = [row for row in self._rows if row.get(by) is not None]
        null_rows = [row for row in self._rows if row.get(by) is None]
        sorted_rows = sorted(non_null_rows, key=lambda row: row.get(by), reverse=descending) + null_rows
        return GeoPromptFrame._from_internal_rows([dict(r) for r in sorted_rows], geometry_column=self.geometry_column, crs=self.crs)

    def describe(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for column in self.columns:
            if column == self.geometry_column:
                continue
            values = [row[column] for row in self._rows if column in row and row[column] is not None]
            numeric = [float(v) for v in values if isinstance(v, (int, float))]
            if not numeric:
                continue
            stats[column] = {
                "count": len(numeric),
                "min": min(numeric),
                "max": max(numeric),
                "mean": sum(numeric) / len(numeric),
                "sum": sum(numeric),
            }
        return stats

    def fishnet(
        self,
        cell_width: float,
        cell_height: float | None = None,
        include_empty: bool = False,
        grid_id_column: str = "grid_id",
    ) -> "GeoPromptFrame":
        if cell_width <= 0:
            raise ValueError("cell_width must be greater than zero")
        resolved_cell_height = cell_height if cell_height is not None else cell_width
        if resolved_cell_height <= 0:
            raise ValueError("cell_height must be greater than zero")
        if not self._rows:
            return GeoPromptFrame._from_internal_rows([], geometry_column=self.geometry_column, crs=self.crs)

        bounds = self.bounds()
        geometry_index = self.spatial_index(mode="geometry")
        rows: list[Record] = []

        row_index = 0
        y_value = bounds.min_y
        while y_value < bounds.max_y or math.isclose(y_value, bounds.max_y):
            column_index = 0
            x_value = bounds.min_x
            while x_value < bounds.max_x or math.isclose(x_value, bounds.max_x):
                cell_geometry = _rectangle_polygon(
                    x_value,
                    y_value,
                    min(x_value + cell_width, bounds.max_x),
                    min(y_value + resolved_cell_height, bounds.max_y),
                )
                cell_bounds = geometry_bounds(cell_geometry)
                candidate_indexes = geometry_index.query(cell_bounds)
                if include_empty or any(
                    geometry_intersects(cell_geometry, self._rows[index][self.geometry_column])
                    for index in candidate_indexes
                ):
                    rows.append(
                        {
                            grid_id_column: f"cell-{row_index:04d}-{column_index:04d}",
                            "grid_row": row_index,
                            "grid_column": column_index,
                            "cell_width": cell_width,
                            "cell_height": resolved_cell_height,
                            self.geometry_column: cell_geometry,
                        }
                    )
                column_index += 1
                x_value += cell_width
                if x_value > bounds.max_x and not math.isclose(x_value, bounds.max_x):
                    break
            row_index += 1
            y_value += resolved_cell_height
            if y_value > bounds.max_y and not math.isclose(y_value, bounds.max_y):
                break

        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def hexbin(
        self,
        size: float,
        include_empty: bool = False,
        grid_id_column: str = "grid_id",
    ) -> "GeoPromptFrame":
        if size <= 0:
            raise ValueError("size must be greater than zero")
        if not self._rows:
            return GeoPromptFrame._from_internal_rows([], geometry_column=self.geometry_column, crs=self.crs)

        bounds = self.bounds()
        geometry_index = self.spatial_index(mode="geometry")
        hex_width = math.sqrt(3.0) * size
        vertical_step = 1.5 * size
        rows: list[Record] = []

        row_index = 0
        center_y = bounds.min_y + size
        while center_y <= bounds.max_y + size:
            offset_x = hex_width / 2.0 if row_index % 2 else 0.0
            column_index = 0
            center_x = bounds.min_x + offset_x
            while center_x <= bounds.max_x + hex_width:
                hex_geometry = _hexagon_polygon(center_x, center_y, size)
                hex_bounds = geometry_bounds(hex_geometry)
                candidate_indexes = geometry_index.query(hex_bounds)
                if include_empty or any(
                    geometry_intersects(hex_geometry, self._rows[index][self.geometry_column])
                    for index in candidate_indexes
                ):
                    rows.append(
                        {
                            grid_id_column: f"hex-{row_index:04d}-{column_index:04d}",
                            "grid_row": row_index,
                            "grid_column": column_index,
                            "hex_size": size,
                            self.geometry_column: hex_geometry,
                        }
                    )
                column_index += 1
                center_x += hex_width
            row_index += 1
            center_y += vertical_step

        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def hotspot_grid(
        self,
        cell_size: float,
        shape: GridShape = "fishnet",
        value_column: str | None = None,
        aggregation: AggregationName = "count",
        include_empty: bool = False,
        include_diagnostics: bool = False,
        hotspot_suffix: str = "hotspot",
    ) -> "GeoPromptFrame":
        if cell_size <= 0:
            raise ValueError("cell_size must be greater than zero")
        if aggregation not in {"count", "sum", "mean", "min", "max"}:
            raise ValueError("aggregation must be 'count', 'sum', 'mean', 'min', or 'max'")
        if value_column is None and aggregation != "count":
            raise ValueError("value_column is required when aggregation is not 'count'")
        if value_column is not None:
            self._require_column(value_column)

        grid = self.fishnet(cell_size, include_empty=include_empty) if shape == "fishnet" else self.hexbin(cell_size, include_empty=include_empty)
        centroid_index = self.spatial_index(mode="centroid", cell_size=cell_size) if self._rows else None
        centroids = self._centroids()
        rows: list[Record] = []

        for cell_row in grid.to_records():
            cell_geometry = cell_row[self.geometry_column]
            cell_bounds = geometry_bounds(cell_geometry)
            candidate_indexes = centroid_index.query(cell_bounds) if centroid_index is not None else []
            matched_indexes = [
                index
                for index in candidate_indexes
                if geometry_within({"type": "Point", "coordinates": centroids[index]}, cell_geometry)
            ]
            matched_rows = [self._rows[index] for index in matched_indexes]

            resolved = dict(cell_row)
            resolved[f"count_{hotspot_suffix}"] = len(matched_rows)
            resolved[f"aggregation_{hotspot_suffix}"] = aggregation
            if include_diagnostics:
                resolved[f"candidate_count_{hotspot_suffix}"] = len(candidate_indexes)
                resolved[f"pruning_ratio_{hotspot_suffix}"] = 1.0 - (len(candidate_indexes) / len(self._rows) if self._rows else 0.0)
            if value_column is None:
                resolved[f"value_{hotspot_suffix}"] = float(len(matched_rows))
            else:
                values = [float(row[value_column]) for row in matched_rows if row.get(value_column) is not None]
                if not values:
                    resolved[f"{value_column}_{aggregation}_{hotspot_suffix}"] = None
                    resolved[f"value_{hotspot_suffix}"] = None
                elif aggregation == "sum":
                    resolved[f"{value_column}_sum_{hotspot_suffix}"] = sum(values)
                    resolved[f"value_{hotspot_suffix}"] = sum(values)
                elif aggregation == "mean":
                    resolved[f"{value_column}_mean_{hotspot_suffix}"] = sum(values) / len(values)
                    resolved[f"value_{hotspot_suffix}"] = sum(values) / len(values)
                elif aggregation == "min":
                    resolved[f"{value_column}_min_{hotspot_suffix}"] = min(values)
                    resolved[f"value_{hotspot_suffix}"] = min(values)
                elif aggregation == "max":
                    resolved[f"{value_column}_max_{hotspot_suffix}"] = max(values)
                    resolved[f"value_{hotspot_suffix}"] = max(values)
                else:
                    resolved[f"{value_column}_count_{hotspot_suffix}"] = len(values)
                    resolved[f"value_{hotspot_suffix}"] = float(len(values))
            rows.append(resolved)

        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def network_build(
        self,
        id_column: str = "site_id",
        edge_id_prefix: str = "edge",
        node_id_prefix: str = "node",
        distance_method: str = "euclidean",
    ) -> "GeoPromptFrame":
        if distance_method not in {"euclidean", "haversine"}:
            raise ValueError("distance_method must be 'euclidean' or 'haversine'")
        if not self._rows:
            return GeoPromptFrame._from_internal_rows([], geometry_column=self.geometry_column, crs=self.crs)
        self._require_column(id_column)
        if any(geometry_type(row[self.geometry_column]) != "LineString" for row in self._rows):
            raise ValueError("network_build requires all geometries to be LineString")

        segment_records: list[dict[str, Any]] = []
        for row_index, row in enumerate(self._rows):
            vertices = list(row[self.geometry_column]["coordinates"])
            for segment_index in range(1, len(vertices)):
                start = _as_coordinate(vertices[segment_index - 1])
                end = _as_coordinate(vertices[segment_index])
                segment_records.append(
                    {
                        "source_row": row,
                        "source_index": row_index,
                        "source_id": str(row[id_column]),
                        "segment_index": segment_index - 1,
                        "start": start,
                        "end": end,
                        "bounds": (
                            min(start[0], end[0]),
                            min(start[1], end[1]),
                            max(start[0], end[0]),
                            max(start[1], end[1]),
                        ),
                    }
                )

        segment_index = SpatialIndex([segment["bounds"] for segment in segment_records])
        cut_points: list[set[Coordinate]] = [
            {segment["start"], segment["end"]}
            for segment in segment_records
        ]
        for index, segment in enumerate(segment_records):
            for other_index in segment_index.query(segment["bounds"]):
                if other_index <= index:
                    continue
                other = segment_records[other_index]
                for point in _segment_intersection_points(segment["start"], segment["end"], other["start"], other["end"]):
                    cut_points[index].add(point)
                    cut_points[other_index].add(point)

        edge_parts: list[dict[str, Any]] = []
        for segment, points in zip(segment_records, cut_points, strict=True):
            ordered_points = sorted(points, key=lambda point: _point_parameter(point, segment["start"], segment["end"]))
            for start_point, end_point in zip(ordered_points, ordered_points[1:]):
                if _same_coordinate(start_point, end_point):
                    continue
                edge_parts.append(
                    {
                        "source_id": segment["source_id"],
                        "source_segment_index": segment["segment_index"],
                        "start": start_point,
                        "end": end_point,
                    }
                )

        unique_nodes = sorted(
            {_coordinate_key(edge["start"]): edge["start"] for edge in edge_parts} | {_coordinate_key(edge["end"]): edge["end"] for edge in edge_parts}
        )
        node_lookup = {
            node_key: f"{node_id_prefix}-{index:05d}"
            for index, node_key in enumerate(unique_nodes)
        }

        rows: list[Record] = []
        for edge_index, edge in enumerate(edge_parts):
            start_key = _coordinate_key(edge["start"])
            end_key = _coordinate_key(edge["end"])
            start_node_id = node_lookup[start_key]
            end_node_id = node_lookup[end_key]
            line_geometry: Geometry = {"type": "LineString", "coordinates": (edge["start"], edge["end"])}
            rows.append(
                {
                    "edge_id": f"{edge_id_prefix}-{edge_index:05d}",
                    "source_id": edge["source_id"],
                    "source_segment_index": edge["source_segment_index"],
                    "from_node_id": start_node_id,
                    "to_node_id": end_node_id,
                    "from_node": edge["start"],
                    "to_node": edge["end"],
                    "edge_length": geometry_length(line_geometry) if distance_method == "euclidean" else coordinate_distance(edge["start"], edge["end"], method=distance_method),
                    self.geometry_column: line_geometry,
                }
            )

        rows.sort(key=lambda row: (str(row["source_id"]), int(row["source_segment_index"]), str(row["from_node_id"]), str(row["to_node_id"])))
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def shortest_path(
        self,
        origin: str | Coordinate,
        destination: str | Coordinate,
        edge_id_column: str = "edge_id",
        from_node_id_column: str = "from_node_id",
        to_node_id_column: str = "to_node_id",
        from_node_column: str = "from_node",
        to_node_column: str = "to_node",
        cost_column: str = "edge_length",
        directed: bool = False,
        path_suffix: str = "path",
    ) -> "GeoPromptFrame":
        for column in (edge_id_column, from_node_id_column, to_node_id_column, from_node_column, to_node_column, cost_column):
            self._require_column(column)
        if not self._rows:
            return GeoPromptFrame._from_internal_rows([], geometry_column=self.geometry_column, crs=self.crs)

        origin_node_id = self._resolve_network_node(origin, from_node_id_column, to_node_id_column, from_node_column, to_node_column)
        destination_node_id = self._resolve_network_node(destination, from_node_id_column, to_node_id_column, from_node_column, to_node_column)

        adjacency: dict[str, list[tuple[str, float, int]]] = {}
        for index, row in enumerate(self._rows):
            from_node_id = str(row[from_node_id_column])
            to_node_id = str(row[to_node_id_column])
            edge_cost = float(row[cost_column])
            adjacency.setdefault(from_node_id, []).append((to_node_id, edge_cost, index))
            if not directed:
                adjacency.setdefault(to_node_id, []).append((from_node_id, edge_cost, index))

        distances: dict[str, float] = {origin_node_id: 0.0}
        previous: dict[str, tuple[str, int]] = {}
        queue: list[tuple[float, str]] = [(0.0, origin_node_id)]
        visited: set[str] = set()

        while queue:
            current_cost, current_node = heappop(queue)
            if current_node in visited:
                continue
            visited.add(current_node)
            if current_node == destination_node_id:
                break
            for next_node, edge_cost, edge_index in adjacency.get(current_node, []):
                path_cost = current_cost + edge_cost
                if path_cost < distances.get(next_node, float("inf")):
                    distances[next_node] = path_cost
                    previous[next_node] = (current_node, edge_index)
                    heappush(queue, (path_cost, next_node))

        if destination_node_id not in distances:
            return GeoPromptFrame._from_internal_rows([], geometry_column=self.geometry_column, crs=self.crs)

        path_edge_indexes: list[int] = []
        node_sequence: list[str] = [destination_node_id]
        current_node = destination_node_id
        while current_node != origin_node_id:
            previous_node, edge_index = previous[current_node]
            path_edge_indexes.append(edge_index)
            node_sequence.append(previous_node)
            current_node = previous_node

        path_edge_indexes.reverse()
        node_sequence.reverse()
        total_cost = distances[destination_node_id]
        rows: list[Record] = []
        for step_index, edge_index in enumerate(path_edge_indexes, start=1):
            resolved = dict(self._rows[edge_index])
            resolved[f"step_{path_suffix}"] = step_index
            resolved[f"origin_node_{path_suffix}"] = origin_node_id
            resolved[f"destination_node_{path_suffix}"] = destination_node_id
            resolved[f"total_cost_{path_suffix}"] = total_cost
            resolved[f"edge_count_{path_suffix}"] = len(path_edge_indexes)
            resolved[f"node_sequence_{path_suffix}"] = list(node_sequence)
            rows.append(resolved)

        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def _resolve_network_node(
        self,
        node: str | Coordinate,
        from_node_id_column: str,
        to_node_id_column: str,
        from_node_column: str,
        to_node_column: str,
    ) -> str:
        if isinstance(node, str):
            node_ids = {str(row[from_node_id_column]) for row in self._rows} | {str(row[to_node_id_column]) for row in self._rows}
            if node not in node_ids:
                raise KeyError(f"network node '{node}' was not found")
            return node

        target_coordinate = _as_coordinate(normalize_geometry(node)["coordinates"])
        node_coordinates: dict[str, Coordinate] = {}
        for row in self._rows:
            node_coordinates[str(row[from_node_id_column])] = _as_coordinate(row[from_node_column])
            node_coordinates[str(row[to_node_id_column])] = _as_coordinate(row[to_node_column])
        return min(
            node_coordinates,
            key=lambda node_id: (
                coordinate_distance(target_coordinate, node_coordinates[node_id], method="euclidean"),
                node_id,
            ),
        )

    def corridor_reach(
        self,
        corridors: "GeoPromptFrame",
        max_distance: float,
        corridor_id_column: str = "site_id",
        aggregations: dict[str, AggregationName] | None = None,
        how: SpatialJoinMode = "left",
        distance_method: str = "euclidean",
        distance_mode: CorridorDistanceMode = "direct",
        score_mode: CorridorScoreMode = "distance",
        weight_column: str | None = None,
        preferred_bearing: float | None = None,
        path_anchor: CorridorPathAnchor = "start",
        scale: float = 1.0,
        power: float = 2.0,
        reach_suffix: str = "reach",
    ) -> "GeoPromptFrame":
        if max_distance < 0:
            raise ValueError("max_distance must be zero or greater")
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if distance_method not in {"euclidean", "haversine"}:
            raise ValueError("distance_method must be 'euclidean' or 'haversine'")
        if distance_mode not in {"direct", "network"}:
            raise ValueError("distance_mode must be 'direct' or 'network'")
        if score_mode not in {"distance", "strength", "alignment", "combined"}:
            raise ValueError("score_mode must be 'distance', 'strength', 'alignment', or 'combined'")
        if path_anchor not in {"start", "end", "nearest"}:
            raise ValueError("path_anchor must be 'start', 'end', or 'nearest'")
        if scale <= 0:
            raise ValueError("scale must be greater than zero")
        if power <= 0:
            raise ValueError("power must be greater than zero")
        if self.crs and corridors.crs and self.crs != corridors.crs:
            raise ValueError("frames must share the same CRS before corridor reach analysis")

        corridors._require_column(corridor_id_column)
        if weight_column is not None:
            corridors._require_column(weight_column)
        corridor_rows = list(corridors._rows)

        rows: list[Record] = []
        for left_row in self._rows:
            left_centroid = geometry_centroid(left_row[self.geometry_column])
            matched_corridors: list[Record] = []
            matched_distances: list[float] = []
            corridor_scores: list[Record] = []
            for corridor_row in corridor_rows:
                corridor_geometry = corridor_row[corridors.geometry_column]
                corridor_vertices = list(corridor_geometry.get("coordinates", []))  # type: ignore[union-attr]
                if not corridor_vertices:
                    continue
                min_dist, along_distance = _point_to_polyline_distance_details(left_centroid, corridor_vertices, method=distance_method)
                corridor_length_value = _polyline_length(corridor_vertices, method=distance_method)
                anchor_distance = _resolve_anchor_distance(along_distance, corridor_length_value, path_anchor)
                if distance_mode == "network":
                    min_dist = min_dist + anchor_distance
                if min_dist <= max_distance:
                    matched_corridors.append(corridor_row)
                    matched_distances.append(min_dist)

                    alignment_score = None
                    if preferred_bearing is not None and len(corridor_vertices) >= 2:
                        alignment_score = (directional_alignment(corridor_vertices[0], corridor_vertices[-1], preferred_bearing) + 1.0) / 2.0
                    weight_value = float(corridor_row[weight_column]) if weight_column is not None else 1.0
                    distance_score = prompt_decay(distance_value=min_dist, scale=scale, power=power)
                    if score_mode == "distance":
                        corridor_score = weight_value * distance_score
                    elif score_mode == "strength":
                        corridor_score = corridor_strength(weight=weight_value, corridor_length=corridor_length_value, distance_value=min_dist, scale=scale, power=power)
                    elif score_mode == "alignment":
                        corridor_score = weight_value * distance_score * (alignment_score if alignment_score is not None else 1.0)
                    else:
                        corridor_score = corridor_strength(weight=weight_value, corridor_length=corridor_length_value, distance_value=min_dist, scale=scale, power=power) * (alignment_score if alignment_score is not None else 1.0)

                    corridor_scores.append(
                        {
                            "corridor_id": str(corridor_row[corridor_id_column]),
                            "distance": min_dist,
                            "along_distance": along_distance,
                            "anchor_distance": anchor_distance,
                            "corridor_length": corridor_length_value,
                            "alignment_score": alignment_score,
                            "score": corridor_score,
                        }
                    )

            if not matched_corridors and how == "inner":
                continue

            corridor_scores.sort(key=lambda item: (-float(item["score"]), float(item["distance"]), str(item["corridor_id"])))

            resolved_row = dict(left_row)
            resolved_row[f"{corridor_id_column}s_{reach_suffix}"] = [
                str(row[corridor_id_column]) for row in matched_corridors
            ]
            resolved_row[f"count_{reach_suffix}"] = len(matched_corridors)
            resolved_row[f"distance_min_{reach_suffix}"] = min(matched_distances) if matched_distances else None
            resolved_row[f"distance_max_{reach_suffix}"] = max(matched_distances) if matched_distances else None
            resolved_row[f"distance_mean_{reach_suffix}"] = (
                sum(matched_distances) / len(matched_distances) if matched_distances else None
            )
            resolved_row[f"distance_limit_{reach_suffix}"] = max_distance
            resolved_row[f"distance_method_{reach_suffix}"] = distance_method
            resolved_row[f"distance_mode_{reach_suffix}"] = distance_mode
            resolved_row[f"path_anchor_{reach_suffix}"] = path_anchor
            resolved_row[f"score_mode_{reach_suffix}"] = score_mode
            resolved_row[f"corridor_scores_{reach_suffix}"] = corridor_scores
            resolved_row[f"best_corridor_{reach_suffix}"] = corridor_scores[0]["corridor_id"] if corridor_scores else None
            resolved_row[f"best_score_{reach_suffix}"] = corridor_scores[0]["score"] if corridor_scores else None

            total_corridor_length = sum(
                _polyline_length(list(row[corridors.geometry_column].get("coordinates", [])), method=distance_method) if row[corridors.geometry_column].get("type") == "LineString" else geometry_length(row[corridors.geometry_column])
                for row in matched_corridors
            )
            resolved_row[f"corridor_length_total_{reach_suffix}"] = total_corridor_length

            resolved_row.update(
                self._aggregate_rows(matched_corridors, aggregations=aggregations, suffix=reach_suffix)
            )
            rows.append(resolved_row)

        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs or corridors.crs)

    def zone_fit_score(
        self,
        zones: "GeoPromptFrame",
        zone_id_column: str = "region_id",
        weight_column: str | None = None,
        max_distance: float | None = None,
        distance_method: str = "euclidean",
        score_weights: dict[str, float] | None = None,
        preferred_bearing: float | None = None,
        group_by: str | None = None,
        group_aggregation: ZoneGroupAggregation = "max",
        top_n: int | None = None,
        score_callback: Any | None = None,
        score_suffix: str = "fit",
    ) -> "GeoPromptFrame":
        if max_distance is not None and max_distance < 0:
            raise ValueError("max_distance must be zero or greater")
        if self.crs and zones.crs and self.crs != zones.crs:
            raise ValueError("frames must share the same CRS before zone fit scoring")
        if group_aggregation not in {"max", "mean", "sum"}:
            raise ValueError("group_aggregation must be 'max', 'mean', or 'sum'")
        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be greater than zero")

        zones._require_column(zone_id_column)
        if group_by is not None:
            zones._require_column(group_by)
        component_weights = _resolve_zone_fit_weights(score_weights)
        zone_rows = list(zones._rows)
        zone_centroids = zones._centroids()
        zone_areas = [geometry_area(row[zones.geometry_column]) for row in zone_rows]

        rows: list[Record] = []
        for left_row in self._rows:
            left_centroid = geometry_centroid(left_row[self.geometry_column])
            left_area = geometry_area(left_row[self.geometry_column])

            best_score = -1.0
            best_zone_id: str | None = None
            zone_scores: list[Record] = []

            for zone_row, zone_centroid, zone_area in zip(zone_rows, zone_centroids, zone_areas, strict=True):
                dist = coordinate_distance(left_centroid, zone_centroid, method=distance_method)
                if max_distance is not None and dist > max_distance:
                    continue

                containment = 1.0 if geometry_within(left_row[self.geometry_column], zone_row[zones.geometry_column]) else 0.0
                overlap = 1.0 if geometry_intersects(left_row[self.geometry_column], zone_row[zones.geometry_column]) else 0.0
                size_ratio = area_similarity(
                    origin_area=left_area,
                    destination_area=zone_area,
                    distance_value=dist,
                    scale=1.0,
                    power=1.0,
                )
                access_score = prompt_decay(distance_value=dist, scale=max_distance or 1.0, power=1.0)
                alignment_score = None
                if preferred_bearing is not None:
                    alignment_score = (directional_alignment(left_centroid, zone_centroid, preferred_bearing) + 1.0) / 2.0

                component_scores = {
                    "containment": containment,
                    "overlap": overlap,
                    "size": size_ratio,
                    "access": access_score,
                    "alignment": alignment_score if alignment_score is not None else 0.0,
                }

                weight = float(left_row.get(weight_column, 1.0)) if weight_column else 1.0
                weighted_total = sum(component_scores[name] * component_weights[name] for name in component_weights)
                total_weight = sum(component_weights.values())
                score = weight * (weighted_total / total_weight)
                if score_callback is not None:
                    score = float(score_callback(left_row, zone_row, dict(component_scores), score))

                zone_scores.append({
                    "zone_id": str(zone_row[zone_id_column]),
                    "group": str(zone_row[group_by]) if group_by is not None else None,
                    "score": score,
                    "distance": dist,
                    "containment": containment,
                    "overlap": overlap,
                    "size_ratio": size_ratio,
                    "access_score": access_score,
                    "alignment_score": alignment_score,
                })
                if score > best_score:
                    best_score = score
                    best_zone_id = str(zone_row[zone_id_column])

            zone_scores.sort(key=lambda item: (-float(item["score"]), float(item["distance"]), str(item["zone_id"])))
            if top_n is not None:
                zone_scores = zone_scores[:top_n]

            resolved_row = dict(left_row)
            resolved_row[f"best_zone_{score_suffix}"] = best_zone_id
            resolved_row[f"best_score_{score_suffix}"] = best_score if best_zone_id is not None else None
            resolved_row[f"zone_count_{score_suffix}"] = len(zone_scores)
            resolved_row[f"score_weights_{score_suffix}"] = dict(component_weights)
            resolved_row[f"zone_scores_{score_suffix}"] = zone_scores
            if group_by is not None:
                grouped_scores: dict[str, dict[str, Any]] = {}
                for zone_score in zone_scores:
                    group_value = str(zone_score["group"])
                    bucket = grouped_scores.setdefault(group_value, {"group": group_value, "zone_ids": [], "scores": []})
                    bucket["zone_ids"].append(zone_score["zone_id"])
                    bucket["scores"].append(zone_score["score"])
                group_rankings = []
                for group_score in grouped_scores.values():
                    if group_aggregation == "max":
                        score_value = max(group_score["scores"])
                    elif group_aggregation == "mean":
                        score_value = sum(group_score["scores"]) / len(group_score["scores"])
                    else:
                        score_value = sum(group_score["scores"])
                    group_rankings.append(
                        {
                            "group": group_score["group"],
                            "score": score_value,
                            "zone_ids": group_score["zone_ids"],
                            "zone_count": len(group_score["zone_ids"]),
                        }
                    )
                group_rankings.sort(key=lambda item: (-float(item["score"]), str(item["group"])))
                resolved_row[f"group_scores_{score_suffix}"] = group_rankings
                resolved_row[f"best_group_{score_suffix}"] = group_rankings[0]["group"] if group_rankings else None
            rows.append(resolved_row)

        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs or zones.crs)

    def centroid_cluster(
        self,
        k: int,
        id_column: str = "site_id",
        distance_method: str = "euclidean",
        max_iterations: int = 50,
    ) -> "GeoPromptFrame":
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if k > len(self._rows):
            raise ValueError("k must not exceed frame length")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero")

        centroids_list = self._centroids()
        sorted_indices = sorted(range(len(self._rows)), key=lambda index: _cluster_sort_key(self._rows[index], index, id_column))
        seed_positions = [int(step * len(sorted_indices) / k) for step in range(k)]
        seed_indices = [sorted_indices[min(position, len(sorted_indices) - 1)] for position in seed_positions]
        cluster_centers = [centroids_list[i] for i in seed_indices]

        assignments = [0] * len(centroids_list)

        for _ in range(max_iterations):
            changed = False
            for point_index, point in enumerate(centroids_list):
                best_cluster = assignments[point_index]
                best_dist = float("inf")
                for cluster_index, center in enumerate(cluster_centers):
                    dist = coordinate_distance(point, center, method=distance_method)
                    if dist < best_dist - 1e-12 or (abs(dist - best_dist) <= 1e-12 and cluster_index < best_cluster):
                        best_dist = dist
                        best_cluster = cluster_index
                if assignments[point_index] != best_cluster:
                    assignments[point_index] = best_cluster
                    changed = True

            if not changed:
                break

            new_centers: list[Coordinate] = []
            for cluster_index in range(k):
                members = [centroids_list[i] for i, a in enumerate(assignments) if a == cluster_index]
                if members:
                    cx = sum(p[0] for p in members) / len(members)
                    cy = sum(p[1] for p in members) / len(members)
                    new_centers.append((cx, cy))
                else:
                    new_centers.append(cluster_centers[cluster_index])
            cluster_centers = new_centers

        cluster_member_indices = {
            cluster_index: [index for index, assignment in enumerate(assignments) if assignment == cluster_index]
            for cluster_index in range(k)
        }
        point_distances = [
            coordinate_distance(centroid_point, cluster_centers[assignments[index]], method=distance_method)
            for index, centroid_point in enumerate(centroids_list)
        ]
        cluster_sse = {
            cluster_index: sum(point_distances[index] ** 2 for index in members)
            for cluster_index, members in cluster_member_indices.items()
        }
        cluster_mean_distance = {
            cluster_index: (sum(point_distances[index] for index in members) / len(members)) if members else 0.0
            for cluster_index, members in cluster_member_indices.items()
        }
        silhouette_scores = _cluster_silhouette_scores(centroids_list, assignments, distance_method=distance_method)
        overall_sse = sum(cluster_sse.values())
        overall_silhouette = sum(silhouette_scores) / len(silhouette_scores) if silhouette_scores else 0.0

        rows: list[Record] = []
        for index, (row, cluster_id, centroid_point) in enumerate(zip(self._rows, assignments, centroids_list, strict=True)):
            resolved = dict(row)
            resolved["cluster_id"] = cluster_id
            resolved["cluster_center"] = cluster_centers[cluster_id]
            resolved["cluster_distance"] = point_distances[index]
            resolved["cluster_size"] = len(cluster_member_indices[cluster_id])
            resolved["cluster_mean_distance"] = cluster_mean_distance[cluster_id]
            resolved["cluster_sse"] = cluster_sse[cluster_id]
            resolved["cluster_silhouette"] = silhouette_scores[index]
            resolved["cluster_sse_total"] = overall_sse
            resolved["cluster_silhouette_mean"] = overall_silhouette
            rows.append(resolved)

        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def envelopes(self) -> "GeoPromptFrame":
        rows: list[Record] = []
        for row in self._rows:
            new_row = dict(row)
            new_row[self.geometry_column] = geometry_envelope(row[self.geometry_column])
            rows.append(new_row)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def convex_hulls(self) -> "GeoPromptFrame":
        rows: list[Record] = []
        for row in self._rows:
            new_row = dict(row)
            new_row[self.geometry_column] = geometry_convex_hull(row[self.geometry_column])
            rows.append(new_row)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def gravity_table(
        self,
        origin_weight: str,
        destination_weight: str,
        id_column: str = "site_id",
        friction: float = 2.0,
        distance_method: str = "euclidean",
    ) -> list[Record]:
        self._require_column(origin_weight)
        self._require_column(destination_weight)
        self._require_column(id_column)

        interactions: list[Record] = []
        for origin in self._rows:
            for destination in self._rows:
                if origin is destination:
                    continue
                distance_value = geometry_distance(
                    origin[self.geometry_column],
                    destination[self.geometry_column],
                    method=distance_method,
                )
                interactions.append({
                    "origin": origin[id_column],
                    "destination": destination[id_column],
                    "distance": distance_value,
                    "gravity": gravity_model(
                        origin_weight=float(origin[origin_weight]),
                        destination_weight=float(destination[destination_weight]),
                        distance_value=distance_value,
                        friction=friction,
                    ),
                    "distance_method": distance_method,
                })
        return interactions

    def accessibility_scores(
        self,
        targets: "GeoPromptFrame",
        weight_column: str,
        friction: float = 2.0,
        id_column: str = "site_id",
        distance_method: str = "euclidean",
    ) -> list[float]:
        targets._require_column(weight_column)
        target_centroids = targets._centroids()
        target_weights = [float(row[weight_column]) for row in targets._rows]
        origin_centroids = self._centroids()

        scores: list[float] = []
        for origin_centroid in origin_centroids:
            distances = [
                coordinate_distance(origin_centroid, tc, method=distance_method)
                for tc in target_centroids
            ]
            scores.append(accessibility_index(target_weights, distances, friction=friction))
        return scores

    def cluster_diagnostics(
        self,
        k_values: Sequence[int],
        id_column: str = "site_id",
        distance_method: str = "euclidean",
        max_iterations: int = 50,
    ) -> list[Record]:
        if not k_values:
            raise ValueError("k_values must contain at least one cluster count")

        unique_k_values = sorted(set(int(value) for value in k_values))
        diagnostics: list[Record] = []
        previous_sse: float | None = None
        for k_value in unique_k_values:
            clustered = self.centroid_cluster(
                k=k_value,
                id_column=id_column,
                distance_method=distance_method,
                max_iterations=max_iterations,
            )
            records = clustered.to_records()
            cluster_sizes = sorted({int(record["cluster_size"]) for record in records}, reverse=True)
            total_sse = float(records[0]["cluster_sse_total"]) if records else 0.0
            silhouette_mean = float(records[0]["cluster_silhouette_mean"]) if records else 0.0
            diagnostics.append(
                {
                    "k": k_value,
                    "cluster_count": len({int(record["cluster_id"]) for record in records}),
                    "cluster_sizes": cluster_sizes,
                    "sse_total": total_sse,
                    "sse_improvement": (previous_sse - total_sse) if previous_sse is not None else None,
                    "silhouette_mean": silhouette_mean,
                }
            )
            previous_sse = total_sse

        recommended_by_silhouette = max(diagnostics, key=lambda item: (float(item["silhouette_mean"]), -int(item["k"])))
        recommended_by_sse = min(diagnostics, key=lambda item: (float(item["sse_total"]), int(item["k"])))
        for item in diagnostics:
            item["recommended_silhouette"] = int(item["k"]) == int(recommended_by_silhouette["k"])
            item["recommended_sse"] = int(item["k"]) == int(recommended_by_sse["k"])
        return diagnostics

    def recommend_cluster_count(
        self,
        k_values: Sequence[int],
        metric: ClusterRecommendMetric = "silhouette",
        id_column: str = "site_id",
        distance_method: str = "euclidean",
        max_iterations: int = 50,
    ) -> Record:
        diagnostics = self.cluster_diagnostics(
            k_values=k_values,
            id_column=id_column,
            distance_method=distance_method,
            max_iterations=max_iterations,
        )
        if metric == "silhouette":
            return next(item for item in diagnostics if item["recommended_silhouette"])
        if metric == "sse":
            return next(item for item in diagnostics if item["recommended_sse"])
        raise ValueError("metric must be 'silhouette' or 'sse'")

    def summarize_clusters(
        self,
        cluster_column: str = "cluster_id",
        group_by: str | None = None,
        aggregations: dict[str, AggregationName] | None = None,
    ) -> "GeoPromptFrame":
        self._require_column(cluster_column)
        if group_by is not None:
            self._require_column(group_by)

        grouped_rows: dict[int, list[Record]] = {}
        for row in self._rows:
            grouped_rows.setdefault(int(row[cluster_column]), []).append(row)

        rows: list[Record] = []
        for cluster_id, cluster_rows in grouped_rows.items():
            centroid_x = sum(float(row["cluster_center"][0]) for row in cluster_rows if "cluster_center" in row) / len(cluster_rows)
            centroid_y = sum(float(row["cluster_center"][1]) for row in cluster_rows if "cluster_center" in row) / len(cluster_rows)
            summary_row: Record = {
                cluster_column: cluster_id,
                "cluster_member_count": len(cluster_rows),
                "cluster_site_ids": [str(row.get("site_id", row.get("region_id", ""))) for row in cluster_rows],
                "cluster_center_summary": (centroid_x, centroid_y),
                "cluster_sse_total": sum(float(row.get("cluster_sse", 0.0)) for row in cluster_rows),
                "cluster_mean_distance_summary": sum(float(row.get("cluster_distance", 0.0)) for row in cluster_rows) / len(cluster_rows),
                self.geometry_column: {"type": "Point", "coordinates": (centroid_x, centroid_y)},
            }
            if group_by is not None:
                group_counts: dict[str, int] = {}
                for row in cluster_rows:
                    group_counts[str(row[group_by])] = group_counts.get(str(row[group_by]), 0) + 1
                summary_row[f"{group_by}_counts"] = group_counts
                summary_row[f"dominant_{group_by}"] = max(group_counts.items(), key=lambda item: (int(item[1]), str(item[0])))[0] if group_counts else None
            summary_row.update(self._aggregate_rows(cluster_rows, aggregations=aggregations, suffix="cluster"))
            rows.append(summary_row)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def overlay_group_comparison(
        self,
        other: "GeoPromptFrame",
        group_by: str,
        right_id_column: str = "region_id",
        normalize_by: OverlayNormalizeMode = "both",
        comparison_suffix: str = "overlay_compare",
    ) -> "GeoPromptFrame":
        summary = self.overlay_summary(
            other,
            right_id_column=right_id_column,
            group_by=group_by,
            normalize_by=normalize_by,
            how="left",
            summary_suffix=comparison_suffix,
        )

        rows: list[Record] = []
        for row in summary.to_records():
            groups = list(row.get(f"groups_{comparison_suffix}", []))
            groups.sort(key=lambda item: (-float(item["area_overlap"]), -float(item["length_overlap"]), str(item["group"])))
            top_group = groups[0] if len(groups) >= 1 else None
            second_group = groups[1] if len(groups) >= 2 else None
            resolved = dict(row)
            resolved[f"top_group_{comparison_suffix}"] = top_group["group"] if top_group else None
            resolved[f"runner_up_group_{comparison_suffix}"] = second_group["group"] if second_group else None
            resolved[f"area_gap_{comparison_suffix}"] = (top_group["area_overlap"] - second_group["area_overlap"]) if top_group and second_group else None
            resolved[f"length_gap_{comparison_suffix}"] = (top_group["length_overlap"] - second_group["length_overlap"]) if top_group and second_group else None
            resolved[f"comparison_strength_{comparison_suffix}"] = (
                (top_group["area_overlap"] + top_group["length_overlap"]) - (second_group["area_overlap"] + second_group["length_overlap"])
            ) if top_group and second_group else None
            rows.append(resolved)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs or other.crs)

    def corridor_diagnostics(
        self,
        corridors: "GeoPromptFrame",
        max_distance: float,
        corridor_id_column: str = "site_id",
        distance_method: str = "euclidean",
        distance_mode: CorridorDistanceMode = "network",
        score_mode: CorridorScoreMode = "combined",
        path_anchor: CorridorPathAnchor = "start",
        weight_column: str | None = None,
        preferred_bearing: float | None = None,
        scale: float = 1.0,
        power: float = 2.0,
    ) -> "GeoPromptFrame":
        reach = self.corridor_reach(
            corridors,
            max_distance=max_distance,
            corridor_id_column=corridor_id_column,
            distance_method=distance_method,
            distance_mode=distance_mode,
            score_mode=score_mode,
            weight_column=weight_column,
            preferred_bearing=preferred_bearing,
            path_anchor=path_anchor,
            scale=scale,
            power=power,
            how="left",
            reach_suffix="diagnostics",
        )
        corridor_scores_by_id: dict[str, list[Record]] = {}
        for row in reach.to_records():
            for item in row.get("corridor_scores_diagnostics", []):
                corridor_scores_by_id.setdefault(str(item["corridor_id"]), []).append(item)

        rows: list[Record] = []
        for corridor_row in corridors.to_records():
            corridor_id = str(corridor_row[corridor_id_column])
            items = corridor_scores_by_id.get(corridor_id, [])
            diagnostic_row = dict(corridor_row)
            diagnostic_row["served_feature_count"] = len(items)
            diagnostic_row["best_match_count"] = sum(1 for row in reach.to_records() if row.get("best_corridor_diagnostics") == corridor_id)
            diagnostic_row["score_sum"] = sum(float(item["score"]) for item in items)
            diagnostic_row["score_mean"] = (sum(float(item["score"]) for item in items) / len(items)) if items else None
            diagnostic_row["distance_mean"] = (sum(float(item["distance"]) for item in items) / len(items)) if items else None
            diagnostic_row["anchor_distance_mean"] = (sum(float(item["anchor_distance"]) for item in items) / len(items)) if items else None
            diagnostic_row["path_anchor"] = path_anchor
            diagnostic_row["distance_mode"] = distance_mode
            diagnostic_row["score_mode"] = score_mode
            rows.append(diagnostic_row)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=corridors.geometry_column, crs=self.crs or corridors.crs)


def _resolve_zone_fit_weights(score_weights: dict[str, float] | None) -> dict[str, float]:
    default_weights = {
        "containment": 0.4,
        "overlap": 0.3,
        "size": 0.3,
        "access": 0.0,
        "alignment": 0.0,
    }
    if score_weights is None:
        return default_weights

    resolved = dict(default_weights)
    for name, value in score_weights.items():
        if name not in resolved:
            raise ValueError(f"unsupported zone fit weight: {name}")
        if value < 0:
            raise ValueError("zone fit weights must be zero or greater")
        resolved[name] = float(value)

    if sum(resolved.values()) <= 0:
        raise ValueError("at least one zone fit weight must be greater than zero")
    return resolved


def _cluster_sort_key(row: Record, index: int, id_column: str) -> tuple[str, int]:
    if id_column in row:
        return (str(row[id_column]), index)
    return (str(index), index)


def _cluster_silhouette_scores(
    centroids: Sequence[Coordinate],
    assignments: Sequence[int],
    distance_method: str,
) -> list[float]:
    unique_clusters = sorted(set(assignments))
    if len(unique_clusters) <= 1:
        return [0.0 for _ in centroids]

    members_by_cluster = {
        cluster_id: [index for index, assignment in enumerate(assignments) if assignment == cluster_id]
        for cluster_id in unique_clusters
    }
    scores: list[float] = []
    for index, point in enumerate(centroids):
        own_cluster = assignments[index]
        own_members = [member for member in members_by_cluster[own_cluster] if member != index]
        if own_members:
            intra_distance = sum(
                coordinate_distance(point, centroids[member], method=distance_method)
                for member in own_members
            ) / len(own_members)
        else:
            intra_distance = 0.0

        nearest_other_distance = min(
            sum(coordinate_distance(point, centroids[member], method=distance_method) for member in members_by_cluster[cluster_id]) / len(members_by_cluster[cluster_id])
            for cluster_id in unique_clusters
            if cluster_id != own_cluster and members_by_cluster[cluster_id]
        )
        denominator = max(intra_distance, nearest_other_distance)
        if denominator == 0.0:
            scores.append(0.0)
        else:
            scores.append((nearest_other_distance - intra_distance) / denominator)
    return scores


def _polyline_length(vertices: Sequence[Coordinate], method: str = "euclidean") -> float:
    if len(vertices) < 2:
        return 0.0
    return sum(coordinate_distance(vertices[index - 1], vertices[index], method=method) for index in range(1, len(vertices)))


def _point_to_polyline_distance_details(point: Coordinate, vertices: Sequence[Coordinate], method: str = "euclidean") -> tuple[float, float]:
    if not vertices:
        return (float("inf"), float("inf"))
    if len(vertices) == 1:
        return (coordinate_distance(point, vertices[0], method=method), 0.0)

    if method == "haversine":
        projected_point, projected_vertices = _project_to_local_tangent_plane(point, vertices)
        return _point_to_polyline_distance_details(projected_point, projected_vertices, method="euclidean")

    best_offset = float("inf")
    best_along = float("inf")
    cumulative_length = 0.0
    for index in range(1, len(vertices)):
        segment_start = vertices[index - 1]
        segment_end = vertices[index]
        offset_distance, projection, t_value = _point_to_segment_projection(point, segment_start, segment_end)
        segment_length = coordinate_distance(segment_start, segment_end, method="euclidean")
        along_distance = cumulative_length + (segment_length * t_value)
        if offset_distance < best_offset:
            best_offset = offset_distance
            best_along = along_distance
        cumulative_length += segment_length
    return (best_offset, best_along)


def _point_to_segment_projection(point: Coordinate, seg_start: Coordinate, seg_end: Coordinate) -> tuple[float, Coordinate, float]:
    sx, sy = seg_start
    ex, ey = seg_end
    px, py = point
    dx, dy = ex - sx, ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return (coordinate_distance(point, seg_start, method="euclidean"), seg_start, 0.0)
    t_value = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    projection = (sx + (t_value * dx), sy + (t_value * dy))
    return (coordinate_distance(point, projection, method="euclidean"), projection, t_value)


def _project_to_local_tangent_plane(point: Coordinate, vertices: Sequence[Coordinate]) -> tuple[Coordinate, list[Coordinate]]:
    all_coordinates = [point, *vertices]
    reference_lat = math.radians(sum(coordinate[1] for coordinate in all_coordinates) / len(all_coordinates))

    def project(coordinate: Coordinate) -> Coordinate:
        lon_radians = math.radians(coordinate[0])
        lat_radians = math.radians(coordinate[1])
        x_value = 6371.0088 * lon_radians * math.cos(reference_lat)
        y_value = 6371.0088 * lat_radians
        return (x_value, y_value)

    return (project(point), [project(vertex) for vertex in vertices])


def _resolve_anchor_distance(along_distance: float, total_length: float, path_anchor: CorridorPathAnchor) -> float:
    if path_anchor == "start":
        return along_distance
    if path_anchor == "end":
        return max(0.0, total_length - along_distance)
    return 0.0


def _rectangle_polygon(min_x: float, min_y: float, max_x: float, max_y: float) -> Geometry:
    return {
        "type": "Polygon",
        "coordinates": (
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y),
        ),
    }


def _hexagon_polygon(center_x: float, center_y: float, size: float) -> Geometry:
    vertices = []
    for angle_degrees in (90, 150, 210, 270, 330, 30):
        angle_radians = math.radians(angle_degrees)
        vertices.append((center_x + (size * math.cos(angle_radians)), center_y + (size * math.sin(angle_radians))))
    vertices.append(vertices[0])
    return {"type": "Polygon", "coordinates": tuple(vertices)}


def _as_coordinate(value: Any) -> Coordinate:
    return (float(value[0]), float(value[1]))


def _coordinate_key(point: Coordinate) -> str:
    return f"{point[0]:.12f},{point[1]:.12f}"


def _same_coordinate(left: Coordinate, right: Coordinate, tolerance: float = 1e-9) -> bool:
    return abs(left[0] - right[0]) <= tolerance and abs(left[1] - right[1]) <= tolerance


def _point_parameter(point: Coordinate, start: Coordinate, end: Coordinate) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return 0.0
    return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq


def _segment_intersection_points(first_start: Coordinate, first_end: Coordinate, second_start: Coordinate, second_end: Coordinate) -> list[Coordinate]:
    points: dict[str, Coordinate] = {}
    denominator = ((first_start[0] - first_end[0]) * (second_start[1] - second_end[1])) - ((first_start[1] - first_end[1]) * (second_start[0] - second_end[0]))
    if abs(denominator) <= 1e-12:
        for point in (first_start, first_end, second_start, second_end):
            if _point_on_segment(point, first_start, first_end) and _point_on_segment(point, second_start, second_end):
                points[_coordinate_key(point)] = point
        return list(points.values())

    first_det = (first_start[0] * first_end[1]) - (first_start[1] * first_end[0])
    second_det = (second_start[0] * second_end[1]) - (second_start[1] * second_end[0])
    x_value = ((first_det * (second_start[0] - second_end[0])) - ((first_start[0] - first_end[0]) * second_det)) / denominator
    y_value = ((first_det * (second_start[1] - second_end[1])) - ((first_start[1] - first_end[1]) * second_det)) / denominator
    intersection = (float(x_value), float(y_value))
    if _point_on_segment(intersection, first_start, first_end) and _point_on_segment(intersection, second_start, second_end):
        points[_coordinate_key(intersection)] = intersection
    return list(points.values())


def _point_on_segment(point: Coordinate, start: Coordinate, end: Coordinate, tolerance: float = 1e-9) -> bool:
    cross = ((point[1] - start[1]) * (end[0] - start[0])) - ((point[0] - start[0]) * (end[1] - start[1]))
    if abs(cross) > tolerance:
        return False
    return (
        min(start[0], end[0]) - tolerance <= point[0] <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance <= point[1] <= max(start[1], end[1]) + tolerance
    )


def _point_to_segment_distance(point: Coordinate, seg_start: Any, seg_end: Any, method: str = "euclidean") -> float:
    sx, sy = float(seg_start[0]), float(seg_start[1])
    ex, ey = float(seg_end[0]), float(seg_end[1])
    px, py = point
    if method == "haversine":
        return _point_to_segment_distance_haversine((px, py), (sx, sy), (ex, ey))

    dx, dy = ex - sx, ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return coordinate_distance(point, (sx, sy), method=method)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    proj_x = sx + t * dx
    proj_y = sy + t * dy
    return coordinate_distance(point, (proj_x, proj_y), method=method)


def _point_to_segment_distance_haversine(point: Coordinate, seg_start: Coordinate, seg_end: Coordinate) -> float:
    reference_lat = math.radians((point[1] + seg_start[1] + seg_end[1]) / 3.0)

    def project(coordinate: Coordinate) -> Coordinate:
        lon_radians = math.radians(coordinate[0])
        lat_radians = math.radians(coordinate[1])
        x_value = 6371.0088 * lon_radians * math.cos(reference_lat)
        y_value = 6371.0088 * lat_radians
        return (x_value, y_value)

    return _point_to_segment_distance(project(point), project(seg_start), project(seg_end), method="euclidean")


__all__ = ["Bounds", "GeoPromptFrame"]