from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from .equations import area_similarity, corridor_strength, directional_alignment, prompt_influence, prompt_interaction
from .geometry import Geometry, geometry_area, geometry_bounds, geometry_centroid, geometry_contains, geometry_distance, geometry_intersects, geometry_intersects_bounds, geometry_length, geometry_type, geometry_within, geometry_within_bounds, normalize_geometry, transform_geometry
from .overlay import clip_geometries, dissolve_geometries, overlay_intersections


Record = dict[str, Any]
BoundsQueryMode = Literal["intersects", "within", "centroid"]
SpatialJoinPredicate = Literal["intersects", "within", "contains"]
SpatialJoinMode = Literal["inner", "left"]
AggregationName = Literal["sum", "mean", "min", "max", "first", "count"]


Coordinate = tuple[float, float]


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

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

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

        nearest: list[Record] = []
        for origin in self._rows:
            candidates = [
                {
                    "origin": origin[id_column],
                    "neighbor": destination[id_column],
                    "distance": geometry_distance(origin[self.geometry_column], destination[self.geometry_column], method=distance_method),
                    "origin_geometry_type": geometry_type(origin[self.geometry_column]),
                    "neighbor_geometry_type": geometry_type(destination[self.geometry_column]),
                }
                for destination in self._rows
                if destination is not origin
            ]
            candidates.sort(key=lambda item: (float(item["distance"]), str(item["neighbor"])))
            for rank, item in enumerate(candidates[:k], start=1):
                nearest.append({**item, "rank": rank, "distance_method": distance_method})
        return nearest

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
        for row in self._rows:
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

        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs)

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
    ) -> "GeoPromptFrame":
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before spatial joins")

        right_columns = [column for column in other.columns if column != other.geometry_column]
        joined_rows: list[Record] = []

        def matches(left_geometry: Geometry, right_geometry: Geometry) -> bool:
            if predicate == "intersects":
                return geometry_intersects(left_geometry, right_geometry)
            if predicate == "within":
                return geometry_within(left_geometry, right_geometry)
            if predicate == "contains":
                return geometry_contains(left_geometry, right_geometry)
            raise ValueError(f"unsupported spatial join predicate: {predicate}")

        for left_row in self._rows:
            row_matches = []
            for right_row in other._rows:
                if matches(left_row[self.geometry_column], right_row[other.geometry_column]):
                    row_matches.append(right_row)

            if not row_matches and how == "left":
                merged_row = dict(left_row)
                for column in right_columns:
                    target_name = column if column not in merged_row else f"{column}_{rsuffix}"
                    merged_row[target_name] = None
                merged_row[f"join_predicate_{rsuffix}"] = predicate
                joined_rows.append(merged_row)

            for right_row in row_matches:
                merged_row = dict(left_row)
                for column in right_columns:
                    target_name = column if column not in merged_row else f"{column}_{rsuffix}"
                    merged_row[target_name] = right_row[column]
                merged_row[f"{other.geometry_column}_{rsuffix}"] = right_row[other.geometry_column]
                merged_row[f"join_predicate_{rsuffix}"] = predicate
                joined_rows.append(merged_row)

        return GeoPromptFrame(rows=joined_rows, geometry_column=self.geometry_column, crs=self.crs or other.crs)

    def clip(self, mask: "GeoPromptFrame") -> "GeoPromptFrame":
        if self.crs and mask.crs and self.crs != mask.crs:
            raise ValueError("frames must share the same CRS before clip operations")

        clipped_groups = clip_geometries(
            [row[self.geometry_column] for row in self._rows],
            [row[mask.geometry_column] for row in mask],
        )
        rows: list[Record] = []
        for row, clipped_geometries in zip(self._rows, clipped_groups, strict=True):
            for clipped_geometry in clipped_geometries:
                clipped_row = dict(row)
                clipped_row[self.geometry_column] = clipped_geometry
                rows.append(clipped_row)
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs or mask.crs)

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
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs or other.crs)

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


__all__ = ["Bounds", "GeoPromptFrame"]