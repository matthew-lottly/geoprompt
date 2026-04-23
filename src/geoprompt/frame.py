"""Core spatial frame object and operations for geographic feature analysis.

GeoPromptFrame provides a dataclass-like geospatial table structure supporting
points, lines, and polygons. Built-in operations include spatial joins, network
analysis, distance-based queries, and custom spatial equations without external
dependencies (geometry engines like Shapely/GeoPandas are opt-in via methods).
"""
from __future__ import annotations

import ast
import importlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from heapq import nlargest, nsmallest
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

from .equations import DistanceMethod, area_similarity, coordinate_distance, corridor_strength, directional_alignment, prompt_influence, prompt_interaction
from .geometry import Geometry, geometry_area, geometry_bounds, geometry_centroid, geometry_contains, geometry_distance, geometry_intersects, geometry_intersects_bounds, geometry_length, geometry_type, geometry_within, geometry_within_bounds, normalize_geometry, repair_geometry, validate_geometry
from .overlay import buffer_geometries, clip_geometries, dissolve_geometries, overlay_intersections
from .safe_expression import ExpressionExecutionError, ExpressionValidationError, evaluate_safe_expression
from .table import PromptTable
from .tools import batch_accessibility_scores, vectorized_gravity_interaction, vectorized_service_probability


Record = dict[str, Any]
BoundsQueryMode = Literal["intersects", "within", "centroid"]
SpatialJoinPredicate = Literal["intersects", "within", "contains"]
SpatialJoinMode = Literal["inner", "left"]
AggregationName = Literal["sum", "mean", "min", "max", "first", "count"]
FillMethod = Literal["ffill", "bfill"]


Coordinate = tuple[float, float]


def _normalize_crs(crs: object | None) -> str | None:
    if crs is None:
        return None
    try:
        pyproj = importlib.import_module("pyproj")
        resolved = pyproj.CRS.from_user_input(crs)
        authority = resolved.to_authority()
        if authority is not None:
            return f"{authority[0]}:{authority[1]}"
        return str(resolved.to_string())
    except (ImportError, ValueError, AttributeError):
        text = str(crs).strip()
        if not text:
            return None
        if text.isdigit():
            return f"EPSG:{text}"
        if text.lower().startswith("epsg:"):
            return f"EPSG:{text.split(':', 1)[1].strip()}"
        return text.upper()


def _row_sort_key(row: Record) -> str:
    return str(row.get("site_id", row.get("region_id", "")))


def _cast_frame_value(value: Any, dtype: str) -> Any:
    if value is None:
        return None

    normalized = dtype.lower()
    if normalized == "int":
        return int(value)
    if normalized == "float":
        return float(value)
    if normalized == "str":
        return str(value)
    if normalized in {"bool", "boolean"}:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "t", "yes", "y", "1", "on"}:
            return True
        if text in {"false", "f", "no", "n", "0", "off", ""}:
            return False
        raise ValueError(f"cannot cast value {value!r} to bool")
    if normalized == "date":
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return datetime.fromisoformat(str(value)).date()
    if normalized == "datetime":
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    raise ValueError(f"unsupported dtype: {dtype}")


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


_ZIP_SENTINEL = object()


def _zip_strict(*iterables: Iterable[Any]) -> Iterable[tuple[Any, ...]]:
    for items in zip_longest(*iterables, fillvalue=_ZIP_SENTINEL):
        if _ZIP_SENTINEL in items:
            raise ValueError("zip() arguments must have equal length")
        yield items


@dataclass(frozen=True)
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


class GeoPromptGeometryAccessor:
    """Pandas-like geometry accessor for row-wise geometry metrics."""

    def __init__(self, frame: "GeoPromptFrame", geometry_column: str | None = None) -> None:
        self._frame = frame
        self._geometry_column = geometry_column or frame.geometry_column
        frame._require_column(self._geometry_column)

    @property
    def geometry_column(self) -> str:
        """Column this accessor operates on."""
        return self._geometry_column

    def _geometry_values(self) -> list[Geometry]:
        values: list[Geometry] = []
        for row in self._frame:
            value = row.get(self._geometry_column)
            if value is None:
                raise ValueError(f"geometry column '{self._geometry_column}' contains null values")
            values.append(value)
        return values

    def area(self) -> list[float]:
        return [geometry_area(geom) for geom in self._geometry_values()]

    def length(self) -> list[float]:
        return [geometry_length(geom) for geom in self._geometry_values()]

    def centroid(self) -> list[Coordinate]:
        return [geometry_centroid(geom) for geom in self._geometry_values()]

    def bounds(self) -> list[tuple[float, float, float, float]]:
        return [geometry_bounds(geom) for geom in self._geometry_values()]

    def validity(self) -> list[dict[str, object]]:
        return [validate_geometry(geom) for geom in self._geometry_values()]

    def types(self) -> list[str]:
        return [geometry_type(geom) for geom in self._geometry_values()]

    def buffer(self, distance: float, resolution: int = 16) -> "GeoPromptFrame":
        """Buffer geometries in this accessor column and return an updated frame."""
        buffered_groups = buffer_geometries(
            self._geometry_values(),
            distance=distance,
            resolution=resolution,
        )
        rows: list[Record] = []
        for row, buffered_geometries in _zip_strict(self._frame._rows, buffered_groups):
            for buffered_geometry in buffered_geometries:
                buffered_row = dict(row)
                buffered_row[self._geometry_column] = buffered_geometry
                rows.append(buffered_row)
        return self._frame._clone_with_rows(rows)


class GeoPromptCoordinateIndexer:
    """Coordinate-based row selection similar to ``GeoPandas.cx``."""

    def __init__(self, frame: "GeoPromptFrame") -> None:
        self._frame = frame

    def __getitem__(self, item: Any) -> "GeoPromptFrame":
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("cx indexer expects [xmin:xmax, ymin:ymax]")
        x_slice, y_slice = item
        if not isinstance(x_slice, slice) or not isinstance(y_slice, slice):
            raise TypeError("cx indexer requires slice objects for x and y ranges")
        if x_slice.start is None or x_slice.stop is None or y_slice.start is None or y_slice.stop is None:
            raise ValueError("cx slices require explicit xmin:xmax and ymin:ymax bounds")

        xmin, xmax = float(x_slice.start), float(x_slice.stop)
        ymin, ymax = float(y_slice.start), float(y_slice.stop)
        rows = [
            dict(row)
            for row in self._frame._rows
            if geometry_intersects_bounds(row[self._frame.geometry_column], xmin, ymin, xmax, ymax)
        ]
        return self._frame._clone_with_rows(rows)


class GeoPromptStringAccessor:
    """String helper accessor for common analyst cleanup workflows."""

    def __init__(self, frame: "GeoPromptFrame") -> None:
        self._frame = frame

    def _apply(self, column: str, func: Any, *, new_column: str | None = None) -> "GeoPromptFrame":
        return self._frame.map_column(
            column,
            lambda value: None if value is None else func(str(value)),
            new_column=new_column,
        )

    def strip(self, column: str, *, new_column: str | None = None) -> "GeoPromptFrame":
        return self._apply(column, lambda value: value.strip(), new_column=new_column)

    def upper(self, column: str, *, new_column: str | None = None) -> "GeoPromptFrame":
        return self._apply(column, lambda value: value.upper(), new_column=new_column)

    def lower(self, column: str, *, new_column: str | None = None) -> "GeoPromptFrame":
        return self._apply(column, lambda value: value.lower(), new_column=new_column)

    def title(self, column: str, *, new_column: str | None = None) -> "GeoPromptFrame":
        return self._apply(column, lambda value: value.title(), new_column=new_column)

    def contains(self, column: str, pattern: str, *, case: bool = True) -> list[bool | None]:
        self._frame._require_column(column)
        needle = pattern if case else pattern.lower()
        results: list[bool | None] = []
        for value in self._frame[column]:
            if value is None:
                results.append(None)
                continue
            haystack = str(value) if case else str(value).lower()
            results.append(needle in haystack)
        return results


class GeoPromptDateTimeAccessor:
    """Datetime helper accessor for common analyst derivations."""

    def __init__(self, frame: "GeoPromptFrame") -> None:
        self._frame = frame

    def _apply(self, column: str, func: Any, *, new_column: str | None = None) -> "GeoPromptFrame":
        self._frame._require_column(column)
        return self._frame.map_column(
            column,
            lambda value: None if value is None else func(_cast_frame_value(value, "datetime")),
            new_column=new_column,
        )

    def to_datetime(self, column: str, *, new_column: str | None = None) -> "GeoPromptFrame":
        return self._apply(column, lambda value: value, new_column=new_column)

    def year(self, column: str, *, new_column: str | None = None) -> "GeoPromptFrame":
        return self._apply(column, lambda value: value.year, new_column=new_column)

    def month(self, column: str, *, new_column: str | None = None) -> "GeoPromptFrame":
        return self._apply(column, lambda value: value.month, new_column=new_column)

    def day(self, column: str, *, new_column: str | None = None) -> "GeoPromptFrame":
        return self._apply(column, lambda value: value.day, new_column=new_column)

    def weekday(self, column: str, *, new_column: str | None = None) -> "GeoPromptFrame":
        return self._apply(column, lambda value: value.weekday(), new_column=new_column)


class GeoPromptStyleAccessor:
    """Minimal style accessor for HTML-oriented analyst reporting workflows."""

    def __init__(self, frame: "GeoPromptFrame") -> None:
        self._frame = frame
        self._formats: dict[str, Callable[[Any], str]] = {}
        self._cell_styles: dict[tuple[int, str], str] = {}

    def format(self, formatter: dict[str, Any] | Callable[[Any], Any]) -> "GeoPromptStyleAccessor":
        if callable(formatter):
            for column in self._frame.columns:
                if column != self._frame.geometry_column:
                    self._formats[column] = lambda value, func=formatter: "" if value is None else str(func(value))
            return self

        for column, func in formatter.items():
            if callable(func):
                self._formats[column] = lambda value, inner=func: "" if value is None else str(inner(value))
            else:
                self._formats[column] = lambda value, fmt=func: "" if value is None else format(value, fmt)
        return self

    def highlight_max(self, subset: Sequence[str] | None = None, color: str = "#fff3b0") -> "GeoPromptStyleAccessor":
        return self._highlight_extrema(subset=subset, color=color, highest=True)

    def highlight_min(self, subset: Sequence[str] | None = None, color: str = "#d9f2e6") -> "GeoPromptStyleAccessor":
        return self._highlight_extrema(subset=subset, color=color, highest=False)

    def _highlight_extrema(
        self,
        *,
        subset: Sequence[str] | None,
        color: str,
        highest: bool,
    ) -> "GeoPromptStyleAccessor":
        columns = list(subset) if subset is not None else [c for c in self._frame.columns if c != self._frame.geometry_column]
        for column in columns:
            numeric_values = [
                (index, row.get(column))
                for index, row in enumerate(self._frame._rows)
                if isinstance(row.get(column), (int, float))
            ]
            if not numeric_values:
                continue
            target = max(value for _, value in numeric_values) if highest else min(value for _, value in numeric_values)
            for index, value in numeric_values:
                if value == target:
                    self._cell_styles[(index, column)] = f"background-color: {color};"
        return self

    def to_html(self, columns: list[str] | None = None) -> str:
        import html as _html

        cols = columns or [c for c in self._frame.columns if c != self._frame.geometry_column]
        rows_html = [
            "<table border='1'>",
            "<thead><tr>" + "".join(f"<th>{_html.escape(c)}</th>" for c in cols) + "</tr></thead>",
            "<tbody>",
        ]
        for row_index, row in enumerate(self._frame._rows):
            cells: list[str] = []
            for column in cols:
                raw_value = row.get(column, "")
                formatter = self._formats.get(column)
                value = formatter(raw_value) if formatter is not None else ("" if raw_value is None else str(raw_value))
                style = self._cell_styles.get((row_index, column), "")
                style_attr = f" style='{_html.escape(style)}'" if style else ""
                cells.append(f"<td{style_attr}>{_html.escape(value)}</td>")
            rows_html.append(f"<tr>{''.join(cells)}</tr>")
        rows_html += ["</tbody></table>"]
        return "\n".join(rows_html)


def _compile_frame_query(expression: str) -> Any:
    tree = ast.parse(expression, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.And,
        ast.Or,
        ast.Not,
        ast.In,
        ast.NotIn,
        ast.Eq,
        ast.NotEq,
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Call,
    )
    allowed_functions = {"abs", "len", "min", "max", "round"}
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("unsupported query syntax")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_functions:
                raise ValueError("query only allows simple built-in functions")
    return compile(tree, "<geoprompt-query>", "eval")


@dataclass(frozen=True)
class GeoPromptSpatialIndex:
    cell_size: float
    row_bounds: tuple[tuple[float, float, float, float], ...]
    row_centroids: tuple[Coordinate, ...]
    buckets: dict[tuple[int, int], tuple[int, ...]]
    centroid_buckets: dict[tuple[int, int], tuple[int, ...]]

    @classmethod
    def from_frame(cls, frame: "GeoPromptFrame", cell_size: float | None = None) -> "GeoPromptSpatialIndex":
        row_bounds = tuple(geometry_bounds(row[frame.geometry_column]) for row in frame)
        row_centroids = tuple(geometry_centroid(row[frame.geometry_column]) for row in frame)
        if cell_size is None:
            frame_bounds = frame.bounds()
            span = max(frame_bounds.max_x - frame_bounds.min_x, frame_bounds.max_y - frame_bounds.min_y, 1.0)
            cell_size = max(span / max(len(frame), 1) ** 0.5, 1e-9)
        if cell_size <= 0:
            raise ValueError("cell_size must be greater than zero")

        mutable_buckets: dict[tuple[int, int], set[int]] = {}
        mutable_centroid_buckets: dict[tuple[int, int], set[int]] = {}
        for index, (min_x, min_y, max_x, max_y) in enumerate(row_bounds):
            min_col = int(min_x // cell_size)
            max_col = int(max_x // cell_size)
            min_row = int(min_y // cell_size)
            max_row = int(max_y // cell_size)
            for col in range(min_col, max_col + 1):
                for row in range(min_row, max_row + 1):
                    mutable_buckets.setdefault((col, row), set()).add(index)
            centroid_x, centroid_y = row_centroids[index]
            centroid_bucket = (int(centroid_x // cell_size), int(centroid_y // cell_size))
            mutable_centroid_buckets.setdefault(centroid_bucket, set()).add(index)

        return cls(
            cell_size=float(cell_size),
            row_bounds=row_bounds,
            row_centroids=row_centroids,
            buckets={key: tuple(sorted(values)) for key, values in mutable_buckets.items()},
            centroid_buckets={key: tuple(sorted(values)) for key, values in mutable_centroid_buckets.items()},
        )

    def query_centroids(self, min_x: float, min_y: float, max_x: float, max_y: float) -> list[int]:
        min_col = int(min_x // self.cell_size)
        max_col = int(max_x // self.cell_size)
        min_row = int(min_y // self.cell_size)
        max_row = int(max_y // self.cell_size)
        candidate_indices: set[int] = set()
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                candidate_indices.update(self.centroid_buckets.get((col, row), ()))
        return sorted(
            index
            for index in candidate_indices
            if min_x <= self.row_centroids[index][0] <= max_x and min_y <= self.row_centroids[index][1] <= max_y
        )

    def nearest(self, origin_centroid: Coordinate, k: int, max_distance: float | None = None) -> list[int]:
        if k <= 0:
            raise ValueError("k must be greater than zero")

        if max_distance is not None:
            return self.query_centroids(
                origin_centroid[0] - max_distance,
                origin_centroid[1] - max_distance,
                origin_centroid[0] + max_distance,
                origin_centroid[1] + max_distance,
            )

        origin_col = int(origin_centroid[0] // self.cell_size)
        origin_row = int(origin_centroid[1] // self.cell_size)
        candidate_indices: set[int] = set()
        max_ring = max(1, len(self.centroid_buckets))

        for ring in range(max_ring + 1):
            for col in range(origin_col - ring, origin_col + ring + 1):
                for row in range(origin_row - ring, origin_row + ring + 1):
                    if max(abs(col - origin_col), abs(row - origin_row)) != ring:
                        continue
                    candidate_indices.update(self.centroid_buckets.get((col, row), ()))

            if len(candidate_indices) < k:
                continue

            distances = sorted(
                coordinate_distance(origin_centroid, self.row_centroids[index])
                for index in candidate_indices
            )
            kth_distance = distances[k - 1]
            next_ring = ring + 1
            lower_bound = max(0.0, next_ring * self.cell_size - self.cell_size)
            if kth_distance <= lower_bound:
                break

        return sorted(candidate_indices)

    def query(self, min_x: float, min_y: float, max_x: float, max_y: float, mode: BoundsQueryMode = "intersects") -> list[int]:
        min_col = int(min_x // self.cell_size)
        max_col = int(max_x // self.cell_size)
        min_row = int(min_y // self.cell_size)
        max_row = int(max_y // self.cell_size)
        candidate_indices: set[int] = set()
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                candidate_indices.update(self.buckets.get((col, row), ()))

        if mode == "intersects":
            return sorted(
                index for index in candidate_indices if _bounds_intersect(self.row_bounds[index], (min_x, min_y, max_x, max_y))
            )
        if mode == "within":
            return sorted(
                index for index in candidate_indices if _bounds_within(self.row_bounds[index], (min_x, min_y, max_x, max_y))
            )
        if mode == "centroid":
            return sorted(
                index
                for index in candidate_indices
                if min_x <= self.row_centroids[index][0] <= max_x and min_y <= self.row_centroids[index][1] <= max_y
            )
        raise ValueError(f"unsupported bounds query mode: {mode}")


class GeoPromptFrame:
    def __init__(self, rows: Sequence[Record], geometry_column: str = "geometry", crs: str | None = None) -> None:
        """Create a GeoPromptFrame from a sequence of row dicts.

        Each row must contain a geometry value under ``geometry_column``.
        Geometries are normalised to the canonical internal representation on
        construction; GeoJSON dicts, WKT strings, and coordinate tuples are
        all accepted.

        Args:
            rows: Sequence of dicts, each representing one spatial feature.
            geometry_column: Name of the key that holds geometry data.
            crs: Optional CRS string (e.g. ``"EPSG:4326"``).  Used for
                consistency checks and reprojection — it does not transform
                coordinates on its own.
        """
        self.geometry_column = geometry_column
        self.crs = _normalize_crs(crs)
        self._index_column: str | None = None
        self._spatial_index: GeoPromptSpatialIndex | None = None
        self._rows = [dict(row) for row in rows]
        for row in self._rows:
            row[self.geometry_column] = normalize_geometry(row[self.geometry_column])
        self.cx = GeoPromptCoordinateIndexer(self)

    @classmethod
    def from_records(cls, records: Iterable[Record], geometry: str = "geometry", crs: str | None = None) -> "GeoPromptFrame":
        """Construct a frame from any iterable of row dicts.

        Convenience alternative to the constructor when the input is a
        generator or other lazy iterable that must be materialised first.

        Args:
            records: Iterable of row dicts.
            geometry: Geometry column name.
            crs: Optional CRS string.

        Returns:
            A new :class:`GeoPromptFrame`.
        """
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
        frame.crs = _normalize_crs(crs)
        frame._index_column = None
        frame._spatial_index = None
        frame._rows = [dict(row) for row in rows]
        frame.cx = GeoPromptCoordinateIndexer(frame)
        return frame

    def _clone_with_rows(
        self,
        rows: Sequence[Record],
        *,
        geometry_column: str | None = None,
        crs: str | None = None,
        index_column: Any = None,
    ) -> "GeoPromptFrame":
        frame = GeoPromptFrame._from_internal_rows(
            rows,
            geometry_column=geometry_column or self.geometry_column,
            crs=self.crs if crs is None else crs,
        )
        if index_column is False:
            frame._index_column = None  # type: ignore[attr-defined]
        else:
            resolved_index = getattr(self, "_index_column", None) if index_column is None else index_column
            if resolved_index and all(resolved_index in row for row in frame._rows):
                frame._index_column = resolved_index  # type: ignore[attr-defined]
        return frame

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        """Iterate over rows as dictionaries.

        Each row is a dict mapping column names to values (including the
        geometry column).
        """
        return iter(self._rows)

    def __getitem__(self, item: Any) -> Any:
        """Return rows, row slices, columns, or filtered subsets.

        Supported patterns include integer positions, slices, column names,
        lists of column names, integer-position lists, and boolean masks.
        """
        if isinstance(item, slice):
            return [dict(row) for row in self._rows[item]]
        if isinstance(item, int):
            return dict(self._rows[item])
        if isinstance(item, str):
            self._require_column(item)
            return [row.get(item) for row in self._rows]
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            values = list(item)
            if not values:
                return self.select_columns([])
            if all(isinstance(value, bool) for value in values):
                if len(values) != len(self._rows):
                    raise ValueError("boolean mask length must match the number of rows")
                rows = [dict(row) for row, include in _zip_strict(self._rows, values) if include]
                return self._clone_with_rows(rows)
            if all(isinstance(value, int) for value in values):
                rows = [dict(self._rows[index]) for index in values]
                return self._clone_with_rows(rows)
            if all(isinstance(value, str) for value in values):
                return self.select_columns(values)
        raise TypeError("unsupported selection; use an int, slice, column name, column list, row index list, or boolean mask")

    def __setitem__(self, key: str, value: Any) -> None:
        """Set or replace a column in place using scalar or row-aligned values."""
        if not isinstance(key, str):
            raise TypeError("column name must be a string")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values = list(value)
            if len(values) != len(self._rows):
                raise ValueError("column length must match the frame length")
        else:
            values = [value for _ in self._rows]
        for row, item_value in _zip_strict(self._rows, values):
            row[key] = normalize_geometry(item_value) if key == self.geometry_column else item_value
        self.clear_spatial_index()

    @property
    def columns(self) -> list[str]:
        """Return stable column order across all rows in the frame."""
        ordered: list[str] = []
        seen: set[str] = set()
        for row in self._rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        return ordered

    @property
    def geom(self) -> GeoPromptGeometryAccessor:
        """Return a geometry accessor for row-wise geometry metrics and validation."""
        return GeoPromptGeometryAccessor(self)

    def geometry(self, column: str | None = None) -> GeoPromptGeometryAccessor:
        """Return a geometry accessor bound to ``column`` (or active geometry)."""
        return GeoPromptGeometryAccessor(self, geometry_column=column or self.geometry_column)

    @property
    def str(self) -> GeoPromptStringAccessor:
        """Return a string accessor for Pandas-like cleanup workflows."""
        return GeoPromptStringAccessor(self)

    @property
    def dt(self) -> GeoPromptDateTimeAccessor:
        """Return a datetime accessor for analyst time-derived columns."""
        return GeoPromptDateTimeAccessor(self)

    @property
    def style(self) -> GeoPromptStyleAccessor:
        """Return a minimal style accessor for conditional HTML formatting."""
        return GeoPromptStyleAccessor(self)

    @property
    def spatial_index(self) -> GeoPromptSpatialIndex | None:
        """Return the cached spatial index if one has been built."""
        return self._spatial_index

    def build_spatial_index(self, cell_size: float | None = None) -> GeoPromptSpatialIndex:
        """Build and cache a spatial index for the frame."""
        self._spatial_index = GeoPromptSpatialIndex.from_frame(self, cell_size=cell_size)
        return self._spatial_index

    def clear_spatial_index(self) -> None:
        """Clear any cached spatial index from the frame."""
        self._spatial_index = None

    def copy(self) -> "GeoPromptFrame":
        """Return a shallow structural copy of the frame."""
        return self._clone_with_rows(self.to_records())

    def head(self, count: int = 5) -> list[Record]:
        """Return the first ``count`` rows as a list of dicts.

        Args:
            count: Number of rows to return (default 5).

        Returns:
            List of row dicts, each a shallow copy.
        """
        return [dict(row) for row in self._rows[:count]]

    def tail(self, count: int = 5) -> list[Record]:
        """Return the last ``count`` rows as a list of dicts."""
        if count <= 0:
            return []
        return [dict(row) for row in self._rows[-count:]]

    def iloc(self, index: int) -> Record:
        """Return one row by zero-based integer position."""
        return dict(self._rows[index])

    def loc(self, key: Any) -> Record | "GeoPromptFrame":
        """Return rows matching the current index-like column value."""
        index_column = getattr(self, "_index_column", None)
        if not index_column:
            raise ValueError("no index column set; call set_index() first")
        matches = [dict(row) for row in self._rows if row.get(index_column) == key]
        if not matches:
            raise KeyError(key)
        if len(matches) == 1:
            return matches[0]
        frame = GeoPromptFrame._from_internal_rows(matches, geometry_column=self.geometry_column, crs=self.crs)
        frame._index_column = index_column  # type: ignore[attr-defined]
        return frame

    def drop_duplicates(self, subset: Sequence[str] | None = None, keep: str = "first") -> "GeoPromptFrame":
        """Return a new frame with duplicate rows removed."""
        if keep not in {"first", "last"}:
            raise ValueError("keep must be 'first' or 'last'")
        columns = list(subset) if subset else [c for c in self.columns]
        seen: dict[tuple[Any, ...], dict[str, Any]] = {}
        ordered_keys: list[tuple[Any, ...]] = []
        for row in self._rows:
            key = tuple(row.get(col) for col in columns)
            if key not in seen:
                ordered_keys.append(key)
            seen[key] = dict(row)
        if keep == "first":
            unique_rows: list[Record] = []
            emitted: set[tuple[Any, ...]] = set()
            for row in self._rows:
                key = tuple(row.get(col) for col in columns)
                if key in emitted:
                    continue
                emitted.add(key)
                unique_rows.append(dict(row))
        else:
            unique_rows = [seen[key] for key in ordered_keys]
        return self._clone_with_rows(unique_rows)

    def to_records(self) -> list[Record]:
        """Return all rows as a list of shallow-copied dicts."""
        return [dict(row) for row in self._rows]

    def to_markdown(self, max_rows: int | None = 20) -> str:
        """Render the frame as a Markdown table.

        Geometry values are replaced with their type name to keep output
        compact.  Use ``max_rows`` to cap the number of body rows
        (``None`` means unlimited).
        """
        if not self._rows:
            return "| |\n| --- |\n"
        cols = self.columns
        header = "| " + " | ".join(cols) + " |"
        separator = "| " + " | ".join("---" for _ in cols) + " |"
        display_rows = self._rows if max_rows is None else self._rows[:max_rows]
        lines = [header, separator]
        for row in display_rows:
            cells: list[str] = []
            for col in cols:
                val = row.get(col)
                if col == self.geometry_column and isinstance(val, dict):
                    cells.append(str(val.get("type", "")))
                else:
                    cells.append(str(val) if val is not None else "")
            lines.append("| " + " | ".join(cells) + " |")
        if max_rows is not None and len(self._rows) > max_rows:
            lines.append(f"| ... ({len(self._rows) - max_rows} more rows) |")
        return "\n".join(lines) + "\n"

    def summary(self) -> Record:
        """Return a dict summarising the frame contents.

        Keys include ``row_count``, ``column_count``, ``columns``, ``crs``,
        ``geometry_types``, ``bounds``, and per-column ``column_stats`` with
        dtype/null_count/unique_count and numeric min/max/mean when applicable.
        """
        geom_types: set[str] = set()
        col_values: dict[str, list[Any]] = {col: [] for col in self.columns}
        for row in self._rows:
            geom = row.get(self.geometry_column)
            if isinstance(geom, dict):
                geom_types.add(str(geom.get("type", "Unknown")))
            for col in self.columns:
                col_values[col].append(row.get(col))
        bounds = None
        if self._rows:
            try:
                b = self.bounds()
                bounds = {"min_x": b.min_x, "min_y": b.min_y, "max_x": b.max_x, "max_y": b.max_y}
            except (ValueError, AttributeError, TypeError):
                pass
        col_stats: list[Record] = []
        for col in self.columns:
            if col == self.geometry_column:
                continue
            vals = col_values[col]
            non_null = [v for v in vals if v is not None]
            numerics = [float(v) for v in non_null if isinstance(v, (int, float)) and not isinstance(v, bool)]
            stat: Record = {
                "column": col,
                "dtype": "numeric" if numerics and len(numerics) == len(non_null) else ("string" if non_null and all(isinstance(v, str) for v in non_null) else "mixed"),
                "null_count": len(vals) - len(non_null),
                "unique_count": len(set(str(v) for v in non_null)),
            }
            if numerics:
                stat["min"] = min(numerics)
                stat["max"] = max(numerics)
                stat["mean"] = sum(numerics) / len(numerics)
            col_stats.append(stat)
        return {
            "row_count": len(self._rows),
            "column_count": len(self.columns),
            "columns": self.columns,
            "crs": self.crs,
            "geometry_column": self.geometry_column,
            "geometry_types": sorted(geom_types),
            "bounds": bounds,
            "column_stats": col_stats,
        }

    def to_json(self, output_path: str | Path, indent: int = 2) -> str:
        """Write the frame rows to JSON and return the output path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_records(), indent=indent), encoding="utf-8")
        return str(path)

    def to_pandas(self) -> Any:
        """Export the frame as a Pandas DataFrame."""
        from .interop import to_pandas as _to_pandas

        return _to_pandas(self)

    @classmethod
    def from_pandas(
        cls,
        dataframe: Any,
        *,
        geometry_column: str = "geometry",
        crs: str | None = None,
        x_column: str | None = None,
        y_column: str | None = None,
    ) -> "GeoPromptFrame":
        """Create a frame from a Pandas DataFrame."""
        from .interop import from_pandas as _from_pandas

        return _from_pandas(
            dataframe,
            geometry_column=geometry_column,
            crs=crs,
            x_column=x_column,
            y_column=y_column,
        )

    def to_polars(self) -> Any:
        """Export the frame as a Polars DataFrame."""
        from .interop import to_polars as _to_polars

        return _to_polars(self)

    @classmethod
    def from_polars(
        cls,
        dataframe: Any,
        *,
        geometry_column: str = "geometry",
        crs: str | None = None,
        x_column: str | None = None,
        y_column: str | None = None,
    ) -> "GeoPromptFrame":
        """Create a frame from a Polars DataFrame."""
        from .interop import from_polars as _from_polars

        return _from_polars(
            dataframe,
            geometry_column=geometry_column,
            crs=crs,
            x_column=x_column,
            y_column=y_column,
        )

    def to_arrow(self) -> Any:
        """Export the frame as a PyArrow table."""
        from .interop import to_arrow as _to_arrow

        return _to_arrow(self)

    @classmethod
    def from_arrow(
        cls,
        table: Any,
        *,
        geometry_column: str = "geometry",
        crs: str | None = None,
        x_column: str | None = None,
        y_column: str | None = None,
    ) -> "GeoPromptFrame":
        """Create a frame from a PyArrow table."""
        from .interop import from_arrow as _from_arrow

        return _from_arrow(
            table,
            geometry_column=geometry_column,
            crs=crs,
            x_column=x_column,
            y_column=y_column,
        )

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> Any:
        """Return a dataframe interchange object for compatible consumers."""
        from .interop import dataframe_protocol

        return dataframe_protocol(self, nan_as_null=nan_as_null, allow_copy=allow_copy)

    def select_columns(self, columns: Sequence[str]) -> "GeoPromptFrame":
        """Return a new frame containing the requested columns and the geometry column."""
        resolved = list(columns)
        for column in resolved:
            self._require_column(column)
        if self.geometry_column not in resolved:
            resolved.append(self.geometry_column)
        rows = [{column: row.get(column) for column in resolved} for row in self._rows]
        current_index = getattr(self, "_index_column", None)
        preserved_index = current_index if current_index and current_index in resolved else False
        return self._clone_with_rows(rows, index_column=preserved_index)

    def where(self, predicate: Any | None = None, **equals: Any) -> "GeoPromptFrame":
        """Filter rows by equality conditions, a callable predicate, or a boolean mask list."""
        mask_values: list[bool] | None = None
        if predicate is not None and not callable(predicate):
            if not isinstance(predicate, (list, tuple)):
                raise TypeError("predicate must be a callable or boolean mask sequence")
            if len(predicate) != len(self._rows):
                raise ValueError("boolean mask length must match the number of rows")
            mask_values = [bool(value) for value in predicate]

        rows: list[Record] = []
        for index, row in enumerate(self._rows):
            if any(row.get(key) != value for key, value in equals.items()):
                continue
            if mask_values is not None and not mask_values[index]:
                continue
            if callable(predicate) and not bool(predicate(row)):
                continue
            rows.append(dict(row))
        return self._clone_with_rows(rows)

    def query(self, expression: str) -> "GeoPromptFrame":
        """Filter rows using a small Pandas-like expression language.

        Example:
            ``frame.query("priority >= 2 and status == 'open'")``
        """
        rows: list[Record] = []
        for row in self._rows:
            try:
                if bool(evaluate_safe_expression(expression, dict(row))):
                    rows.append(dict(row))
            except (ExpressionValidationError, ExpressionExecutionError) as exc:
                raise ValueError(f"invalid query expression: {expression}") from exc
        return self._clone_with_rows(rows)


    def clip_values(self, columns: str | Sequence[str], min_val: float | None = None, max_val: float | None = None) -> "GeoPromptFrame":
        """Clip (limit) values in specified columns to [min_val, max_val] bounds.

        Args:
            columns: Column name(s) to clip.
            min_val: Minimum value (values below are replaced).
            max_val: Maximum value (values above are replaced).
            
        Returns:
            New GeoPromptFrame with values clipped.
        """
        if isinstance(columns, str):
            columns = [columns]
        else:
            columns = list(columns)
        
        rows: list[Record] = []
        for row in self._rows:
            new_row = dict(row)
            for col in columns:
                if col in new_row and new_row[col] is not None:
                    val = new_row[col]
                    if isinstance(val, (int, float)):
                        if min_val is not None:
                            val = max(val, min_val)
                        if max_val is not None:
                            val = min(val, max_val)
                        new_row[col] = val
            rows.append(new_row)
        return self._clone_with_rows(rows)

    def applymap(self, func: Callable[[Any], Any]) -> "GeoPromptFrame":
        """Apply function element-wise to all values (except geometry).

        Args:
            func: Function to apply to each element.
            
        Returns:
            New GeoPromptFrame with func applied element-wise.
        """
        rows: list[Record] = []
        for row in self._rows:
            new_row = {}
            for key, val in row.items():
                if key == 'geometry' or val is None:
                    new_row[key] = val
                else:
                    try:
                        new_row[key] = func(val)
                    except Exception:
                        new_row[key] = val
            rows.append(new_row)
        return self._clone_with_rows(rows)

    def map(self, func: Callable[[Any], Any]) -> "GeoPromptFrame":
        """Alias for applymap() for pandas compatibility."""
        return self.applymap(func)

    @property
    def row_bounds(self) -> list[dict[str, float]]:
        """Return per-row bounds as list of dicts with minx, miny, maxx, maxy keys."""
        bounds_list = []
        for row in self._rows:
            geom = row.get('geometry')
            if geom is None:
                bounds_list.append({'minx': None, 'miny': None, 'maxx': None, 'maxy': None})
            else:
                try:
                    from .geometry import geometry_bounds
                    minx, miny, maxx, maxy = geometry_bounds(geom)
                    bounds_list.append({'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy})
                except Exception:
                    bounds_list.append({'minx': None, 'miny': None, 'maxx': None, 'maxy': None})
        return bounds_list

    def mask(self, cond: Any | None = None, other: Any = None) -> "GeoPromptFrame":
        """Replace values where condition is True with other (default: None/NA).
        
        Inverse of where(). Replaces True values in condition with other.
        
        Args:
            cond: Boolean mask (list/callable) where True indicates replacement positions.
            other: Value to replace with. Defaults to None.
            
        Returns:
            New GeoPromptFrame with conditional replacement applied.
        """
        mask_values: list[bool] | None = None
        if cond is not None and not callable(cond):
            if not isinstance(cond, (list, tuple)):
                raise TypeError("cond must be a callable or boolean mask sequence")
            if len(cond) != len(self._rows):
                raise ValueError("boolean mask length must match the number of rows")
            mask_values = [bool(value) for value in cond]
        
        rows: list[Record] = []
        for index, row in enumerate(self._rows):
            new_row = dict(row)
            
            # Determine if this row should have values replaced
            should_mask = False
            if mask_values is not None:
                should_mask = mask_values[index]
            elif callable(cond):
                should_mask = bool(cond(row))
            
            if should_mask:
                # Replace all non-geometry values with other
                for key in new_row:
                    if key != 'geometry':
                        new_row[key] = other
            
            rows.append(new_row)
        return self._clone_with_rows(rows)

    def resample(self, freq: str, on: str | None = None, agg: str | Callable | dict | None = None) -> "GeoPromptFrame":
        """Resample time-series data to a different frequency.
        
        Groups rows by time bucket (freq) and optionally aggregates values.
        
        Args:
            freq: Frequency string ('D'=daily, 'H'=hourly, 'W'=weekly, 'M'=monthly, etc).
            on: Column name for time bucketing. If None, uses first datetime column.
            agg: Aggregation function ('mean', 'sum', 'first', 'last', etc) or callable.
            
        Returns:
            New GeoPromptFrame with resampled data.
        """
        from datetime import datetime, timedelta
        
        # Find datetime column if not specified
        if on is None:
            for col in self.columns:
                if col != 'geometry':
                    if self and isinstance(self[0].get(col), datetime):
                        on = col
                        break
            if on is None:
                raise ValueError("No datetime column found. Specify 'on' parameter.")
        
        if on not in self.columns:
            raise ValueError(f"Column '{on}' not found in frame")
        
        # Parse frequency string to timedelta
        freq_char = freq[-1].upper()
        freq_count = int(freq[:-1]) if len(freq) > 1 else 1
        
        freq_map = {
            'S': timedelta(seconds=freq_count),
            'T': timedelta(minutes=freq_count),
            'H': timedelta(hours=freq_count),
            'D': timedelta(days=freq_count),
            'W': timedelta(weeks=freq_count),
            'M': timedelta(days=30*freq_count),
            'Y': timedelta(days=365*freq_count),
        }
        
        if freq_char not in freq_map:
            raise ValueError(f"Unknown frequency: {freq}")
        
        bucket_size = freq_map[freq_char]
        
        # Group rows by time bucket
        if not self:
            return self._clone_with_rows([])
        
        buckets: dict[int, list[Record]] = {}
        min_time = None
        
        for row in self._rows:
            ts = row.get(on)
            if not isinstance(ts, datetime):
                continue
            
            if min_time is None:
                min_time = ts
            
            bucket_idx = int((ts - min_time) / bucket_size)
            if bucket_idx not in buckets:
                buckets[bucket_idx] = []
            buckets[bucket_idx].append(row)
        
        if not buckets:
            return self._clone_with_rows([])
        
        # Create resampled rows
        rows: list[Record] = []
        for bucket_idx in sorted(buckets.keys()):
            bucket_rows = buckets[bucket_idx]
            if not bucket_rows:
                continue
            
            # Create aggregated row
            agg_row: Record = {}
            
            # Handle timestamp
            agg_row[on] = bucket_rows[0].get(on)
            
            # Aggregate other columns
            for col in self.columns:
                if col == on or col == 'geometry':
                    continue
                
                values = [row.get(col) for row in bucket_rows if row.get(col) is not None]
                
                if not values:
                    agg_row[col] = None
                elif callable(agg):
                    agg_row[col] = agg(values)
                elif isinstance(agg, str):
                    if agg == 'mean':
                        numeric_vals = [v for v in values if isinstance(v, (int, float))]
                        agg_row[col] = sum(numeric_vals) / len(numeric_vals) if numeric_vals else None
                    elif agg == 'sum':
                        numeric_vals = [v for v in values if isinstance(v, (int, float))]
                        agg_row[col] = sum(numeric_vals) if numeric_vals else None
                    elif agg == 'count':
                        agg_row[col] = len(values)
                    elif agg == 'first':
                        agg_row[col] = values[0]
                    elif agg == 'last':
                        agg_row[col] = values[-1]
                    elif agg == 'min':
                        numeric_vals = [v for v in values if isinstance(v, (int, float))]
                        agg_row[col] = min(numeric_vals) if numeric_vals else None
                    elif agg == 'max':
                        numeric_vals = [v for v in values if isinstance(v, (int, float))]
                        agg_row[col] = max(numeric_vals) if numeric_vals else None
                    else:
                        agg_row[col] = values[0]
                else:
                    agg_row[col] = values[0]
            
            # Preserve geometry from first row if present
            if 'geometry' in bucket_rows[0]:
                agg_row['geometry'] = bucket_rows[0]['geometry']
            
            rows.append(agg_row)
        
        return self._clone_with_rows(rows)



    def sort_values(self, by: str, descending: bool = False) -> "GeoPromptFrame":
        """Return a new frame sorted by a column, with nulls placed last."""
        self._require_column(by)
        non_null_rows = [dict(row) for row in self._rows if row.get(by) is not None]
        null_rows = [dict(row) for row in self._rows if row.get(by) is None]
        non_null_sorted = sorted(non_null_rows, key=lambda row: (row.get(by), _row_sort_key(row)), reverse=descending)
        return self._clone_with_rows([*non_null_sorted, *null_rows])

    def nlargest(self, n: int, columns: str | Sequence[str]) -> "GeoPromptFrame":
        """Return the top ``n`` rows ranked by one or more columns."""
        if n <= 0:
            return self._clone_with_rows([])
        rank_columns = [columns] if isinstance(columns, str) else list(columns)
        for column in rank_columns:
            self._require_column(column)
        candidates = [dict(row) for row in self._rows if all(row.get(column) is not None for column in rank_columns)]
        ranked = nlargest(n, candidates, key=lambda row: tuple(row.get(column) for column in rank_columns))
        return self._clone_with_rows(ranked)

    def nsmallest(self, n: int, columns: str | Sequence[str]) -> "GeoPromptFrame":
        """Return the bottom ``n`` rows ranked by one or more columns."""
        if n <= 0:
            return self._clone_with_rows([])
        rank_columns = [columns] if isinstance(columns, str) else list(columns)
        for column in rank_columns:
            self._require_column(column)
        candidates = [dict(row) for row in self._rows if all(row.get(column) is not None for column in rank_columns)]
        ranked = nsmallest(n, candidates, key=lambda row: tuple(row.get(column) for column in rank_columns))
        return self._clone_with_rows(ranked)

    def melt(
        self,
        *,
        id_vars: Sequence[str] | None = None,
        value_vars: Sequence[str] | None = None,
        var_name: str = "variable",
        value_name: str = "value",
    ) -> "GeoPromptFrame":
        """Unpivot wide columns into a longer analyst-friendly layout."""
        id_columns = list(id_vars or [])
        for column in id_columns:
            self._require_column(column)

        if self.geometry_column not in id_columns:
            id_columns_with_geometry = [*id_columns, self.geometry_column]
        else:
            id_columns_with_geometry = id_columns

        selected_values = list(value_vars) if value_vars is not None else [
            column for column in self.columns if column not in id_columns_with_geometry
        ]
        for column in selected_values:
            self._require_column(column)

        rows: list[Record] = []
        for row in self._rows:
            base = {column: row.get(column) for column in id_columns_with_geometry}
            for column in selected_values:
                melted = dict(base)
                melted[var_name] = column
                melted[value_name] = row.get(column)
                rows.append(melted)
        current_index = getattr(self, "_index_column", None)
        preserved_index = current_index if current_index and current_index in id_columns_with_geometry else False
        return self._clone_with_rows(rows, index_column=preserved_index)

    def bounds(self) -> Bounds:
        """Return the bounding box of all geometries in the frame.

        Returns:
            :class:`Bounds` with ``min_x``, ``min_y``, ``max_x``, ``max_y``.
        """
        xs: list[float] = []
        ys: list[float] = []
        for row in self._rows:
            min_x, min_y, max_x, max_y = geometry_bounds(row[self.geometry_column])
            xs.extend([min_x, max_x])
            ys.extend([min_y, max_y])
        return Bounds(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))

    def centroid(self) -> Coordinate:
        """Return the mean centroid of all geometries as an ``(x, y)`` tuple."""
        centroids = [geometry_centroid(row[self.geometry_column]) for row in self._rows]
        xs = [coord[0] for coord in centroids]
        ys = [coord[1] for coord in centroids]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _centroids(self, rows: Sequence[Record] | None = None, geometry_column: str | None = None) -> list[Coordinate]:
        active_rows = rows if rows is not None else self._rows
        active_geometry_column = geometry_column or self.geometry_column
        return [geometry_centroid(row[active_geometry_column]) for row in active_rows]

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

    def distance_matrix(self, distance_method: DistanceMethod = "euclidean") -> list[list[float]]:
        """Return a full NxN pairwise distance matrix between all rows.

        **Note:** Computing distance matrices for large frames (100K+ rows) can
        consume significant memory (e.g., 100K rows = 80GB for float64 matrix).
        Consider using :meth:`nearest_neighbors` with a smaller ``k`` value or
        streaming approaches for large datasets.

        Args:
            distance_method: ``"euclidean"`` (coordinate units) or
                ``"haversine"`` (kilometres, expects lon/lat coordinates).

        Returns:
            List of N lists of N floats.  ``result[i][j]`` is the distance
            from row *i* to row *j*.
        """
        if len(self._rows) > 100000:
            raise MemoryError(
                f"Distance matrix requires ~{(len(self._rows) ** 2 * 8) // (1024**3)}GB for {len(self._rows)} rows. "
                "Consider nearest_neighbors() or streaming instead."
            )
        return [
            [
                geometry_distance(origin=row[self.geometry_column], destination=other[self.geometry_column], method=distance_method)
                for other in self._rows
            ]
            for row in self._rows
        ]

    def geometry_types(self) -> list[str]:
        """Return the geometry type string for each row (e.g. ``"Point"``, ``"Polygon"``)."""
        return [geometry_type(row[self.geometry_column]) for row in self._rows]

    def geometry_lengths(self) -> list[float]:
        """Return the geometry length for each row.

        For Points, length is 0.  For LineStrings, total arc length.
        For Polygons, perimeter length.
        """
        return [geometry_length(row[self.geometry_column]) for row in self._rows]

    def geometry_areas(self) -> list[float]:
        """Return the geometry area for each row.

        For Points and LineStrings, area is 0.  For Polygons, the
        absolute planar area in coordinate-unit\u00b2.
        """
        return [geometry_area(row[self.geometry_column]) for row in self._rows]

    def geometry_validity(self, id_column: str | None = None) -> PromptTable:
        """Return a per-row report describing geometry validity and repair guidance."""
        if id_column is not None:
            self._require_column(id_column)

        report_rows: list[Record] = []
        for index, row in enumerate(self._rows):
            validity = validate_geometry(row[self.geometry_column])
            report: Record = {
                "row_index": index,
                "geometry_type": validity["geometry_type"],
                "is_valid": validity["is_valid"],
                "issue_count": validity["issue_count"],
                "issues": validity["issues"],
                "suggested_fix": validity["suggested_fix"],
            }
            if id_column is not None:
                report[id_column] = row[id_column]
            report_rows.append(report)
        return PromptTable(report_rows)

    def fix_geometries(self, drop_invalid: bool = False) -> "GeoPromptFrame":
        """Apply lightweight repairs and optionally drop rows that remain invalid."""
        repaired_rows: list[Record] = []
        for row in self._rows:
            updated = dict(row)
            updated[self.geometry_column] = repair_geometry(row[self.geometry_column])
            validity = validate_geometry(updated[self.geometry_column])
            if drop_invalid and not bool(validity["is_valid"]):
                continue
            repaired_rows.append(updated)
        return GeoPromptFrame(rows=repaired_rows, geometry_column=self.geometry_column, crs=self.crs)

    def nearest_neighbors(
        self,
        id_column: str = "site_id",
        k: int = 1,
        distance_method: DistanceMethod = "euclidean",
        use_spatial_index: bool = True,
    ) -> list[Record]:
        """Find the *k* nearest neighbors for every row within the same frame.

        Returns a flat list of records describing each (origin, neighbor) pair.
        For a frame with N rows and ``k=1`` the output has N records.

        Args:
            id_column: Column used as the feature identifier in result records.
            k: Number of nearest neighbors to find per origin (must be ≥ 1).
            distance_method: ``"euclidean"`` or ``"haversine"``.

        Returns:
            List of dicts with keys: ``origin``, ``neighbor``, ``distance``,
            ``origin_geometry_type``, ``neighbor_geometry_type``, ``rank``,
            ``distance_method``.
        """
        self._require_column(id_column)
        if k <= 0:
            raise ValueError("k must be greater than zero")

        centroids = self._centroids()
        geometry_types = self.geometry_types()
        spatial_index = self.build_spatial_index() if use_spatial_index and distance_method == "euclidean" and len(self._rows) > 1 else None
        nearest: list[Record] = []
        for origin_index, origin in enumerate(self._rows):
            candidate_indices: Sequence[int] = range(len(self._rows))
            if spatial_index is not None:
                resolved_candidates = [index for index in spatial_index.nearest(centroids[origin_index], k + 1) if index != origin_index]
                if len(resolved_candidates) >= k:
                    candidate_indices = resolved_candidates
            candidates = [
                (
                    self._rows[destination_index],
                    geometry_types[destination_index],
                    coordinate_distance(centroids[origin_index], centroids[destination_index], method=distance_method),
                )
                for destination_index in candidate_indices
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
        distance_method: DistanceMethod,
        max_distance: float | None = None,
        candidate_indices: Sequence[int] | None = None,
    ) -> list[tuple[Record, float]]:
        candidates: list[tuple[Record, float]] = []
        indices = candidate_indices if candidate_indices is not None else range(len(right_rows))
        for index in indices:
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
        distance_method: DistanceMethod = "euclidean",
        use_spatial_index: bool = True,
    ) -> "GeoPromptFrame":
        """Join each left row to its ``k`` nearest rows in ``other``.

        Each left row produces up to *k* output rows (one per nearest-right
        match).  With ``how="left"``, left rows that have no match within
        ``max_distance`` are kept with ``None`` values for all right columns.

        Args:
            other: The right frame to match against.
            k: Number of nearest neighbors to return per left row.
            how: ``"inner"`` drops unmatched left rows; ``"left"`` keeps them.
            lsuffix: Suffix appended to colliding left column names.
            rsuffix: Suffix appended to colliding right column names and to the
                distance/rank columns added to the output.
            max_distance: Optional upper bound on distance.  Left rows with no
                right match within this distance are treated as unmatched.
            distance_method: ``"euclidean"`` or ``"haversine"``.

        Returns:
            New :class:`GeoPromptFrame`.  See *output-columns.md* for the full
            column spec.
        """
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
        right_index = (
            other.build_spatial_index()
            if use_spatial_index and distance_method == "euclidean" and right_rows
            else None
        )
        joined_rows: list[Record] = []

        for left_row, left_centroid in _zip_strict(self._rows, left_centroids):
            candidate_indices = None if right_index is None else right_index.nearest(left_centroid, k, max_distance=max_distance)
            row_matches = self._nearest_row_matches(
                origin_centroid=left_centroid,
                right_rows=right_rows,
                right_centroids=right_centroids,
                k=k,
                distance_method=distance_method,
                max_distance=max_distance,
                candidate_indices=candidate_indices,
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
        distance_method: DistanceMethod = "euclidean",
        origin_suffix: str = "origin",
        use_spatial_index: bool = True,
    ) -> "GeoPromptFrame":
        """Assign each target to its nearest origin in ``self``.

        This is the reverse of :meth:`nearest_join` — ``self`` contains the
        origins (e.g. service locations) and ``targets`` are the demand points
        to be assigned.  Each target row appears once in the output, merged
        with the nearest origin's columns.

        Args:
            targets: Frame of demand points to assign.
            how: ``"inner"`` drops unassigned targets; ``"left"`` keeps them.
            max_distance: Optional maximum assignment distance.
            distance_method: ``"euclidean"`` or ``"haversine"``.
            origin_suffix: Suffix appended to origin columns on name collision.

        Returns:
            New :class:`GeoPromptFrame` with one row per target.
        """
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
            use_spatial_index=use_spatial_index,
        )

    def summarize_assignments(
        self,
        targets: "GeoPromptFrame",
        origin_id_column: str = "site_id",
        target_id_column: str = "site_id",
        aggregations: dict[str, AggregationName] | None = None,
        how: SpatialJoinMode = "left",
        max_distance: float | None = None,
        distance_method: DistanceMethod = "euclidean",
        assignment_suffix: str = "assigned",
    ) -> "GeoPromptFrame":
        """Summarise how many targets are assigned to each origin.

        Each target is assigned to its single nearest origin.  The output has
        one row per origin, augmented with assignment counts, distance stats,
        and any custom aggregations over the assigned targets.

        Args:
            targets: Frame of demand points to assign.
            origin_id_column: ID column in ``self`` (origins).
            target_id_column: ID column in ``targets``.
            aggregations: Optional dict mapping target column names to an
                aggregation operation (``"sum"``, ``"mean"``, ``"min"``,
                ``"max"``, ``"first"``, ``"count"``).  Each produces a column
                named ``{col}_{op}_{assignment_suffix}``.
            how: ``"left"`` keeps origins with zero assignments;
                ``"inner"`` drops them.
            max_distance: Optional maximum assignment distance.
            distance_method: ``"euclidean"`` or ``"haversine"``.
            assignment_suffix: Suffix for all added summary columns.

        Returns:
            New :class:`GeoPromptFrame` with one row per origin.  See
            *output-columns.md* for the full column spec.
        """
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
        assignments: dict[str, list[tuple[Record, float]]] = {}

        for target_row, target_centroid in _zip_strict(target_rows, target_centroids):
            row_matches = self._nearest_row_matches(
                origin_centroid=target_centroid,
                right_rows=self._rows,
                right_centroids=origin_centroids,
                k=1,
                distance_method=distance_method,
                max_distance=max_distance,
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

    def query_radius(
        self,
        anchor: str | Geometry | Coordinate,
        max_distance: float,
        id_column: str = "site_id",
        include_anchor: bool = False,
        distance_method: DistanceMethod = "euclidean",
        use_spatial_index: bool = True,
    ) -> "GeoPromptFrame":
        """Return all rows within ``max_distance`` of an anchor, sorted by distance.

        Args:
            anchor: An ID string looked up in ``id_column``, a geometry dict,
                or a raw ``(x, y)`` coordinate tuple.
            max_distance: Radius threshold (inclusive).  Must be ≥ 0.
            id_column: Column used to look up string anchors.
            include_anchor: If the anchor is specified by ID, include its own
                row in the results.
            distance_method: ``"euclidean"`` or ``"haversine"``.

        Returns:
            New :class:`GeoPromptFrame` with a ``distance`` column appended,
            sorted closest-first.
        """
        if max_distance < 0:
            raise ValueError("max_distance must be zero or greater")

        anchor_geometry, anchor_id = self._resolve_anchor_geometry(anchor, id_column=id_column)
        anchor_centroid = geometry_centroid(anchor_geometry)

        spatial_index = self.build_spatial_index() if use_spatial_index and distance_method == "euclidean" and self._rows else None
        candidate_indices: Sequence[int] = range(len(self._rows))
        if spatial_index is not None:
            candidate_indices = spatial_index.query_centroids(
                anchor_centroid[0] - max_distance,
                anchor_centroid[1] - max_distance,
                anchor_centroid[0] + max_distance,
                anchor_centroid[1] + max_distance,
            )

        rows: list[Record] = []
        for index in candidate_indices:
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
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs)

    def within_distance(
        self,
        anchor: str | Geometry | Coordinate,
        max_distance: float,
        id_column: str = "site_id",
        include_anchor: bool = False,
        distance_method: DistanceMethod = "euclidean",
        use_spatial_index: bool = True,
    ) -> list[bool]:
        """Return a boolean mask indicating which rows are within ``max_distance`` of the anchor.

        Args:
            anchor: An ID string, geometry dict, or ``(x, y)`` coordinate.
            max_distance: Distance threshold (inclusive).  Must be ≥ 0.
            id_column: Column used to look up string anchors.
            include_anchor: Whether to mark the anchor row itself as ``True``.
            distance_method: ``"euclidean"`` or ``"haversine"``.

        Returns:
            List of bools aligned with the frame rows.
        """
        if max_distance < 0:
            raise ValueError("max_distance must be zero or greater")

        anchor_geometry, anchor_id = self._resolve_anchor_geometry(anchor, id_column=id_column)
        anchor_centroid = geometry_centroid(anchor_geometry)
        centroids = self._centroids()
        matches = [False] * len(self._rows)

        candidate_indices: Sequence[int] = range(len(self._rows))
        if use_spatial_index and distance_method == "euclidean" and self._rows:
            spatial_index = self.build_spatial_index()
            candidate_indices = spatial_index.query_centroids(
                anchor_centroid[0] - max_distance,
                anchor_centroid[1] - max_distance,
                anchor_centroid[0] + max_distance,
                anchor_centroid[1] + max_distance,
            )

        for index in candidate_indices:
            row = self._rows[index]
            centroid = centroids[index]
            if anchor_id is not None and not include_anchor and id_column in row and str(row[id_column]) == anchor_id:
                matches[index] = False
                continue
            matches[index] = coordinate_distance(anchor_centroid, centroid, method=distance_method) <= max_distance
        return matches

    def proximity_join(
        self,
        other: "GeoPromptFrame",
        max_distance: float,
        how: SpatialJoinMode = "inner",
        lsuffix: str = "left",
        rsuffix: str = "right",
        distance_method: DistanceMethod = "euclidean",
        use_spatial_index: bool = True,
    ) -> "GeoPromptFrame":
        """Join each left row to every right row within ``max_distance``.

        Unlike :meth:`nearest_join`, this can produce multiple output rows per
        left row — one for each right row within range.  Results are
        distance-sorted within each left group.

        Args:
            other: Right frame.
            max_distance: Distance threshold (inclusive).  Must be ≥ 0.
            how: ``"inner"`` drops left rows with no matches;
                ``"left"`` keeps them with ``None`` right values.
            lsuffix: Suffix for colliding left columns.
            rsuffix: Suffix for colliding right columns and for the added
                ``distance_{rsuffix}`` column.
            distance_method: ``"euclidean"`` or ``"haversine"``.

        Returns:
            New :class:`GeoPromptFrame`.
        """
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
        right_index = (
            other.build_spatial_index()
            if use_spatial_index and distance_method == "euclidean" and right_rows
            else None
        )
        joined_rows: list[Record] = []

        for left_row, left_centroid in _zip_strict(self._rows, left_centroids):
            row_matches: list[tuple[Record, float]] = []
            candidate_indices = range(len(right_rows))
            if right_index is not None:
                candidate_indices = right_index.query_centroids(
                    left_centroid[0] - max_distance,
                    left_centroid[1] - max_distance,
                    left_centroid[0] + max_distance,
                    left_centroid[1] + max_distance,
                )
            for index in candidate_indices:
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
        """Return a new frame with each geometry replaced by its buffer polygon.

        Requires Shapely (``pip install -e .[overlay]``).

        Args:
            distance: Buffer radius in coordinate units.  Must be ≥ 0.
            resolution: Number of segments used to approximate each quarter
                circle (higher = smoother buffer polygon, default 16).

        Returns:
            New :class:`GeoPromptFrame` where each row's geometry is the
            expanded buffer shape.
        """
        buffered_groups = buffer_geometries(
            [row[self.geometry_column] for row in self._rows],
            distance=distance,
            resolution=resolution,
        )
        rows: list[Record] = []
        for row, buffered_geometries in _zip_strict(self._rows, buffered_groups):
            for buffered_geometry in buffered_geometries:
                buffered_row = dict(row)
                buffered_row[self.geometry_column] = buffered_geometry
                rows.append(buffered_row)
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs)

    def buffer_join(
        self,
        other: "GeoPromptFrame",
        distance: float,
        how: SpatialJoinMode = "inner",
        lsuffix: str = "left",
        rsuffix: str = "right",
        resolution: int = 16,
    ) -> "GeoPromptFrame":
        """Join by expanding each left geometry to a buffer and intersecting with ``other``.

        Useful for proximity queries on polygon or line geometries where
        centroid-based distance does not capture the true footprint.
        Requires Shapely (``pip install -e .[overlay]``).

        Args:
            other: Right frame.
            distance: Buffer radius.  Must be ≥ 0.
            how: ``"inner"`` or ``"left"``.
            lsuffix: Suffix for colliding left columns and for the added
                ``buffer_geometry_{lsuffix}`` / ``buffer_distance_{lsuffix}`` columns.
            rsuffix: Suffix for colliding right column names.
            resolution: Buffer polygon resolution (default 16).

        Returns:
            New :class:`GeoPromptFrame`.
        """
        if distance < 0:
            raise ValueError("distance must be zero or greater")
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before buffer joins")

        right_rows = list(other._rows)
        right_columns = [column for column in other.columns if column != other.geometry_column]
        right_bounds = [geometry_bounds(row[other.geometry_column]) for row in right_rows]
        buffered_groups = buffer_geometries(
            [row[self.geometry_column] for row in self._rows],
            distance=distance,
            resolution=resolution,
        )

        joined_rows: list[Record] = []
        for left_row, buffered_geometries in _zip_strict(self._rows, buffered_groups):
            row_matches: list[tuple[Record, Geometry]] = []
            for buffered_geometry in buffered_geometries:
                buffered_bounds = geometry_bounds(buffered_geometry)
                for right_row, right_bound in _zip_strict(right_rows, right_bounds):
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

        return GeoPromptFrame(rows=joined_rows, geometry_column=self.geometry_column, crs=self.crs or other.crs)

    def coverage_summary(
        self,
        targets: "GeoPromptFrame",
        predicate: SpatialJoinPredicate = "intersects",
        target_id_column: str = "site_id",
        aggregations: dict[str, AggregationName] | None = None,
        rsuffix: str = "covered",
        use_spatial_index: bool = True,
    ) -> "GeoPromptFrame":
        """Count targets whose geometry satisfies a predicate with each self geometry.

        The output has one row per ``self`` row with a ``count_{rsuffix}``
        column and a list of matching target IDs.

        Args:
            targets: Frame of target features to test against.
            predicate: Spatial relationship to test — ``"intersects"``
                (default), ``"within"``, or ``"contains"``.
            target_id_column: ID column in ``targets``.
            aggregations: Optional column → operation dict for numeric
                aggregations over the matched targets.
            rsuffix: Suffix appended to all summary columns added to the output.

        Returns:
            New :class:`GeoPromptFrame` with coverage summary columns.
        """
        if self.crs and targets.crs and self.crs != targets.crs:
            raise ValueError("frames must share the same CRS before coverage summaries")
        if predicate not in {"intersects", "within", "contains"}:
            raise ValueError(f"unsupported spatial join predicate: {predicate}")

        if targets._rows:
            targets._require_column(target_id_column)
        target_rows = list(targets._rows)
        target_bounds = [geometry_bounds(row[targets.geometry_column]) for row in target_rows]
        target_index = targets.build_spatial_index() if use_spatial_index and target_rows else None
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
            candidate_indices: Sequence[int] = range(len(target_rows))
            if target_index is not None:
                candidate_indices = target_index.query(left_bounds[0], left_bounds[1], left_bounds[2], left_bounds[3], mode="intersects")
            for index in candidate_indices:
                target_row = target_rows[index]
                target_bound = target_bounds[index]
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

        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs or targets.crs)

    def query_bounds(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        mode: BoundsQueryMode = "intersects",
    ) -> "GeoPromptFrame":
        """Return all rows whose geometry passes the bounding-box filter.

        Args:
            min_x: Left edge of the query box.
            min_y: Bottom edge of the query box.
            max_x: Right edge of the query box.
            max_y: Top edge of the query box.
            mode: ``"intersects"`` (default) — geometry bounding box overlaps
                the query box; ``"within"`` — geometry bounding box fits
                entirely inside; ``"centroid"`` — geometry centroid falls
                inside the query box.

        Returns:
            Filtered :class:`GeoPromptFrame` preserving all columns.
        """
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

    def query_bounds_indexed(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        mode: BoundsQueryMode = "intersects",
        spatial_index: GeoPromptSpatialIndex | None = None,
    ) -> "GeoPromptFrame":
        """Return a bounds-filtered frame using a reusable spatial index."""
        if min_x > max_x or min_y > max_y:
            raise ValueError("query bounds must be ordered from minimum to maximum")
        index = spatial_index or self.build_spatial_index()
        rows = [dict(self._rows[row_index]) for row_index in index.query(min_x, min_y, max_x, max_y, mode=mode)]
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs)

    def set_crs(self, crs: str | int, allow_override: bool = False) -> "GeoPromptFrame":
        """Attach a CRS label to the frame without transforming coordinates.

        Args:
            crs: CRS string to attach (e.g. ``"EPSG:4326"``).
            allow_override: If ``True``, replace an existing CRS.  If
                ``False`` (default), raises ``ValueError`` if CRS is already
                set to a different value.

        Returns:
            New :class:`GeoPromptFrame` with the CRS set.
        """
        normalized_crs = _normalize_crs(crs)
        if self.crs is not None and self.crs != normalized_crs and not allow_override:
            raise ValueError("frame already has a CRS; pass allow_override=True to replace it")
        return GeoPromptFrame(rows=self.to_records(), geometry_column=self.geometry_column, crs=normalized_crs)

    def to_crs(self, target_crs: str | int) -> "GeoPromptFrame":
        """Reproject all geometries to a different CRS.

        Requires PyProj (``pip install -e .[projection]``).
        The frame must already have a CRS set via :meth:`set_crs`.

        Args:
            target_crs: Destination CRS string (e.g. ``"EPSG:3857"``).

        Returns:
            New :class:`GeoPromptFrame` with reprojected coordinates and
            ``crs=target_crs``.
        """
        if self.crs is None:
            raise ValueError("frame CRS is not set; call set_crs before reprojecting")
        normalized_target_crs = _normalize_crs(target_crs)
        if self.crs == normalized_target_crs:
            return GeoPromptFrame(rows=self.to_records(), geometry_column=self.geometry_column, crs=self.crs)

        from .crs import reproject_geometry

        rows = self.to_records()
        for row in rows:
            row[self.geometry_column] = reproject_geometry(row[self.geometry_column], self.crs, normalized_target_crs)
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=normalized_target_crs)

    def spatial_join(
        self,
        other: "GeoPromptFrame",
        predicate: SpatialJoinPredicate = "intersects",
        how: SpatialJoinMode = "inner",
        lsuffix: str = "left",
        rsuffix: str = "right",
        use_spatial_index: bool = True,
    ) -> "GeoPromptFrame":
        """Join rows by a spatial predicate test on their geometries.

        Produces one output row per (left, right) pair whose geometries satisfy
        the predicate.  Uses bounding-box pre-filtering for efficiency.

        Args:
            other: Right frame.
            predicate: ``"intersects"`` (default), ``"within"``, or
                ``"contains"``.
            how: ``"inner"`` drops left rows with no match;
                ``"left"`` keeps them with ``None`` right values.
            lsuffix: Suffix for colliding left column names.
            rsuffix: Suffix for colliding right column names and for the added
                ``{geometry}_{rsuffix}`` and ``join_predicate_{rsuffix}`` columns.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before spatial joins")

        right_columns = [column for column in other.columns if column != other.geometry_column]
        joined_rows: list[Record] = []
        right_rows = list(other._rows)
        right_bounds = [geometry_bounds(row[other.geometry_column]) for row in right_rows]
        right_index = other.build_spatial_index() if use_spatial_index and right_rows else None

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

        for left_row in self._rows:
            left_geometry = left_row[self.geometry_column]
            left_bounds = geometry_bounds(left_geometry)
            row_matches = []
            candidate_indices = range(len(right_rows))
            if right_index is not None:
                candidate_indices = right_index.query(left_bounds[0], left_bounds[1], left_bounds[2], left_bounds[3], mode="intersects")
            for index in candidate_indices:
                right_row = right_rows[index]
                right_bound = right_bounds[index]
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
        """Clip all geometries in ``self`` to the union of ``mask`` geometries.

        Requires Shapely (``pip install -e .[overlay]``).

        Args:
            mask: Frame whose geometries form the clip boundary.

        Returns:
            New :class:`GeoPromptFrame` with geometries trimmed to the mask.
            Rows whose geometry has no overlap with the mask are omitted.
        """
        if self.crs and mask.crs and self.crs != mask.crs:
            raise ValueError("frames must share the same CRS before clip operations")

        mask_rows = list(mask)
        clipped_groups = clip_geometries(
            [row[self.geometry_column] for row in self._rows],
            [row[mask.geometry_column] for row in mask_rows],
        )
        rows: list[Record] = []
        for row, clipped_geometries in _zip_strict(self._rows, clipped_groups):
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
        """Compute geometric intersections of all crossing left/right geometry pairs.

        The left geometry column in the output is replaced by the intersection
        geometry.  Produces one row per intersecting (left, right) pair.
        Requires Shapely (``pip install -e .[overlay]``).

        Args:
            other: Right frame.
            lsuffix: Suffix for colliding left column names.
            rsuffix: Suffix for colliding right column names and for the added
                ``{geometry}_{rsuffix}`` column.

        Returns:
            New :class:`GeoPromptFrame` with intersection geometries.
        """
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
        """Group rows by a column and merge their geometries (union).

        Non-geometry columns without an aggregation rule are kept by taking the
        first value in the group.  Requires Shapely for polygon/line merging
        (``pip install -e .[overlay]``).

        Args:
            by: Column name to group by.
            aggregations: Optional dict mapping column names to aggregation
                operations (``"sum"``, ``"mean"``, ``"min"``, ``"max"``,
                ``"first"``, ``"count"``).

        Returns:
            New :class:`GeoPromptFrame` with one row per unique ``by`` value.
        """
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

    def explode(self) -> "GeoPromptFrame":
        """Split multi-part geometries into individual single-part rows.

        Each part inherits the attributes of the original row.  Single-part
        geometries pass through unchanged.

        Returns:
            New :class:`GeoPromptFrame` with one row per geometry part.
        """
        _MULTI_MAP = {
            "MultiPoint": ("Point", True),
            "MultiLineString": ("LineString", False),
            "MultiPolygon": ("Polygon", False),
        }
        rows: list[Record] = []
        for row in self._rows:
            geom = row[self.geometry_column]
            if not isinstance(geom, dict):
                rows.append(dict(row))
                continue
            gtype = geom.get("type", "")
            if gtype not in _MULTI_MAP:
                rows.append(dict(row))
                continue
            single_type, flat_coords = _MULTI_MAP[gtype]
            coords = geom.get("coordinates", [])
            for part in coords:
                child: Record = {k: v for k, v in row.items() if k != self.geometry_column}
                if flat_coords:
                    child[self.geometry_column] = {"type": single_type, "coordinates": part}
                else:
                    child[self.geometry_column] = {"type": single_type, "coordinates": part}
                rows.append(child)
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs)

    def unary_union(self) -> dict[str, Any]:
        """Return the union of all geometries as a single GeoJSON geometry dict.

        Requires Shapely (``pip install shapely``).
        """
        shapely_ops = importlib.import_module("shapely.ops")
        shapely_geometry = importlib.import_module("shapely.geometry")
        shapes = [shapely_geometry.shape(row[self.geometry_column]) for row in self._rows if isinstance(row.get(self.geometry_column), dict)]
        if not shapes:
            raise ValueError("frame has no geometries to union")
        merged = shapely_ops.unary_union(shapes)
        return dict(merged.__geo_interface__)

    def with_column(self, name: str, values: Sequence[Any]) -> "GeoPromptFrame":
        """Return a new frame with ``name`` added (or replaced) from ``values``.

        Args:
            name: Column name to add or overwrite.
            values: Sequence of values, one per row.  Must have the same
                length as the frame.

        Returns:
            New :class:`GeoPromptFrame` with the column added.
        """
        if len(values) != len(self._rows):
            raise ValueError("column length must match the frame length")
        rows = self.to_records()
        for row, value in _zip_strict(rows, values):
            row[name] = value
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs)

    def assign(self, **columns: Any) -> "GeoPromptFrame":
        """Return a new frame with one or more columns added or replaced.

        Each keyword argument is a new column.  The value can be:

        * A scalar — broadcast to every row.
        * A sequence of length ``len(self)`` — one value per row.
        * A callable ``(frame) -> value_or_sequence`` — evaluated against
          the *current* frame state (columns are applied sequentially, so
          later columns can reference earlier ones).

        Args:
            **columns: Column names mapped to scalars, sequences, or callables.

        Returns:
            New :class:`GeoPromptFrame`.
        """
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

    def batch_accessibility_scores(
        self,
        supply_columns: Sequence[str],
        travel_cost_columns: Sequence[str],
        decay_method: str = "power",
        *,
        scale: float = 1.0,
        power: float = 2.0,
        rate: float = 1.0,
        sigma: float = 1.0,
    ) -> list[float]:
        """Compute accessibility scores row-wise from aligned supply and travel cost columns."""
        for column in [*supply_columns, *travel_cost_columns]:
            self._require_column(column)
        supply_rows = [[float(row[column]) for column in supply_columns] for row in self._rows]
        travel_cost_rows = [[float(row[column]) for column in travel_cost_columns] for row in self._rows]
        return batch_accessibility_scores(
            supply_rows,
            travel_cost_rows,
            decay_method=decay_method,
            scale=scale,
            power=power,
            rate=rate,
            sigma=sigma,
        )

    def batch_accessibility_table(
        self,
        supply_columns: Sequence[str],
        travel_cost_columns: Sequence[str],
        decay_method: str = "power",
        *,
        id_column: str = "site_id",
        scale: float = 1.0,
        power: float = 2.0,
        rate: float = 1.0,
        sigma: float = 1.0,
    ) -> PromptTable:
        self._require_column(id_column)
        scores = self.batch_accessibility_scores(
            supply_columns,
            travel_cost_columns,
            decay_method=decay_method,
            scale=scale,
            power=power,
            rate=rate,
            sigma=sigma,
        )
        return PromptTable.from_records(
            {
                id_column: str(row[id_column]),
                "accessibility_score": score,
                "decay_method": decay_method,
            }
            for row, score in _zip_strict(self._rows, scores)
        )

    def gravity_interaction_series(
        self,
        origin_mass_column: str,
        destination_mass_column: str,
        cost_column: str,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        scale_factor: float = 1.0,
    ) -> list[float]:
        """Compute gravity-model interaction values row-wise from frame columns."""
        self._require_column(origin_mass_column)
        self._require_column(destination_mass_column)
        self._require_column(cost_column)
        return vectorized_gravity_interaction(
            [float(row[origin_mass_column]) for row in self._rows],
            [float(row[destination_mass_column]) for row in self._rows],
            [float(row[cost_column]) for row in self._rows],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            scale_factor=scale_factor,
        )

    def gravity_interaction_table(
        self,
        origin_mass_column: str,
        destination_mass_column: str,
        cost_column: str,
        *,
        id_column: str = "site_id",
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        scale_factor: float = 1.0,
    ) -> PromptTable:
        self._require_column(id_column)
        values = self.gravity_interaction_series(
            origin_mass_column,
            destination_mass_column,
            cost_column,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            scale_factor=scale_factor,
        )
        return PromptTable.from_records(
            {
                id_column: str(row[id_column]),
                origin_mass_column: float(row[origin_mass_column]),
                destination_mass_column: float(row[destination_mass_column]),
                cost_column: float(row[cost_column]),
                "gravity_interaction": value,
            }
            for row, value in _zip_strict(self._rows, values)
        )

    def service_probability_series(
        self,
        predictor_columns: Sequence[str],
        coefficients: dict[str, float],
        intercept: float = 0.0,
    ) -> list[float]:
        """Compute logistic service probabilities row-wise from named predictor columns."""
        for column in predictor_columns:
            self._require_column(column)
        predictor_rows = [
            {column: float(row[column]) for column in predictor_columns}
            for row in self._rows
        ]
        return vectorized_service_probability(predictor_rows, coefficients, intercept=intercept)

    def service_probability_table(
        self,
        predictor_columns: Sequence[str],
        coefficients: dict[str, float],
        intercept: float = 0.0,
        *,
        id_column: str = "site_id",
    ) -> PromptTable:
        self._require_column(id_column)
        probabilities = self.service_probability_series(
            predictor_columns,
            coefficients,
            intercept=intercept,
        )
        return PromptTable.from_records(
            {
                id_column: str(row[id_column]),
                **{column: float(row[column]) for column in predictor_columns},
                "service_probability": probability,
            }
            for row, probability in _zip_strict(self._rows, probabilities)
        )

    def build_spatial_index(self, cell_size: float | None = None) -> GeoPromptSpatialIndex:
        """Build and cache a lightweight bounds index for repeated spatial queries."""
        self._spatial_index = GeoPromptSpatialIndex.from_frame(self, cell_size=cell_size)
        return self._spatial_index

    def set_geometry(self, column: str, *, validate: bool = True) -> "GeoPromptFrame":
        """Switch the active geometry column while keeping all geometry columns in rows.

        This enables workflows with multiple geometry columns (for example
        ``origin_geom`` and ``dest_geom``) by changing which geometry column is
        treated as active for frame-level spatial operations.
        """
        self._require_column(column)
        rows: list[Record] = []
        for row in self._rows:
            if column not in row:
                raise KeyError(f"column '{column}' is not present in one or more rows")
            updated = dict(row)
            if validate and updated[column] is not None:
                updated[column] = normalize_geometry(updated[column])
            rows.append(updated)
        return self._clone_with_rows(rows, geometry_column=column)

    def _require_column(self, name: str) -> None:
        if name not in self.columns:
            available = ", ".join(self.columns) if self.columns else "(no columns available)"
            raise KeyError(f"column '{name}' is not present; available columns: {available}")

    def catchment_competition(
        self,
        targets: "GeoPromptFrame",
        distance: float,
        target_id_column: str = "site_id",
        distance_method: DistanceMethod = "euclidean",
    ) -> "GeoPromptFrame":
        """Summarize exclusive and contested target coverage by provider catchments."""
        if distance < 0:
            raise ValueError("distance must be zero or greater")
        if self.crs and targets.crs and self.crs != targets.crs:
            raise ValueError("frames must share the same CRS before catchment competition")
        if targets._rows:
            targets._require_column(target_id_column)

        assignments: dict[int, list[Record]] = {index: [] for index in range(len(self._rows))}
        contested: dict[int, list[str]] = {index: [] for index in range(len(self._rows))}
        for target in targets._rows:
            matches: list[tuple[int, float]] = []
            target_centroid = geometry_centroid(target[targets.geometry_column])
            for index, provider in enumerate(self._rows):
                provider_centroid = geometry_centroid(provider[self.geometry_column])
                gap = coordinate_distance(provider_centroid, target_centroid, method=distance_method)
                if gap <= distance:
                    matches.append((index, gap))
            if len(matches) == 1:
                assignments[matches[0][0]].append(target)
            elif len(matches) > 1:
                target_id = str(target[target_id_column])
                for index, _ in matches:
                    assignments[index].append(target)
                    contested[index].append(target_id)

        rows: list[Record] = []
        for index, row in enumerate(self._rows):
            matched = assignments[index]
            contested_ids = sorted(set(contested[index]))
            resolved = dict(row)
            resolved[f"{target_id_column}s_competition"] = [str(item[target_id_column]) for item in matched]
            resolved["count_competition"] = len(matched)
            resolved["contested_target_ids_competition"] = contested_ids
            resolved["exclusive_count_competition"] = max(0, len(matched) - len(contested_ids))
            resolved["contested_count_competition"] = len(contested_ids)
            rows.append(resolved)
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs or targets.crs)

    def overlay_summary(
        self,
        other: "GeoPromptFrame",
        target_id_column: str = "site_id",
        rsuffix: str = "overlay",
    ) -> "GeoPromptFrame":
        """Summarize overlay results as counts and share metrics instead of raw intersections."""
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before overlay summaries")
        if other._rows:
            other._require_column(target_id_column)

        rows: list[Record] = []
        for row in self._rows:
            left_geometry = row[self.geometry_column]
            left_area = geometry_area(left_geometry)
            matched_rows: list[Record] = []
            for other_row in other._rows:
                if geometry_intersects(left_geometry, other_row[other.geometry_column]):
                    matched_rows.append(other_row)
            resolved = dict(row)
            resolved[f"{target_id_column}s_{rsuffix}"] = [str(item[target_id_column]) for item in matched_rows]
            resolved[f"count_{rsuffix}"] = len(matched_rows)
            resolved[f"area_share_{rsuffix}"] = 1.0 if left_area > 0 and matched_rows else 0.0
            rows.append(resolved)
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs or other.crs)

    def corridor_reach(
        self,
        targets: "GeoPromptFrame",
        max_distance: float,
        target_id_column: str = "site_id",
        rsuffix: str = "reach",
    ) -> "GeoPromptFrame":
        """Summarize which targets fall within a corridor-style reach distance of each geometry."""
        if max_distance < 0:
            raise ValueError("max_distance must be zero or greater")
        if self.crs and targets.crs and self.crs != targets.crs:
            raise ValueError("frames must share the same CRS before corridor reach")
        if targets._rows:
            targets._require_column(target_id_column)

        rows: list[Record] = []
        for row in self._rows:
            left_geometry = row[self.geometry_column]
            matched_rows: list[Record] = []
            left_bounds = geometry_bounds(left_geometry)
            expanded_bounds = (
                left_bounds[0] - max_distance,
                left_bounds[1] - max_distance,
                left_bounds[2] + max_distance,
                left_bounds[3] + max_distance,
            )
            for target in targets._rows:
                target_geometry = target[targets.geometry_column]
                if not geometry_intersects_bounds(target_geometry, *expanded_bounds):
                    continue
                if geometry_intersects(left_geometry, target_geometry) or geometry_distance(left_geometry, target_geometry) <= max_distance:
                    matched_rows.append(target)
            resolved = dict(row)
            resolved[f"{target_id_column}s_{rsuffix}"] = [str(item[target_id_column]) for item in matched_rows]
            resolved[f"count_{rsuffix}"] = len(matched_rows)
            rows.append(resolved)
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs or targets.crs)

    def zone_fit_scoring(
        self,
        zones: "GeoPromptFrame",
        zone_id_column: str = "site_id",
        demand_column: str | None = None,
        rsuffix: str = "zone_fit",
    ) -> "GeoPromptFrame":
        """Score how well each feature fits the nearest candidate zone."""
        if self.crs and zones.crs and self.crs != zones.crs:
            raise ValueError("frames must share the same CRS before zone fit scoring")
        if zones._rows:
            zones._require_column(zone_id_column)

        rows: list[Record] = []
        for row in self._rows:
            left_geometry = row[self.geometry_column]
            left_area = max(geometry_area(left_geometry), 1e-9)
            best_zone_id: str | None = None
            best_score = -1.0
            for zone in zones._rows:
                zone_geometry = zone[zones.geometry_column]
                distance_value = geometry_distance(left_geometry, zone_geometry)
                similarity = area_similarity(
                    origin_area=left_area,
                    destination_area=max(geometry_area(zone_geometry), left_area),
                    distance_value=distance_value,
                    scale=1.0,
                    power=1.0,
                )
                demand_factor = float(zone.get(demand_column, 1.0)) if demand_column is not None and demand_column in zone else 1.0
                score = similarity * demand_factor
                if score > best_score:
                    best_score = score
                    best_zone_id = str(zone[zone_id_column])
            resolved = dict(row)
            resolved[f"{zone_id_column}_{rsuffix}"] = best_zone_id
            resolved[f"score_{rsuffix}"] = best_score if best_score >= 0 else None
            resolved["zone_fit_score"] = best_score if best_score >= 0 else None
            rows.append(resolved)
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs or zones.crs)

    def multi_scale_clustering(
        self,
        distance_threshold: float,
        min_cluster_size: int = 1,
        distance_method: DistanceMethod = "euclidean",
    ) -> "GeoPromptFrame":
        """Cluster nearby features using a simple deterministic connected-components approach."""
        if distance_threshold < 0:
            raise ValueError("distance_threshold must be zero or greater")
        if min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be >= 1")

        centroids = self._centroids()
        cluster_ids = [-1] * len(self._rows)
        current_cluster = 0
        for index in range(len(self._rows)):
            if cluster_ids[index] != -1:
                continue
            queue = [index]
            members: list[int] = []
            cluster_ids[index] = current_cluster
            while queue:
                current = queue.pop()
                members.append(current)
                for other in range(len(self._rows)):
                    if cluster_ids[other] != -1:
                        continue
                    if coordinate_distance(centroids[current], centroids[other], method=distance_method) <= distance_threshold:
                        cluster_ids[other] = current_cluster
                        queue.append(other)
            if len(members) < min_cluster_size:
                for member in members:
                    cluster_ids[member] = -1
            else:
                current_cluster += 1

        rows = []
        for row, cluster_id in _zip_strict(self._rows, cluster_ids):
            resolved = dict(row)
            resolved["cluster_id"] = cluster_id
            rows.append(resolved)
        return GeoPromptFrame(rows=rows, geometry_column=self.geometry_column, crs=self.crs)

    def neighborhood_pressure(
        self,
        weight_column: str,
        scale: float = 1.0,
        power: float = 2.0,
        include_self: bool = False,
        distance_method: DistanceMethod = "euclidean",
    ) -> list[float]:
        """Compute cumulative influence pressure on each row from all neighbors.

        Applies the :func:`~geoprompt.equations.prompt_influence` decay formula
        for every (row, neighbor) pair and sums the result.  Good for mapping
        congestion or density gradients.

        Args:
            weight_column: Column containing numeric weights (e.g. demand).
            scale: Distance decay scale parameter.
            power: Distance decay exponent.
            include_self: Whether a row contributes to its own pressure score.
            distance_method: ``"euclidean"`` or ``"haversine"``.

        Returns:
            List of floats aligned with the frame rows.
        """
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
        distance_method: DistanceMethod = "euclidean",
    ) -> list[float]:
        """Compute how strongly each row influences (or is influenced by) a single anchor.

        Uses the :func:`~geoprompt.equations.prompt_influence` formula with
        each row's weight and its distance to the anchor row.  Useful for
        ranked site scoring relative to a hub.

        Args:
            weight_column: Column containing numeric weights.
            anchor: ID of the anchor row in ``id_column``.
            id_column: Column used to locate the anchor.
            scale: Distance decay scale parameter.
            power: Distance decay exponent.
            distance_method: ``"euclidean"`` or ``"haversine"``.

        Returns:
            List of floats aligned with the frame rows.
        """
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
        distance_method: DistanceMethod = "euclidean",
    ) -> list[float]:
        """Compute accessibility scores weighted by corridor length and decay from an anchor.

        Uses :func:`~geoprompt.equations.corridor_strength`, which extends
        :func:`~geoprompt.equations.prompt_influence` by incorporating the
        physical length of each geometry (a LineString corridor).  Suitable for
        scoring road or utility corridor access from a hub.

        Args:
            weight_column: Column containing numeric weights (e.g. capacity).
            anchor: ID of the anchor row in ``id_column``.
            id_column: Column used to locate the anchor.
            scale: Distance decay scale parameter.
            power: Distance decay exponent.
            distance_method: ``"euclidean"`` or ``"haversine"``.

        Returns:
            List of floats aligned with the frame rows.
        """
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
        distance_method: DistanceMethod = "euclidean",
    ) -> list[Record]:
        """Return a pairwise table of area-weighted similarity scores.

        Uses :func:`~geoprompt.equations.area_similarity`.  Useful for
        comparing polygon features by both size and proximity.

        Args:
            id_column: Column used as the identifier in result records.
            scale: Distance decay scale parameter.
            power: Distance decay exponent.
            distance_method: ``"euclidean"`` or ``"haversine"``.

        Returns:
            List of dicts with keys: ``origin``, ``destination``,
            ``area_similarity``, ``distance_method``.
        """
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
        distance_method: DistanceMethod = "euclidean",
    ) -> list[Record]:
        """Return a pairwise gravity-model interaction table.

        Applies :func:`~geoprompt.equations.prompt_interaction` to every
        (origin, destination) pair.  When ``preferred_bearing`` is set, a
        ``directional_alignment`` score is also included using
        :func:`~geoprompt.equations.directional_alignment`.

        Args:
            origin_weight: Column holding the origin mass/weight.
            destination_weight: Column holding the destination mass/weight.
            id_column: Column used as the identifier in result records.
            scale: Distance decay scale parameter.
            power: Distance decay exponent.
            preferred_bearing: Optional compass bearing (degrees from north,
                clockwise) used to score directional alignment.
            distance_method: ``"euclidean"`` or ``"haversine"``.

        Returns:
            List of dicts with keys: ``origin``, ``destination``, ``distance``,
            ``interaction``, ``distance_method``, and optionally
            ``directional_alignment``.
        """
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

    # --- Analyst ergonomics methods ---

    def groupby(self, by: str | Sequence[str]) -> "GroupedGeoPromptFrame":
        """Group rows by one or more columns for aggregation.

        Args:
            by: Column name or sequence of column names to group by.

        Returns:
            :class:`GroupedGeoPromptFrame` supporting ``.agg()`` calls.
        """
        group_columns = [by] if isinstance(by, str) else list(by)
        if not group_columns:
            raise ValueError("groupby requires at least one column")
        for col in group_columns:
            self._require_column(col)
        return GroupedGeoPromptFrame(self._rows, group_columns, self.geometry_column, self.crs)

    def agg(self, aggregations: dict[str, str | Sequence[str]]) -> "GeoPromptFrame":
        """Aggregate the full frame into a single summary row."""
        from .table import _apply_aggregation

        summary: Record = {"row_count": len(self._rows)}
        if self._rows:
            summary[self.geometry_column] = self._rows[0].get(self.geometry_column)
        for column, ops in aggregations.items():
            self._require_column(column)
            op_list = [ops] if isinstance(ops, str) else list(ops)
            values = [row.get(column) for row in self._rows if column in row]
            for op in op_list:
                summary[f"{column}_{op}"] = _apply_aggregation(values, op)
        return GeoPromptFrame._from_internal_rows([summary], geometry_column=self.geometry_column, crs=self.crs)

    def aggregate(self, aggregations: dict[str, str | Sequence[str]]) -> "GeoPromptFrame":
        """Alias for :meth:`agg` for pandas-style parity."""
        return self.agg(aggregations)

    def stack(
        self,
        columns: Sequence[str] | None = None,
        *,
        var_name: str = "variable",
        value_name: str = "value",
    ) -> "GeoPromptFrame":
        """Stack selected columns into a longer row-wise layout."""
        value_columns = list(columns) if columns is not None else [c for c in self.columns if c != self.geometry_column]
        for column in value_columns:
            self._require_column(column)
        id_columns = [c for c in self.columns if c not in value_columns and c != self.geometry_column]
        return self.melt(
            id_vars=id_columns,
            value_vars=value_columns,
            var_name=var_name,
            value_name=value_name,
        )

    def unstack(
        self,
        *,
        index: str,
        columns: str,
        values: str,
        agg: str = "first",
    ) -> PromptTable:
        """Unstack a long layout back into a table by pivoting it wider."""
        return self.pivot(index=index, columns=columns, values=values, agg=agg)

    def wide_to_long(
        self,
        *,
        stubnames: Sequence[str],
        i: str | Sequence[str],
        j: str,
        sep: str = "_",
    ) -> "GeoPromptFrame":
        """Convert repeated wide stub columns like sales_2024 into a long layout."""
        id_columns = [i] if isinstance(i, str) else list(i)
        for column in id_columns:
            self._require_column(column)

        suffixes: set[str] = set()
        for column in self.columns:
            for stub in stubnames:
                prefix = f"{stub}{sep}"
                if column.startswith(prefix):
                    suffixes.add(column[len(prefix):])

        rows: list[Record] = []
        ordered_suffixes = sorted(suffixes)
        for row in self._rows:
            base = {column: row.get(column) for column in id_columns}
            base[self.geometry_column] = row.get(self.geometry_column)
            for suffix in ordered_suffixes:
                long_row = dict(base)
                long_row[j] = suffix
                populated = False
                for stub in stubnames:
                    key = f"{stub}{sep}{suffix}"
                    if key in row:
                        long_row[stub] = row.get(key)
                        populated = True
                if populated:
                    rows.append(long_row)
        current_index = getattr(self, "_index_column", None)
        preserved_index = current_index if current_index and current_index in id_columns else False
        return self._clone_with_rows(rows, index_column=preserved_index)

    def pivot(
        self,
        index: str,
        columns: str,
        values: str,
        agg: str = "sum",
    ) -> PromptTable:
        """Pivot the frame into a cross-tabulated :class:`PromptTable`.

        Args:
            index: Column whose unique values become rows.
            columns: Column whose unique values become new columns.
            values: Column to aggregate.
            agg: Aggregation function name.

        Returns:
            A :class:`PromptTable` with the pivoted data.
        """
        for col in [index, columns, values]:
            self._require_column(col)
        table = PromptTable(self.to_records())
        return table.pivot(index=index, columns=columns, values=values, agg=agg)

    def value_counts(self, column: str, normalize: bool = False, dropna: bool = True) -> PromptTable:
        """Count unique values in a column.

        Args:
            column: Column to count.
            normalize: If ``True``, return shares instead of counts.
            dropna: If ``True``, exclude ``None`` values.

        Returns:
            A :class:`PromptTable` with value counts.
        """
        self._require_column(column)
        table = PromptTable(self.to_records())
        return table.value_counts(column=column, normalize=normalize, dropna=dropna)

    def describe(self, columns: Sequence[str] | None = None) -> PromptTable:
        """Return descriptive statistics for selected columns.

        Args:
            columns: List of columns to describe.  Defaults to all
                non-geometry columns.

        Returns:
            A :class:`PromptTable` with per-column statistics.
        """
        selected = list(columns) if columns is not None else [c for c in self.columns if c != self.geometry_column]
        table = PromptTable(self.to_records())
        return table.describe(columns=selected)

    def isna(self) -> PromptTable:
        """Return a boolean table showing which values are missing."""
        rows: list[Record] = []
        for row in self._rows:
            rows.append({column: row.get(column) is None for column in self.columns})
        return PromptTable(rows)

    def notna(self) -> PromptTable:
        """Return a boolean table showing which values are present."""
        rows: list[Record] = []
        for row in self._rows:
            rows.append({column: row.get(column) is not None for column in self.columns})
        return PromptTable(rows)

    def profile(self) -> Record:
        """Return an analyst-friendly profile summary of the frame."""
        summary = self.summary()
        null_counts = {
            column: sum(1 for row in self._rows if row.get(column) is None)
            for column in self.columns
        }
        unique_counts = {
            column: len({json.dumps(row.get(column), sort_keys=True, default=str) for row in self._rows})
            for column in self.columns
            if column != self.geometry_column
        }
        summary["null_counts"] = null_counts
        summary["unique_counts"] = unique_counts
        summary["preview"] = self.head(5)
        return summary

    def fillna(self, value: Any = None, column: str | None = None, method: FillMethod | None = None) -> "GeoPromptFrame":
        """Fill null values in the frame.

        Args:
            value: Scalar replacement value, or a dict mapping column names to
                replacement values.
            column: If given, only fill nulls in this column.
            method: ``"ffill"`` (forward fill) or ``"bfill"`` (backward fill).

        Returns:
            New :class:`GeoPromptFrame` with nulls replaced.
        """
        rows = self.to_records()
        if method == "ffill":
            last: dict[str, Any] = {}
            for row in rows:
                targets = [column] if column else [k for k in row if k != self.geometry_column]
                for col in targets:
                    if row.get(col) is None and col in last:
                        row[col] = last[col]
                    elif row.get(col) is not None:
                        last[col] = row[col]
            return self._clone_with_rows(rows)
        if method == "bfill":
            last = {}
            for row in reversed(rows):
                targets = [column] if column else [k for k in row if k != self.geometry_column]
                for col in targets:
                    if row.get(col) is None and col in last:
                        row[col] = last[col]
                    elif row.get(col) is not None:
                        last[col] = row[col]
            return self._clone_with_rows(rows)

        fill_map: dict[str, Any] = {}
        if isinstance(value, dict):
            fill_map = value
        elif column is not None:
            fill_map = {column: value}
        else:
            for col in (self.columns if self._rows else []):
                if col != self.geometry_column:
                    fill_map[col] = value

        for row in rows:
            for col, fill_val in fill_map.items():
                if row.get(col) is None:
                    row[col] = fill_val
        return self._clone_with_rows(rows)

    def dropna(self, subset: Sequence[str] | None = None, how: str = "any") -> "GeoPromptFrame":
        """Drop rows with null values.

        Args:
            subset: Columns to check for nulls.  Defaults to all
                non-geometry columns.
            how: ``"any"`` drops rows with any null; ``"all"`` drops only
                rows where every checked column is null.

        Returns:
            New :class:`GeoPromptFrame` with null rows removed.
        """
        if how not in {"any", "all"}:
            raise ValueError("how must be 'any' or 'all'")
        check_cols = list(subset) if subset else [c for c in self.columns if c != self.geometry_column]
        rows: list[Record] = []
        for row in self._rows:
            nulls = [row.get(c) is None for c in check_cols]
            if how == "any" and any(nulls):
                continue
            if how == "all" and all(nulls):
                continue
            rows.append(dict(row))
        return self._clone_with_rows(rows)

    def rename_columns(self, mapping: dict[str, str]) -> "GeoPromptFrame":
        """Return a new frame with columns renamed according to ``mapping``.

        The geometry column name is updated if it appears in the mapping.

        Args:
            mapping: Dict mapping old column names to new names.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        new_geom = mapping.get(self.geometry_column, self.geometry_column)
        rows: list[Record] = []
        for row in self._rows:
            rows.append({mapping.get(k, k): v for k, v in row.items()})
        current_index = getattr(self, "_index_column", None)
        renamed_index = mapping.get(current_index, current_index) if current_index else None
        return self._clone_with_rows(rows, geometry_column=new_geom, index_column=renamed_index)

    def drop_columns(self, columns: Sequence[str]) -> "GeoPromptFrame":
        """Return a new frame with the specified columns removed.

        The geometry column cannot be dropped.

        Args:
            columns: Column names to remove.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        drop_set = set(columns)
        if self.geometry_column in drop_set:
            raise ValueError("cannot drop the geometry column")
        rows = [{k: v for k, v in row.items() if k not in drop_set} for row in self._rows]
        current_index = getattr(self, "_index_column", None)
        preserved_index = current_index if current_index and current_index not in drop_set else False
        return self._clone_with_rows(rows, index_column=preserved_index)

    def reorder_columns(self, columns: Sequence[str]) -> "GeoPromptFrame":
        """Return a new frame with columns reordered.

        Any columns not listed are appended in their original order.
        The geometry column is always preserved.

        Args:
            columns: Desired column order.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        ordered = list(columns)
        remaining = [c for c in self.columns if c not in ordered]
        full_order = ordered + remaining
        if self.geometry_column not in full_order:
            full_order.append(self.geometry_column)
        rows = [{col: row.get(col) for col in full_order} for row in self._rows]
        return self._clone_with_rows(rows)

    def astype(self, column: str, dtype: str) -> "GeoPromptFrame":
        """Cast a column to a new type.

        Args:
            column: Column to cast.
            dtype: Target type name — supports ``"int"``, ``"float"``, ``"str"``, ``"bool"``, ``"date"``, and ``"datetime"``.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        self._require_column(column)
        rows = self.to_records()
        for row in rows:
            if row.get(column) is not None:
                row[column] = _cast_frame_value(row[column], dtype)
        return self._clone_with_rows(rows)

    def replace(self, column: str, mapping: dict[Any, Any]) -> "GeoPromptFrame":
        """Replace values in a column according to a mapping.

        Args:
            column: Column to apply replacements to.
            mapping: Dict mapping old values to new values.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        self._require_column(column)
        rows = self.to_records()
        for row in rows:
            val = row.get(column)
            if val in mapping:
                row[column] = mapping[val]
        return self._clone_with_rows(rows)

    def map_column(self, column: str, func: Any, *, new_column: str | None = None) -> "GeoPromptFrame":
        """Map a callable across a single column.

        Args:
            column: Source column to transform.
            func: Callable taking the current value and returning a new value.
            new_column: Optional target column name. Defaults to overwriting
                ``column``.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        self._require_column(column)
        if not callable(func):
            raise TypeError("func must be callable")
        target = new_column or column
        rows = self.to_records()
        for row in rows:
            row[target] = func(row.get(column))
        return self._clone_with_rows(rows)

    def apply_rows(self, func: Any) -> "GeoPromptFrame":
        """Apply a row-wise transformation callable to each record."""
        if not callable(func):
            raise TypeError("func must be callable")
        rows = [dict(func(dict(row))) for row in self._rows]
        return self._clone_with_rows(rows)

    def pipe(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Pipe the frame through a callable for chainable workflows."""
        if not callable(func):
            raise TypeError("func must be callable")
        return func(self, *args, **kwargs)

    def set_index(self, column: str) -> "GeoPromptFrame":
        """Set a column as an index-like identifier, stored in ``_index_column``.

        This does not restructure the data but records which column acts as
        the logical row identifier for downstream operations.

        Args:
            column: Column to designate as the index.

        Returns:
            New :class:`GeoPromptFrame` with ``_index_column`` set.
        """
        self._require_column(column)
        frame = GeoPromptFrame._from_internal_rows(
            [dict(r) for r in self._rows],
            geometry_column=self.geometry_column,
            crs=self.crs,
        )
        frame._index_column = column  # type: ignore[attr-defined]
        return frame

    def reset_index(self, drop: bool = False) -> "GeoPromptFrame":
        """Reset the index column.

        If the frame has an ``_index_column`` set by :meth:`set_index`, this
        method clears it.  When ``drop`` is ``False`` a numeric ``"index"``
        column is added.

        Args:
            drop: If ``True`` remove the index entirely.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        rows = self.to_records()
        if not drop:
            for i, row in enumerate(rows):
                row.setdefault("index", i)
        frame = GeoPromptFrame._from_internal_rows(
            rows, geometry_column=self.geometry_column, crs=self.crs,
        )
        frame._index_column = None  # type: ignore[attr-defined]
        return frame

    def crosstab(
        self,
        index: str,
        columns: str,
        values: str | None = None,
        agg: str = "count",
    ) -> PromptTable:
        """Cross-tabulate two categorical columns.

        When *values* is ``None``, cells contain counts of co-occurrences.

        Args:
            index: Column whose unique values become rows.
            columns: Column whose unique values become new columns.
            values: Optional numeric column to aggregate.
            agg: Aggregation function (default ``"count"``).

        Returns:
            A :class:`PromptTable` with cross-tabulated results.
        """
        self._require_column(index)
        self._require_column(columns)
        if values is not None:
            self._require_column(values)

        buckets: dict[Any, dict[Any, list[Any]]] = {}
        col_values: set[Any] = set()
        for row in self._rows:
            idx = row.get(index)
            col = row.get(columns)
            col_values.add(col)
            bucket = buckets.setdefault(idx, {})
            bucket.setdefault(col, []).append(row.get(values, 1))

        sorted_cols = sorted(col_values, key=str)
        result_rows: list[Record] = []
        for idx_val, cols_bucket in buckets.items():
            result: Record = {index: idx_val}
            for cv in sorted_cols:
                vals = cols_bucket.get(cv, [])
                if not vals:
                    result[str(cv)] = 0 if agg == "count" else None
                elif agg == "count":
                    result[str(cv)] = len(vals)
                elif agg == "sum":
                    result[str(cv)] = sum(float(v) for v in vals if v is not None)
                elif agg == "mean":
                    numeric = [float(v) for v in vals if v is not None]
                    result[str(cv)] = (sum(numeric) / len(numeric)) if numeric else None
                elif agg == "min":
                    result[str(cv)] = min(vals)
                elif agg == "max":
                    result[str(cv)] = max(vals)
                elif agg == "first":
                    result[str(cv)] = vals[0]
                else:
                    raise ValueError(f"unsupported aggregation: {agg}")
            result_rows.append(result)
        return PromptTable(result_rows)

    def merge(
        self,
        other: "GeoPromptFrame",
        on: str | Sequence[str],
        how: str = "inner",
        rsuffix: str = "right",
        *,
        lsuffix: str = "left",
        suffixes: tuple[str, str] | None = None,
        indicator: bool = False,
        validate: str | None = None,
    ) -> "GeoPromptFrame":
        """Merge with another frame on shared key column(s).

        Args:
            other: Right frame.
            on: Column name or list of column names to join on.
            how: ``"inner"`` or ``"left"``.
            rsuffix: Suffix for colliding right column names (default).
            lsuffix: Suffix for colliding left column names.
            suffixes: Explicit ``(left_suffix, right_suffix)`` tuple.  When
                provided, overrides *lsuffix* / *rsuffix*.
            indicator: If ``True``, add a ``_merge`` column showing whether
                each row came from ``"both"`` or ``"left_only"``.
            validate: Optional join-cardinality check: ``"one_to_one"``,
                ``"one_to_many"``, or ``"many_to_one"``.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        if suffixes is not None:
            lsuffix, rsuffix = suffixes

        on_cols: list[str] = [on] if isinstance(on, str) else list(on)
        for col in on_cols:
            self._require_column(col)
            other._require_column(col)

        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")
        if validate not in {None, "one_to_one", "one_to_many", "many_to_one"}:
            raise ValueError("validate must be one of: None, 'one_to_one', 'one_to_many', 'many_to_one'")

        def _key(row: Record) -> tuple:
            return tuple(row.get(c) for c in on_cols)

        left_counts: dict[tuple, int] = {}
        for row in self._rows:
            k = _key(row)
            left_counts[k] = left_counts.get(k, 0) + 1
        right_counts: dict[tuple, int] = {}
        for row in other._rows:
            k = _key(row)
            right_counts[k] = right_counts.get(k, 0) + 1

        if validate == "one_to_one":
            if any(count > 1 for count in left_counts.values()) or any(count > 1 for count in right_counts.values()):
                raise ValueError("merge validation failed: expected one-to-one keys")
        elif validate == "one_to_many":
            if any(count > 1 for count in left_counts.values()):
                raise ValueError("merge validation failed: expected one-to-many keys with unique left keys")
        elif validate == "many_to_one":
            if any(count > 1 for count in right_counts.values()):
                raise ValueError("merge validation failed: expected many-to-one keys with unique right keys")

        right_index: dict[tuple, list[Record]] = {}
        for row in other._rows:
            right_index.setdefault(_key(row), []).append(row)

        on_set = set(on_cols)
        right_columns = [c for c in other.columns if c not in on_set and c != other.geometry_column]
        collision_cols = set(self.columns) & set(right_columns)

        def _suffixed(col: str, sfx: str) -> str:
            sep = "" if sfx.startswith("_") else "_"
            return f"{col}{sep}{sfx}"

        rows: list[Record] = []
        for left_row in self._rows:
            key = _key(left_row)
            right_rows = right_index.get(key, [])
            if not right_rows and how == "left":
                merged: Record = {}
                for col, val in left_row.items():
                    target = _suffixed(col, lsuffix) if col in collision_cols else col
                    merged[target] = val
                for col in right_columns:
                    target = _suffixed(col, rsuffix) if col in collision_cols else col
                    merged[target] = None
                if indicator:
                    merged["_merge"] = "left_only"
                rows.append(merged)
            for right_row in right_rows:
                merged = {}
                for col, val in left_row.items():
                    target = _suffixed(col, lsuffix) if col in collision_cols else col
                    merged[target] = val
                for col in right_columns:
                    target = _suffixed(col, rsuffix) if col in collision_cols else col
                    merged[target] = right_row[col]
                if indicator:
                    merged["_merge"] = "both"
                rows.append(merged)

        geom_col = self.geometry_column
        if geom_col in collision_cols:
            geom_col = _suffixed(geom_col, lsuffix)
        preserved_index = getattr(self, "_index_column", None)
        if preserved_index and preserved_index not in rows[0] if rows else True:
            preserved_index = False
        return self._clone_with_rows(rows, geometry_column=geom_col, index_column=preserved_index)

    @staticmethod
    def concat(frames: Sequence["GeoPromptFrame"]) -> "GeoPromptFrame":
        """Concatenate multiple frames into one.

        All frames should share the same geometry column name.  The CRS of the
        first frame is used.

        Args:
            frames: Sequence of frames to concatenate.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        if not frames:
            return GeoPromptFrame([], geometry_column="geometry")
        geom_col = frames[0].geometry_column
        crs = frames[0].crs
        ordered_columns: list[str] = []
        seen: set[str] = set()
        for frame in frames:
            for column in frame.columns:
                if column not in seen:
                    seen.add(column)
                    ordered_columns.append(column)
        all_rows: list[Record] = []
        for frame in frames:
            for row in frame.to_records():
                all_rows.append({column: row.get(column) for column in ordered_columns})
        combined = GeoPromptFrame._from_internal_rows(all_rows, geometry_column=geom_col, crs=crs)
        common_index = getattr(frames[0], "_index_column", None)
        if common_index and all(getattr(frame, "_index_column", None) == common_index for frame in frames):
            combined._index_column = common_index  # type: ignore[attr-defined]
        return combined

    def sample(self, n: int, seed: int | None = None) -> "GeoPromptFrame":
        """Return a random sample of ``n`` rows.

        Args:
            n: Number of rows to sample.
            seed: Optional random seed for reproducibility.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        import random
        rng = random.Random(seed)
        n = min(n, len(self._rows))
        sampled = rng.sample(self._rows, n)
        return self._clone_with_rows(sampled)

    def unique(self, column: str) -> list[Any]:
        """Return unique values from a column.

        Args:
            column: Column name.

        Returns:
            List of unique values.
        """
        self._require_column(column)
        seen: set[str] = set()
        result: list[Any] = []
        for row in self._rows:
            val = row.get(column)
            key = json.dumps(val, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                result.append(val)
        return result

    def set_geometry(self, column: str) -> "GeoPromptFrame":
        """Return a new frame with a different active geometry column.

        Args:
            column: Name of the column to use as the geometry column.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        if not self._rows or column not in self._rows[0]:
            raise KeyError(f"column '{column}' is not present")
        return GeoPromptFrame._from_internal_rows(
            [dict(row) for row in self._rows],
            geometry_column=column,
            crs=self.crs,
        )

    def to_prompt_table(self, drop_geometry: bool = True) -> PromptTable:
        """Convert to a :class:`PromptTable`, optionally dropping geometry.

        Args:
            drop_geometry: If ``True``, remove the geometry column.

        Returns:
            A :class:`PromptTable`.
        """
        if drop_geometry:
            rows = [{k: v for k, v in row.items() if k != self.geometry_column} for row in self._rows]
        else:
            rows = self.to_records()
        return PromptTable(rows)

    # --- Geometry convenience methods ---

    def simplify(self, tolerance: float) -> "GeoPromptFrame":
        """Simplify all geometries using the Douglas-Peucker algorithm.

        Requires Shapely (``pip install -e .[overlay]``).

        Args:
            tolerance: Maximum distance a simplified edge may deviate from
                the original geometry.

        Returns:
            New :class:`GeoPromptFrame` with simplified geometries.
        """
        from .overlay import geometry_to_shapely, geometry_from_shapely
        rows: list[Record] = []
        for row in self._rows:
            geom = row[self.geometry_column]
            shape = geometry_to_shapely(geom)
            simplified = shape.simplify(tolerance)
            parts = geometry_from_shapely(simplified)
            if parts:
                for part in parts:
                    new_row = dict(row)
                    new_row[self.geometry_column] = part
                    rows.append(new_row)
            else:
                rows.append(dict(row))
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def convex_hull(self) -> "GeoPromptFrame":
        """Replace each geometry with its convex hull.

        Requires Shapely (``pip install -e .[overlay]``).

        Returns:
            New :class:`GeoPromptFrame` with convex hull geometries.
        """
        from .overlay import geometry_to_shapely, geometry_from_shapely
        rows: list[Record] = []
        for row in self._rows:
            shape = geometry_to_shapely(row[self.geometry_column])
            hull = shape.convex_hull
            parts = geometry_from_shapely(hull)
            if parts:
                new_row = dict(row)
                new_row[self.geometry_column] = parts[0]
                rows.append(new_row)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def envelope(self) -> "GeoPromptFrame":
        """Replace each geometry with its bounding-box rectangle.

        Returns:
            New :class:`GeoPromptFrame` with envelope polygons.
        """
        rows: list[Record] = []
        for row in self._rows:
            min_x, min_y, max_x, max_y = geometry_bounds(row[self.geometry_column])
            env = {
                "type": "Polygon",
                "coordinates": (
                    (min_x, min_y), (max_x, min_y), (max_x, max_y),
                    (min_x, max_y), (min_x, min_y),
                ),
            }
            new_row = dict(row)
            new_row[self.geometry_column] = env
            rows.append(new_row)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    # --- Overlay convenience methods ---

    def difference(self, other: "GeoPromptFrame") -> "GeoPromptFrame":
        """Compute the geometric difference of each geometry against the union of ``other``.

        Requires Shapely (``pip install -e .[overlay]``).

        Args:
            other: Frame whose geometries are subtracted.

        Returns:
            New :class:`GeoPromptFrame`.
        """
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before difference operations")
        from .overlay import geometry_to_shapely, geometry_from_shapely, _load_shapely
        _, _, unary_union, _ = _load_shapely()
        mask = unary_union([geometry_to_shapely(row[other.geometry_column]) for row in other._rows])
        rows: list[Record] = []
        for row in self._rows:
            shape = geometry_to_shapely(row[self.geometry_column])
            diff = shape.difference(mask)
            parts = geometry_from_shapely(diff)
            for part in parts:
                new_row = dict(row)
                new_row[self.geometry_column] = part
                rows.append(new_row)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def symmetric_difference(self, other: "GeoPromptFrame") -> "GeoPromptFrame":
        """Compute the symmetric difference between the union of geometries in both frames.

        Requires Shapely (``pip install -e .[overlay]``).

        Args:
            other: Other frame.

        Returns:
            New :class:`GeoPromptFrame` with symmetric-difference geometries.
        """
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before symmetric difference operations")
        from .overlay import geometry_to_shapely, geometry_from_shapely, _load_shapely
        _, _, unary_union, _ = _load_shapely()
        left_union = unary_union([geometry_to_shapely(row[self.geometry_column]) for row in self._rows])
        right_union = unary_union([geometry_to_shapely(row[other.geometry_column]) for row in other._rows])
        result = left_union.symmetric_difference(right_union)
        parts = geometry_from_shapely(result)
        rows: list[Record] = []
        for part in parts:
            rows.append({self.geometry_column: part})
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    def union(self, other: "GeoPromptFrame") -> "GeoPromptFrame":
        """Compute the geometric union of both frames.

        Requires Shapely (``pip install -e .[overlay]``).

        Args:
            other: Other frame.

        Returns:
            New :class:`GeoPromptFrame` with union geometries.
        """
        if self.crs and other.crs and self.crs != other.crs:
            raise ValueError("frames must share the same CRS before union operations")
        from .overlay import geometry_to_shapely, geometry_from_shapely, _load_shapely
        _, _, unary_union, _ = _load_shapely()
        all_shapes = [geometry_to_shapely(row[self.geometry_column]) for row in self._rows]
        all_shapes += [geometry_to_shapely(row[other.geometry_column]) for row in other._rows]
        merged = unary_union(all_shapes)
        parts = geometry_from_shapely(merged)
        rows: list[Record] = []
        for part in parts:
            rows.append({self.geometry_column: part})
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self.geometry_column, crs=self.crs)

    # ------------------------------------------------------------------
    # G2 additions — DataFrame fundamentals and spatial-specific methods
    # ------------------------------------------------------------------

    def corr(self, method: str = "pearson") -> dict[str, dict[str, float]]:
        """Return pairwise correlation of numeric columns.

        Returns a dict-of-dicts mapping ``{col: {col: correlation}}``.
        Only numeric (int/float) columns are included.
        """
        import math as _math

        numeric_cols = [c for c in self.columns if c != self.geometry_column
                        and all(isinstance(row.get(c), (int, float)) for row in self._rows if row.get(c) is not None)]
        result: dict[str, dict[str, float]] = {}
        for col_a in numeric_cols:
            result[col_a] = {}
            vals_a = [float(row[col_a]) for row in self._rows if row.get(col_a) is not None]
            mean_a = sum(vals_a) / len(vals_a) if vals_a else 0.0
            std_a = _math.sqrt(sum((v - mean_a) ** 2 for v in vals_a) / len(vals_a)) if len(vals_a) > 1 else 0.0
            for col_b in numeric_cols:
                if std_a == 0.0:
                    result[col_a][col_b] = float("nan")
                    continue
                pairs = [(float(r[col_a]), float(r[col_b])) for r in self._rows
                         if r.get(col_a) is not None and r.get(col_b) is not None]
                if not pairs:
                    result[col_a][col_b] = float("nan")
                    continue
                mean_b = sum(p[1] for p in pairs) / len(pairs)
                std_b = _math.sqrt(sum((p[1] - mean_b) ** 2 for p in pairs) / len(pairs)) if len(pairs) > 1 else 0.0
                if std_b == 0.0:
                    result[col_a][col_b] = float("nan")
                    continue
                cov = sum((p[0] - mean_a) * (p[1] - mean_b) for p in pairs) / len(pairs)
                result[col_a][col_b] = cov / (std_a * std_b)
        return result

    def cov(self) -> dict[str, dict[str, float]]:
        """Return pairwise covariance of numeric columns."""
        import math as _math

        numeric_cols = [c for c in self.columns if c != self.geometry_column
                        and all(isinstance(row.get(c), (int, float)) for row in self._rows if row.get(c) is not None)]
        result: dict[str, dict[str, float]] = {}
        for col_a in numeric_cols:
            result[col_a] = {}
            vals_a = [float(row[col_a]) for row in self._rows if row.get(col_a) is not None]
            mean_a = sum(vals_a) / len(vals_a) if vals_a else 0.0
            for col_b in numeric_cols:
                pairs = [(float(r[col_a]), float(r[col_b])) for r in self._rows
                         if r.get(col_a) is not None and r.get(col_b) is not None]
                if len(pairs) < 2:
                    result[col_a][col_b] = float("nan")
                    continue
                mean_b = sum(p[1] for p in pairs) / len(pairs)
                cov = sum((p[0] - mean_a) * (p[1] - mean_b) for p in pairs) / (len(pairs) - 1)
                result[col_a][col_b] = cov
        return result

    def cumsum(self, column: str) -> list[float | None]:
        """Return cumulative sum of *column* values."""
        running = 0.0
        result: list[float | None] = []
        for row in self._rows:
            v = row.get(column)
            if v is None:
                result.append(None)
            else:
                running += float(v)
                result.append(running)
        return result

    def cumprod(self, column: str) -> list[float | None]:
        """Return cumulative product of *column* values."""
        running = 1.0
        result: list[float | None] = []
        for row in self._rows:
            v = row.get(column)
            if v is None:
                result.append(None)
            else:
                running *= float(v)
                result.append(running)
        return result

    def cummax(self, column: str) -> list[float | None]:
        """Return cumulative maximum of *column* values."""
        running: float | None = None
        result: list[float | None] = []
        for row in self._rows:
            v = row.get(column)
            if v is None:
                result.append(running)
            else:
                fv = float(v)
                running = fv if running is None else max(running, fv)
                result.append(running)
        return result

    def cummin(self, column: str) -> list[float | None]:
        """Return cumulative minimum of *column* values."""
        running: float | None = None
        result: list[float | None] = []
        for row in self._rows:
            v = row.get(column)
            if v is None:
                result.append(running)
            else:
                fv = float(v)
                running = fv if running is None else min(running, fv)
                result.append(running)
        return result

    def diff(self, column: str, periods: int = 1) -> list[float | None]:
        """Return first discrete difference of *column* by *periods*."""
        vals = [row.get(column) for row in self._rows]
        result: list[float | None] = []
        for i, v in enumerate(vals):
            if i < periods or v is None or vals[i - periods] is None:
                result.append(None)
            else:
                result.append(float(v) - float(vals[i - periods]))  # type: ignore[arg-type]
        return result

    def pct_change(self, column: str, periods: int = 1) -> list[float | None]:
        """Return percentage change of *column* between rows."""
        vals = [row.get(column) for row in self._rows]
        result: list[float | None] = []
        for i, v in enumerate(vals):
            prev = vals[i - periods] if i >= periods else None
            if v is None or prev is None or float(prev) == 0.0:
                result.append(None)
            else:
                result.append((float(v) - float(prev)) / abs(float(prev)))  # type: ignore[arg-type]
        return result

    def rolling(self, column: str, window: int, func: str = "mean") -> list[float | None]:
        """Apply a rolling window aggregation to *column*.

        *func* may be ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, or ``"std"``.
        """
        import math as _math
        vals = [row.get(column) for row in self._rows]
        result: list[float | None] = []
        for i in range(len(vals)):
            if i < window - 1:
                result.append(None)
                continue
            window_vals = [float(v) for v in vals[i - window + 1:i + 1] if v is not None]
            if not window_vals:
                result.append(None)
                continue
            if func == "mean":
                result.append(sum(window_vals) / len(window_vals))
            elif func == "sum":
                result.append(sum(window_vals))
            elif func == "min":
                result.append(min(window_vals))
            elif func == "max":
                result.append(max(window_vals))
            elif func == "std":
                mean = sum(window_vals) / len(window_vals)
                result.append(_math.sqrt(sum((v - mean) ** 2 for v in window_vals) / len(window_vals)))
            else:
                result.append(None)
        return result

    def expanding(self, column: str, func: str = "mean") -> list[float | None]:
        """Apply an expanding window aggregation to *column*.

        *func* may be ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, or ``"std"``.
        """
        import math as _math
        vals = [row.get(column) for row in self._rows]
        result: list[float | None] = []
        seen: list[float] = []
        for v in vals:
            if v is not None:
                seen.append(float(v))
            if not seen:
                result.append(None)
                continue
            if func == "mean":
                result.append(sum(seen) / len(seen))
            elif func == "sum":
                result.append(sum(seen))
            elif func == "min":
                result.append(min(seen))
            elif func == "max":
                result.append(max(seen))
            elif func == "std":
                if len(seen) < 2:
                    result.append(0.0)
                else:
                    mean = sum(seen) / len(seen)
                    result.append(_math.sqrt(sum((x - mean) ** 2 for x in seen) / len(seen)))
            else:
                result.append(None)
        return result

    def isin(self, column: str, values: Sequence[Any]) -> list[bool]:
        """Return a boolean list: True where *column* value is in *values*."""
        self._require_column(column)
        value_set = set(values)
        return [row.get(column) in value_set for row in self._rows]

    def between(self, column: str, left: Any, right: Any, inclusive: str = "both") -> list[bool | None]:
        """Return a boolean list: True where *column* value is between *left* and *right*."""
        self._require_column(column)
        result: list[bool | None] = []
        for row in self._rows:
            v = row.get(column)
            if v is None:
                result.append(None)
                continue
            try:
                fv = float(v)
                if inclusive == "both":
                    result.append(float(left) <= fv <= float(right))
                elif inclusive == "left":
                    result.append(float(left) <= fv < float(right))
                elif inclusive == "right":
                    result.append(float(left) < fv <= float(right))
                else:
                    result.append(float(left) < fv < float(right))
            except (TypeError, ValueError):
                result.append(None)
        return result

    def transform(self, column: str, func: Any, *, new_column: str | None = None) -> "GeoPromptFrame":
        """Apply *func* element-wise to *column*, returning the same-length frame.

        Unlike :meth:`map_column`, this method always returns a full frame with
        the transformed values replacing (or extending) the column.
        """
        return self.map_column(column, func, new_column=new_column)

    def eval(self, expression: str) -> list[Any]:
        """Evaluate a string *expression* against the frame columns.

        Supports simple arithmetic and comparison expressions referencing
        column names as Python identifiers.  Each row is evaluated separately.

        Example::

            frame.eval("score * 2 + offset")
        """
        results: list[Any] = []
        for row in self._rows:
            namespace: dict[str, Any] = {k: v for k, v in row.items() if isinstance(k, str)}
            try:
                results.append(evaluate_safe_expression(expression, namespace))
            except (ExpressionValidationError, ExpressionExecutionError):
                results.append(None)
        return results

    def iterrows(self):
        """Iterate over rows as ``(index, row_dict)`` tuples."""
        for i, row in enumerate(self._rows):
            yield i, dict(row)

    def itertuples(self, index: bool = True, name: str | None = "Row"):
        """Iterate over rows as named tuples (or plain tuples when *name* is None)."""
        from collections import namedtuple

        cols = self.columns
        if name:
            try:
                nt = namedtuple(name, (["Index"] + cols) if index else cols)  # type: ignore[misc]
            except ValueError:
                nt = None
        else:
            nt = None

        for i, row in enumerate(self._rows):
            vals = [row.get(c) for c in cols]
            if index:
                vals = [i] + vals
            if nt is not None:
                try:
                    yield nt(*vals)
                    continue
                except (TypeError, ValueError, AttributeError):
                    pass
            yield tuple(vals)

    def to_dict(self, orient: str = "records") -> Any:
        """Convert the frame to a dict representation.

        *orient* mirrors the pandas convention:

        - ``"records"`` — list of row dicts.
        - ``"list"`` — dict of column → list of values.
        - ``"dict"`` — dict of column → dict of index → value.
        - ``"index"`` — dict of index → row dict.
        """
        if orient == "records":
            return [dict(row) for row in self._rows]
        cols = self.columns
        if orient == "list":
            return {c: [row.get(c) for row in self._rows] for c in cols}
        if orient == "dict":
            return {c: {i: row.get(c) for i, row in enumerate(self._rows)} for c in cols}
        if orient == "index":
            return {i: dict(row) for i, row in enumerate(self._rows)}
        raise ValueError(f"unsupported orient: {orient!r}")

    def to_csv(self, path: str | None = None, *, sep: str = ",",
               include_geometry: bool = True, **kwargs: Any) -> str | None:
        """Serialize the frame to CSV format.

        When *path* is given the output is written to that file; otherwise the
        CSV string is returned.  Geometry columns are serialised as WKT.
        """
        import csv
        import io as _io

        cols = [c for c in self.columns if include_geometry or c != self.geometry_column]
        buf = _io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=cols, delimiter=sep,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in self._rows:
            serialized: Record = {}
            for c in cols:
                v = row.get(c)
                if c == self.geometry_column and isinstance(v, dict):
                    try:
                        from .geometry import geometry_wkt_write
                        v = geometry_wkt_write(v)
                    except (ImportError, AttributeError, ValueError):
                        v = str(v)
                serialized[c] = "" if v is None else v
            writer.writerow(serialized)
        result = buf.getvalue()
        if path:
            Path(path).write_text(result, encoding="utf-8")
            return None
        return result

    def to_excel(self, path: str, sheet_name: str = "Sheet1", **kwargs: Any) -> None:
        """Write the frame to an Excel file via *openpyxl*.

        Requires ``openpyxl`` to be installed.  Geometry is written as WKT.
        """
        import importlib as _il
        openpyxl = _il.import_module("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = sheet_name
        cols = self.columns
        ws.append(cols)
        for row in self._rows:
            row_vals = []
            for c in cols:
                v = row.get(c)
                if c == self.geometry_column and isinstance(v, dict):
                    try:
                        from .geometry import geometry_wkt_write
                        v = geometry_wkt_write(v)
                    except (ImportError, AttributeError, ValueError):
                        v = str(v)
                row_vals.append("" if v is None else v)
            ws.append(row_vals)
        wb.save(path)

    def to_latex(self, columns: list[str] | None = None, **kwargs: Any) -> str:
        """Return a LaTeX tabular representation of the frame."""
        cols = columns or [c for c in self.columns if c != self.geometry_column]
        header = " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\"
        lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline", header, "\\hline"]
        for row in self._rows:
            line = " & ".join(str(row.get(c, "")).replace("_", "\\_") for c in cols) + " \\\\"
            lines.append(line)
        lines += ["\\hline", "\\end{tabular}"]
        return "\n".join(lines)

    def to_html(self, columns: list[str] | None = None, **kwargs: Any) -> str:
        """Return an HTML table representation of the frame."""
        import html as _html
        cols = columns or [c for c in self.columns if c != self.geometry_column]
        rows_html = ["<table border='1'>",
                     "<thead><tr>" + "".join(f"<th>{_html.escape(c)}</th>" for c in cols) + "</tr></thead>",
                     "<tbody>"]
        for row in self._rows:
            cells = "".join(f"<td>{_html.escape(str(row.get(c, '')))}</td>" for c in cols)
            rows_html.append(f"<tr>{cells}</tr>")
        rows_html += ["</tbody></table>"]
        return "\n".join(rows_html)

    def to_clipboard(self, sep: str = "\t", **kwargs: Any) -> None:
        """Copy the frame to the system clipboard as TSV."""
        import subprocess
        text = self.to_csv(sep=sep, include_geometry=False) or ""
        try:
            subprocess.run(["clip"], input=text.encode("utf-8"), check=True)  # noqa: S603,S607
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            try:
                import pyperclip  # type: ignore[import]
                pyperclip.copy(text)
            except (TypeError, ValueError, AttributeError):
                pass  # silently skip if clipboard not available

    def to_sql(self, table_name: str, connection: Any, if_exists: str = "fail",
               index: bool = False, **kwargs: Any) -> None:
        """Write the frame to a SQL database table.

        Geometry is serialised as WKT text.  Requires a DB-API 2.0 compatible
        *connection* object (e.g. ``sqlite3``, ``psycopg2``).
        """
        cols = self.columns
        col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
        cursor = connection.cursor()
        if if_exists == "replace":
            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        if if_exists in {"replace", "fail"}:
            cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')
        placeholders = ", ".join("?" for _ in cols)
        for row in self._rows:
            vals = []
            for c in cols:
                v = row.get(c)
                if c == self.geometry_column and isinstance(v, dict):
                    try:
                        from .geometry import geometry_wkt_write
                        v = geometry_wkt_write(v)
                    except (ImportError, ValueError, TypeError, AttributeError):
                        v = str(v)
                vals.append(None if v is None else str(v))
            cursor.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders})', vals)  # noqa: S608
        connection.commit()

    def to_file(self, path: str | Path, **kwargs: Any) -> Path:
        """Write this frame using extension-based format auto-detection."""
        from .io import write_data

        return write_data(path, self, **kwargs)

    def memory_usage(self, deep: bool = False) -> dict[str, int]:
        """Return an estimated memory usage in bytes for each column."""
        import sys
        result: dict[str, int] = {}
        for col in self.columns:
            total = 0
            for row in self._rows:
                v = row.get(col)
                total += sys.getsizeof(v)
            result[col] = total
        return result

    def info(self) -> str:
        """Return a summary string of frame dtypes, non-null counts, and memory."""
        import sys
        lines = [
            f"GeoPromptFrame: {len(self._rows)} rows × {len(self.columns)} columns",
            f"Geometry column: {self.geometry_column!r}",
            f"CRS: {self.crs or 'unknown'}",
            "",
            f"{'Column':<30} {'Non-Null':>10} {'Dtype':<15} {'Memory':>10}",
            "-" * 70,
        ]
        for col in self.columns:
            values = [row.get(col) for row in self._rows]
            non_null = sum(1 for v in values if v is not None)
            sample = next((v for v in values if v is not None), None)
            dtype = type(sample).__name__ if sample is not None else "object"
            mem = sum(sys.getsizeof(v) for v in values)
            lines.append(f"{col:<30} {non_null:>10} {dtype:<15} {mem:>10}")
        return "\n".join(lines)

    def nunique(self, column: str | None = None) -> int | dict[str, int]:
        """Return the number of unique non-null values per column.

        If *column* is given, returns a single count; otherwise returns a dict.
        """
        if column is not None:
            self._require_column(column)
            return len({row.get(column) for row in self._rows if row.get(column) is not None})
        return {c: len({row.get(c) for row in self._rows if row.get(c) is not None}) for c in self.columns}

    def idxmin(self, column: str) -> int | None:
        """Return the integer index of the minimum value in *column*."""
        self._require_column(column)
        best_i: int | None = None
        best_v: float | None = None
        for i, row in enumerate(self._rows):
            v = row.get(column)
            if v is None:
                continue
            fv = float(v)
            if best_v is None or fv < best_v:
                best_v = fv
                best_i = i
        return best_i

    def idxmax(self, column: str) -> int | None:
        """Return the integer index of the maximum value in *column*."""
        self._require_column(column)
        best_i: int | None = None
        best_v: float | None = None
        for i, row in enumerate(self._rows):
            v = row.get(column)
            if v is None:
                continue
            fv = float(v)
            if best_v is None or fv > best_v:
                best_v = fv
                best_i = i
        return best_i

    def mode(self, column: str) -> list[Any]:
        """Return the most frequent non-null values in *column*."""
        self._require_column(column)
        from collections import Counter
        counts = Counter(row.get(column) for row in self._rows if row.get(column) is not None)
        if not counts:
            return []
        max_count = max(counts.values())
        return [v for v, c in counts.items() if c == max_count]

    def sem(self, column: str) -> float:
        """Return the standard error of the mean for *column*."""
        import math as _math
        vals = [float(row[column]) for row in self._rows if row.get(column) is not None]
        if len(vals) < 2:
            return float("nan")
        mean = sum(vals) / len(vals)
        std = _math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
        return std / _math.sqrt(len(vals))

    def skew(self, column: str) -> float:
        """Return the skewness of *column* values."""
        import math as _math
        vals = [float(row[column]) for row in self._rows if row.get(column) is not None]
        n = len(vals)
        if n < 3:
            return float("nan")
        mean = sum(vals) / n
        std = _math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
        if std == 0:
            return float("nan")
        return (sum((v - mean) ** 3 for v in vals) / n) / (std ** 3)

    def kurtosis(self, column: str) -> float:
        """Return the excess kurtosis of *column* values."""
        import math as _math
        vals = [float(row[column]) for row in self._rows if row.get(column) is not None]
        n = len(vals)
        if n < 4:
            return float("nan")
        mean = sum(vals) / n
        std = _math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
        if std == 0:
            return float("nan")
        return (sum((v - mean) ** 4 for v in vals) / n) / (std ** 4) - 3.0

    def rank(self, column: str, ascending: bool = True, method: str = "average") -> list[float | None]:
        """Return rank of values in *column*.

        *method* controls tie-breaking: ``"average"``, ``"min"``, ``"max"``,
        ``"first"``, or ``"dense"``.
        """
        vals = [(i, row.get(column)) for i, row in enumerate(self._rows)]
        non_null = [(i, v) for i, v in vals if v is not None]
        sorted_vals = sorted(non_null, key=lambda x: float(x[1]), reverse=not ascending)  # type: ignore[arg-type]

        rank_map: dict[int, float] = {}
        i = 0
        while i < len(sorted_vals):
            j = i
            val = sorted_vals[i][1]
            while j < len(sorted_vals) and sorted_vals[j][1] == val:
                j += 1
            group = sorted_vals[i:j]
            group_rank_start = i + 1
            group_rank_end = j
            for k, (idx, _) in enumerate(group):
                if method == "average":
                    rank_map[idx] = (group_rank_start + group_rank_end) / 2
                elif method == "min":
                    rank_map[idx] = float(group_rank_start)
                elif method == "max":
                    rank_map[idx] = float(group_rank_end)
                elif method == "first":
                    rank_map[idx] = float(group_rank_start + k)
                elif method == "dense":
                    rank_map[idx] = float(i + 1)
            i = j
        return [rank_map.get(i) for i in range(len(self._rows))]

    def combine_first(self, other: "GeoPromptFrame") -> "GeoPromptFrame":
        """Update null values from *other*, combining two frames row-by-row.

        Rows are matched by position.  *other* must have the same number of rows.
        """
        if len(other) != len(self):
            raise ValueError("frames must have the same number of rows to combine_first")
        rows: list[Record] = []
        for self_row, other_row in zip(self._rows, other._rows):
            merged: Record = dict(other_row)
            merged.update({k: v for k, v in self_row.items() if v is not None})
            rows.append(merged)
        return self._clone_with_rows(rows)

    def compare(self, other: "GeoPromptFrame", columns: list[str] | None = None) -> list[dict[str, Any]]:
        """Return a list of diffs between this frame and *other*.

        Each entry in the result describes a cell that differs, with keys
        ``row``, ``column``, ``self``, and ``other``.
        """
        if len(other) != len(self):
            raise ValueError("frames must have the same number of rows to compare")
        cols = columns or [c for c in self.columns if c != self.geometry_column]
        diffs: list[dict[str, Any]] = []
        for i, (self_row, other_row) in enumerate(zip(self._rows, other._rows)):
            for col in cols:
                sv = self_row.get(col)
                ov = other_row.get(col)
                if sv != ov:
                    diffs.append({"row": i, "column": col, "self": sv, "other": ov})
        return diffs

    def update(self, other: "GeoPromptFrame", overwrite: bool = True) -> "GeoPromptFrame":
        """Update values in this frame from *other* at matching positions.

        When *overwrite* is True (default) all values from *other* overwrite
        this frame's values.  When False, only null values are replaced.
        """
        if len(other) != len(self):
            raise ValueError("frames must have the same number of rows to update")
        rows: list[Record] = []
        for self_row, other_row in zip(self._rows, other._rows):
            merged = dict(self_row)
            for k, v in other_row.items():
                if overwrite or merged.get(k) is None:
                    merged[k] = v
            rows.append(merged)
        return self._clone_with_rows(rows)

    # --- Spatial-specific additions ---

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        """Return ``(minx, miny, maxx, maxy)`` for the entire frame."""
        if not self._rows:
            return (0.0, 0.0, 0.0, 0.0)
        xmins, ymins, xmaxs, ymaxs = [], [], [], []
        for row in self._rows:
            b = geometry_bounds(row[self.geometry_column])
            xmins.append(b[0])
            ymins.append(b[1])
            xmaxs.append(b[2])
            ymaxs.append(b[3])
        return (min(xmins), min(ymins), max(xmaxs), max(ymaxs))

    @property
    def area(self) -> list[float]:
        """Return row-wise geometry areas directly on the frame."""
        return [geometry_area(row[self.geometry_column]) for row in self._rows]

    @property
    def length(self) -> list[float]:
        """Return row-wise geometry lengths directly on the frame."""
        return [geometry_length(row[self.geometry_column]) for row in self._rows]

    @property
    def geom_type(self) -> list[str]:
        """Return a list of geometry type strings for each row."""
        return [geometry_type(row[self.geometry_column]) for row in self._rows]

    @property
    def is_valid(self) -> list[bool]:
        """Return a list of booleans indicating geometry validity per row."""
        return [bool(validate_geometry(row[self.geometry_column]).get("is_valid", False)) for row in self._rows]

    @property
    def is_empty(self) -> list[bool]:
        """Return a list of booleans indicating whether each geometry is empty."""
        from .geometry import geometry_is_empty
        return [geometry_is_empty(row[self.geometry_column]) for row in self._rows]

    @property
    def active_geometry_name(self) -> str:
        """Return the name of the active geometry column."""
        return self.geometry_column

    def rename_geometry(self, new_name: str) -> "GeoPromptFrame":
        """Return a new frame with the geometry column renamed to *new_name*."""
        rows = [{new_name if k == self.geometry_column else k: v
                 for k, v in row.items()} for row in self._rows]
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=new_name, crs=self.crs)

    @property
    def __geo_interface__(self) -> dict[str, Any]:
        """Return a GeoJSON FeatureCollection dict for this frame."""
        features = [
            {
                "type": "Feature",
                "geometry": dict(row[self.geometry_column]),
                "properties": {k: v for k, v in row.items() if k != self.geometry_column},
            }
            for row in self._rows
        ]
        return {"type": "FeatureCollection", "features": features}

    def iterfeatures(self):
        """Yield each row as a GeoJSON-like Feature dict."""
        for row in self._rows:
            yield {
                "type": "Feature",
                "geometry": dict(row[self.geometry_column]),
                "properties": {k: v for k, v in row.items() if k != self.geometry_column},
            }

    def estimate_utm_crs(self) -> str:
        """Return the EPSG code of the best UTM zone for this frame's extent.

        Derived from the centroid longitude of :attr:`total_bounds`.
        """
        minx, miny, maxx, maxy = self.total_bounds
        lon = (minx + maxx) / 2
        lat = (miny + maxy) / 2
        zone = int((lon + 180) / 6) + 1
        if lat >= 0:
            return f"EPSG:{32600 + zone}"
        return f"EPSG:{32700 + zone}"

    def explore(self, column: str | None = None, **kwargs: Any) -> Any:
        """Return an interactive folium map for this frame.

        Requires ``folium`` to be installed.  The map is centred on
        :attr:`total_bounds` and renders each feature as a GeoJSON layer.
        *column* is used for choropleth-style fill if provided.
        """
        import importlib as _il
        folium = _il.import_module("folium")
        minx, miny, maxx, maxy = self.total_bounds
        center = [(miny + maxy) / 2, (minx + maxx) / 2]
        m = folium.Map(location=center, zoom_start=kwargs.pop("zoom_start", 10))
        geojson_data = self.__geo_interface__
        if column and column in self.columns:
            values = [row.get(column) for row in self._rows]
            numeric = [float(v) for v in values if v is not None and isinstance(v, (int, float))]
            if numeric:
                vmin, vmax = min(numeric), max(numeric)
                def _style(feature: Any) -> dict[str, Any]:
                    val = feature.get("properties", {}).get(column)
                    if val is None or vmax == vmin:
                        return {"fillColor": "#3388ff", "fillOpacity": 0.5, "weight": 1}
                    norm = (float(val) - vmin) / (vmax - vmin)
                    r = int(255 * norm)
                    b = int(255 * (1 - norm))
                    return {"fillColor": f"#{r:02x}00{b:02x}", "fillOpacity": 0.6, "weight": 1}
                folium.GeoJson(geojson_data, style_function=_style).add_to(m)
                return m
        folium.GeoJson(geojson_data).add_to(m)
        return m

    def plot(self, column: str | None = None, ax: Any = None, color: str = "#3388ff", **kwargs: Any) -> Any:
        """Render a lightweight matplotlib plot for the frame geometries."""
        matplotlib_pyplot = importlib.import_module("matplotlib.pyplot")
        axes = ax if ax is not None else matplotlib_pyplot.subplots()[1]
        for row in self._rows:
            geometry = row[self.geometry_column]
            kind = geometry_type(geometry)
            coords = geometry.get("coordinates")
            if kind == "Point":
                axes.scatter([coords[0]], [coords[1]], color=color, **kwargs)
            elif kind == "LineString":
                xs = [coord[0] for coord in coords]
                ys = [coord[1] for coord in coords]
                axes.plot(xs, ys, color=color, **kwargs)
            elif kind == "Polygon":
                ring = coords[0] if coords and isinstance(coords[0][0], (tuple, list)) else coords
                xs = [coord[0] for coord in ring]
                ys = [coord[1] for coord in ring]
                axes.fill(xs, ys, facecolor=color, alpha=0.35)
                axes.plot(xs, ys, color=color)
        if column and column in self.columns:
            axes.set_title(str(column))
        return axes


class GroupedGeoPromptFrame:
    """Grouped spatial frame rows with aggregation support preserving geometry."""

    def __init__(
        self,
        rows: Sequence[Record],
        group_columns: Sequence[str],
        geometry_column: str,
        crs: str | None,
    ) -> None:
        self._rows = list(rows)
        self._group_columns = list(group_columns)
        self._geometry_column = geometry_column
        self._crs = crs

    def _group_items(self) -> list[tuple[tuple[Any, ...], list[Record]]]:
        grouped: dict[tuple[Any, ...], list[Record]] = {}
        for row in self._rows:
            key = tuple(row.get(col) for col in self._group_columns)
            grouped.setdefault(key, []).append(dict(row))
        return list(grouped.items())

    def _group_frame(self, rows: Sequence[Record]) -> GeoPromptFrame:
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self._geometry_column, crs=self._crs)

    def agg(self, aggregations: dict[str, str | Sequence[str]]) -> "GeoPromptFrame":
        """Aggregate each group and return a new :class:`GeoPromptFrame`.

        The geometry of each group is taken from the first row in the group.

        Args:
            aggregations: Dict mapping column names to aggregation operation(s).

        Returns:
            New :class:`GeoPromptFrame` with one row per group.
        """
        from .table import _apply_aggregation

        result_rows: list[Record] = []
        for key, group_rows in self._group_items():
            summary: Record = {col: val for col, val in zip(self._group_columns, key)}
            summary["row_count"] = len(group_rows)
            summary[self._geometry_column] = group_rows[0][self._geometry_column]
            for col, ops in aggregations.items():
                op_list = [ops] if isinstance(ops, str) else list(ops)
                values = [row.get(col) for row in group_rows if col in row]
                for op in op_list:
                    summary[f"{col}_{op}"] = _apply_aggregation(values, op)
            result_rows.append(summary)

        return GeoPromptFrame._from_internal_rows(
            result_rows, geometry_column=self._geometry_column, crs=self._crs,
        )

    def apply(self, func: Callable[[GeoPromptFrame], Any]) -> "GeoPromptFrame":
        """Apply a callable to each group and combine results into a frame."""
        result_rows: list[Record] = []
        for key, group_rows in self._group_items():
            group_frame = self._group_frame(group_rows)
            result = func(group_frame)
            if isinstance(result, GeoPromptFrame):
                result_rows.extend(result.to_records())
                continue
            if isinstance(result, dict):
                row = {col: val for col, val in zip(self._group_columns, key)}
                row.update(result)
                row.setdefault(self._geometry_column, group_rows[0].get(self._geometry_column))
                result_rows.append(row)
                continue
            if isinstance(result, list) and all(isinstance(item, dict) for item in result):
                result_rows.extend(dict(item) for item in result)
                continue
            row = {col: val for col, val in zip(self._group_columns, key)}
            row["result"] = result
            row[self._geometry_column] = group_rows[0].get(self._geometry_column)
            result_rows.append(row)
        return GeoPromptFrame._from_internal_rows(result_rows, geometry_column=self._geometry_column, crs=self._crs)

    def transform(
        self,
        column: str | Callable[[list[Any]], Any],
        func: Callable[[list[Any]], Any] | None = None,
        *,
        new_column: str | None = None,
    ) -> "GeoPromptFrame":
        """Transform grouped values and return a same-length frame."""
        target_column: str | None
        transform_func: Callable[[list[Any]], Any]
        if callable(column) and func is None:
            target_column = None
            transform_func = column
        else:
            target_column = str(column)
            transform_func = func if func is not None else (lambda values: values)

        rows: list[Record] = []
        output_column = new_column or (target_column if target_column is not None else "transform")
        for _, group_rows in self._group_items():
            values = group_rows if target_column is None else [row.get(target_column) for row in group_rows]
            transformed = transform_func(values)
            if isinstance(transformed, Sequence) and not isinstance(transformed, (str, bytes, dict)):
                transformed_values = list(transformed)
                if len(transformed_values) != len(group_rows):
                    raise ValueError("transform must return a scalar or a sequence matching group length")
            else:
                transformed_values = [transformed] * len(group_rows)
            for row, value in zip(group_rows, transformed_values):
                new_row = dict(row)
                new_row[output_column] = value
                rows.append(new_row)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self._geometry_column, crs=self._crs)

    def filter(self, func: Callable[[GeoPromptFrame], bool]) -> "GeoPromptFrame":
        """Keep only groups whose frame satisfies the predicate."""
        rows: list[Record] = []
        for _, group_rows in self._group_items():
            if bool(func(self._group_frame(group_rows))):
                rows.extend(dict(row) for row in group_rows)
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self._geometry_column, crs=self._crs)

    def first(self) -> "GeoPromptFrame":
        """Return the first row from each group."""
        rows = [dict(group_rows[0]) for _, group_rows in self._group_items() if group_rows]
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self._geometry_column, crs=self._crs)

    def last(self) -> "GeoPromptFrame":
        """Return the last row from each group."""
        rows = [dict(group_rows[-1]) for _, group_rows in self._group_items() if group_rows]
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self._geometry_column, crs=self._crs)

    def nth(self, n: int) -> "GeoPromptFrame":
        """Return the nth row from each group when present."""
        rows: list[Record] = []
        for _, group_rows in self._group_items():
            index = n if n >= 0 else len(group_rows) + n
            if 0 <= index < len(group_rows):
                rows.append(dict(group_rows[index]))
        return GeoPromptFrame._from_internal_rows(rows, geometry_column=self._geometry_column, crs=self._crs)

    def cumcount(self) -> list[int]:
        """Return per-row cumulative counts within each group."""
        counts: dict[tuple[Any, ...], int] = {}
        results: list[int] = []
        for row in self._rows:
            key = tuple(row.get(col) for col in self._group_columns)
            current = counts.get(key, 0)
            results.append(current)
            counts[key] = current + 1
        return results

    def ngroup(self) -> list[int]:
        """Return per-row dense group ids in original row order."""
        labels: dict[tuple[Any, ...], int] = {}
        results: list[int] = []
        for row in self._rows:
            key = tuple(row.get(col) for col in self._group_columns)
            if key not in labels:
                labels[key] = len(labels)
            results.append(labels[key])
        return results


geopromptframe = GeoPromptFrame
geopromptspatialindex = GeoPromptSpatialIndex
groupedgeopromptframe = GroupedGeoPromptFrame


__all__ = [
    "Bounds",
    "GeoPromptFrame",
    "GeoPromptGeometryAccessor",
    "GeoPromptSpatialIndex",
    "GroupedGeoPromptFrame",
    "geopromptframe",
    "geopromptspatialindex",
    "groupedgeopromptframe",
]