"""Table query engine: cursors, SQL-like expressions, field calculations, and data management.

Covers search/insert/update/delete cursors, SQL expression parsing,
where-clause building, select-by-attributes, select-by-location,
frequency distribution, and data management helpers.
"""
from __future__ import annotations

import json
import math
import operator
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .safe_expression import evaluate_safe_expression

Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Expression engine  (item 1074)
# ---------------------------------------------------------------------------

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    "<>": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

_EXPR_PATTERN = re.compile(
    r"""^\s*(\w+)\s*(==|!=|<>|<=|>=|<|>|=)\s*(?:'([^']*)'|"([^"]*)"|(\S+))\s*$""",
    re.IGNORECASE,
)

_LIKE_PATTERN = re.compile(r"^\s*(\w+)\s+LIKE\s+'(.*)'\s*$", re.IGNORECASE)
_IN_PATTERN = re.compile(r"^\s*(\w+)\s+IN\s*\((.+)\)\s*$", re.IGNORECASE)
_IS_NULL_PATTERN = re.compile(r"^\s*(\w+)\s+IS\s+NULL\s*$", re.IGNORECASE)
_IS_NOT_NULL_PATTERN = re.compile(r"^\s*(\w+)\s+IS\s+NOT\s+NULL\s*$", re.IGNORECASE)
_BETWEEN_PATTERN = re.compile(
    r"^\s*(\w+)\s+BETWEEN\s+(\S+)\s+AND\s+(\S+)\s*$", re.IGNORECASE
)


def _coerce_value(text: str) -> Any:
    """Coerce a string token to int/float/bool/None or leave as str."""
    if text.lower() == "null" or text.lower() == "none":
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def parse_expression(expr: str) -> Callable[[Record], bool]:
    """Parse a simple SQL-like expression into a row predicate.

    Supports: field = value, field != value, field < value, field LIKE 'pattern',
    field IN (a, b, c), field IS NULL, field IS NOT NULL, field BETWEEN a AND b.
    """
    # IS NOT NULL
    m = _IS_NOT_NULL_PATTERN.match(expr)
    if m:
        field = m.group(1)
        return lambda row, f=field: row.get(f) is not None

    # IS NULL
    m = _IS_NULL_PATTERN.match(expr)
    if m:
        field = m.group(1)
        return lambda row, f=field: row.get(f) is None

    # BETWEEN
    m = _BETWEEN_PATTERN.match(expr)
    if m:
        field = m.group(1)
        lo = _coerce_value(m.group(2))
        hi = _coerce_value(m.group(3))
        return lambda row, f=field, l=lo, h=hi: l <= (row.get(f) or 0) <= h

    # IN
    m = _IN_PATTERN.match(expr)
    if m:
        field = m.group(1)
        vals = {_coerce_value(v.strip().strip("'\"")) for v in m.group(2).split(",")}
        return lambda row, f=field, v=vals: row.get(f) in v

    # LIKE
    m = _LIKE_PATTERN.match(expr)
    if m:
        field = m.group(1)
        pattern = m.group(2).replace("%", ".*").replace("_", ".")
        regex = re.compile(f"^{pattern}$", re.IGNORECASE)
        return lambda row, f=field, r=regex: bool(r.match(str(row.get(f, ""))))

    # Comparison operators
    m = _EXPR_PATTERN.match(expr)
    if m:
        field = m.group(1)
        op_str = m.group(2)
        raw_val = m.group(3) if m.group(3) is not None else (m.group(4) if m.group(4) is not None else m.group(5))
        value = _coerce_value(raw_val)
        op_fn = _OPS.get(op_str, operator.eq)
        return lambda row, f=field, v=value, fn=op_fn: fn(row.get(f), v)

    raise ValueError(f"cannot parse expression: {expr!r}")


def where_clause(expressions: Sequence[str], combine: str = "AND") -> Callable[[Record], bool]:
    """Build a composite predicate from multiple SQL-like expressions.

    *combine* can be 'AND' or 'OR'.
    """
    predicates = [parse_expression(e) for e in expressions]
    if combine.upper() == "OR":
        return lambda row: any(p(row) for p in predicates)
    return lambda row: all(p(row) for p in predicates)


# ---------------------------------------------------------------------------
# Where-clause builder  (item 1075)
# ---------------------------------------------------------------------------

def build_where_clause(
    *,
    field: str | None = None,
    op: str = "=",
    value: Any = None,
    and_clauses: Sequence[str] | None = None,
    or_clauses: Sequence[str] | None = None,
) -> str:
    """Programmatically build a SQL WHERE clause string."""
    parts: list[str] = []
    if field is not None and value is not None:
        if isinstance(value, str):
            parts.append(f"{field} {op} '{value}'")
        else:
            parts.append(f"{field} {op} {value}")
    if and_clauses:
        parts.extend(and_clauses)
    clause = " AND ".join(parts) if parts else "1=1"
    if or_clauses:
        or_part = " OR ".join(or_clauses)
        clause = f"({clause}) OR ({or_part})"
    return clause


# ---------------------------------------------------------------------------
# Search cursor  (items 1061, 1068)
# ---------------------------------------------------------------------------

class SearchCursor:
    """Read-only cursor that iterates over rows matching an optional where clause.

    Mimics ArcPy's da.SearchCursor.
    """

    def __init__(
        self,
        rows: Sequence[Record],
        fields: Sequence[str] | None = None,
        where: str | None = None,
        order_by: str | None = None,
        spatial_filter: dict[str, Any] | None = None,
    ) -> None:
        self._rows = list(rows)
        self._fields = list(fields) if fields else None
        self._predicate = parse_expression(where) if where else None
        self._spatial_filter = spatial_filter
        self._order_by = order_by

    def __iter__(self) -> Iterator[Record]:
        rows = self._rows
        if self._predicate:
            rows = [r for r in rows if self._predicate(r)]

        if self._spatial_filter:
            bbox = self._spatial_filter.get("bbox")
            if bbox:
                min_x, min_y, max_x, max_y = bbox
                filtered: list[Record] = []
                for r in rows:
                    geom = r.get("geometry")
                    if geom and "coordinates" in geom:
                        coords = geom["coordinates"]
                        if isinstance(coords[0], (int, float)):
                            if min_x <= coords[0] <= max_x and min_y <= coords[1] <= max_y:
                                filtered.append(r)
                        else:
                            filtered.append(r)  # non-point — include
                    else:
                        filtered.append(r)
                rows = filtered

        if self._order_by:
            desc = self._order_by.startswith("-")
            field = self._order_by.lstrip("-")
            rows = sorted(rows, key=lambda r: (r.get(field) is None, r.get(field)), reverse=desc)

        for row in rows:
            if self._fields:
                yield {f: row.get(f) for f in self._fields}
            else:
                yield dict(row)

    def __enter__(self) -> "SearchCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Insert cursor  (items 1062, 1069)
# ---------------------------------------------------------------------------

class InsertCursor:
    """Cursor that collects new rows for batch insertion.

    Mimics ArcPy's da.InsertCursor.
    """

    def __init__(self, target: list[Record], fields: Sequence[str] | None = None) -> None:
        self._target = target
        self._fields = list(fields) if fields else None
        self._count = 0

    def insertRow(self, values: Sequence[Any] | Record) -> None:  # noqa: N802
        if isinstance(values, dict):
            self._target.append(dict(values))
        elif self._fields:
            self._target.append(dict(zip(self._fields, values)))
        else:
            raise ValueError("must supply fields or pass dict values")
        self._count += 1

    @property
    def insert_count(self) -> int:
        return self._count

    def __enter__(self) -> "InsertCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Update cursor  (items 1063, 1070)
# ---------------------------------------------------------------------------

class UpdateCursor:
    """Cursor that iterates rows and allows in-place mutation/deletion.

    Mimics ArcPy's da.UpdateCursor.
    """

    def __init__(
        self,
        rows: list[Record],
        fields: Sequence[str] | None = None,
        where: str | None = None,
    ) -> None:
        self._rows = rows
        self._fields = list(fields) if fields else None
        self._predicate = parse_expression(where) if where else None
        self._idx = 0
        self._pending_delete: set[int] = set()

    def __iter__(self) -> Iterator[Record]:
        for i, row in enumerate(self._rows):
            if self._predicate and not self._predicate(row):
                continue
            self._idx = i
            if self._fields:
                yield {f: row.get(f) for f in self._fields}
            else:
                yield dict(row)

    def updateRow(self, values: Record) -> None:  # noqa: N802
        self._rows[self._idx].update(values)

    def deleteRow(self) -> None:  # noqa: N802
        self._pending_delete.add(self._idx)

    def flush(self) -> int:
        """Apply pending deletes. Returns count of deleted rows."""
        if not self._pending_delete:
            return 0
        count = len(self._pending_delete)
        self._rows[:] = [r for i, r in enumerate(self._rows) if i not in self._pending_delete]
        self._pending_delete.clear()
        return count

    def __enter__(self) -> "UpdateCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.flush()


# ---------------------------------------------------------------------------
# Delete cursor  (item 1064)
# ---------------------------------------------------------------------------

def delete_rows(rows: list[Record], where: str) -> int:
    """Delete rows matching a where-clause expression. Returns count deleted."""
    pred = parse_expression(where)
    original = len(rows)
    rows[:] = [r for r in rows if not pred(r)]
    return original - len(rows)


# ---------------------------------------------------------------------------
# Select by attributes  (item 978)
# ---------------------------------------------------------------------------

def select_by_attributes(
    rows: Sequence[Record],
    expression: str,
    mode: str = "new",
    existing_selection: set[int] | None = None,
) -> set[int]:
    """Select row indices by a SQL-like attribute expression.

    *mode*: 'new', 'add', 'remove', 'subset'.
    Returns a set of selected row indices.
    """
    pred = parse_expression(expression)
    matched = {i for i, r in enumerate(rows) if pred(r)}

    if existing_selection is None:
        existing_selection = set()

    if mode == "new":
        return matched
    elif mode == "add":
        return existing_selection | matched
    elif mode == "remove":
        return existing_selection - matched
    elif mode == "subset":
        return existing_selection & matched
    else:
        raise ValueError(f"unknown select mode: {mode}")


# ---------------------------------------------------------------------------
# Select by location  (item 979)
# ---------------------------------------------------------------------------

def select_by_location(
    rows: Sequence[Record],
    bbox: tuple[float, float, float, float],
    geometry_field: str = "geometry",
) -> set[int]:
    """Select row indices where the geometry intersects a bounding box."""
    min_x, min_y, max_x, max_y = bbox
    selected: set[int] = set()
    for i, row in enumerate(rows):
        geom = row.get(geometry_field)
        if not geom or "coordinates" not in geom:
            continue
        coords = geom["coordinates"]
        if isinstance(coords[0], (int, float)):
            if min_x <= coords[0] <= max_x and min_y <= coords[1] <= max_y:
                selected.add(i)
        else:
            # For non-point geometries, check any coordinate
            for c in _iter_all_coords(coords):
                if min_x <= c[0] <= max_x and min_y <= c[1] <= max_y:
                    selected.add(i)
                    break
    return selected


def _iter_all_coords(coords: Any) -> Iterator[list[float]]:
    if not coords:
        return
    if isinstance(coords[0], (int, float)):
        yield coords
    else:
        for c in coords:
            yield from _iter_all_coords(c)


# ---------------------------------------------------------------------------
# Switch / clear selection  (items 981, 982)
# ---------------------------------------------------------------------------

def switch_selection(rows: Sequence[Record], selection: set[int]) -> set[int]:
    """Invert a selection."""
    return set(range(len(rows))) - selection


def clear_selection() -> set[int]:
    """Return an empty selection set."""
    return set()


# ---------------------------------------------------------------------------
# Field calculation  (item 956)
# ---------------------------------------------------------------------------

def calculate_field(
    rows: list[Record],
    field: str,
    expression: str | Callable[[Record], Any],
) -> int:
    """Calculate/update a field for all rows.

    *expression* can be a callable(row) → value, or a simple string expression
    like "!field1! + !field2!" (ArcPy-style) or "field1 * 2".

    Returns the count of updated rows.
    """
    if callable(expression):
        func = expression
    else:
        # Parse ArcPy-style !field! references
        text = re.sub(r"!(\w+)!", r"\1", str(expression))

        def _expr(row: Record, _t: str = text) -> Any:
            namespace = dict(row)
            namespace["math"] = math
            return evaluate_safe_expression(_t, namespace, allowed_attribute_roots={"math"})

        func = _expr

    count = 0
    for row in rows:
        row[field] = func(row)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Alter field  (item 954)
# ---------------------------------------------------------------------------

def alter_field(
    rows: list[Record],
    field: str,
    *,
    new_name: str | None = None,
    new_type: str | None = None,
    alias: str | None = None,
) -> list[Record]:
    """Rename a field and/or change its type across all rows.

    Returns the modified rows list.
    """
    type_map: dict[str, Callable[[Any], Any]] = {
        "string": str,
        "integer": lambda v: int(float(v)) if v is not None else None,
        "float": lambda v: float(v) if v is not None else None,
        "boolean": lambda v: bool(v) if v is not None else None,
    }

    for row in rows:
        if field not in row:
            continue
        value = row[field]
        if new_type and new_type in type_map and value is not None:
            value = type_map[new_type](value)
        if new_name and new_name != field:
            row[new_name] = value
            del row[field]
        else:
            row[field] = value

    return rows


# ---------------------------------------------------------------------------
# Frequency distribution  (item 1060)
# ---------------------------------------------------------------------------

def frequency_distribution(
    rows: Sequence[Record],
    fields: Sequence[str],
) -> list[Record]:
    """Compute frequency counts for unique value combinations.

    Returns a list of dicts with the field values plus a 'count' field.
    """
    from collections import Counter
    keys: list[tuple[Any, ...]] = []
    for row in rows:
        keys.append(tuple(row.get(f) for f in fields))

    counts = Counter(keys)
    result: list[Record] = []
    for combo, count in counts.most_common():
        entry: Record = dict(zip(fields, combo))
        entry["count"] = count
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Extent object  (item 1078)
# ---------------------------------------------------------------------------

class Extent:
    """Represents a spatial bounding box."""

    __slots__ = ("min_x", "min_y", "max_x", "max_y")

    def __init__(self, min_x: float, min_y: float, max_x: float, max_y: float) -> None:
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    @classmethod
    def from_features(cls, features: Sequence[Record], geometry_field: str = "geometry") -> "Extent":
        """Compute extent from a set of features."""
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for feat in features:
            geom = feat.get(geometry_field, {})
            coords = geom.get("coordinates", [])
            for c in _iter_all_coords(coords):
                min_x = min(min_x, c[0])
                max_x = max(max_x, c[0])
                min_y = min(min_y, c[1])
                max_y = max(max_y, c[1])
        return cls(min_x, min_y, max_x, max_y)

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> tuple[float, float]:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def intersects(self, other: "Extent") -> bool:
        return not (
            self.max_x < other.min_x or self.min_x > other.max_x
            or self.max_y < other.min_y or self.min_y > other.max_y
        )

    def expand(self, distance: float) -> "Extent":
        return Extent(
            self.min_x - distance, self.min_y - distance,
            self.max_x + distance, self.max_y + distance,
        )

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_x, self.min_y, self.max_x, self.max_y)

    def __repr__(self) -> str:
        return f"Extent({self.min_x}, {self.min_y}, {self.max_x}, {self.max_y})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Extent):
            return NotImplemented
        return self.to_tuple() == other.to_tuple()

    def __hash__(self) -> int:
        return hash(self.to_tuple())


# ---------------------------------------------------------------------------
# Change detection / spatial diff  (item 1008)
# ---------------------------------------------------------------------------

def spatial_diff(
    old_rows: Sequence[Record],
    new_rows: Sequence[Record],
    id_field: str = "id",
) -> dict[str, list[Record]]:
    """Compare two feature sets and classify changes.

    Returns dict with keys: added, removed, modified, unchanged.
    """
    old_map = {r.get(id_field): r for r in old_rows if r.get(id_field) is not None}
    new_map = {r.get(id_field): r for r in new_rows if r.get(id_field) is not None}

    added: list[Record] = []
    removed: list[Record] = []
    modified: list[Record] = []
    unchanged: list[Record] = []

    for k, v in new_map.items():
        if k not in old_map:
            added.append(v)
        elif v != old_map[k]:
            modified.append(v)
        else:
            unchanged.append(v)

    for k, v in old_map.items():
        if k not in new_map:
            removed.append(v)

    return {"added": added, "removed": removed, "modified": modified, "unchanged": unchanged}


# ---------------------------------------------------------------------------
# Bulk update  (item 1072)
# ---------------------------------------------------------------------------

def bulk_update(
    rows: list[Record],
    updates: dict[str, Any],
    where: str | None = None,
) -> int:
    """Update field values on all rows (or rows matching *where*). Returns count updated."""
    pred = parse_expression(where) if where else None
    count = 0
    for row in rows:
        if pred and not pred(row):
            continue
        row.update(updates)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Data dictionary export  (item 1099)
# ---------------------------------------------------------------------------

def data_dictionary(
    rows: Sequence[Record],
    geometry_field: str = "geometry",
) -> list[Record]:
    """Generate a data dictionary from feature records.

    Returns a list of field descriptors: name, type, nullable, sample_values.
    """
    if not rows:
        return []

    fields: dict[str, dict[str, Any]] = {}
    for row in rows:
        for k, v in row.items():
            if k == geometry_field:
                continue
            if k not in fields:
                fields[k] = {"name": k, "types": set(), "nullable": False, "samples": []}
            if v is None:
                fields[k]["nullable"] = True
            else:
                fields[k]["types"].add(type(v).__name__)
                if len(fields[k]["samples"]) < 3:
                    fields[k]["samples"].append(v)

    result: list[Record] = []
    for info in fields.values():
        types = sorted(info["types"])
        result.append({
            "name": info["name"],
            "type": types[0] if len(types) == 1 else "/".join(types),
            "nullable": info["nullable"],
            "sample_values": info["samples"],
        })
    return result


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    # Expression engine
    "parse_expression",
    "where_clause",
    "build_where_clause",
    # Cursors
    "SearchCursor",
    "InsertCursor",
    "UpdateCursor",
    "delete_rows",
    # Selection
    "select_by_attributes",
    "select_by_location",
    "switch_selection",
    "clear_selection",
    # Field ops
    "calculate_field",
    "alter_field",
    # Frequency
    "frequency_distribution",
    # Extent
    "Extent",
    # Change detection
    "spatial_diff",
    # Bulk
    "bulk_update",
    # Data dictionary
    "data_dictionary",
]
