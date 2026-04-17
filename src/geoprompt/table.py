from __future__ import annotations

import csv
import json
from collections import Counter
from numbers import Real
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence


Record = dict[str, Any]


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _infer_column_dtype(values: Sequence[Any]) -> str:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return "empty"
    if all(_is_numeric_value(value) for value in non_null):
        return "numeric"
    if all(isinstance(value, str) for value in non_null):
        return "string"
    return "mixed"


def _apply_aggregation(values: Sequence[Any], operation: str) -> Any:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None

    if operation == "count":
        return len(cleaned)
    if operation == "first":
        return cleaned[0]
    if operation == "unique_count":
        return len({json.dumps(value, sort_keys=True, default=str) for value in cleaned})

    if operation in {"sum", "mean", "median"}:
        numeric = [float(value) for value in cleaned if _is_numeric_value(value)]
        if len(numeric) != len(cleaned):
            raise ValueError(f"aggregation '{operation}' requires numeric values")
        if operation == "sum":
            return sum(numeric)
        if operation == "mean":
            return sum(numeric) / len(numeric)
        return median(numeric)

    if operation == "min":
        return min(cleaned)
    if operation == "max":
        return max(cleaned)

    raise ValueError(f"unsupported aggregation: {operation}")


class PromptTable:
    """Lightweight non-spatial table for model outputs and comparisons."""

    def __init__(self, rows: Sequence[Record]) -> None:
        self._rows = [dict(row) for row in rows]

    @classmethod
    def from_records(cls, records: Iterable[Record]) -> "PromptTable":
        return cls(list(records))

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, item: int | slice) -> Record | list[Record]:
        if isinstance(item, slice):
            return [dict(row) for row in self._rows[item]]
        return dict(self._rows[item])

    @property
    def columns(self) -> list[str]:
        return list(self._rows[0].keys()) if self._rows else []

    def head(self, count: int = 5) -> list[Record]:
        return [dict(row) for row in self._rows[:count]]

    def to_records(self) -> list[Record]:
        return [dict(row) for row in self._rows]

    def sort_values(self, by: str, descending: bool = False) -> "PromptTable":
        if by not in self.columns:
            raise KeyError(f"column '{by}' is not present")
        return PromptTable(
            sorted(self._rows, key=lambda row: (row[by] is None, row[by]), reverse=descending)
        )

    def select_columns(self, columns: Sequence[str]) -> "PromptTable":
        for column in columns:
            if column not in self.columns:
                raise KeyError(f"column '{column}' is not present")
        return PromptTable([{column: row[column] for column in columns} for row in self._rows])

    def where(
        self,
        predicate: Any | None = None,
        **equals: Any,
    ) -> "PromptTable":
        if predicate is not None and not callable(predicate):
            raise TypeError("predicate must be callable when provided")

        rows: list[Record] = []
        for row in self._rows:
            if any(row.get(key) != value for key, value in equals.items()):
                continue
            if predicate is not None and not bool(predicate(row)):
                continue
            rows.append(dict(row))
        return PromptTable(rows)

    def join(
        self,
        other: "PromptTable",
        on: str,
        how: str = "inner",
        rsuffix: str = "right",
    ) -> "PromptTable":
        if on not in self.columns:
            raise KeyError(f"column '{on}' is not present in left table")
        if on not in other.columns:
            raise KeyError(f"column '{on}' is not present in right table")
        if how not in {"inner", "left"}:
            raise ValueError("how must be 'inner' or 'left'")

        right_index: dict[Any, list[Record]] = {}
        for row in other:
            right_index.setdefault(row[on], []).append(dict(row))

        right_columns = [column for column in other.columns if column != on]
        rows: list[Record] = []
        for left_row in self._rows:
            right_rows = right_index.get(left_row.get(on), [])
            if not right_rows and how == "left":
                merged = dict(left_row)
                for column in right_columns:
                    target = column if column not in merged else f"{column}_{rsuffix}"
                    merged[target] = None
                rows.append(merged)
                continue

            for right_row in right_rows:
                merged = dict(left_row)
                for column in right_columns:
                    target = column if column not in merged else f"{column}_{rsuffix}"
                    merged[target] = right_row[column]
                rows.append(merged)

        return PromptTable(rows)

    def pivot(
        self,
        index: str,
        columns: str,
        values: str,
        agg: str = "sum",
    ) -> "PromptTable":
        for column in [index, columns, values]:
            if column not in self.columns:
                raise KeyError(f"column '{column}' is not present")

        grouped: dict[Any, dict[Any, list[Any]]] = {}
        column_values: set[Any] = set()
        for row in self._rows:
            grouped.setdefault(row[index], {}).setdefault(row[columns], []).append(row[values])
            column_values.add(row[columns])

        sorted_columns = sorted(column_values, key=str)
        rows: list[Record] = []
        for index_value, bucket in grouped.items():
            pivot_row: Record = {index: index_value}
            for column_value in sorted_columns:
                values_bucket = bucket.get(column_value, [])
                if not values_bucket:
                    pivot_row[str(column_value)] = None
                elif agg == "sum":
                    pivot_row[str(column_value)] = sum(float(value) for value in values_bucket)
                elif agg == "mean":
                    pivot_row[str(column_value)] = sum(float(value) for value in values_bucket) / len(values_bucket)
                elif agg == "min":
                    pivot_row[str(column_value)] = min(values_bucket)
                elif agg == "max":
                    pivot_row[str(column_value)] = max(values_bucket)
                elif agg == "first":
                    pivot_row[str(column_value)] = values_bucket[0]
                elif agg == "count":
                    pivot_row[str(column_value)] = len(values_bucket)
                else:
                    raise ValueError(f"unsupported aggregation: {agg}")
            rows.append(pivot_row)

        return PromptTable(rows)

    def summarize(self, by: str, aggregations: dict[str, str] | None = None) -> "PromptTable":
        if by not in self.columns:
            raise KeyError(f"column '{by}' is not present")
        return self.groupby(by).agg(aggregations or {})

    def groupby(self, by: str | Sequence[str]) -> "GroupedPromptTable":
        group_columns = [by] if isinstance(by, str) else list(by)
        if not group_columns:
            raise ValueError("groupby requires at least one column")
        for column in group_columns:
            if column not in self.columns:
                raise KeyError(f"column '{column}' is not present")
        return GroupedPromptTable(self._rows, group_columns)

    def value_counts(self, column: str, normalize: bool = False, dropna: bool = True) -> "PromptTable":
        if column not in self.columns:
            raise KeyError(f"column '{column}' is not present")

        values = [row.get(column) for row in self._rows]
        if dropna:
            values = [value for value in values if value is not None]

        counts = Counter(values)
        total = sum(counts.values())
        rows = [
            {
                column: value,
                "count": count,
                "share": (count / total) if total else 0.0,
            }
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
        ]
        if normalize:
            return PromptTable([{column: row[column], "share": row["share"]} for row in rows])
        return PromptTable(rows)

    def describe(self, columns: Sequence[str] | None = None) -> "PromptTable":
        selected = list(columns) if columns is not None else self.columns
        for column in selected:
            if column not in self.columns:
                raise KeyError(f"column '{column}' is not present")

        rows: list[Record] = []
        for column in selected:
            values = [row.get(column) for row in self._rows]
            non_null = [value for value in values if value is not None]
            dtype = _infer_column_dtype(values)
            numeric = [float(value) for value in non_null if _is_numeric_value(value)]
            top_value = Counter(non_null).most_common(1)[0][0] if non_null else None
            rows.append(
                {
                    "column": column,
                    "dtype": dtype,
                    "row_count": len(values),
                    "non_null_count": len(non_null),
                    "null_count": len(values) - len(non_null),
                    "unique_count": len({json.dumps(value, sort_keys=True, default=str) for value in non_null}),
                    "top": top_value,
                    "sum": sum(numeric) if dtype == "numeric" else None,
                    "mean": (sum(numeric) / len(numeric)) if dtype == "numeric" and numeric else None,
                    "median": median(numeric) if dtype == "numeric" and numeric else None,
                    "min": min(non_null) if non_null else None,
                    "max": max(non_null) if non_null else None,
                }
            )

        return PromptTable(rows)

    def to_markdown(self) -> str:
        if not self._rows:
            return "| |\n| --- |\n"
        columns = self.columns
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in self._rows]
        return "\n".join([header, separator, *body]) + "\n"

    def to_csv(self, output_path: str | Path) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self._rows)
        return str(path)

    def to_json(self, output_path: str | Path, indent: int = 2) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._rows, indent=indent), encoding="utf-8")
        return str(path)

    def to_html(self, output_path: str | Path, conditional: dict[str, Any] | None = None) -> str:
        """Write the table to an HTML file.

        Args:
            output_path: Destination file path.
            conditional: Optional formatting rules.  Supported keys:

                ``"column"``
                    Column to evaluate.
                ``"low_color"``
                    Background color for values below ``threshold`` (default
                    ``"#fca5a5"``).
                ``"high_color"``
                    Background color for values at or above ``threshold``
                    (default ``"#86efac"``).
                ``"threshold"``
                    Numeric threshold (default ``0``).

        Returns:
            The resolved output path string.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            html = "<table></table>"
        else:
            cond_col = conditional.get("column") if conditional else None
            low_color = (conditional or {}).get("low_color", "#fca5a5")
            high_color = (conditional or {}).get("high_color", "#86efac")
            threshold = float((conditional or {}).get("threshold", 0))

            header = "".join(f"<th>{column}</th>" for column in self.columns)
            body_parts: list[str] = []
            for row in self._rows:
                cells: list[str] = []
                for column in self.columns:
                    val = row.get(column, "")
                    style = ""
                    if cond_col and column == cond_col and _is_numeric_value(val):
                        bg = high_color if float(val) >= threshold else low_color
                        style = f' style="background:{bg}"'
                    cells.append(f"<td{style}>{val}</td>")
                body_parts.append("<tr>" + "".join(cells) + "</tr>")
            body = "".join(body_parts)
            html = f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
        path.write_text(html, encoding="utf-8")
        return str(path)

    def crosstab(
        self,
        index: str,
        columns: str,
        values: str | None = None,
        agg: str = "count",
    ) -> "PromptTable":
        """Cross-tabulate two categorical columns.

        Args:
            index: Column whose unique values become rows.
            columns: Column whose unique values become new columns.
            values: Optional column to aggregate (counts by default).
            agg: Aggregation function name.

        Returns:
            A new :class:`PromptTable`.
        """
        for col in [index, columns]:
            if col not in self.columns:
                raise KeyError(f"column '{col}' is not present")
        if values is not None and values not in self.columns:
            raise KeyError(f"column '{values}' is not present")

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


class GroupedPromptTable:
    """Grouped PromptTable rows with lightweight aggregation support."""

    def __init__(self, rows: Sequence[Record], group_columns: Sequence[str]) -> None:
        self._rows = [dict(row) for row in rows]
        self._group_columns = list(group_columns)

    def agg(self, aggregations: dict[str, str | Sequence[str]]) -> PromptTable:
        grouped_rows: dict[tuple[Any, ...], list[Record]] = {}
        for row in self._rows:
            key = tuple(row.get(column) for column in self._group_columns)
            grouped_rows.setdefault(key, []).append(row)

        summary_rows: list[Record] = []
        for key, rows in grouped_rows.items():
            summary_row: Record = {
                column: value for column, value in zip(self._group_columns, key)
            }
            summary_row["row_count"] = len(rows)
            for column, operations in aggregations.items():
                ops = [operations] if isinstance(operations, str) else list(operations)
                values = [row.get(column) for row in rows if column in row]
                for operation in ops:
                    summary_row[f"{column}_{operation}"] = _apply_aggregation(values, operation)
            summary_rows.append(summary_row)

        return PromptTable(summary_rows)


__all__ = ["PromptTable", "GroupedPromptTable"]