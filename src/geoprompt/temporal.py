"""Lightweight temporal analysis helpers for GeoPrompt workflows."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Sequence


Record = dict[str, Any]


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"unsupported timestamp value: {value}") from exc


def sort_by_time(rows: Sequence[Record], time_column: str) -> list[Record]:
    """Return records sorted by a timestamp column."""
    return sorted((dict(row) for row in rows), key=lambda row: _parse_timestamp(row.get(time_column)))


def resample_time_series(
    rows: Sequence[Record],
    *,
    time_column: str,
    value_column: str,
    freq: str = "day",
) -> list[Record]:
    """Aggregate a simple time series by day, month, or year."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        dt = _parse_timestamp(row.get(time_column))
        value = float(row.get(value_column, 0))
        if freq == "day":
            key = dt.strftime("%Y-%m-%d")
        elif freq == "month":
            key = dt.strftime("%Y-%m")
        elif freq == "year":
            key = dt.strftime("%Y")
        else:
            raise ValueError("freq must be 'day', 'month', or 'year'")
        grouped[key].append(value)

    results: list[Record] = []
    for key in sorted(grouped):
        vals = grouped[key]
        results.append({
            "period": key,
            "count": len(vals),
            "sum": sum(vals),
            "mean": sum(vals) / len(vals) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        })
    return results


def rolling_window_stats(
    rows: Sequence[Record],
    *,
    value_column: str,
    window: int = 3,
) -> list[Record]:
    """Compute rolling window mean/min/max over ordered records."""
    if window <= 0:
        raise ValueError("window must be >= 1")
    ordered = [dict(row) for row in rows]
    values = [float(row.get(value_column, 0)) for row in ordered]
    results: list[Record] = []
    for idx, row in enumerate(ordered):
        start = max(0, idx - window + 1)
        bucket = values[start:idx + 1]
        new_row = dict(row)
        new_row["rolling_mean"] = sum(bucket) / len(bucket)
        new_row["rolling_min"] = min(bucket)
        new_row["rolling_max"] = max(bucket)
        results.append(new_row)
    return results


__all__ = ["resample_time_series", "rolling_window_stats", "sort_by_time"]
