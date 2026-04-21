"""Lightweight temporal analysis helpers for GeoPrompt workflows."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from datetime import datetime, timedelta
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


# ---------------------------------------------------------------------------
# Section I: 3D, Temporal, and Utility-Network Expansion
# ---------------------------------------------------------------------------


# --- 3D Geometry Support ---


def geometry_3d(
    coordinates: Sequence[Sequence[float]],
    geometry_type: str = "Point",
) -> dict[str, Any]:
    """Create a Z-aware GeoJSON-like geometry.

    Coordinates should include Z values: ``[(x, y, z), ...]``.
    """
    if geometry_type == "Point":
        return {"type": "Point", "coordinates": list(coordinates[0]) if coordinates else [0, 0, 0]}
    if geometry_type == "LineString":
        return {"type": "LineString", "coordinates": [list(c) for c in coordinates]}
    if geometry_type == "Polygon":
        return {"type": "Polygon", "coordinates": [[list(c) for c in coordinates]]}
    raise ValueError(f"unsupported geometry type: {geometry_type}")


def extract_z_values(geometry: dict[str, Any]) -> list[float]:
    """Extract Z values from a Z-aware geometry."""
    coords = geometry.get("coordinates", [])
    zs: list[float] = []
    _collect_z(coords, zs)
    return zs


def _collect_z(coords: Any, zs: list[float]) -> None:
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        if len(coords) >= 3:
            zs.append(float(coords[2]))
    elif isinstance(coords, (list, tuple)):
        for c in coords:
            _collect_z(c, zs)


def geometry_3d_bounds(geometry: dict[str, Any]) -> dict[str, float]:
    """Return 3D bounding box of a Z-aware geometry."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    _collect_xyz(geometry.get("coordinates", []), xs, ys, zs)
    return {
        "min_x": min(xs) if xs else 0.0,
        "min_y": min(ys) if ys else 0.0,
        "min_z": min(zs) if zs else 0.0,
        "max_x": max(xs) if xs else 0.0,
        "max_y": max(ys) if ys else 0.0,
        "max_z": max(zs) if zs else 0.0,
    }


def _collect_xyz(coords: Any, xs: list, ys: list, zs: list) -> None:
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        xs.append(float(coords[0]))
        ys.append(float(coords[1]))
        if len(coords) >= 3:
            zs.append(float(coords[2]))
    elif isinstance(coords, (list, tuple)):
        for c in coords:
            _collect_xyz(c, xs, ys, zs)


# --- Surface Sampling & Profile ---


def surface_profile(
    route_points: Sequence[tuple[float, float]],
    elevation_func: Any,
) -> list[dict[str, float]]:
    """Extract an elevation profile along a route.

    *elevation_func* takes (x, y) and returns a z value.

    Returns list of dicts with ``"distance"``, ``"x"``, ``"y"``, ``"z"``.
    """
    profile: list[dict[str, float]] = []
    cum_dist = 0.0
    for i, (x, y) in enumerate(route_points):
        if i > 0:
            cum_dist += math.hypot(x - route_points[i - 1][0], y - route_points[i - 1][1])
        z = elevation_func(x, y)
        profile.append({"distance": cum_dist, "x": x, "y": y, "z": float(z)})
    return profile


def surface_sample_grid(
    bounds: tuple[float, float, float, float],
    elevation_func: Any,
    *,
    cell_size: float = 1.0,
) -> dict[str, Any]:
    """Sample elevation over a grid, returning a raster-like dict."""
    min_x, min_y, max_x, max_y = bounds
    cols = max(1, int((max_x - min_x) / cell_size))
    rows = max(1, int((max_y - min_y) / cell_size))
    grid: list[list[float]] = []
    for r in range(rows):
        row_vals: list[float] = []
        y = max_y - (r + 0.5) * cell_size
        for c in range(cols):
            x = min_x + (c + 0.5) * cell_size
            row_vals.append(float(elevation_func(x, y)))
        grid.append(row_vals)
    return {"data": grid, "transform": (min_x, max_y, cell_size, cell_size), "rows": rows, "cols": cols}


# --- Voxel / Volumetric ---


def voxel_grid(
    points_3d: Sequence[tuple[float, float, float]],
    values: Sequence[float],
    *,
    cell_size: float = 1.0,
) -> dict[str, Any]:
    """Create a simple voxel grid from 3D point data.

    Returns dict with ``"voxels"`` (list of (ix, iy, iz, value) tuples)
    and grid metadata.
    """
    if not points_3d:
        return {"voxels": [], "cell_size": cell_size}

    xs = [p[0] for p in points_3d]
    ys = [p[1] for p in points_3d]
    zs = [p[2] for p in points_3d]
    origin = (min(xs), min(ys), min(zs))

    buckets: dict[tuple[int, int, int], list[float]] = {}
    for (x, y, z), v in zip(points_3d, values):
        ix = int((x - origin[0]) / cell_size)
        iy = int((y - origin[1]) / cell_size)
        iz = int((z - origin[2]) / cell_size)
        buckets.setdefault((ix, iy, iz), []).append(v)

    voxels = [(ix, iy, iz, sum(vs) / len(vs)) for (ix, iy, iz), vs in sorted(buckets.items())]
    return {"voxels": voxels, "origin": origin, "cell_size": cell_size, "voxel_count": len(voxels)}


# --- Temporal Joins & Windowing ---


def temporal_join(
    left: Sequence[Record],
    right: Sequence[Record],
    *,
    left_time: str,
    right_time: str,
    tolerance: str = "1h",
    join_type: str = "nearest",
) -> list[Record]:
    """Join two record sets by temporal proximity.

    *tolerance* is a string like ``"30m"``, ``"1h"``, ``"1d"``.
    *join_type*: ``"nearest"`` matches closest in time; ``"window"`` keeps
    all within tolerance.
    """
    tol_seconds = _parse_duration(tolerance)
    results: list[Record] = []

    for lrec in left:
        lt = _parse_timestamp(lrec.get(left_time))
        best_rec: Record | None = None
        best_diff = float("inf")

        matches: list[Record] = []
        for rrec in right:
            rt = _parse_timestamp(rrec.get(right_time))
            diff = abs((lt - rt).total_seconds())
            if diff <= tol_seconds:
                matches.append(rrec)
                if diff < best_diff:
                    best_diff = diff
                    best_rec = rrec

        if join_type == "nearest" and best_rec:
            merged = {**lrec, **{f"right_{k}": v for k, v in best_rec.items()}}
            merged["_time_diff_seconds"] = best_diff
            results.append(merged)
        elif join_type == "window":
            for m in matches:
                merged = {**lrec, **{f"right_{k}": v for k, v in m.items()}}
                results.append(merged)

    return results


def temporal_window(
    records: Sequence[Record],
    *,
    time_column: str,
    window_size: str = "1h",
    step: str | None = None,
) -> list[list[Record]]:
    """Sliding temporal window over records.

    Returns list of windows (each a list of records).
    """
    win_sec = _parse_duration(window_size)
    step_sec = _parse_duration(step) if step else win_sec
    sorted_recs = sorted(records, key=lambda r: _parse_timestamp(r.get(time_column)))
    if not sorted_recs:
        return []

    start = _parse_timestamp(sorted_recs[0].get(time_column))
    end = _parse_timestamp(sorted_recs[-1].get(time_column))

    windows: list[list[Record]] = []
    current = start
    while current <= end:
        win_end = current + timedelta(seconds=win_sec)
        window = [
            r for r in sorted_recs
            if current <= _parse_timestamp(r.get(time_column)) < win_end
        ]
        windows.append(window)
        current += timedelta(seconds=step_sec)

    return windows


def _parse_duration(s: str) -> float:
    """Parse a duration string like '30m', '1h', '2d' to seconds."""
    s = s.strip()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s[-1] in units:
        return float(s[:-1]) * units[s[-1]]
    return float(s)


def animation_frames(
    records: Sequence[Record],
    *,
    time_column: str,
    frame_duration: str = "1h",
) -> list[dict[str, Any]]:
    """Prepare animation-ready frames from temporal data.

    Returns a list of frame dicts with ``"start"``, ``"end"``, ``"records"``.
    """
    windows = temporal_window(records, time_column=time_column, window_size=frame_duration)
    sorted_recs = sorted(records, key=lambda r: _parse_timestamp(r.get(time_column)))
    if not sorted_recs:
        return []

    start = _parse_timestamp(sorted_recs[0].get(time_column))
    dur = _parse_duration(frame_duration)
    frames: list[dict[str, Any]] = []
    for i, window in enumerate(windows):
        frame_start = start + timedelta(seconds=i * dur)
        frames.append({
            "frame": i,
            "start": frame_start.isoformat(),
            "end": (frame_start + timedelta(seconds=dur)).isoformat(),
            "record_count": len(window),
            "records": window,
        })
    return frames


# --- Event Tracking ---


class EventTracker:
    """Track outages, incidents, and restoration events over time."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def add_event(
        self,
        event_type: str,
        *,
        start_time: str,
        end_time: str | None = None,
        asset_id: str | None = None,
        severity: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._events.append({
            "event_type": event_type,
            "start_time": start_time,
            "end_time": end_time,
            "asset_id": asset_id,
            "severity": severity,
            "duration_hours": self._duration_hours(start_time, end_time),
            "metadata": metadata or {},
        })

    def _duration_hours(self, start: str, end: str | None) -> float | None:
        if not end:
            return None
        try:
            s = _parse_timestamp(start)
            e = _parse_timestamp(end)
            return (e - s).total_seconds() / 3600
        except Exception:
            return None

    def get_events(
        self,
        *,
        event_type: str | None = None,
        asset_id: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        results = self._events
        if event_type:
            results = [e for e in results if e["event_type"] == event_type]
        if asset_id:
            results = [e for e in results if e["asset_id"] == asset_id]
        if severity:
            results = [e for e in results if e["severity"] == severity]
        return results

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        durations: list[float] = []
        for e in self._events:
            by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1
            by_severity[e["severity"]] = by_severity.get(e["severity"], 0) + 1
            if e["duration_hours"] is not None:
                durations.append(e["duration_hours"])
        return {
            "total_events": len(self._events),
            "by_type": by_type,
            "by_severity": by_severity,
            "mean_duration_hours": sum(durations) / len(durations) if durations else None,
            "total_downtime_hours": sum(durations),
        }


# --- Asset Criticality Ranking ---


def asset_criticality_ranking(
    assets: Sequence[Record],
    *,
    consequence_field: str = "consequence",
    likelihood_field: str = "likelihood",
    weight_consequence: float = 0.6,
    weight_likelihood: float = 0.4,
) -> list[dict[str, Any]]:
    """Rank assets by criticality score (consequence × likelihood).

    Returns ranked list with ``"asset"``, ``"criticality_score"``, ``"rank"``.
    """
    scored: list[dict[str, Any]] = []
    for a in assets:
        c = float(a.get(consequence_field, 0))
        l_val = float(a.get(likelihood_field, 0))
        score = weight_consequence * c + weight_likelihood * l_val
        scored.append({**a, "criticality_score": score})

    scored.sort(key=lambda x: x["criticality_score"], reverse=True)
    for rank, item in enumerate(scored, 1):
        item["rank"] = rank
    return scored


# --- Failure Simulation (Monte Carlo) ---


def failure_simulation_monte_carlo(
    assets: Sequence[Record],
    *,
    failure_rate_field: str = "failure_rate",
    consequence_field: str = "consequence",
    simulations: int = 1000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Monte Carlo failure simulation for asset portfolios.

    Each simulation randomly fails assets based on their failure rate and
    sums consequences.

    Returns dict with ``"mean_loss"``, ``"std_loss"``, ``"percentiles"``,
    ``"asset_failure_counts"``.
    """
    if seed is not None:
        random.seed(seed)

    n = len(assets)
    rates = [float(a.get(failure_rate_field, 0)) for a in assets]
    consequences = [float(a.get(consequence_field, 0)) for a in assets]
    failure_counts = [0] * n

    losses: list[float] = []
    for _ in range(simulations):
        total_loss = 0.0
        for i in range(n):
            if random.random() < rates[i]:
                total_loss += consequences[i]
                failure_counts[i] += 1
        losses.append(total_loss)

    losses.sort()
    mean_loss = sum(losses) / len(losses)
    var = sum((loss - mean_loss) ** 2 for loss in losses) / len(losses)

    percentiles = {}
    for p in (5, 25, 50, 75, 95):
        idx = int(len(losses) * p / 100)
        percentiles[f"p{p}"] = losses[min(idx, len(losses) - 1)]

    return {
        "mean_loss": mean_loss,
        "std_loss": math.sqrt(var),
        "min_loss": losses[0],
        "max_loss": losses[-1],
        "percentiles": percentiles,
        "simulations": simulations,
        "asset_failure_counts": {
            str(assets[i].get("id", i)): failure_counts[i]
            for i in range(n)
        },
    }


# --- Capital Planning Optimization ---


def capital_planning_prioritize(
    projects: Sequence[Record],
    *,
    benefit_field: str = "benefit",
    cost_field: str = "cost",
    budget: float,
) -> dict[str, Any]:
    """Budget-constrained capital planning prioritization (knapsack-style).

    Uses a greedy benefit/cost ratio approach.

    Returns dict with ``"selected_projects"``, ``"total_benefit"``,
    ``"total_cost"``, ``"budget_remaining"``.
    """
    items = [
        {**p, "_ratio": float(p.get(benefit_field, 0)) / max(float(p.get(cost_field, 1)), 1e-12)}
        for p in projects
    ]
    items.sort(key=lambda x: x["_ratio"], reverse=True)

    selected: list[Record] = []
    total_cost = 0.0
    total_benefit = 0.0

    for item in items:
        c = float(item.get(cost_field, 0))
        if total_cost + c <= budget:
            selected.append({k: v for k, v in item.items() if k != "_ratio"})
            total_cost += c
            total_benefit += float(item.get(benefit_field, 0))

    return {
        "selected_projects": selected,
        "total_benefit": total_benefit,
        "total_cost": total_cost,
        "budget_remaining": budget - total_cost,
        "selected_count": len(selected),
    }


__all__ = [
    "animation_frames",
    "asset_criticality_ranking",
    "capital_planning_prioritize",
    "EventTracker",
    "extract_z_values",
    "failure_simulation_monte_carlo",
    "geometry_3d",
    "geometry_3d_bounds",
    "resample_time_series",
    "rolling_window_stats",
    "sort_by_time",
    "surface_profile",
    "surface_sample_grid",
    "temporal_join",
    "temporal_window",
    "voxel_grid",
    # G13b additions
    "temporal_buffer",
    "temporal_filter",
    "time_series_trend",
]


# ---------------------------------------------------------------------------
# G13b additions — temporal analysis utilities
# ---------------------------------------------------------------------------

from typing import Any as _Any


def temporal_buffer(frame: _Any, duration: float, *,
                    time_column: str = "timestamp",
                    unit: str = "days") -> _Any:
    """Expand each feature's temporal extent by *duration* on each side.

    Adds ``time_start`` and ``time_end`` columns derived from *time_column*
    ± *duration*.

    Args:
        frame: A :class:`~geoprompt.GeoPromptFrame` with a *time_column*.
        duration: Buffer duration in *unit*.
        time_column: Column containing ISO 8601 timestamps.
        unit: Time unit: ``"seconds"``, ``"minutes"``, ``"hours"``,
              ``"days"``, ``"weeks"``.

    Returns:
        A new frame with ``time_start`` and ``time_end`` columns added.
    """
    import datetime
    unit_map = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400, "weeks": 604800}
    seconds = duration * unit_map.get(unit, 86400)
    delta = datetime.timedelta(seconds=seconds)

    rows = list(frame)
    out = []
    for r in rows:
        ts_str = str(r.get(time_column, ""))
        try:
            ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            try:
                ts = datetime.datetime.strptime(ts_str, "%Y-%m-%d")
            except Exception:
                ts = datetime.datetime.min
        nr = dict(r)
        nr["time_start"] = (ts - delta).isoformat()
        nr["time_end"] = (ts + delta).isoformat()
        out.append(nr)
    return type(frame).from_records(out)


def temporal_filter(frame: _Any, start: str, end: str, *,
                    time_column: str = "timestamp") -> _Any:
    """Filter a frame to features whose timestamp falls within [*start*, *end*].

    Args:
        frame: A :class:`~geoprompt.GeoPromptFrame` with a *time_column*.
        start: ISO 8601 start datetime string (inclusive).
        end: ISO 8601 end datetime string (inclusive).
        time_column: Column containing timestamps.

    Returns:
        A new frame with only the features in the time range.
    """
    rows = list(frame)
    selected = [r for r in rows if start <= str(r.get(time_column, "")) <= end]
    return type(frame).from_records(selected)


def time_series_trend(values: list[float], timestamps: list[str] | None = None) -> dict:
    """Compute a simple linear trend for a time series.

    Uses ordinary least squares (OLS) regression of *values* against an
    integer time index (or ordinal timestamps if provided).

    Args:
        values: Ordered list of numeric observations.
        timestamps: Optional ISO 8601 timestamp strings (same length as
            *values*).  Used to compute equal-spaced time indices.

    Returns:
        Dict with ``slope``, ``intercept``, ``r_squared``, ``trend``
        (``"increasing"``, ``"decreasing"``, or ``"flat"``), and
        ``predicted`` (list of fitted values).
    """
    import datetime
    n = len(values)
    if n < 2:
        return {"slope": 0.0, "intercept": values[0] if values else 0.0, "r_squared": 0.0, "trend": "flat", "predicted": values[:]}

    if timestamps:
        def _to_ord(ts: str) -> float:
            try:
                return float(datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
            except Exception:
                return 0.0
        xs = [_to_ord(t) for t in timestamps]
        # Normalise to [0, 1]
        x_min, x_max = min(xs), max(xs)
        rng = x_max - x_min or 1.0
        xs = [(x - x_min) / rng for x in xs]
    else:
        xs = list(range(n))

    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    sxy = sum((xs[i] - x_mean) * (float(values[i]) - y_mean) for i in range(n))
    sxx = sum((x - x_mean) ** 2 for x in xs)
    slope = sxy / sxx if sxx != 0 else 0.0
    intercept = y_mean - slope * x_mean
    predicted = [slope * x + intercept for x in xs]
    ss_res = sum((float(values[i]) - predicted[i]) ** 2 for i in range(n))
    ss_tot = sum((float(v) - y_mean) ** 2 for v in values)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    trend = "flat" if abs(slope) < 1e-9 else ("increasing" if slope > 0 else "decreasing")
    return {"slope": slope, "intercept": intercept, "r_squared": r2, "trend": trend, "predicted": predicted}
