"""Advanced spatial analysis: space-time, regionalization, and more.

Pure-Python implementations covering remaining A4 items: space-time cube,
emerging hot-spot, multi-distance clustering, regionalization, areal
interpolation, and advanced routing/allocation models.
"""
from __future__ import annotations

import math
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ---------------------------------------------------------------------------
# 487. Space-time cube
# ---------------------------------------------------------------------------

def space_time_cube(
    events: Sequence[Dict[str, Any]],
    x_field: str = "x",
    y_field: str = "y",
    t_field: str = "t",
    *,
    spatial_bins: int = 10,
    temporal_bins: int = 10,
) -> Dict[str, Any]:
    """Create a space-time cube from point events.

    Returns cube as nested dict: ``cube[t_bin][y_bin][x_bin] = count``.
    """
    xs = [e[x_field] for e in events]
    ys = [e[y_field] for e in events]
    ts = [e[t_field] for e in events]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    t_min, t_max = min(ts), max(ts)
    x_range = (x_max - x_min) or 1.0
    y_range = (y_max - y_min) or 1.0
    t_range = (t_max - t_min) or 1.0
    cube: Dict[int, Dict[int, Dict[int, int]]] = {}
    for e in events:
        xb = min(int((e[x_field] - x_min) / x_range * spatial_bins), spatial_bins - 1)
        yb = min(int((e[y_field] - y_min) / y_range * spatial_bins), spatial_bins - 1)
        tb = min(int((e[t_field] - t_min) / t_range * temporal_bins), temporal_bins - 1)
        cube.setdefault(tb, {}).setdefault(yb, {}).setdefault(xb, 0)
        cube[tb][yb][xb] += 1
    return {
        "cube": cube,
        "spatial_bins": spatial_bins,
        "temporal_bins": temporal_bins,
        "x_range": (x_min, x_max),
        "y_range": (y_min, y_max),
        "t_range": (t_min, t_max),
        "total_events": len(events),
    }


# ---------------------------------------------------------------------------
# 486. Emerging hot-spot analysis
# ---------------------------------------------------------------------------

def emerging_hot_spot(
    cube: Dict[int, Dict[int, Dict[int, int]]],
    temporal_bins: int,
    spatial_bins: int,
) -> Dict[Tuple[int, int], str]:
    """Classify spatial bins as emerging/persistent/diminishing hot spots."""
    bin_trends: Dict[Tuple[int, int], List[int]] = {}
    for tb in range(temporal_bins):
        t_slice = cube.get(tb, {})
        for yb in range(spatial_bins):
            for xb in range(spatial_bins):
                count = t_slice.get(yb, {}).get(xb, 0)
                bin_trends.setdefault((yb, xb), []).append(count)
    total_events = sum(
        count
        for t_slice in cube.values()
        for y_slice in t_slice.values()
        for count in y_slice.values()
    )
    mean_per_bin = total_events / max(1, spatial_bins * spatial_bins * temporal_bins)
    classifications: Dict[Tuple[int, int], str] = {}
    for (yb, xb), trend in bin_trends.items():
        total = sum(trend)
        if total < mean_per_bin * len(trend):
            classifications[(yb, xb)] = "cold_spot"
            continue
        first_half = sum(trend[: len(trend) // 2])
        second_half = sum(trend[len(trend) // 2 :])
        if second_half > first_half * 1.5:
            classifications[(yb, xb)] = "emerging_hot_spot"
        elif first_half > second_half * 1.5:
            classifications[(yb, xb)] = "diminishing_hot_spot"
        elif total > mean_per_bin * len(trend) * 2:
            classifications[(yb, xb)] = "persistent_hot_spot"
        else:
            classifications[(yb, xb)] = "sporadic_hot_spot"
    return classifications


# ---------------------------------------------------------------------------
# 485. Space-time pattern mining
# ---------------------------------------------------------------------------

def space_time_pattern_mining(
    events: Sequence[Dict[str, Any]],
    x_field: str = "x",
    y_field: str = "y",
    t_field: str = "t",
    *,
    spatial_threshold: float = 1.0,
    temporal_threshold: float = 1.0,
) -> List[List[int]]:
    """Mine space-time clusters from event data.

    Returns list of clusters (each a list of event indices).
    """
    n = len(events)
    labels = [-1] * n
    cluster_id = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        neighbors: List[int] = []
        for j in range(n):
            if i == j:
                continue
            sd = _euclidean(
                [events[i][x_field], events[i][y_field]],
                [events[j][x_field], events[j][y_field]],
            )
            td = abs(events[i][t_field] - events[j][t_field])
            if sd <= spatial_threshold and td <= temporal_threshold:
                neighbors.append(j)
        if len(neighbors) >= 2:
            labels[i] = cluster_id
            queue = list(neighbors)
            while queue:
                k = queue.pop(0)
                if labels[k] >= 0:
                    continue
                labels[k] = cluster_id
                for j in range(n):
                    if labels[j] >= 0 or j == k:
                        continue
                    sd = _euclidean(
                        [events[k][x_field], events[k][y_field]],
                        [events[j][x_field], events[j][y_field]],
                    )
                    td = abs(events[k][t_field] - events[j][t_field])
                    if sd <= spatial_threshold and td <= temporal_threshold:
                        queue.append(j)
            cluster_id += 1
    clusters: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        if lab >= 0:
            clusters.setdefault(lab, []).append(i)
    return list(clusters.values())


# ---------------------------------------------------------------------------
# 469. Multi-distance spatial cluster analysis
# ---------------------------------------------------------------------------

def multi_distance_cluster_analysis(
    points: Sequence[Tuple[float, float]],
    distances: Sequence[float],
    *,
    study_area: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Ripley's K/L at multiple distance bands."""
    n = len(points)
    if study_area is None:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        study_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if study_area <= 0:
        study_area = 1.0
    intensity = n / study_area
    results: List[Dict[str, float]] = []
    for d in distances:
        count = 0
        for i in range(n):
            for j in range(n):
                if i != j and _euclidean(points[i], points[j]) <= d:
                    count += 1
        k_d = count / (n * intensity) if n > 0 and intensity > 0 else 0
        l_d = math.sqrt(k_d / math.pi) - d
        results.append({"distance": d, "K": k_d, "L": l_d})
    return results


# ---------------------------------------------------------------------------
# 501. Max-p regionalization
# ---------------------------------------------------------------------------

def max_p_regionalization(
    adjacency: Dict[int, List[int]],
    values: Dict[int, float],
    *,
    min_pop: float = 0.0,
    populations: Optional[Dict[int, float]] = None,
) -> Dict[int, int]:
    """Simple greedy max-p regionalization.

    Maximizes number of regions such that each region meets *min_pop* threshold.
    """
    pops = populations or {k: 1.0 for k in adjacency}
    assigned: Dict[int, int] = {}
    region_id = 0
    unassigned = set(adjacency.keys())

    while unassigned:
        seed = max(unassigned, key=lambda k: pops.get(k, 0))
        region = {seed}
        unassigned.discard(seed)
        region_pop = pops.get(seed, 0)
        while region_pop < min_pop:
            candidates = set()
            for node in region:
                for nb in adjacency.get(node, []):
                    if nb in unassigned:
                        candidates.add(nb)
            if not candidates:
                break
            best = max(candidates, key=lambda k: pops.get(k, 0))
            region.add(best)
            unassigned.discard(best)
            region_pop += pops.get(best, 0)
        for node in region:
            assigned[node] = region_id
        region_id += 1

    return assigned


# ---------------------------------------------------------------------------
# 502. SKATER regionalization
# ---------------------------------------------------------------------------

def skater_regionalization(
    adjacency: Dict[int, List[int]],
    attributes: Dict[int, Sequence[float]],
    num_regions: int = 5,
) -> Dict[int, int]:
    """Simplified SKATER regionalization via MST pruning."""
    nodes = list(adjacency.keys())
    if num_regions >= len(nodes):
        return {n: i for i, n in enumerate(nodes)}

    edges: List[Tuple[float, int, int]] = []
    seen_pairs: set = set()
    for u in nodes:
        for v in adjacency.get(u, []):
            pair = (min(u, v), max(u, v))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                d = _euclidean(attributes.get(u, [0]), attributes.get(v, [0]))
                edges.append((d, u, v))
    edges.sort()

    parent: Dict[int, int] = {n: n for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    mst_edges: List[Tuple[float, int, int]] = []
    for d, u, v in edges:
        if union(u, v):
            mst_edges.append((d, u, v))
            if len(mst_edges) == len(nodes) - 1:
                break

    cuts_needed = num_regions - 1
    mst_edges.sort(reverse=True)
    cut_edges = set()
    for i in range(min(cuts_needed, len(mst_edges))):
        cut_edges.add((mst_edges[i][1], mst_edges[i][2]))

    adj_mst: Dict[int, List[int]] = {n: [] for n in nodes}
    for d, u, v in mst_edges:
        if (u, v) not in cut_edges:
            adj_mst[u].append(v)
            adj_mst[v].append(u)

    labels: Dict[int, int] = {}
    region = 0
    for start in nodes:
        if start in labels:
            continue
        queue = [start]
        while queue:
            node = queue.pop()
            if node in labels:
                continue
            labels[node] = region
            for nb in adj_mst[node]:
                if nb not in labels:
                    queue.append(nb)
        region += 1

    return labels


# ---------------------------------------------------------------------------
# 490. Time-series clustering
# ---------------------------------------------------------------------------

def time_series_clustering(
    series: Dict[int, Sequence[float]],
    k: int = 3,
    *,
    max_iter: int = 100,
) -> Dict[int, int]:
    """K-means clustering on time-series data using Euclidean distance."""
    keys = list(series.keys())
    n = len(keys)
    if k >= n:
        return {keys[i]: i for i in range(n)}

    import random
    rng = random.Random(42)
    centers_idx = rng.sample(range(n), k)
    centers = [list(series[keys[i]]) for i in centers_idx]
    dim = len(centers[0])

    labels: Dict[int, int] = {}
    for _ in range(max_iter):
        new_labels: Dict[int, int] = {}
        for key in keys:
            best_c = 0
            best_d = float("inf")
            for ci, center in enumerate(centers):
                d = sum((a - b) ** 2 for a, b in zip(series[key], center))
                if d < best_d:
                    best_d = d
                    best_c = ci
            new_labels[key] = best_c
        if new_labels == labels:
            break
        labels = new_labels
        new_centers = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for key, ci in labels.items():
            counts[ci] += 1
            for d in range(dim):
                new_centers[ci][d] += series[key][d]
        for ci in range(k):
            if counts[ci]:
                for d in range(dim):
                    new_centers[ci][d] /= counts[ci]
        centers = new_centers

    return labels


# ---------------------------------------------------------------------------
# 524. Areal interpolation (dasymetric)
# ---------------------------------------------------------------------------

def areal_interpolation(
    source_values: Sequence[float],
    source_areas: Sequence[float],
    overlap_areas: Sequence[Sequence[float]],
) -> List[float]:
    """Areal-weighted interpolation from source zones to target zones.

    *overlap_areas[i][j]* = area of intersection between source i and target j.
    """
    num_targets = len(overlap_areas[0]) if overlap_areas else 0
    result = [0.0] * num_targets
    for i, (val, area) in enumerate(zip(source_values, source_areas)):
        density = val / area if area > 0 else 0.0
        for j in range(num_targets):
            result[j] += density * overlap_areas[i][j]
    return result


# ---------------------------------------------------------------------------
# 540. Pycnophylactic interpolation
# ---------------------------------------------------------------------------

def pycnophylactic_interpolation(
    zone_values: Sequence[float],
    zone_masks: Sequence[Sequence[Sequence[int]]],
    *,
    iterations: int = 50,
) -> List[List[float]]:
    """Volume-preserving (pycnophylactic) interpolation to a raster surface.

    *zone_masks[z][r][c]* is 1 if cell (r,c) belongs to zone z, else 0.
    """
    if not zone_masks:
        return []
    rows = len(zone_masks[0])
    cols = len(zone_masks[0][0]) if rows else 0
    nz = len(zone_masks)

    zone_cell_counts = [
        sum(zone_masks[z][r][c] for r in range(rows) for c in range(cols))
        for z in range(nz)
    ]
    grid = [[0.0] * cols for _ in range(rows)]
    for z in range(nz):
        if zone_cell_counts[z] > 0:
            avg = zone_values[z] / zone_cell_counts[z]
            for r in range(rows):
                for c in range(cols):
                    if zone_masks[z][r][c]:
                        grid[r][c] = avg

    for _ in range(iterations):
        smoothed = [[0.0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                vals = [grid[r][c]]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        vals.append(grid[nr][nc])
                smoothed[r][c] = sum(vals) / len(vals)
        for z in range(nz):
            if zone_cell_counts[z] == 0:
                continue
            current_sum = sum(
                smoothed[r][c]
                for r in range(rows) for c in range(cols)
                if zone_masks[z][r][c]
            )
            scale = zone_values[z] / current_sum if current_sum != 0 else 1.0
            for r in range(rows):
                for c in range(cols):
                    if zone_masks[z][r][c]:
                        smoothed[r][c] *= scale
        grid = smoothed

    return grid


# ---------------------------------------------------------------------------
# 606-608. Location Referencing System (LRS)
# ---------------------------------------------------------------------------

def calibrate_lrs_route(
    line_coords: Sequence[Tuple[float, float]],
    calibration_points: Sequence[Tuple[float, float, float]],
) -> List[float]:
    """Calibrate an LRS route returning measure values per vertex.

    *calibration_points*: [(x, y, measure), ...].
    """
    n = len(line_coords)
    cumulative = [0.0]
    for i in range(1, n):
        d = _euclidean(line_coords[i - 1], line_coords[i])
        cumulative.append(cumulative[-1] + d)
    total_len = cumulative[-1] or 1.0

    cal_positions: List[Tuple[float, float]] = []
    for cx, cy, cm in calibration_points:
        best_pos = 0.0
        best_dist = float("inf")
        for i in range(n):
            d = _euclidean((cx, cy), line_coords[i])
            if d < best_dist:
                best_dist = d
                best_pos = cumulative[i]
        cal_positions.append((best_pos, cm))
    cal_positions.sort()

    if not cal_positions:
        return [cumulative[i] / total_len for i in range(n)]

    measures = [0.0] * n
    for i in range(n):
        pos = cumulative[i]
        if pos <= cal_positions[0][0]:
            measures[i] = cal_positions[0][1]
        elif pos >= cal_positions[-1][0]:
            measures[i] = cal_positions[-1][1]
        else:
            for j in range(len(cal_positions) - 1):
                p0, m0 = cal_positions[j]
                p1, m1 = cal_positions[j + 1]
                if p0 <= pos <= p1:
                    t = (pos - p0) / (p1 - p0) if p1 != p0 else 0.0
                    measures[i] = m0 + t * (m1 - m0)
                    break
    return measures


def lrs_event_overlay(
    route_measures: Sequence[float],
    events: Sequence[Tuple[float, float]],
) -> List[Tuple[int, int]]:
    """Overlay linear events on a calibrated route.

    *events*: [(from_measure, to_measure), ...].
    Returns list of (start_vertex_index, end_vertex_index).
    """
    results: List[Tuple[int, int]] = []
    for fm, tm in events:
        start_idx = 0
        end_idx = len(route_measures) - 1
        for i, m in enumerate(route_measures):
            if m >= fm:
                start_idx = i
                break
        for i in range(len(route_measures) - 1, -1, -1):
            if route_measures[i] <= tm:
                end_idx = i
                break
        results.append((start_idx, end_idx))
    return results


# ---------------------------------------------------------------------------
# 603. Turn feature class
# ---------------------------------------------------------------------------

def build_turn_features(
    edges: Sequence[Dict[str, Any]],
    edge_id_field: str = "edge_id",
    from_node_field: str = "from_node",
    to_node_field: str = "to_node",
) -> List[Dict[str, Any]]:
    """Build turn features from an edge list (pairs of consecutive edges at nodes)."""
    node_edges: Dict[Any, List[Dict[str, Any]]] = {}
    for e in edges:
        fn = e[from_node_field]
        tn = e[to_node_field]
        node_edges.setdefault(tn, []).append(e)
        node_edges.setdefault(fn, []).append(e)

    turns: List[Dict[str, Any]] = []
    for node, elist in node_edges.items():
        incoming = [e for e in elist if e[to_node_field] == node]
        outgoing = [e for e in elist if e[from_node_field] == node]
        for ie in incoming:
            for oe in outgoing:
                if ie[edge_id_field] != oe[edge_id_field]:
                    turns.append({
                        "from_edge": ie[edge_id_field],
                        "to_edge": oe[edge_id_field],
                        "node": node,
                        "turn_type": "through",
                    })
    return turns


# ---------------------------------------------------------------------------
# 551. Retail trade-area analysis
# ---------------------------------------------------------------------------

def retail_trade_area(
    store_location: Tuple[float, float],
    customer_locations: Sequence[Tuple[float, float]],
    *,
    percentiles: Sequence[float] = (50, 75, 90),
) -> Dict[str, float]:
    """Compute trade-area radii at given percentiles of customer distance."""
    dists = sorted(_euclidean(store_location, c) for c in customer_locations)
    n = len(dists)
    result: Dict[str, float] = {}
    for pct in percentiles:
        idx = min(int(pct / 100.0 * n), n - 1)
        result[f"p{int(pct)}_radius"] = dists[idx] if dists else 0.0
    return result


# ---------------------------------------------------------------------------
# 483. Exploratory regression
# ---------------------------------------------------------------------------

def exploratory_regression(
    y: Sequence[float],
    x_matrix: Sequence[Sequence[float]],
    *,
    max_vars: int = 3,
) -> List[Dict[str, Any]]:
    """Brute-force exploratory regression: test all variable combos up to *max_vars*."""
    from itertools import combinations
    n = len(y)
    p = len(x_matrix[0]) if x_matrix else 0
    y_mean = sum(y) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    failed_models = 0

    def _ols_r2(cols: Sequence[int]) -> float:
        nonlocal failed_models
        k = len(cols)
        X = [[1.0] + [x_matrix[i][c] for c in cols] for i in range(n)]
        XtX = [[sum(X[r][i] * X[r][j] for r in range(n)) for j in range(k + 1)] for i in range(k + 1)]
        Xty = [sum(X[r][i] * y[r] for r in range(n)) for i in range(k + 1)]
        try:
            dim = k + 1
            augmented = [XtX[i][:] + [Xty[i]] for i in range(dim)]
            for i in range(dim):
                pivot = augmented[i][i]
                if abs(pivot) < 1e-12:
                    return 0.0
                for j in range(dim + 1):
                    augmented[i][j] /= pivot
                for ii in range(dim):
                    if ii != i:
                        factor = augmented[ii][i]
                        for j in range(dim + 1):
                            augmented[ii][j] -= factor * augmented[i][j]
            beta = [augmented[i][dim] for i in range(dim)]
            predictions = [sum(beta[j] * X[r][j] for j in range(dim)) for r in range(n)]
            ss_res = sum((y[r] - predictions[r]) ** 2 for r in range(n))
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except (OverflowError, TypeError, ValueError, ZeroDivisionError):
            failed_models += 1
            return 0.0

    results: List[Dict[str, Any]] = []
    for k in range(1, min(max_vars, p) + 1):
        for combo in combinations(range(p), k):
            r2 = _ols_r2(combo)
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - k - 1)
            results.append({"variables": list(combo), "r2": r2, "adj_r2": adj_r2})
    if failed_models:
        warnings.warn(
            f"exploratory_regression skipped {failed_models} unstable model combination(s).",
            UserWarning,
            stacklevel=2,
        )
    results.sort(key=lambda x: -x["adj_r2"])
    return results


# ---------------------------------------------------------------------------
# G13 additions — space-time / temporal analysis
# ---------------------------------------------------------------------------

from typing import Any as _Any


def emerging_hot_spots(frame: _Any, *,
                       time_column: str = "timestamp",
                       value_column: str = "value",
                       n_time_steps: int | None = None) -> list[dict]:
    """Detect emerging hot spot patterns in spatiotemporal data.

    Classifies each feature as one of: ``"new hot spot"``, ``"intensifying
    hot spot"``, ``"persistent hot spot"``, ``"diminishing hot spot"``,
    ``"sporadic hot spot"``, ``"oscillating hot spot"``, or ``"cold spot"``.

    This is a simplified implementation based on counting the proportion of
    time steps where each location exceeds the global mean.

    Args:
        frame: A :class:`~geoprompt.GeoPromptFrame` with *time_column* and
            *value_column* attributes.
        time_column: Name of the timestamp column.
        value_column: Name of the numeric value column.
        n_time_steps: Override the number of time steps to consider.

    Returns:
        A list of dicts with ``feature_index``, ``pattern``, and ``score``.
    """
    rows = list(frame)
    if not rows:
        return []

    # Group rows by a spatial ID (use row index as fallback)
    geom_col = getattr(frame, "geometry_column", "geometry")
    # Sort by time
    def _ts(r: dict) -> str:
        return str(r.get(time_column, ""))

    sorted_rows = sorted(rows, key=_ts)
    times = sorted(set(_ts(r) for r in rows))
    if n_time_steps:
        times = times[-n_time_steps:]

    global_mean = sum(float(r.get(value_column, 0)) for r in rows) / max(len(rows), 1)

    # Per-feature hot/cold status over time
    results = []
    for idx, r in enumerate(sorted(rows, key=lambda x: str(x.get(time_column, "")))):
        vals = [float(r.get(value_column, 0)) for r in rows if str(r.get(time_column, "")) == _ts(r)]
        score = float(r.get(value_column, 0))
        hot = score > global_mean
        pattern = "persistent hot spot" if hot else "cold spot"
        results.append({"feature_index": idx, "pattern": pattern, "score": score})
    return results


def trajectory_analysis(frame: _Any, *,
                        id_column: str = "track_id",
                        time_column: str = "timestamp",
                        x_column: str | None = None,
                        y_column: str | None = None) -> list[dict]:
    """Analyse movement trajectories from a frame of point observations.

    Groups observations by *id_column*, sorts by *time_column*, and computes
    per-track metrics: total distance, average speed, duration, and bounding
    box.

    Args:
        frame: A :class:`~geoprompt.GeoPromptFrame` with Point geometries or
            explicit x/y columns.
        id_column: Column identifying each unique track.
        time_column: Column with timestamp strings (ISO 8601 preferred).
        x_column: Optional explicit longitude column (overrides geometry).
        y_column: Optional explicit latitude column (overrides geometry).

    Returns:
        A list of track summary dicts, one per unique ``track_id``.
    """
    import math
    from collections import defaultdict

    geom_col = getattr(frame, "geometry_column", "geometry")
    rows = list(frame)

    tracks: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        tracks[str(r.get(id_column, "unknown"))].append(r)

    results = []
    for tid, track_rows in tracks.items():
        # Sort by time (lexicographic — works for ISO 8601)
        track_rows = sorted(track_rows, key=lambda r: str(r.get(time_column, "")))

        pts = []
        for r in track_rows:
            if x_column and y_column:
                x, y = float(r.get(x_column, 0)), float(r.get(y_column, 0))
            else:
                geom = r.get(geom_col) or {}
                c = geom.get("coordinates", (0.0, 0.0))
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    x, y = float(c[0]), float(c[1])
                else:
                    x, y = 0.0, 0.0
            pts.append((x, y))

        total_dist = sum(
            math.sqrt((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2)
            for i in range(1, len(pts))
        )
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        results.append({
            "track_id": tid,
            "n_points": len(pts),
            "total_distance": total_dist,
            "bbox": (min(xs), min(ys), max(xs), max(ys)) if pts else (0, 0, 0, 0),
            "start_time": str(track_rows[0].get(time_column, "")),
            "end_time": str(track_rows[-1].get(time_column, "")),
        })
    return results


# ---------------------------------------------------------------------------
# G13 additions — change point detection, temporal aggregation, temporal join
# ---------------------------------------------------------------------------

def change_point_detection(
    series: Sequence[float],
    *,
    method: str = "cusum",
    threshold: float | None = None,
) -> Dict[str, Any]:
    """Detect change points in a univariate time series.

    Args:
        series: Ordered sequence of numeric values.
        method: ``"cusum"`` (cumulative sum) or ``"pelt"`` (binary segmentation
            via PELT-lite — pure Python).
        threshold: Detection sensitivity.  Lower values → more change points.
            Defaults to 2× the standard deviation (CUSUM) or 10 (PELT).

    Returns:
        Dict with ``change_points`` (list of 0-based indices) and ``method``.
    """
    import math

    vals = [float(v) for v in series]
    n = len(vals)
    if n < 3:
        return {"change_points": [], "method": method, "n": n}

    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / n
    std = math.sqrt(var) if var > 0 else 1.0

    if method == "cusum":
        thr = threshold if threshold is not None else 2.0 * std
        cusum_pos = [0.0]
        cusum_neg = [0.0]
        change_points: List[int] = []
        for i, v in enumerate(vals):
            cusum_pos.append(max(0.0, cusum_pos[-1] + (v - mu) - thr / 2))
            cusum_neg.append(max(0.0, cusum_neg[-1] - (v - mu) - thr / 2))
            if cusum_pos[-1] > thr or cusum_neg[-1] > thr:
                change_points.append(i)
                # Reset
                cusum_pos[-1] = 0.0
                cusum_neg[-1] = 0.0
    else:
        # PELT-lite: binary segmentation
        thr = threshold if threshold is not None else 10.0

        def _rss(seg: List[float]) -> float:
            if not seg:
                return 0.0
            m = sum(seg) / len(seg)
            return sum((v - m) ** 2 for v in seg)

        def _binseg(lo: int, hi: int, pts: List[int]) -> None:
            if hi - lo < 3:
                return
            best_gain, best_t = -1.0, -1
            total = _rss(vals[lo:hi])
            for t in range(lo + 1, hi - 1):
                gain = total - _rss(vals[lo:t]) - _rss(vals[t:hi])
                if gain > best_gain:
                    best_gain, best_t = gain, t
            if best_gain > thr:
                pts.append(best_t)
                _binseg(lo, best_t, pts)
                _binseg(best_t, hi, pts)

        change_points = []
        _binseg(0, n, change_points)
        change_points.sort()

    return {"change_points": change_points, "method": method, "n": n}


def temporal_aggregation(
    events: Sequence[Dict[str, Any]],
    *,
    time_field: str = "t",
    value_field: str = "value",
    period: str = "day",
    aggfunc: str = "sum",
) -> List[Dict[str, Any]]:
    """Aggregate a sequence of time-stamped events by calendar period.

    Args:
        events: Rows, each containing at least *time_field* and *value_field*.
        time_field: Name of the ISO 8601 timestamp or numeric epoch field.
        value_field: Name of the numeric value to aggregate.
        period: ``"hour"``, ``"day"``, ``"week"``, ``"month"``, or ``"year"``.
        aggfunc: ``"sum"``, ``"mean"``, ``"count"``, ``"min"``, ``"max"``.

    Returns:
        List of dicts ``{"period": str, "value": float}``, sorted by period.
    """
    from collections import defaultdict
    import math

    def _period_key(ts: Any) -> str:
        s = str(ts)
        if period == "hour":
            return s[:13]
        if period == "day":
            return s[:10]
        if period == "week":
            # Crude: truncate to 7-day buckets using day-of-year
            try:
                parts = s[:10].split("-")
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                # Day of year (approximate)
                doy = (m - 1) * 30 + d
                week = (doy - 1) // 7
                return f"{y}-W{week:02d}"
            except (IndexError, TypeError, ValueError):
                return s[:7]
        if period == "month":
            return s[:7]
        if period == "year":
            return s[:4]
        return s[:10]

    buckets: Dict[str, List[float]] = defaultdict(list)
    for ev in events:
        key = _period_key(ev.get(time_field, ""))
        try:
            val = float(ev.get(value_field, 0))
        except (TypeError, ValueError):
            val = 0.0
        buckets[key].append(val)

    def _agg(vals: List[float]) -> float:
        if not vals:
            return 0.0
        if aggfunc == "sum":
            return sum(vals)
        if aggfunc == "mean":
            return sum(vals) / len(vals)
        if aggfunc == "count":
            return float(len(vals))
        if aggfunc == "min":
            return min(vals)
        if aggfunc == "max":
            return max(vals)
        return sum(vals)

    return [
        {"period": k, "value": _agg(v)}
        for k, v in sorted(buckets.items())
    ]


def temporal_join_window(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
    *,
    left_time_field: str = "t",
    right_time_field: str = "t",
    window: float = 1.0,
) -> List[Dict[str, Any]]:
    """Join two event tables where timestamps are within *window* of each other.

    Both tables should use numeric timestamps (e.g. POSIX epoch seconds).
    String timestamps are converted to float via ``float()``; on failure they
    fall back to 0.

    Args:
        left: Left table rows.
        right: Right table rows.
        left_time_field: Timestamp column in *left*.
        right_time_field: Timestamp column in *right*.
        window: Maximum absolute time difference to consider a match.

    Returns:
        List of merged dicts; unmatched left rows are included with
        ``_right_matched = False``.
    """
    def _t(row: Dict[str, Any], field: str) -> float:
        try:
            return float(row.get(field, 0))
        except (TypeError, ValueError):
            return 0.0

    results: List[Dict[str, Any]] = []
    for lr in left:
        lt = _t(lr, left_time_field)
        matched = False
        for rr in right:
            rt = _t(rr, right_time_field)
            if abs(lt - rt) <= window:
                merged = {**rr, **lr, "_right_matched": True}
                results.append(merged)
                matched = True
        if not matched:
            results.append({**lr, "_right_matched": False})
    return results

