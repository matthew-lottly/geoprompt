"""Advanced spatial analysis: weights matrices, clustering, interpolation, MCDA, service areas.

Pure-Python implementations of spatial statistics and analysis methods
covering roadmap items from A4 (Spatial Analysis 451-650).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6_371_000
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _centroid_of_points(points: Sequence[Sequence[float]]) -> tuple[float, float]:
    n = len(points)
    if n == 0:
        raise ValueError("cannot compute centroid of empty point set")
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    return (sx / n, sy / n)


# ---------------------------------------------------------------------------
# Spatial weights matrices (472-476)
# ---------------------------------------------------------------------------

def spatial_weights_distance_band(
    points: Sequence[Sequence[float]],
    threshold: float,
    *,
    binary: bool = True,
) -> dict[int, dict[int, float]]:
    """Build a distance-band spatial weights matrix.

    Returns {i: {j: weight, ...}} for all pairs within *threshold*.
    """
    n = len(points)
    w: dict[int, dict[int, float]] = {i: {} for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean(points[i], points[j])
            if d <= threshold:
                val = 1.0 if binary else d
                w[i][j] = val
                w[j][i] = val
    return w


def spatial_weights_inverse_distance(
    points: Sequence[Sequence[float]],
    *,
    power: float = 1.0,
    threshold: float | None = None,
) -> dict[int, dict[int, float]]:
    """Build an inverse-distance-weighted spatial weights matrix."""
    n = len(points)
    w: dict[int, dict[int, float]] = {i: {} for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean(points[i], points[j])
            if d == 0:
                continue
            if threshold is not None and d > threshold:
                continue
            val = 1.0 / (d ** power)
            w[i][j] = val
            w[j][i] = val
    return w


def spatial_weights_kernel(
    points: Sequence[Sequence[float]],
    bandwidth: float,
    *,
    kernel: str = "gaussian",
) -> dict[int, dict[int, float]]:
    """Build a kernel-based spatial weights matrix.

    Supported kernels: gaussian, bisquare, triangular.
    """
    n = len(points)
    w: dict[int, dict[int, float]] = {i: {} for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean(points[i], points[j])
            if d > bandwidth:
                continue
            ratio = d / bandwidth
            if kernel == "gaussian":
                val = math.exp(-0.5 * ratio ** 2)
            elif kernel == "bisquare":
                val = (1 - ratio ** 2) ** 2
            elif kernel == "triangular":
                val = 1 - ratio
            else:
                raise ValueError(f"unknown kernel: {kernel}")
            w[i][j] = val
            w[j][i] = val
    return w


def row_standardize_weights(
    weights: dict[int, dict[int, float]],
) -> dict[int, dict[int, float]]:
    """Row-standardise a spatial weights matrix so each row sums to 1."""
    result: dict[int, dict[int, float]] = {}
    for i, neighbours in weights.items():
        total = sum(neighbours.values())
        if total == 0:
            result[i] = {}
        else:
            result[i] = {j: v / total for j, v in neighbours.items()}
    return result


# ---------------------------------------------------------------------------
# Getis-Ord General G (454)
# ---------------------------------------------------------------------------

def getis_ord_general_g(
    values: Sequence[float],
    weights: dict[int, dict[int, float]],
) -> dict[str, float]:
    """Compute the Getis-Ord General G statistic for spatial autocorrelation."""
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 observations")
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denominator += values[i] * values[j]
            wij = weights.get(i, {}).get(j, 0.0)
            numerator += wij * values[i] * values[j]
    G = numerator / denominator if denominator != 0 else 0.0
    W = sum(sum(nb.values()) for nb in weights.values())
    mean_val = sum(values) / n
    s2 = sum((v - mean_val) ** 2 for v in values) / n
    EG = W / (n * (n - 1)) if n > 1 else 0.0
    z_score = (G - EG) / max(math.sqrt(s2 / n), 1e-15)
    return {"G": G, "expected_G": EG, "z_score": z_score}


# ---------------------------------------------------------------------------
# Ripley's K / L function (456)
# ---------------------------------------------------------------------------

def ripleys_k(
    points: Sequence[Sequence[float]],
    distances: Sequence[float],
    *,
    area: float | None = None,
) -> list[dict[str, float]]:
    """Compute Ripley's K and L functions at given distance thresholds.

    If *area* is not provided it is estimated from the bounding box of *points*.
    """
    n = len(points)
    if n < 2:
        raise ValueError("need at least 2 points")
    if area is None:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        area = max(w * h, 1e-15)
    intensity = n / area
    results: list[dict[str, float]] = []
    for d in distances:
        count = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if _euclidean(points[i], points[j]) <= d:
                    count += 1
        K = area * count / (n * n) if n > 0 else 0.0
        L = math.sqrt(K / math.pi) - d
        results.append({"distance": d, "K": K, "L": L})
    return results


# ---------------------------------------------------------------------------
# Point & line density (458-459)
# ---------------------------------------------------------------------------

def point_density(
    points: Sequence[Sequence[float]],
    cell_size: float,
    *,
    extent: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Simple point-density grid (count per cell).

    Returns dict with 'cells' (row-major list of counts), 'rows', 'cols',
    'extent' (xmin, ymin, xmax, ymax), and 'cell_size'.
    """
    if not points:
        raise ValueError("no points provided")
    if extent is None:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        extent = (min(xs), min(ys), max(xs) + cell_size, max(ys) + cell_size)
    xmin, ymin, xmax, ymax = extent
    cols = max(1, math.ceil((xmax - xmin) / cell_size))
    rows = max(1, math.ceil((ymax - ymin) / cell_size))
    grid = [0] * (rows * cols)
    for p in points:
        c = min(int((p[0] - xmin) / cell_size), cols - 1)
        r = min(int((p[1] - ymin) / cell_size), rows - 1)
        if 0 <= r < rows and 0 <= c < cols:
            grid[r * cols + c] += 1
    return {"cells": grid, "rows": rows, "cols": cols, "extent": extent, "cell_size": cell_size}


def line_density(
    lines: Sequence[Sequence[Sequence[float]]],
    cell_size: float,
    *,
    extent: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Line density: total length of line segments per cell.

    Each line is a sequence of (x, y) coordinates.
    """
    all_pts = [p for line in lines for p in line]
    if not all_pts:
        raise ValueError("no line vertices provided")
    if extent is None:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        extent = (min(xs), min(ys), max(xs) + cell_size, max(ys) + cell_size)
    xmin, ymin, xmax, ymax = extent
    cols = max(1, math.ceil((xmax - xmin) / cell_size))
    rows = max(1, math.ceil((ymax - ymin) / cell_size))
    grid = [0.0] * (rows * cols)
    for line in lines:
        for k in range(len(line) - 1):
            x0, y0 = line[k][0], line[k][1]
            x1, y1 = line[k + 1][0], line[k + 1][1]
            seg_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            c = min(int((mx - xmin) / cell_size), cols - 1)
            r = min(int((my - ymin) / cell_size), rows - 1)
            if 0 <= r < rows and 0 <= c < cols:
                grid[r * cols + c] += seg_len
    return {"cells": grid, "rows": rows, "cols": cols, "extent": extent, "cell_size": cell_size}


# ---------------------------------------------------------------------------
# Directional distribution (466)
# ---------------------------------------------------------------------------

def directional_distribution(
    points: Sequence[Sequence[float]],
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute directional distribution (standard deviational ellipse) parameters.

    Returns centre, rotation angle (radians), semi-major axis, semi-minor axis.
    """
    n = len(points)
    if n < 3:
        raise ValueError("need at least 3 points")
    w = weights if weights is not None else [1.0] * n
    tw = sum(w)
    cx = sum(w[i] * points[i][0] for i in range(n)) / tw
    cy = sum(w[i] * points[i][1] for i in range(n)) / tw
    dx = [points[i][0] - cx for i in range(n)]
    dy = [points[i][1] - cy for i in range(n)]
    sum_xy = sum(w[i] * dx[i] * dy[i] for i in range(n))
    sum_x2 = sum(w[i] * dx[i] ** 2 for i in range(n))
    sum_y2 = sum(w[i] * dy[i] ** 2 for i in range(n))
    A = sum_x2 - sum_y2
    B = math.sqrt(A ** 2 + 4 * sum_xy ** 2)
    theta = math.atan2(2 * sum_xy, A + B) / 2 if (A + B) != 0 else 0.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    sigma_x = math.sqrt(sum(w[i] * (dx[i] * cos_t - dy[i] * sin_t) ** 2 for i in range(n)) / tw)
    sigma_y = math.sqrt(sum(w[i] * (dx[i] * sin_t + dy[i] * cos_t) ** 2 for i in range(n)) / tw)
    return {
        "center": (cx, cy),
        "angle_rad": theta,
        "semi_major": max(sigma_x, sigma_y),
        "semi_minor": min(sigma_x, sigma_y),
    }


# ---------------------------------------------------------------------------
# Clustering algorithms (497-503)
# ---------------------------------------------------------------------------

def dbscan_spatial(
    points: Sequence[Sequence[float]],
    eps: float,
    min_samples: int = 5,
) -> dict[str, Any]:
    """Density-based spatial clustering (DBSCAN).

    Returns dict with 'labels' (list, -1 = noise), 'n_clusters'.
    """
    n = len(points)
    labels = [-1] * n
    cluster_id = 0

    def _neighbors(i: int) -> list[int]:
        return [j for j in range(n) if j != i and _euclidean(points[i], points[j]) <= eps]

    visited = [False] * n
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nb = _neighbors(i)
        if len(nb) < min_samples:
            continue
        labels[i] = cluster_id
        seed = list(nb)
        k = 0
        while k < len(seed):
            j = seed[k]
            if not visited[j]:
                visited[j] = True
                nb_j = _neighbors(j)
                if len(nb_j) >= min_samples:
                    seed.extend(nb_j)
            if labels[j] == -1:
                labels[j] = cluster_id
            k += 1
        cluster_id += 1

    return {"labels": labels, "n_clusters": cluster_id}


def kmeans_spatial(
    points: Sequence[Sequence[float]],
    k: int,
    *,
    max_iter: int = 100,
    seed: int | None = None,
) -> dict[str, Any]:
    """K-means clustering on spatial coordinates.

    Returns dict with 'labels', 'centroids', 'iterations'.
    """
    n = len(points)
    if k <= 0 or k > n:
        raise ValueError("k must be between 1 and number of points")
    rng = random.Random(seed)
    indices = rng.sample(range(n), k)
    centroids = [list(points[i]) for i in indices]
    labels = [0] * n

    for iteration in range(max_iter):
        for i in range(n):
            dists = [_euclidean(points[i], c) for c in centroids]
            labels[i] = dists.index(min(dists))
        new_centroids = []
        for c in range(k):
            members = [points[i] for i in range(n) if labels[i] == c]
            if members:
                new_centroids.append([
                    sum(m[d] for m in members) / len(members)
                    for d in range(len(points[0]))
                ])
            else:
                new_centroids.append(centroids[c])
        if new_centroids == centroids:
            return {"labels": labels, "centroids": [tuple(c) for c in new_centroids], "iterations": iteration + 1}
        centroids = new_centroids

    return {"labels": labels, "centroids": [tuple(c) for c in centroids], "iterations": max_iter}


def optics_ordering(
    points: Sequence[Sequence[float]],
    min_samples: int = 5,
    max_eps: float = float("inf"),
) -> list[dict[str, Any]]:
    """Compute OPTICS ordering for variable-density clustering.

    Returns a list of dicts with 'index', 'reachability_distance', 'core_distance'.
    """
    n = len(points)
    INF = float("inf")
    processed = [False] * n
    reachability = [INF] * n
    core_dist = [INF] * n
    ordering: list[dict[str, Any]] = []

    for i in range(n):
        dists = sorted(
            (_euclidean(points[i], points[j]), j) for j in range(n) if j != i
        )
        if len(dists) >= min_samples:
            core_dist[i] = dists[min_samples - 1][0]

    def _update(idx: int, seeds: list[tuple[float, int]]):
        cd = core_dist[idx]
        if cd == INF:
            return
        for d, j in ((_euclidean(points[idx], points[j]), j) for j in range(n) if j != idx):
            if processed[j]:
                continue
            new_rd = max(cd, d)
            if new_rd < max_eps and new_rd < reachability[j]:
                reachability[j] = new_rd
                seeds.append((new_rd, j))

    for i in range(n):
        if processed[i]:
            continue
        processed[i] = True
        ordering.append({"index": i, "reachability_distance": reachability[i], "core_distance": core_dist[i]})
        if core_dist[i] == INF:
            continue
        seeds: list[tuple[float, int]] = []
        _update(i, seeds)
        while seeds:
            seeds.sort()
            rd, j = seeds.pop(0)
            if processed[j]:
                continue
            processed[j] = True
            reachability[j] = rd
            ordering.append({"index": j, "reachability_distance": rd, "core_distance": core_dist[j]})
            _update(j, seeds)

    return ordering


# ---------------------------------------------------------------------------
# Multi-criteria decision analysis (511-516)
# ---------------------------------------------------------------------------

def analytic_hierarchy_process(
    criteria_matrix: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Analytic Hierarchy Process (AHP).

    *criteria_matrix* is a square pairwise-comparison matrix.
    Returns priority weights and consistency ratio.
    """
    n = len(criteria_matrix)
    if n < 2:
        raise ValueError("need at least 2 criteria")
    col_sums = [sum(criteria_matrix[i][j] for i in range(n)) for j in range(n)]
    norm = [[criteria_matrix[i][j] / col_sums[j] if col_sums[j] != 0 else 0 for j in range(n)] for i in range(n)]
    weights = [sum(norm[i]) / n for i in range(n)]
    weighted_sum = [sum(criteria_matrix[i][j] * weights[j] for j in range(n)) for i in range(n)]
    lambda_max = sum(weighted_sum[i] / weights[i] if weights[i] != 0 else 0 for i in range(n)) / n
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_table.get(n, 1.49)
    cr = ci / ri if ri != 0 else 0.0
    return {"weights": weights, "lambda_max": lambda_max, "consistency_index": ci, "consistency_ratio": cr}


def topsis(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float],
    *,
    beneficial: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).

    *matrix* is alternatives×criteria. Returns scores and ranking.
    """
    m = len(matrix)
    nc = len(weights) if weights else 0
    if m == 0 or nc == 0:
        raise ValueError("empty matrix or weights")
    if beneficial is None:
        beneficial = [True] * nc
    norms = []
    for j in range(nc):
        col_sum = math.sqrt(sum(matrix[i][j] ** 2 for i in range(m)))
        norms.append(col_sum if col_sum != 0 else 1.0)
    norm_matrix = [[matrix[i][j] / norms[j] * weights[j] for j in range(nc)] for i in range(m)]
    ideal_best = [max(norm_matrix[i][j] for i in range(m)) if beneficial[j] else min(norm_matrix[i][j] for i in range(m)) for j in range(nc)]
    ideal_worst = [min(norm_matrix[i][j] for i in range(m)) if beneficial[j] else max(norm_matrix[i][j] for i in range(m)) for j in range(nc)]
    scores = []
    for i in range(m):
        d_best = math.sqrt(sum((norm_matrix[i][j] - ideal_best[j]) ** 2 for j in range(nc)))
        d_worst = math.sqrt(sum((norm_matrix[i][j] - ideal_worst[j]) ** 2 for j in range(nc)))
        s = d_worst / (d_best + d_worst) if (d_best + d_worst) != 0 else 0.0
        scores.append(s)
    ranking = sorted(range(m), key=lambda i: scores[i], reverse=True)
    return {"scores": scores, "ranking": ranking}


def weighted_sum_model(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> list[float]:
    """Weighted Sum Model (WSM) — additive MCDA scoring."""
    return [sum(matrix[i][j] * weights[j] for j in range(len(weights))) for i in range(len(matrix))]


def weighted_product_model(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> list[float]:
    """Weighted Product Model (WPM) — multiplicative MCDA scoring."""
    scores = []
    for i in range(len(matrix)):
        product = 1.0
        for j in range(len(weights)):
            product *= matrix[i][j] ** weights[j] if matrix[i][j] > 0 else 0.0
        scores.append(product)
    return scores


def fuzzy_overlay(
    layers: Sequence[Sequence[float]],
    *,
    method: str = "and",
) -> list[float]:
    """Fuzzy overlay combining membership layers.

    Methods: and (min), or (max), product, sum, gamma.
    Each layer is a list of membership values (0-1) for each cell/feature.
    """
    if not layers:
        raise ValueError("no layers provided")
    n = len(layers[0])
    result = [0.0] * n
    for i in range(n):
        vals = [layer[i] for layer in layers]
        if method == "and":
            result[i] = min(vals)
        elif method == "or":
            result[i] = max(vals)
        elif method == "product":
            p = 1.0
            for v in vals:
                p *= v
            result[i] = p
        elif method == "sum":
            p = 1.0
            for v in vals:
                p *= (1 - v)
            result[i] = 1 - p
        elif method == "gamma":
            prod = 1.0
            for v in vals:
                prod *= v
            anti_prod = 1.0
            for v in vals:
                anti_prod *= (1 - v)
            gamma = 0.9
            result[i] = (prod ** (1 - gamma)) * ((1 - anti_prod) ** gamma)
        else:
            raise ValueError(f"unknown method: {method}")
    return result


def boolean_overlay(
    layers: Sequence[Sequence[bool]],
    *,
    method: str = "and",
) -> list[bool]:
    """Boolean suitability overlay (AND / OR / NOT)."""
    if not layers:
        raise ValueError("no layers provided")
    n = len(layers[0])
    result = [False] * n
    for i in range(n):
        vals = [layer[i] for layer in layers]
        if method == "and":
            result[i] = all(vals)
        elif method == "or":
            result[i] = any(vals)
        elif method == "not":
            result[i] = not vals[0]
        else:
            raise ValueError(f"unknown method: {method}")
    return result


# ---------------------------------------------------------------------------
# Interpolation (520-534)
# ---------------------------------------------------------------------------

def natural_neighbor_interpolation(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    query_point: Sequence[float],
) -> float:
    """Simple natural-neighbour (Sibson) interpolation approximation.

    Uses inverse-distance weighting among the nearest neighbours as a
    pure-Python approximation (true Voronoi-based NN would need a full
    Delaunay implementation).
    """
    if not points:
        raise ValueError("no sample points")
    dists = [(_euclidean(p, query_point), i) for i, p in enumerate(points)]
    dists.sort()
    k = min(len(dists), max(3, int(math.sqrt(len(points)))))
    nearest = dists[:k]
    total_w = 0.0
    weighted_v = 0.0
    for d, idx in nearest:
        w = 1.0 / max(d, 1e-15)
        total_w += w
        weighted_v += w * values[idx]
    return weighted_v / total_w


def local_polynomial_interpolation(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    query_point: Sequence[float],
    *,
    degree: int = 1,
    bandwidth: float | None = None,
) -> float:
    """Local polynomial interpolation (degree 1 = moving plane).

    Fits a weighted least-squares polynomial near *query_point*.
    """
    n = len(points)
    if n < 3:
        raise ValueError("need at least 3 points")
    if bandwidth is None:
        all_d = [_euclidean(p, query_point) for p in points]
        all_d.sort()
        bandwidth = all_d[min(n - 1, max(3, n // 2))] * 1.001
    ws = []
    xs = []
    ys = []
    for i in range(n):
        d = _euclidean(points[i], query_point)
        if d > bandwidth:
            continue
        w = (1 - (d / bandwidth) ** 2) ** 2 if d < bandwidth else 0.0
        ws.append(w)
        xs.append(points[i])
        ys.append(values[i])
    if not ws or sum(ws) < 1e-15:
        return float("nan")
    if degree == 1 and len(xs[0]) >= 2:
        sw = sum(ws)
        swx = sum(ws[i] * xs[i][0] for i in range(len(ws)))
        swy = sum(ws[i] * xs[i][1] for i in range(len(ws)))
        swz = sum(ws[i] * ys[i] for i in range(len(ws)))
        swxx = sum(ws[i] * xs[i][0] ** 2 for i in range(len(ws)))
        swyy = sum(ws[i] * xs[i][1] ** 2 for i in range(len(ws)))
        swxy = sum(ws[i] * xs[i][0] * xs[i][1] for i in range(len(ws)))
        swxz = sum(ws[i] * xs[i][0] * ys[i] for i in range(len(ws)))
        swyz = sum(ws[i] * xs[i][1] * ys[i] for i in range(len(ws)))
        A = [[sw, swx, swy], [swx, swxx, swxy], [swy, swxy, swyy]]
        b = [swz, swxz, swyz]
        det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
               - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
               + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
        if abs(det) < 1e-15:
            return sum(ws[i] * ys[i] for i in range(len(ws))) / sw if sw > 1e-15 else float("nan")
        c0 = ((A[1][1] * A[2][2] - A[1][2] * A[2][1]) * b[0]
               + (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * b[1]
               + (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * b[2]) / det
        c1 = ((A[1][2] * A[2][0] - A[1][0] * A[2][2]) * b[0]
               + (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * b[1]
               + (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * b[2]) / det
        c2 = ((A[1][0] * A[2][1] - A[1][1] * A[2][0]) * b[0]
               + (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * b[1]
               + (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * b[2]) / det
        return c0 + c1 * query_point[0] + c2 * query_point[1]
    sw = sum(ws)
    return sum(ws[i] * ys[i] for i in range(len(ws))) / sw


def radial_basis_function_interpolation(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    query_point: Sequence[float],
    *,
    function: str = "multiquadric",
    epsilon: float = 1.0,
) -> float:
    """Radial Basis Function (RBF) interpolation at a query point.

    Supported functions: multiquadric, inverse_multiquadric, gaussian, linear, cubic, thin_plate.
    """
    n = len(points)
    if n == 0:
        raise ValueError("no sample points")

    def _rbf(r: float) -> float:
        if function == "multiquadric":
            return math.sqrt(1 + (epsilon * r) ** 2)
        elif function == "inverse_multiquadric":
            return 1 / math.sqrt(1 + (epsilon * r) ** 2)
        elif function == "gaussian":
            return math.exp(-(epsilon * r) ** 2)
        elif function == "linear":
            return r
        elif function == "cubic":
            return r ** 3
        elif function == "thin_plate":
            return r ** 2 * math.log(max(r, 1e-15)) if r > 0 else 0.0
        raise ValueError(f"unknown RBF: {function}")

    # Solve using simple weighted average approach for stability
    ws = []
    for i in range(n):
        d = _euclidean(points[i], query_point)
        ws.append(_rbf(d))
    total_w = sum(abs(w) for w in ws)
    if total_w == 0:
        return values[0] if values else 0.0
    return sum(ws[i] * values[i] for i in range(n)) / total_w


def cross_validation_interpolation(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    interpolator: str = "idw",
    *,
    power: float = 2.0,
) -> dict[str, float]:
    """Leave-one-out cross-validation for interpolation quality.

    Returns RMSE and MAE.
    """
    n = len(points)
    if n < 3:
        raise ValueError("need at least 3 points")
    errors = []
    for i in range(n):
        pts = [points[j] for j in range(n) if j != i]
        vals = [values[j] for j in range(n) if j != i]
        if interpolator == "idw":
            dists = [_euclidean(p, points[i]) for p in pts]
            ws = [1 / max(d, 1e-15) ** power for d in dists]
            tw = sum(ws)
            pred = sum(ws[j] * vals[j] for j in range(len(pts))) / tw
        elif interpolator == "nn":
            pred = natural_neighbor_interpolation(pts, vals, points[i])
        else:
            raise ValueError(f"unknown interpolator: {interpolator}")
        errors.append(pred - values[i])
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
    mae = sum(abs(e) for e in errors) / n
    return {"rmse": rmse, "mae": mae, "n": n}


# ---------------------------------------------------------------------------
# Service area helpers (542-548)
# ---------------------------------------------------------------------------

def buffer_service_area(
    center: Sequence[float],
    radius: float,
    *,
    num_segments: int = 36,
) -> dict[str, Any]:
    """Generate a circular buffer service-area polygon in GeoJSON format."""
    coords = []
    for i in range(num_segments + 1):
        angle = 2 * math.pi * i / num_segments
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        coords.append((x, y))
    return {"type": "Polygon", "coordinates": [coords]}


def two_step_floating_catchment_area(
    supply: Sequence[dict[str, Any]],
    demand: Sequence[dict[str, Any]],
    threshold: float,
) -> list[float]:
    """Two-Step Floating Catchment Area (2SFCA) accessibility scores.

    *supply*: list of dicts with 'location' (x,y) and 'capacity'.
    *demand*: list of dicts with 'location' (x,y) and 'population'.
    Returns accessibility score per demand point.
    """
    ns = len(supply)
    nd = len(demand)
    # Step 1: compute supply-to-demand ratios
    ratios = []
    for j in range(ns):
        sloc = supply[j]["location"]
        total_pop = 0.0
        for i in range(nd):
            d = _euclidean(sloc, demand[i]["location"])
            if d <= threshold:
                total_pop += demand[i].get("population", 1)
        ratio = supply[j].get("capacity", 1) / max(total_pop, 1e-15)
        ratios.append(ratio)
    # Step 2: sum ratios for each demand point
    scores = []
    for i in range(nd):
        dloc = demand[i]["location"]
        s = 0.0
        for j in range(ns):
            d = _euclidean(dloc, supply[j]["location"])
            if d <= threshold:
                s += ratios[j]
        scores.append(s)
    return scores


def enhanced_two_step_fca(
    supply: Sequence[dict[str, Any]],
    demand: Sequence[dict[str, Any]],
    threshold: float,
    *,
    decay_function: str = "linear",
) -> list[float]:
    """Enhanced 2SFCA (E2SFCA) with distance-decay weighting."""
    ns = len(supply)
    nd = len(demand)

    def _weight(d: float) -> float:
        if d > threshold:
            return 0.0
        ratio = d / threshold
        if decay_function == "linear":
            return 1 - ratio
        elif decay_function == "gaussian":
            return math.exp(-0.5 * (ratio * 3) ** 2)
        elif decay_function == "power":
            return max(1 - ratio ** 2, 0)
        return 1.0

    ratios = []
    for j in range(ns):
        sloc = supply[j]["location"]
        total_wpop = 0.0
        for i in range(nd):
            d = _euclidean(sloc, demand[i]["location"])
            total_wpop += demand[i].get("population", 1) * _weight(d)
        ratio = supply[j].get("capacity", 1) / max(total_wpop, 1e-15)
        ratios.append(ratio)
    scores = []
    for i in range(nd):
        dloc = demand[i]["location"]
        s = 0.0
        for j in range(ns):
            d = _euclidean(dloc, supply[j]["location"])
            s += ratios[j] * _weight(d)
        scores.append(s)
    return scores


# ---------------------------------------------------------------------------
# Huff model & trade area (550-552)
# ---------------------------------------------------------------------------

def huff_model(
    stores: Sequence[dict[str, Any]],
    demand_points: Sequence[Sequence[float]],
    *,
    attractiveness_field: str = "size",
    distance_exponent: float = 2.0,
) -> list[list[float]]:
    """Huff model probability matrix.

    Returns probability[i][j] = probability that demand point i visits store j.
    """
    nd = len(demand_points)
    ns = len(stores)
    probs: list[list[float]] = []
    for i in range(nd):
        row: list[float] = []
        total = 0.0
        for j in range(ns):
            sloc = stores[j].get("location", (0, 0))
            attr = stores[j].get(attractiveness_field, 1.0)
            d = max(_euclidean(demand_points[i], sloc), 1e-15)
            val = attr / (d ** distance_exponent)
            row.append(val)
            total += val
        if total > 0:
            row = [v / total for v in row]
        probs.append(row)
    return probs


def market_penetration(
    customers: Sequence[Sequence[float]],
    store_location: Sequence[float],
    *,
    distance_bins: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Market penetration analysis: customer counts by distance band."""
    if distance_bins is None:
        distance_bins = [1, 3, 5, 10, 25]
    bins = sorted(distance_bins)
    counts = {f"0-{bins[0]}": 0}
    for k in range(len(bins) - 1):
        counts[f"{bins[k]}-{bins[k + 1]}"] = 0
    counts[f">{bins[-1]}"] = 0
    for c in customers:
        d = _euclidean(c, store_location)
        placed = False
        if d <= bins[0]:
            counts[f"0-{bins[0]}"] += 1
            placed = True
        else:
            for k in range(len(bins) - 1):
                if bins[k] < d <= bins[k + 1]:
                    counts[f"{bins[k]}-{bins[k + 1]}"] += 1
                    placed = True
                    break
        if not placed:
            counts[f">{bins[-1]}"] += 1
    return {"bins": counts, "total_customers": len(customers)}


# ---------------------------------------------------------------------------
# Similarity search (504)
# ---------------------------------------------------------------------------

def similarity_search(
    features: Sequence[dict[str, Any]],
    target: dict[str, Any],
    *,
    fields: Sequence[str] | None = None,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Find features most similar to *target* based on attribute values.

    Uses normalised Euclidean distance across numeric fields.
    """
    if fields is None:
        fields = [k for k, v in target.items() if isinstance(v, (int, float))]
    if not fields:
        return []
    # Compute min/max for normalisation
    all_vals: dict[str, list[float]] = {f: [] for f in fields}
    for feat in features:
        for f in fields:
            v = feat.get(f)
            if isinstance(v, (int, float)):
                all_vals[f].append(float(v))
    ranges: dict[str, tuple[float, float]] = {}
    for f in fields:
        vals = all_vals[f]
        if vals:
            ranges[f] = (min(vals), max(vals))
        else:
            ranges[f] = (0, 1)
    scored: list[tuple[float, int]] = []
    for idx, feat in enumerate(features):
        dist = 0.0
        for f in fields:
            lo, hi = ranges[f]
            span = hi - lo if hi != lo else 1.0
            fv = float(feat.get(f, 0))
            tgt = float(target.get(f, 0))
            dist += ((fv - tgt) / span) ** 2
        scored.append((math.sqrt(dist), idx))
    scored.sort()
    return [{"index": idx, "distance": d, "feature": features[idx]} for d, idx in scored[:top_n]]


# ---------------------------------------------------------------------------
# Quantile / probability maps (536-537)
# ---------------------------------------------------------------------------

def quantile_map(
    values: Sequence[float],
    n_classes: int = 5,
) -> dict[str, Any]:
    """Classify values into quantile-based classes.

    Returns class labels and break values.
    """
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    sorted_vals = sorted(values)
    breaks = []
    for i in range(1, n_classes):
        idx = int(i * n / n_classes)
        breaks.append(sorted_vals[min(idx, n - 1)])
    labels = []
    for v in values:
        cls = 0
        for b in breaks:
            if v > b:
                cls += 1
        labels.append(cls)
    return {"labels": labels, "breaks": breaks, "n_classes": n_classes}


def probability_map(
    values: Sequence[float],
    threshold: float,
    *,
    above: bool = True,
) -> list[float]:
    """Compute exceedance (or non-exceedance) probability per-cell using rank.

    Simple empirical probability based on rank ordering.
    """
    n = len(values)
    sorted_vals = sorted(values)
    probs = []
    for v in values:
        rank = sum(1 for sv in sorted_vals if sv <= v)
        p = rank / n
        if above:
            probs.append(1 - p)
        else:
            probs.append(p)
    return probs
