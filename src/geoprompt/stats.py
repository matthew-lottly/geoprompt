"""Spatial statistics module for autocorrelation, hotspot analysis, and interpolation.

All heavy dependencies (scipy, numpy) are lazily imported and optional.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .geometry import Geometry, geometry_centroid, geometry_distance


Record = dict[str, Any]


def _build_distance_weights(
    centroids: Sequence[tuple[float, float]],
    bandwidth: float | None = None,
    k: int | None = None,
) -> list[list[float]]:
    """Build a spatial weights matrix based on distance or k-nearest neighbors."""
    n = len(centroids)
    weights: list[list[float]] = [[0.0] * n for _ in range(n)]

    if k is not None:
        for i in range(n):
            dists = [(j, math.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])) for j in range(n) if j != i]
            dists.sort(key=lambda x: x[1])
            for j_idx, _ in dists[:k]:
                weights[i][j_idx] = 1.0
    elif bandwidth is not None:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = math.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
                if d <= bandwidth:
                    weights[i][j] = 1.0
    else:
        raise ValueError("either bandwidth or k must be specified")

    return weights


def _row_standardize(weights: list[list[float]]) -> list[list[float]]:
    """Row-standardize a weights matrix."""
    n = len(weights)
    result: list[list[float]] = []
    for i in range(n):
        row_sum = sum(weights[i])
        if row_sum > 0:
            result.append([w / row_sum for w in weights[i]])
        else:
            result.append([0.0] * n)
    return result


def morans_i(
    values: Sequence[float],
    centroids: Sequence[tuple[float, float]],
    bandwidth: float | None = None,
    k: int | None = None,
) -> dict[str, float]:
    """Compute global Moran's I spatial autocorrelation statistic.

    Args:
        values: Numeric values to test for spatial autocorrelation.
        centroids: Paired (x, y) coordinates for each observation.
        bandwidth: Distance threshold for neighbor definition.
        k: Number of nearest neighbors to use (alternative to bandwidth).

    Returns:
        Dict with ``"morans_i"``, ``"expected_i"``, ``"z_score"`` keys.
    """
    n = len(values)
    if n < 3:
        raise ValueError("at least 3 observations are required")
    if n != len(centroids):
        raise ValueError("values and centroids must have equal length")

    weights = _build_distance_weights(centroids, bandwidth=bandwidth, k=k)

    mean_val = sum(values) / n
    deviations = [v - mean_val for v in values]

    s0 = sum(weights[i][j] for i in range(n) for j in range(n))
    if s0 == 0:
        return {"morans_i": 0.0, "expected_i": -1 / (n - 1), "z_score": 0.0}

    numerator = sum(weights[i][j] * deviations[i] * deviations[j] for i in range(n) for j in range(n))
    denominator = sum(d * d for d in deviations)

    if denominator == 0:
        return {"morans_i": 0.0, "expected_i": -1 / (n - 1), "z_score": 0.0}

    I = (n / s0) * (numerator / denominator)
    expected_i = -1.0 / (n - 1)

    s1 = 0.5 * sum((weights[i][j] + weights[j][i]) ** 2 for i in range(n) for j in range(n))
    s2 = sum((sum(weights[i][j] for j in range(n)) + sum(weights[j][i] for j in range(n))) ** 2 for i in range(n))
    k2 = (sum(d ** 4 for d in deviations) / n) / ((sum(d ** 2 for d in deviations) / n) ** 2) if denominator > 0 else 0

    A = n * ((n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * s0 ** 2)
    B = k2 * ((n ** 2 - n) * s1 - 2 * n * s2 + 6 * s0 ** 2)
    C = (n - 1) * (n - 2) * (n - 3) * s0 ** 2

    variance = (A - B) / C - expected_i ** 2 if C > 0 else 0.0
    z_score = (I - expected_i) / math.sqrt(variance) if variance > 0 else 0.0

    return {"morans_i": I, "expected_i": expected_i, "z_score": z_score}


def gearys_c(
    values: Sequence[float],
    centroids: Sequence[tuple[float, float]],
    bandwidth: float | None = None,
    k: int | None = None,
) -> dict[str, float]:
    """Compute Geary's C spatial autocorrelation statistic.

    Args:
        values: Numeric values to test.
        centroids: Paired (x, y) coordinates.
        bandwidth: Distance threshold for neighbor definition.
        k: Number of nearest neighbors.

    Returns:
        Dict with ``"gearys_c"`` and ``"z_score"`` keys.
    """
    n = len(values)
    if n < 3:
        raise ValueError("at least 3 observations are required")

    weights = _build_distance_weights(centroids, bandwidth=bandwidth, k=k)
    mean_val = sum(values) / n
    deviations = [v - mean_val for v in values]

    s0 = sum(weights[i][j] for i in range(n) for j in range(n))
    if s0 == 0:
        return {"gearys_c": 1.0, "z_score": 0.0}

    numerator = sum(weights[i][j] * (values[i] - values[j]) ** 2 for i in range(n) for j in range(n))
    denominator = sum(d * d for d in deviations)

    if denominator == 0:
        return {"gearys_c": 1.0, "z_score": 0.0}

    C = ((n - 1) / (2 * s0)) * (numerator / denominator)
    z_score = (1 - C) / 0.5 if C != 1.0 else 0.0  # simplified z approximation

    return {"gearys_c": C, "z_score": z_score}


def local_morans_i(
    values: Sequence[float],
    centroids: Sequence[tuple[float, float]],
    bandwidth: float | None = None,
    k: int | None = None,
) -> list[dict[str, Any]]:
    """Compute Local Moran's I (LISA) for each observation.

    Args:
        values: Numeric values.
        centroids: Paired (x, y) coordinates.
        bandwidth: Distance threshold.
        k: Number of nearest neighbors.

    Returns:
        List of dicts with ``"local_i"``, ``"z_score"``, ``"cluster_type"``
        (``"HH"``, ``"LL"``, ``"HL"``, ``"LH"``, or ``"NS"``).
    """
    n = len(values)
    if n < 3:
        raise ValueError("at least 3 observations are required")

    weights = _row_standardize(_build_distance_weights(centroids, bandwidth=bandwidth, k=k))
    mean_val = sum(values) / n
    deviations = [v - mean_val for v in values]
    var = sum(d * d for d in deviations) / n

    results: list[dict[str, Any]] = []
    for i in range(n):
        lag = sum(weights[i][j] * deviations[j] for j in range(n))
        local_i = (deviations[i] / var) * lag if var > 0 else 0.0

        z = local_i / math.sqrt(abs(var)) if var > 0 else 0.0

        if abs(z) < 1.96:
            cluster = "NS"
        elif deviations[i] > 0 and lag > 0:
            cluster = "HH"
        elif deviations[i] < 0 and lag < 0:
            cluster = "LL"
        elif deviations[i] > 0 and lag < 0:
            cluster = "HL"
        else:
            cluster = "LH"

        results.append({"local_i": local_i, "z_score": z, "cluster_type": cluster})

    return results


def getis_ord_g(
    values: Sequence[float],
    centroids: Sequence[tuple[float, float]],
    bandwidth: float | None = None,
    k: int | None = None,
) -> list[dict[str, float]]:
    """Compute Getis-Ord Gi* hotspot statistic for each observation.

    Args:
        values: Numeric values.
        centroids: Paired (x, y) coordinates.
        bandwidth: Distance threshold.
        k: Number of nearest neighbors.

    Returns:
        List of dicts with ``"gi_star"`` and ``"z_score"``.
        Positive z-scores indicate hotspots, negative indicate coldspots.
    """
    n = len(values)
    if n < 3:
        raise ValueError("at least 3 observations are required")

    weights = _build_distance_weights(centroids, bandwidth=bandwidth, k=k)
    # Include self-weights for Gi*
    for i in range(n):
        weights[i][i] = 1.0

    mean_val = sum(values) / n
    s = math.sqrt(sum(v ** 2 for v in values) / n - mean_val ** 2) if n > 0 else 0.0

    results: list[dict[str, float]] = []
    for i in range(n):
        wi = weights[i]
        sum_wj = sum(wi)
        sum_wj_xj = sum(wi[j] * values[j] for j in range(n))
        sum_wj2 = sum(wi[j] ** 2 for j in range(n))

        numerator = sum_wj_xj - mean_val * sum_wj
        denom_inner = (n * sum_wj2 - sum_wj ** 2) / (n - 1) if n > 1 else 0.0
        denominator = s * math.sqrt(max(denom_inner, 0.0))

        gi_star = numerator / denominator if denominator > 0 else 0.0
        results.append({"gi_star": gi_star, "z_score": gi_star})

    return results


def kernel_density(
    centroids: Sequence[tuple[float, float]],
    weights: Sequence[float] | None = None,
    bandwidth: float = 1.0,
    cell_size: float = 0.1,
    bounds: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Compute a kernel density estimate grid.

    Uses Gaussian kernel weighting.

    Args:
        centroids: Point locations.
        weights: Optional weights per point (default 1.0 each).
        bandwidth: Kernel bandwidth (standard deviation).
        cell_size: Grid cell size.
        bounds: ``(min_x, min_y, max_x, max_y)`` grid extent; auto-computed if omitted.

    Returns:
        Dict with ``"grid"`` (2D list of density values), ``"bounds"``,
        ``"cell_size"``, ``"rows"``, ``"cols"``.
    """
    if not centroids:
        return {"grid": [], "bounds": (0, 0, 0, 0), "cell_size": cell_size, "rows": 0, "cols": 0}

    w = list(weights) if weights is not None else [1.0] * len(centroids)

    if bounds is None:
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]
        pad = bandwidth * 3
        bounds = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    min_x, min_y, max_x, max_y = bounds
    cols = max(1, int((max_x - min_x) / cell_size))
    rows = max(1, int((max_y - min_y) / cell_size))

    grid: list[list[float]] = [[0.0] * cols for _ in range(rows)]

    for idx, (cx, cy) in enumerate(centroids):
        weight = w[idx]
        for r in range(rows):
            cell_y = min_y + (r + 0.5) * cell_size
            for c in range(cols):
                cell_x = min_x + (c + 0.5) * cell_size
                d2 = (cx - cell_x) ** 2 + (cy - cell_y) ** 2
                grid[r][c] += weight * math.exp(-d2 / (2 * bandwidth ** 2))

    norm = 1.0 / (2 * math.pi * bandwidth ** 2)
    for r in range(rows):
        for c in range(cols):
            grid[r][c] *= norm

    return {"grid": grid, "bounds": bounds, "cell_size": cell_size, "rows": rows, "cols": cols}


def idw_interpolation(
    known_points: Sequence[tuple[float, float]],
    known_values: Sequence[float],
    query_points: Sequence[tuple[float, float]],
    power: float = 2.0,
    max_distance: float | None = None,
) -> list[float | None]:
    """Inverse Distance Weighting interpolation.

    Args:
        known_points: Locations of known observations.
        known_values: Values at known locations.
        query_points: Locations at which to interpolate.
        power: Distance weighting exponent (default 2).
        max_distance: Maximum distance to consider; farther points get ``None``.

    Returns:
        List of interpolated values (or ``None`` if no neighbors in range).
    """
    if len(known_points) != len(known_values):
        raise ValueError("known_points and known_values must have equal length")

    results: list[float | None] = []
    for qx, qy in query_points:
        numer = 0.0
        denom = 0.0
        exact = None
        for (kx, ky), kv in zip(known_points, known_values):
            d = math.hypot(qx - kx, qy - ky)
            if d < 1e-12:
                exact = kv
                break
            if max_distance is not None and d > max_distance:
                continue
            w = 1.0 / (d ** power)
            numer += w * kv
            denom += w

        if exact is not None:
            results.append(exact)
        elif denom > 0:
            results.append(numer / denom)
        else:
            results.append(None)

    return results


def spatial_weights_matrix(
    centroids: Sequence[tuple[float, float]],
    *,
    bandwidth: float | None = None,
    k: int | None = None,
    row_standardize: bool = False,
) -> list[list[float]]:
    """Build and optionally row-standardize a spatial weights matrix."""
    weights = _build_distance_weights(centroids, bandwidth=bandwidth, k=k)
    return _row_standardize(weights) if row_standardize else weights


def spatial_lag(
    values: Sequence[float],
    centroids: Sequence[tuple[float, float]],
    *,
    bandwidth: float | None = None,
    k: int | None = None,
) -> list[float]:
    """Compute the row-standardized spatial lag for each observation."""
    if len(values) != len(centroids):
        raise ValueError("values and centroids must have equal length")
    weights = spatial_weights_matrix(centroids, bandwidth=bandwidth, k=k, row_standardize=True)
    return [sum(weights[i][j] * values[j] for j in range(len(values))) for i in range(len(values))]


def semivariogram(
    centroids: Sequence[tuple[float, float]],
    values: Sequence[float],
    *,
    bins: int = 10,
) -> list[dict[str, float]]:
    """Compute an empirical semivariogram from point values."""
    if len(values) != len(centroids):
        raise ValueError("values and centroids must have equal length")
    if len(values) < 2:
        return []

    pairs: list[tuple[float, float]] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            dist = math.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
            gamma = 0.5 * ((values[i] - values[j]) ** 2)
            pairs.append((dist, gamma))

    max_dist = max(dist for dist, _ in pairs)
    if max_dist == 0:
        return [{"bin": 1.0, "pair_count": float(len(pairs)), "mean_distance": 0.0, "semivariance": 0.0}]
    step = max_dist / max(bins, 1)

    results: list[dict[str, float]] = []
    for idx in range(max(bins, 1)):
        lower = idx * step
        upper = (idx + 1) * step if idx < bins - 1 else max_dist + 1e-12
        bucket = [(dist, gamma) for dist, gamma in pairs if lower <= dist < upper]
        if not bucket:
            continue
        results.append({
            "bin": float(idx + 1),
            "pair_count": float(len(bucket)),
            "mean_distance": sum(dist for dist, _ in bucket) / len(bucket),
            "semivariance": sum(gamma for _, gamma in bucket) / len(bucket),
        })
    return results


def spatial_outliers(
    values: Sequence[float],
    centroids: Sequence[tuple[float, float]],
    bandwidth: float | None = None,
    k: int | None = None,
    threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """Flag spatial outliers based on local deviation from neighbors.

    Args:
        values: Numeric values.
        centroids: Paired (x, y) coordinates.
        bandwidth: Distance threshold for neighbor definition.
        k: Number of nearest neighbors.
        threshold: Standard-deviation threshold for flagging outliers.

    Returns:
        List of dicts with ``"value"``, ``"local_mean"``, ``"deviation"``,
        ``"is_outlier"`` keys.
    """
    n = len(values)
    weights = _row_standardize(_build_distance_weights(centroids, bandwidth=bandwidth, k=k))

    results: list[dict[str, Any]] = []
    all_deviations: list[float] = []
    local_means: list[float] = []

    for i in range(n):
        local_mean = sum(weights[i][j] * values[j] for j in range(n))
        local_means.append(local_mean)
        all_deviations.append(abs(values[i] - local_mean))

    dev_mean = sum(all_deviations) / n if n > 0 else 0.0
    dev_std = math.sqrt(sum((d - dev_mean) ** 2 for d in all_deviations) / n) if n > 1 else 1.0

    for i in range(n):
        z = (all_deviations[i] - dev_mean) / dev_std if dev_std > 0 else 0.0
        results.append({
            "value": values[i],
            "local_mean": local_means[i],
            "deviation": all_deviations[i],
            "is_outlier": abs(z) > threshold,
        })

    return results


__all__ = [
    "gearys_c",
    "getis_ord_g",
    "idw_interpolation",
    "kernel_density",
    "local_morans_i",
    "morans_i",
    "semivariogram",
    "spatial_lag",
    "spatial_outliers",
    "spatial_weights_matrix",
]
