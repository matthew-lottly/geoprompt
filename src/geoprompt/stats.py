"""Spatial statistics module for autocorrelation, hotspot analysis, and interpolation.

All heavy dependencies (scipy, numpy) are lazily imported and optional.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .geometry import geometry_centroid


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

    mi = (n / s0) * (numerator / denominator)
    expected_i = -1.0 / (n - 1)

    s1 = 0.5 * sum((weights[i][j] + weights[j][i]) ** 2 for i in range(n) for j in range(n))
    s2 = sum((sum(weights[i][j] for j in range(n)) + sum(weights[j][i] for j in range(n))) ** 2 for i in range(n))
    k2 = (sum(d ** 4 for d in deviations) / n) / ((sum(d ** 2 for d in deviations) / n) ** 2) if denominator > 0 else 0

    A = n * ((n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * s0 ** 2)
    B = k2 * ((n ** 2 - n) * s1 - 2 * n * s2 + 6 * s0 ** 2)
    C = (n - 1) * (n - 2) * (n - 3) * s0 ** 2

    variance = (A - B) / C - expected_i ** 2 if C > 0 else 0.0
    z_score = (mi - expected_i) / math.sqrt(variance) if variance > 0 else 0.0

    return {"morans_i": mi, "expected_i": expected_i, "z_score": z_score}


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


# ---------------------------------------------------------------------------
# Section D: Advanced Spatial Analysis and Statistics
# ---------------------------------------------------------------------------


def hot_spot_analysis(
    records: Sequence[Record],
    value_field: str,
    *,
    bandwidth: float | None = None,
    k: int = 8,
    geometry_field: str = "geometry",
    confidence_levels: Sequence[float] = (0.90, 0.95, 0.99),
) -> list[dict[str, Any]]:
    """Perform integrated hot spot analysis (Getis-Ord Gi*) on feature records.

    Extracts centroids and values from *records*, runs ``getis_ord_g``, and
    classifies each feature into a confidence bin.

    Returns:
        List of dicts with ``"gi_star"``, ``"z_score"``, ``"p_value"``,
        ``"confidence_bin"`` (``"Hot"``/``"Cold"``/``"Not Significant"``).
    """

    centroids: list[tuple[float, float]] = []
    values: list[float] = []
    for rec in records:
        geom = rec.get(geometry_field)
        c = geometry_centroid(geom) if geom else {"x": 0.0, "y": 0.0}
        centroids.append((c["x"], c["y"]))
        values.append(float(rec.get(value_field, 0)))

    gi_results = getis_ord_g(values, centroids, bandwidth=bandwidth, k=k)
    z_thresholds = sorted(
        [(1 - cl, _z_critical(cl)) for cl in confidence_levels], key=lambda t: t[1], reverse=True
    )

    enriched: list[dict[str, Any]] = []
    for r in gi_results:
        z = r["z_score"]
        p = 2 * (1 - _norm_cdf(abs(z)))
        label = "Not Significant"
        for p_thresh, z_thresh in z_thresholds:
            if abs(z) >= z_thresh:
                label = f"Hot ({int((1-p_thresh)*100)}%)" if z > 0 else f"Cold ({int((1-p_thresh)*100)}%)"
                break
        enriched.append({"gi_star": r["gi_star"], "z_score": z, "p_value": p, "confidence_bin": label})
    return enriched


def _z_critical(confidence: float) -> float:
    """Approximate z-critical value for a two-tailed test."""
    # Rational approx sufficient for 0.90, 0.95, 0.99
    lookup = {0.90: 1.6449, 0.95: 1.96, 0.99: 2.576}
    return lookup.get(confidence, 1.96)


def _norm_cdf(z: float) -> float:
    """Approximate standard normal CDF via the Abramowitz & Stegun formula."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p_coeff = 0.3275911
    sign = 1 if z >= 0 else -1
    z = abs(z) / math.sqrt(2)
    t = 1.0 / (1.0 + p_coeff * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)
    return 0.5 * (1.0 + sign * y)


def cluster_outlier_analysis(
    values: Sequence[float],
    centroids: Sequence[tuple[float, float]],
    *,
    bandwidth: float | None = None,
    k: int | None = None,
    permutations: int = 99,
) -> list[dict[str, Any]]:
    """Anselin Local Moran's I cluster-outlier analysis with pseudo-p-values.

    Runs local Moran's I and adds permutation-based significance.
    """
    import random

    lisa = local_morans_i(values, centroids, bandwidth=bandwidth, k=k)
    len(values)
    vals = list(values)

    for idx, entry in enumerate(lisa):
        count_extreme = 0
        for _ in range(permutations):
            shuffled = vals[:]
            random.shuffle(shuffled)
            perm_lisa = local_morans_i(shuffled, centroids, bandwidth=bandwidth, k=k)
            if abs(perm_lisa[idx]["local_i"]) >= abs(entry["local_i"]):
                count_extreme += 1
        entry["pseudo_p"] = (count_extreme + 1) / (permutations + 1)
        entry["significant"] = entry["pseudo_p"] <= 0.05
    return lisa


def adaptive_kernel_density(
    centroids: Sequence[tuple[float, float]],
    weights: Sequence[float] | None = None,
    *,
    cell_size: float = 0.1,
    bounds: tuple[float, float, float, float] | None = None,
    pilot_bandwidth: float = 1.0,
    sensitivity: float = 0.5,
) -> dict[str, Any]:
    """Adaptive kernel density estimation with location-varying bandwidths.

    A pilot density run determines local density at each point; bandwidths
    are then scaled inversely to local density raised to ``sensitivity``.
    """
    pilot = kernel_density(centroids, weights, bandwidth=pilot_bandwidth, cell_size=cell_size, bounds=bounds)
    g_bounds = pilot["bounds"]

    local_densities: list[float] = []
    for cx, cy in centroids:
        col = max(0, min(int((cx - g_bounds[0]) / cell_size), pilot["cols"] - 1))
        row = max(0, min(int((cy - g_bounds[1]) / cell_size), pilot["rows"] - 1))
        local_densities.append(max(pilot["grid"][row][col], 1e-12))

    geo_mean = math.exp(sum(math.log(d) for d in local_densities) / len(local_densities))
    adaptive_bw = [pilot_bandwidth * (geo_mean / d) ** sensitivity for d in local_densities]

    w = list(weights) if weights is not None else [1.0] * len(centroids)
    if bounds is None:
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]
        pad = max(adaptive_bw) * 3
        bounds = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    min_x, min_y, max_x, max_y = bounds
    cols = max(1, int((max_x - min_x) / cell_size))
    rows = max(1, int((max_y - min_y) / cell_size))
    grid: list[list[float]] = [[0.0] * cols for _ in range(rows)]

    for pidx, (cx, cy) in enumerate(centroids):
        h = adaptive_bw[pidx]
        norm = 1.0 / (2 * math.pi * h ** 2)
        for r in range(rows):
            cell_y = min_y + (r + 0.5) * cell_size
            for c in range(cols):
                cell_x = min_x + (c + 0.5) * cell_size
                d2 = (cx - cell_x) ** 2 + (cy - cell_y) ** 2
                grid[r][c] += w[pidx] * norm * math.exp(-d2 / (2 * h ** 2))

    return {"grid": grid, "bounds": bounds, "cell_size": cell_size, "rows": rows, "cols": cols}


def kriging_interpolation(
    known_points: Sequence[tuple[float, float]],
    known_values: Sequence[float],
    query_points: Sequence[tuple[float, float]],
    *,
    model: str = "spherical",
    sill: float | None = None,
    range_param: float | None = None,
    nugget: float = 0.0,
) -> list[dict[str, float]]:
    """Ordinary kriging interpolation.

    Args:
        known_points: Sampled locations.
        known_values: Values at sampled locations.
        query_points: Target prediction locations.
        model: Variogram model (``"spherical"``, ``"exponential"``, ``"gaussian"``).
        sill: Variogram sill (auto-estimated if *None*).
        range_param: Variogram range (auto-estimated if *None*).
        nugget: Variogram nugget (default 0).

    Returns:
        List of dicts with ``"value"`` and ``"variance"`` per query point.
    """
    n = len(known_points)
    if n != len(known_values):
        raise ValueError("known_points and known_values must have equal length")

    # Auto-estimate sill/range from semivariogram if needed
    sv = semivariogram(known_points, known_values, bins=min(n, 15))
    if sill is None:
        sill = max((b["semivariance"] for b in sv), default=1.0)
    if range_param is None:
        range_param = max((b["mean_distance"] for b in sv), default=1.0)

    variogram = _variogram_model(model, sill, range_param, nugget)

    # Build system matrix
    K = [[variogram(math.hypot(known_points[i][0] - known_points[j][0],
                                known_points[i][1] - known_points[j][1]))
          for j in range(n)] + [1.0] for i in range(n)]
    K.append([1.0] * n + [0.0])

    results: list[dict[str, float]] = []
    for qx, qy in query_points:
        k_vec = [variogram(math.hypot(qx - known_points[i][0], qy - known_points[i][1]))
                 for i in range(n)] + [1.0]
        weights_vec = _solve_linear(K, k_vec)
        if weights_vec is None:
            results.append({"value": sum(known_values) / n, "variance": sill})
            continue
        pred = sum(weights_vec[i] * known_values[i] for i in range(n))
        var_k = sum(weights_vec[i] * k_vec[i] for i in range(n + 1))
        results.append({"value": pred, "variance": max(0.0, sill - var_k)})

    return results


def _variogram_model(model: str, sill: float, range_param: float, nugget: float):
    """Return a variogram function for the given model name."""
    def spherical(h: float) -> float:
        if h == 0:
            return 0.0
        if h >= range_param:
            return nugget + sill
        ratio = h / range_param
        return nugget + sill * (1.5 * ratio - 0.5 * ratio ** 3)

    def exponential(h: float) -> float:
        if h == 0:
            return 0.0
        return nugget + sill * (1 - math.exp(-3 * h / range_param))

    def gaussian_v(h: float) -> float:
        if h == 0:
            return 0.0
        return nugget + sill * (1 - math.exp(-3 * (h / range_param) ** 2))

    models = {"spherical": spherical, "exponential": exponential, "gaussian": gaussian_v}
    return models.get(model, spherical)


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve Ax = b using Gaussian elimination (small systems)."""
    n = len(b)
    M = [row[:] + [bi] for row, bi in zip(A, b)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        if abs(M[col][col]) < 1e-12:
            return None
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def spline_interpolation(
    known_points: Sequence[tuple[float, float]],
    known_values: Sequence[float],
    query_points: Sequence[tuple[float, float]],
    *,
    spline_type: str = "thin_plate",
    smoothing: float = 0.0,
) -> list[float]:
    """Spline-based spatial interpolation.

    Uses thin-plate spline (radial basis function) by default.
    Falls back to IDW if scipy is unavailable.
    """
    try:
        import numpy as np
        from scipy.interpolate import RBFInterpolator

        pts = np.array(known_points)
        vals = np.array(known_values)
        q = np.array(query_points)
        kernel = "thin_plate_spline" if spline_type == "thin_plate" else spline_type
        interp = RBFInterpolator(pts, vals, kernel=kernel, smoothing=smoothing)
        return interp(q).tolist()
    except ImportError:
        return [v if v is not None else 0.0
                for v in idw_interpolation(known_points, known_values, query_points)]


def trend_surface(
    centroids: Sequence[tuple[float, float]],
    values: Sequence[float],
    *,
    order: int = 1,
) -> dict[str, Any]:
    """Fit a polynomial trend surface to spatial data.

    Args:
        order: Polynomial order (1 = linear, 2 = quadratic).

    Returns:
        Dict with ``"coefficients"``, ``"residuals"``, and ``"r_squared"``.
    """
    n = len(values)
    if n != len(centroids):
        raise ValueError("values and centroids must have equal length")

    # Build design matrix
    rows: list[list[float]] = []
    for x, y in centroids:
        row = [1.0, x, y]
        if order >= 2:
            row.extend([x * x, x * y, y * y])
        if order >= 3:
            row.extend([x ** 3, x ** 2 * y, x * y ** 2, y ** 3])
        rows.append(row)

    # Normal equations: (X^T X) c = X^T y
    p = len(rows[0])
    XtX = [[sum(rows[k][i] * rows[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    Xty = [sum(rows[k][i] * values[k] for k in range(n)) for i in range(p)]

    coeffs = _solve_linear(XtX, Xty)
    if coeffs is None:
        coeffs = [0.0] * p

    predicted = [sum(coeffs[j] * rows[i][j] for j in range(p)) for i in range(n)]
    residuals = [values[i] - predicted[i] for i in range(n)]
    ss_res = sum(r ** 2 for r in residuals)
    mean_v = sum(values) / n
    ss_tot = sum((v - mean_v) ** 2 for v in values)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"coefficients": coeffs, "residuals": residuals, "r_squared": r_sq}


def spatial_regression(
    centroids: Sequence[tuple[float, float]],
    dependent: Sequence[float],
    explanatory: Sequence[Sequence[float]],
    *,
    model_type: str = "ols",
    bandwidth: float | None = None,
    k: int | None = None,
) -> dict[str, Any]:
    """Spatial regression (OLS with optional spatial lag diagnostic).

    Args:
        model_type: ``"ols"`` for ordinary least squares, ``"spatial_lag"`` to
            add a spatial lag term.

    Returns:
        Dict with ``"coefficients"``, ``"r_squared"``, ``"residuals"``,
        ``"aic"``, ``"moran_residual"`` keys.
    """
    n = len(dependent)
    p = len(explanatory[0]) if explanatory else 0

    X: list[list[float]] = [[1.0] + [explanatory[i][j] for j in range(p)] for i in range(n)]
    y = list(dependent)

    if model_type == "spatial_lag" and (bandwidth or k):
        lag = spatial_lag(dependent, centroids, bandwidth=bandwidth, k=k)
        for i in range(n):
            X[i].append(lag[i])

    ncols = len(X[0])
    XtX = [[sum(X[r][i] * X[r][j] for r in range(n)) for j in range(ncols)] for i in range(ncols)]
    Xty = [sum(X[r][i] * y[r] for r in range(n)) for i in range(ncols)]
    coeffs = _solve_linear(XtX, Xty) or [0.0] * ncols

    predicted = [sum(coeffs[j] * X[i][j] for j in range(ncols)) for i in range(n)]
    residuals = [y[i] - predicted[i] for i in range(n)]
    ss_res = sum(r ** 2 for r in residuals)
    mean_y = sum(y) / n
    ss_tot = sum((v - mean_y) ** 2 for v in y)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    aic = n * math.log(ss_res / n) + 2 * ncols if ss_res > 0 and n > 0 else float("inf")

    moran_res = None
    if bandwidth or k:
        try:
            mr = morans_i(residuals, centroids, bandwidth=bandwidth, k=k)
            moran_res = mr["morans_i"]
        except Exception:
            pass

    return {
        "coefficients": coeffs,
        "r_squared": r_sq,
        "residuals": residuals,
        "aic": aic,
        "moran_residual": moran_res,
    }


def geographically_weighted_summary(
    centroids: Sequence[tuple[float, float]],
    values: Sequence[float],
    *,
    bandwidth: float | None = None,
    k: int = 15,
) -> list[dict[str, float]]:
    """Compute geographically weighted local mean and std for each observation."""
    n = len(values)
    weights_m = _build_distance_weights(centroids, bandwidth=bandwidth, k=k)

    results: list[dict[str, float]] = []
    for i in range(n):
        dists_w: list[tuple[float, float]] = []
        for j in range(n):
            if weights_m[i][j] > 0 or i == j:
                d = math.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
                bw = bandwidth if bandwidth else max(d for d in [math.hypot(centroids[i][0] - centroids[m][0], centroids[i][1] - centroids[m][1]) for m in range(n) if weights_m[i][m] > 0] if d > 0) or 1.0
                w = math.exp(-0.5 * (d / bw) ** 2) if bw > 0 else 0.0
                dists_w.append((values[j], w))

        total_w = sum(w for _, w in dists_w)
        if total_w == 0:
            results.append({"local_mean": values[i], "local_std": 0.0, "local_n": 0.0})
            continue
        local_mean = sum(v * w for v, w in dists_w) / total_w
        local_var = sum(w * (v - local_mean) ** 2 for v, w in dists_w) / total_w
        results.append({"local_mean": local_mean, "local_std": math.sqrt(local_var), "local_n": float(len(dists_w))})
    return results


def neighborhood_statistics(
    centroids: Sequence[tuple[float, float]],
    values: Sequence[float],
    *,
    bandwidth: float | None = None,
    k: int = 8,
    statistics: Sequence[str] = ("mean", "std", "min", "max", "sum"),
) -> list[dict[str, float]]:
    """Compute neighborhood statistics for each observation.

    For each point, statistics are computed over its neighbors defined by
    *bandwidth* or *k*.
    """
    n = len(values)
    weights_m = _build_distance_weights(centroids, bandwidth=bandwidth, k=k)
    results: list[dict[str, float]] = []
    for i in range(n):
        nbrs = [values[j] for j in range(n) if weights_m[i][j] > 0 or j == i]
        entry: dict[str, float] = {}
        if "mean" in statistics:
            entry["mean"] = sum(nbrs) / len(nbrs) if nbrs else 0.0
        if "std" in statistics:
            m = sum(nbrs) / len(nbrs) if nbrs else 0.0
            entry["std"] = math.sqrt(sum((v - m) ** 2 for v in nbrs) / len(nbrs)) if nbrs else 0.0
        if "min" in statistics:
            entry["min"] = min(nbrs) if nbrs else 0.0
        if "max" in statistics:
            entry["max"] = max(nbrs) if nbrs else 0.0
        if "sum" in statistics:
            entry["sum"] = sum(nbrs)
        if "count" in statistics:
            entry["count"] = float(len(nbrs))
        if "median" in statistics:
            s = sorted(nbrs)
            mid = len(s) // 2
            entry["median"] = (s[mid] + s[~mid]) / 2
        results.append(entry)
    return results


def zonal_statistics_by_class(
    zones: Sequence[Record],
    features: Sequence[Record],
    *,
    zone_field: str = "zone",
    value_field: str = "value",
    class_field: str = "class",
    statistics: Sequence[str] = ("count", "sum", "mean"),
    geometry_field: str = "geometry",
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute zonal statistics broken down by class within each zone.

    Returns nested dicts: ``{zone_value: {class_value: {stat: value}}}``.
    """
    from .geometry import geometry_contains

    zone_polys: list[tuple[Any, Any]] = []
    for z in zones:
        zone_polys.append((z.get(zone_field, "unknown"), z.get(geometry_field)))

    buckets: dict[str, dict[str, list[float]]] = {}
    for feat in features:
        fgeom = feat.get(geometry_field)
        fc = geometry_centroid(fgeom) if fgeom else None
        cls = str(feat.get(class_field, "default"))
        val = float(feat.get(value_field, 0))

        assigned_zone = "unassigned"
        if fc:
            for zname, zpoly in zone_polys:
                if zpoly and geometry_contains(zpoly, {"type": "Point", "coordinates": [fc["x"], fc["y"]]}):
                    assigned_zone = str(zname)
                    break
        buckets.setdefault(assigned_zone, {}).setdefault(cls, []).append(val)

    result: dict[str, dict[str, dict[str, float]]] = {}
    for zone, classes in buckets.items():
        result[zone] = {}
        for cls, vals in classes.items():
            stats: dict[str, float] = {}
            if "count" in statistics:
                stats["count"] = float(len(vals))
            if "sum" in statistics:
                stats["sum"] = sum(vals)
            if "mean" in statistics:
                stats["mean"] = sum(vals) / len(vals) if vals else 0.0
            if "std" in statistics:
                m = sum(vals) / len(vals) if vals else 0.0
                stats["std"] = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if vals else 0.0
            if "min" in statistics:
                stats["min"] = min(vals) if vals else 0.0
            if "max" in statistics:
                stats["max"] = max(vals) if vals else 0.0
            result[zone][cls] = stats
    return result


def point_pattern_analysis(
    centroids: Sequence[tuple[float, float]],
    *,
    study_area: float | None = None,
) -> dict[str, Any]:
    """Basic point pattern analysis: nearest-neighbor index and Ripley's K summary.

    Args:
        centroids: Point locations.
        study_area: Area of the study region (auto-estimated from bounding box if None).

    Returns:
        Dict with ``"nn_index"``, ``"nn_z_score"``, ``"expected_nn_distance"``,
        ``"observed_nn_distance"``, ``"ripley_k"`` (list of distance/K pairs).
    """
    n = len(centroids)
    if n < 2:
        return {"nn_index": 1.0, "nn_z_score": 0.0, "expected_nn_distance": 0.0,
                "observed_nn_distance": 0.0, "ripley_k": []}

    if study_area is None:
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]
        study_area = max((max(xs) - min(xs)) * (max(ys) - min(ys)), 1e-12)

    nn_dists: list[float] = []
    for i in range(n):
        min_d = float("inf")
        for j in range(n):
            if i == j:
                continue
            d = math.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
            if d < min_d:
                min_d = d
        nn_dists.append(min_d)

    obs_mean = sum(nn_dists) / n
    density = n / study_area
    expected_mean = 0.5 / math.sqrt(density) if density > 0 else 0.0
    se = 0.26136 / math.sqrt(n * density) if density > 0 else 1.0
    nn_index = obs_mean / expected_mean if expected_mean > 0 else 1.0
    z = (obs_mean - expected_mean) / se if se > 0 else 0.0

    # Ripley's K at several distances
    max_d = math.sqrt(study_area) / 2
    step = max_d / 10
    ripley: list[dict[str, float]] = []
    for t_idx in range(1, 11):
        t = step * t_idx
        count = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = math.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
                if d <= t:
                    count += 1
        K = study_area * count / (n * n)
        L = math.sqrt(K / math.pi) - t
        ripley.append({"distance": t, "K": K, "L_minus_t": L})

    return {
        "nn_index": nn_index,
        "nn_z_score": z,
        "expected_nn_distance": expected_mean,
        "observed_nn_distance": obs_mean,
        "ripley_k": ripley,
    }


def suitability_model(
    layers: Sequence[Sequence[float]],
    weights: Sequence[float],
    *,
    constraints: Sequence[Sequence[bool]] | None = None,
) -> list[float]:
    """Weighted overlay suitability model.

    Each layer is a sequence of scores (0-1) for each cell/feature.
    Weights are normalized to sum to 1. Optional boolean constraint layers
    mask out infeasible locations (False = excluded).

    Returns:
        Combined suitability scores.
    """
    n = len(layers[0]) if layers else 0
    total_w = sum(weights)
    norm_w = [w / total_w for w in weights] if total_w > 0 else [0.0] * len(weights)

    result: list[float] = []
    for i in range(n):
        if constraints:
            if any(not c[i] for c in constraints):
                result.append(0.0)
                continue
        score = sum(norm_w[k] * layers[k][i] for k in range(len(layers)))
        result.append(max(0.0, min(1.0, score)))
    return result


def multi_criteria_decision_analysis(
    alternatives: Sequence[str],
    criteria: Sequence[str],
    scores: Sequence[Sequence[float]],
    weights: Sequence[float],
    *,
    method: str = "weighted_sum",
) -> list[dict[str, Any]]:
    """Multi-criteria decision analysis (MCDA) with explainable scoring.

    Args:
        alternatives: Alternative names.
        criteria: Criteria names.
        scores: Matrix of scores [alternative][criterion].
        weights: Criteria weights.
        method: ``"weighted_sum"`` or ``"topsis"``.

    Returns:
        Ranked list of dicts with ``"alternative"``, ``"score"``,
        ``"rank"``, ``"breakdown"`` (per-criterion weighted scores).
    """
    n_alt = len(alternatives)
    n_crit = len(criteria)
    total_w = sum(weights)
    norm_w = [w / total_w for w in weights] if total_w > 0 else [0.0] * n_crit

    if method == "topsis":
        # Normalize scores
        col_norms = []
        for j in range(n_crit):
            ss = math.sqrt(sum(scores[i][j] ** 2 for i in range(n_alt)))
            col_norms.append(ss if ss > 0 else 1.0)
        normed = [[scores[i][j] / col_norms[j] for j in range(n_crit)] for i in range(n_alt)]
        weighted = [[normed[i][j] * norm_w[j] for j in range(n_crit)] for i in range(n_alt)]

        ideal = [max(weighted[i][j] for i in range(n_alt)) for j in range(n_crit)]
        anti_ideal = [min(weighted[i][j] for i in range(n_alt)) for j in range(n_crit)]

        results: list[dict[str, Any]] = []
        for i in range(n_alt):
            d_plus = math.sqrt(sum((weighted[i][j] - ideal[j]) ** 2 for j in range(n_crit)))
            d_minus = math.sqrt(sum((weighted[i][j] - anti_ideal[j]) ** 2 for j in range(n_crit)))
            closeness = d_minus / (d_plus + d_minus) if (d_plus + d_minus) > 0 else 0.0
            breakdown = {criteria[j]: weighted[i][j] for j in range(n_crit)}
            results.append({"alternative": alternatives[i], "score": closeness, "breakdown": breakdown})
    else:
        results = []
        for i in range(n_alt):
            weighted_scores = {criteria[j]: norm_w[j] * scores[i][j] for j in range(n_crit)}
            total = sum(weighted_scores.values())
            results.append({"alternative": alternatives[i], "score": total, "breakdown": weighted_scores})

    results.sort(key=lambda r: r["score"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["rank"] = rank
    return results


def territory_design(
    centroids: Sequence[tuple[float, float]],
    demands: Sequence[float],
    n_territories: int,
    *,
    max_iterations: int = 100,
) -> list[dict[str, Any]]:
    """Simple territory balancing using k-means style assignment.

    Assigns points to *n_territories* territories to balance total demand
    while maintaining spatial contiguity.

    Returns:
        List of dicts with ``"territory"``, ``"total_demand"``,
        ``"member_count"``, ``"centroid"`` keys.
    """
    import random

    n = len(centroids)
    if n < n_territories:
        raise ValueError("more territories requested than points")

    # Initialize centers using k-means++
    centers: list[tuple[float, float]] = [centroids[random.randint(0, n - 1)]]
    for _ in range(n_territories - 1):
        dists = [min(math.hypot(c[0] - cx, c[1] - cy) for cx, cy in centers) for c in centroids]
        total_d = sum(dists)
        r = random.random() * total_d
        cumsum = 0.0
        for idx, d in enumerate(dists):
            cumsum += d
            if cumsum >= r:
                centers.append(centroids[idx])
                break
        else:
            centers.append(centroids[-1])

    assignments = [0] * n
    for _ in range(max_iterations):
        changed = False
        for i in range(n):
            dists = [math.hypot(centroids[i][0] - centers[t][0], centroids[i][1] - centers[t][1])
                     for t in range(n_territories)]
            best = min(range(n_territories), key=lambda t: dists[t])
            if assignments[i] != best:
                assignments[i] = best
                changed = True
        if not changed:
            break
        for t in range(n_territories):
            members = [i for i in range(n) if assignments[i] == t]
            if members:
                centers[t] = (
                    sum(centroids[i][0] for i in members) / len(members),
                    sum(centroids[i][1] for i in members) / len(members),
                )

    territories: list[dict[str, Any]] = []
    for t in range(n_territories):
        members = [i for i in range(n) if assignments[i] == t]
        territories.append({
            "territory": t,
            "total_demand": sum(demands[i] for i in members),
            "member_count": len(members),
            "centroid": centers[t],
        })
    return territories


def catchment_analysis(
    facility_centroids: Sequence[tuple[float, float]],
    demand_centroids: Sequence[tuple[float, float]],
    demand_values: Sequence[float],
    *,
    max_distance: float,
) -> list[dict[str, Any]]:
    """Compute service catchments for facilities.

    Each demand point is assigned to the nearest facility within *max_distance*.

    Returns:
        Per-facility dict with ``"facility_index"``, ``"total_demand"``,
        ``"demand_count"``, ``"mean_distance"``.
    """
    n_fac = len(facility_centroids)
    buckets: dict[int, list[tuple[float, float]]] = {i: [] for i in range(n_fac)}

    for di, (dx, dy) in enumerate(demand_centroids):
        best_fac = -1
        best_dist = float("inf")
        for fi, (fx, fy) in enumerate(facility_centroids):
            d = math.hypot(dx - fx, dy - fy)
            if d <= max_distance and d < best_dist:
                best_dist = d
                best_fac = fi
        if best_fac >= 0:
            buckets[best_fac].append((demand_values[di], best_dist))

    results: list[dict[str, Any]] = []
    for fi in range(n_fac):
        items = buckets[fi]
        results.append({
            "facility_index": fi,
            "total_demand": sum(v for v, _ in items),
            "demand_count": len(items),
            "mean_distance": sum(d for _, d in items) / len(items) if items else 0.0,
        })
    return results


def location_allocation(
    candidate_facilities: Sequence[tuple[float, float]],
    demand_centroids: Sequence[tuple[float, float]],
    demand_values: Sequence[float],
    *,
    n_facilities: int,
    max_distance: float | None = None,
) -> dict[str, Any]:
    """Greedy location-allocation (p-median style) for facility siting.

    Selects *n_facilities* from candidates to minimize total weighted distance.

    Returns:
        Dict with ``"selected_facilities"``, ``"total_cost"``,
        ``"assignments"`` (demand index → facility index).
    """
    selected: list[int] = []
    remaining = list(range(len(candidate_facilities)))

    for _ in range(n_facilities):
        best_fac = -1
        best_cost = float("inf")
        for fi in remaining:
            trial = selected + [fi]
            cost = _allocation_cost(candidate_facilities, demand_centroids, demand_values, trial, max_distance)
            if cost < best_cost:
                best_cost = cost
                best_fac = fi
        if best_fac >= 0:
            selected.append(best_fac)
            remaining.remove(best_fac)

    assignments: dict[int, int] = {}
    for di, (dx, dy) in enumerate(demand_centroids):
        best = -1
        best_d = float("inf")
        for fi in selected:
            fx, fy = candidate_facilities[fi]
            d = math.hypot(dx - fx, dy - fy)
            if (max_distance is None or d <= max_distance) and d < best_d:
                best_d = d
                best = fi
        assignments[di] = best

    cost = _allocation_cost(candidate_facilities, demand_centroids, demand_values, selected, max_distance)
    return {
        "selected_facilities": selected,
        "total_cost": cost,
        "assignments": assignments,
    }


def _allocation_cost(
    facilities: Sequence[tuple[float, float]],
    demands: Sequence[tuple[float, float]],
    weights: Sequence[float],
    selected: list[int],
    max_distance: float | None,
) -> float:
    """Compute total weighted distance cost for a facility set."""
    total = 0.0
    for di, (dx, dy) in enumerate(demands):
        best_d = float("inf")
        for fi in selected:
            fx, fy = facilities[fi]
            d = math.hypot(dx - fx, dy - fy)
            if d < best_d:
                best_d = d
        if max_distance and best_d > max_distance:
            best_d = max_distance * 10  # penalty
        total += weights[di] * best_d
    return total


def od_matrix_optimization(
    origins: Sequence[tuple[float, float]],
    destinations: Sequence[tuple[float, float]],
    *,
    cost_field: str = "distance",
    n_nearest: int | None = None,
) -> dict[str, Any]:
    """Build and optionally optimize an origin-destination cost matrix.

    Returns:
        Dict with ``"matrix"`` (2D costs), ``"summary"`` per origin.
    """
    n_o = len(origins)
    n_d = len(destinations)
    matrix: list[list[float]] = []

    for ox, oy in origins:
        row = [math.hypot(ox - dx, oy - dy) for dx, dy in destinations]
        matrix.append(row)

    summary: list[dict[str, Any]] = []
    for i in range(n_o):
        sorted_dests = sorted(range(n_d), key=lambda j: matrix[i][j])
        nearest = sorted_dests[:n_nearest] if n_nearest else sorted_dests
        summary.append({
            "origin_index": i,
            "nearest_destinations": nearest,
            "nearest_costs": [matrix[i][j] for j in nearest],
            "mean_cost": sum(matrix[i]) / n_d if n_d > 0 else 0.0,
        })

    return {"matrix": matrix, "summary": summary}


def route_alternative_scoring(
    routes: Sequence[dict[str, Any]],
    *,
    criteria_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Score and rank route alternatives by weighted criteria.

    Each route dict should contain numeric fields used as criteria.
    Default weights give equal importance to all criteria.
    """
    if not routes:
        return []

    fields = [k for k in routes[0] if isinstance(routes[0][k], (int, float))]
    if criteria_weights is None:
        criteria_weights = {f: 1.0 for f in fields}

    # Normalize each field to 0-1 range
    ranges: dict[str, tuple[float, float]] = {}
    for f in fields:
        vals = [float(r.get(f, 0)) for r in routes]
        ranges[f] = (min(vals), max(vals))

    scored: list[dict[str, Any]] = []
    for route in routes:
        total = 0.0
        breakdown: dict[str, float] = {}
        for f in fields:
            lo, hi = ranges[f]
            norm = (float(route.get(f, 0)) - lo) / (hi - lo) if hi > lo else 0.5
            w = criteria_weights.get(f, 0.0)
            breakdown[f] = norm * w
            total += norm * w
        scored.append({**route, "composite_score": total, "score_breakdown": breakdown})

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    for rank, r in enumerate(scored, 1):
        r["rank"] = rank
    return scored


def scenario_comparison_engine(
    scenarios: Sequence[dict[str, Any]],
    *,
    metric_fields: Sequence[str] | None = None,
    baseline_index: int = 0,
) -> dict[str, Any]:
    """Compare multiple scenarios and compute metric deltas.

    Args:
        scenarios: Each scenario is a dict with name + numeric metric fields.
        metric_fields: Fields to compare (auto-detected if *None*).
        baseline_index: Which scenario is the baseline.

    Returns:
        Dict with ``"deltas"`` (per-scenario vs baseline), ``"ranking"``
        (best to worst by average improvement), and ``"recommendations"``.
    """
    if not scenarios:
        return {"deltas": [], "ranking": [], "recommendations": []}

    if metric_fields is None:
        metric_fields = [k for k in scenarios[0] if isinstance(scenarios[0].get(k), (int, float))]

    baseline = scenarios[baseline_index]
    deltas: list[dict[str, Any]] = []
    improvements: list[tuple[int, float]] = []

    for i, sc in enumerate(scenarios):
        sc_deltas: dict[str, float] = {}
        total_improvement = 0.0
        for f in metric_fields:
            base_v = float(baseline.get(f, 0))
            sc_v = float(sc.get(f, 0))
            delta = sc_v - base_v
            pct = (delta / base_v * 100) if base_v != 0 else 0.0
            sc_deltas[f] = pct
            total_improvement += pct
        deltas.append({"scenario_index": i, "name": sc.get("name", f"Scenario {i}"), "deltas_pct": sc_deltas})
        improvements.append((i, total_improvement / len(metric_fields) if metric_fields else 0.0))

    improvements.sort(key=lambda t: t[1], reverse=True)
    ranking = [{"scenario_index": idx, "avg_improvement_pct": imp} for idx, imp in improvements]

    recommendations: list[str] = []
    if improvements:
        best_idx = improvements[0][0]
        recommendations.append(f"Scenario '{scenarios[best_idx].get('name', best_idx)}' shows best overall improvement.")
        for f in metric_fields:
            best_f = max(range(len(scenarios)), key=lambda i: float(scenarios[i].get(f, 0)))
            if best_f != best_idx:
                recommendations.append(f"For '{f}', Scenario '{scenarios[best_f].get('name', best_f)}' is better.")

    return {"deltas": deltas, "ranking": ranking, "recommendations": recommendations}


# ---------------------------------------------------------------------------
# Descriptive spatial statistics (A4 items 461-468)
# ---------------------------------------------------------------------------


def mean_center(
    features: Sequence[Record],
    *,
    weight_field: str | None = None,
    geometry_column: str = "geometry",
) -> tuple[float, float]:
    """Return the (weighted) mean center of a set of features.

    Parameters
    ----------
    features : sequence of records
        Each record must have a *geometry_column* key with a valid geometry.
    weight_field : str, optional
        If provided, the centre is weighted by this numeric field.
    geometry_column : str
        Name of the geometry key in each record. Default ``"geometry"``.

    Returns
    -------
    tuple[float, float]
        ``(x, y)`` of the mean centre.
    """
    if not features:
        raise ValueError("features must not be empty")
    sum_x = 0.0
    sum_y = 0.0
    sum_w = 0.0
    for feat in features:
        cx, cy = geometry_centroid(feat[geometry_column])
        w = float(feat.get(weight_field, 1.0)) if weight_field else 1.0
        sum_x += cx * w
        sum_y += cy * w
        sum_w += w
    if sum_w == 0.0:
        raise ValueError("total weight is zero")
    return (sum_x / sum_w, sum_y / sum_w)


def median_center(
    features: Sequence[Record],
    *,
    geometry_column: str = "geometry",
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> tuple[float, float]:
    """Return the geometric median (Weiszfeld's algorithm) of feature centroids.

    The geometric median minimises the sum of Euclidean distances to all points.

    Parameters
    ----------
    features : sequence of records
        Each record must have a geometry key.
    geometry_column : str
        Name of the geometry key. Default ``"geometry"``.
    max_iterations : int
        Maximum iterations for Weiszfeld's algorithm. Default 1000.
    tolerance : float
        Convergence tolerance. Default 1e-8.

    Returns
    -------
    tuple[float, float]
        ``(x, y)`` of the median centre.
    """
    if not features:
        raise ValueError("features must not be empty")
    points = [geometry_centroid(f[geometry_column]) for f in features]
    # Start from mean
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    for _ in range(max_iterations):
        num_x = 0.0
        num_y = 0.0
        denom = 0.0
        for px, py in points:
            d = math.hypot(px - cx, py - cy)
            if d < 1e-15:
                continue
            w = 1.0 / d
            num_x += px * w
            num_y += py * w
            denom += w
        if denom == 0.0:
            break
        nx = num_x / denom
        ny = num_y / denom
        if math.hypot(nx - cx, ny - cy) < tolerance:
            cx, cy = nx, ny
            break
        cx, cy = nx, ny
    return (cx, cy)


def central_feature(
    features: Sequence[Record],
    *,
    geometry_column: str = "geometry",
) -> Record:
    """Return the feature whose centroid has the smallest total distance to all others.

    Parameters
    ----------
    features : sequence of records
        Must contain at least one feature.
    geometry_column : str
        Name of the geometry key. Default ``"geometry"``.

    Returns
    -------
    Record
        The feature that minimises total distance.
    """
    if not features:
        raise ValueError("features must not be empty")
    centroids = [geometry_centroid(f[geometry_column]) for f in features]
    best_idx = 0
    best_total = float("inf")
    for i, (xi, yi) in enumerate(centroids):
        total = sum(math.hypot(xi - xj, yi - yj) for j, (xj, yj) in enumerate(centroids) if j != i)
        if total < best_total:
            best_total = total
            best_idx = i
    return features[best_idx]


def standard_distance(
    features: Sequence[Record],
    *,
    geometry_column: str = "geometry",
) -> float:
    """Return the standard distance (dispersion around mean center).

    Standard distance is the spatial equivalent of standard deviation.

    Parameters
    ----------
    features : sequence of records
    geometry_column : str

    Returns
    -------
    float
        Standard distance in the coordinate system's units.
    """
    if not features:
        raise ValueError("features must not be empty")
    cx, cy = mean_center(features, geometry_column=geometry_column)
    n = len(features)
    sum_sq = 0.0
    for feat in features:
        px, py = geometry_centroid(feat[geometry_column])
        sum_sq += (px - cx) ** 2 + (py - cy) ** 2
    return math.sqrt(sum_sq / n)


def standard_deviational_ellipse(
    features: Sequence[Record],
    *,
    geometry_column: str = "geometry",
) -> dict[str, Any]:
    """Return the standard deviational ellipse parameters.

    Parameters
    ----------
    features : sequence of records
    geometry_column : str

    Returns
    -------
    dict
        Keys: ``center`` (x,y), ``sigma_x``, ``sigma_y``, ``rotation`` (radians).
    """
    if len(features) < 2:
        raise ValueError("need at least 2 features")
    cx, cy = mean_center(features, geometry_column=geometry_column)
    dx_list: list[float] = []
    dy_list: list[float] = []
    for feat in features:
        px, py = geometry_centroid(feat[geometry_column])
        dx_list.append(px - cx)
        dy_list.append(py - cy)

    n = len(features)
    sum_dx2 = sum(d * d for d in dx_list)
    sum_dy2 = sum(d * d for d in dy_list)
    sum_dxdy = sum(dx_list[i] * dy_list[i] for i in range(n))

    # Rotation angle
    a = sum_dx2 - sum_dy2
    b = math.sqrt(a * a + 4 * sum_dxdy * sum_dxdy)
    theta = math.atan2(a + b, 2 * sum_dxdy) if abs(sum_dxdy) > 1e-15 else 0.0

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    sigma_x_sq = sum((dx_list[i] * cos_t - dy_list[i] * sin_t) ** 2 for i in range(n)) / n
    sigma_y_sq = sum((dx_list[i] * sin_t + dy_list[i] * cos_t) ** 2 for i in range(n)) / n

    return {
        "center": (cx, cy),
        "sigma_x": math.sqrt(sigma_x_sq),
        "sigma_y": math.sqrt(sigma_y_sq),
        "rotation": theta,
    }


def directional_mean(
    features: Sequence[Record],
    *,
    geometry_column: str = "geometry",
) -> float:
    """Return the linear directional mean of line features (in radians, 0 = east, CCW).

    Parameters
    ----------
    features : sequence of records
        Each geometry should be a LineString.
    geometry_column : str

    Returns
    -------
    float
        Mean direction in radians.
    """
    if not features:
        raise ValueError("features must not be empty")
    sin_sum = 0.0
    cos_sum = 0.0
    for feat in features:
        geom = feat[geometry_column]
        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue
        sx, sy = coords[0]
        ex, ey = coords[-1]
        angle = math.atan2(ey - sy, ex - sx)
        sin_sum += math.sin(angle)
        cos_sum += math.cos(angle)
    return math.atan2(sin_sum, cos_sum)


def average_nearest_neighbor(
    features: Sequence[Record],
    *,
    geometry_column: str = "geometry",
    study_area: float | None = None,
) -> dict[str, float]:
    """Compute the average nearest-neighbour statistic.

    Parameters
    ----------
    features : sequence of records
    geometry_column : str
    study_area : float, optional
        Area of the study region. If omitted, the bounding-box area is used.

    Returns
    -------
    dict
        Keys: ``observed_mean_distance``, ``expected_mean_distance``,
        ``nearest_neighbor_ratio``, ``z_score``.
    """
    if len(features) < 2:
        raise ValueError("need at least 2 features")
    centroids = [geometry_centroid(f[geometry_column]) for f in features]
    n = len(centroids)

    # Compute nearest-neighbour distances
    nn_dists: list[float] = []
    for i in range(n):
        min_d = float("inf")
        for j in range(n):
            if i == j:
                continue
            d = math.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
            if d < min_d:
                min_d = d
        nn_dists.append(min_d)

    observed = sum(nn_dists) / n

    # Study area
    if study_area is None:
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]
        study_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if study_area <= 0:
        study_area = 1.0  # degenerate fallback

    density = n / study_area
    expected = 0.5 / math.sqrt(density) if density > 0 else 0.0
    se = 0.26136 / math.sqrt(n * density) if density > 0 else 1.0
    z_score = (observed - expected) / se if se > 0 else 0.0

    return {
        "observed_mean_distance": observed,
        "expected_mean_distance": expected,
        "nearest_neighbor_ratio": observed / expected if expected > 0 else 0.0,
        "z_score": z_score,
    }


def spatial_weights_queen(
    features: Sequence[Record],
    *,
    geometry_column: str = "geometry",
) -> list[list[int]]:
    """Build queen contiguity weights (shared edge or vertex).

    Returns an adjacency list: ``result[i]`` is the list of indices that share any
    boundary with feature *i*.
    """
    from .geometry import geometry_touches

    n = len(features)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if geometry_touches(features[i][geometry_column], features[j][geometry_column]):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def spatial_weights_rook(
    features: Sequence[Record],
    *,
    geometry_column: str = "geometry",
) -> list[list[int]]:
    """Build rook contiguity weights (shared edge only, not just a point).

    Two polygons are rook-neighbours if their intersection has dimension >= 1
    (i.e. they share an edge, not just a corner).
    """
    from .geometry import geometry_intersects

    n = len(features)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        gi = features[i][geometry_column]
        coords_i = set()
        for ring in gi.get("coordinates", []):
            if isinstance(ring, (list, tuple)):
                for pt in ring:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        coords_i.add((round(pt[0], 10), round(pt[1], 10)))
        for j in range(i + 1, n):
            gj = features[j][geometry_column]
            if not geometry_intersects(gi, gj):
                continue
            # Check shared point count; rook needs >= 2 shared boundary points
            coords_j = set()
            for ring in gj.get("coordinates", []):
                if isinstance(ring, (list, tuple)):
                    for pt in ring:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            coords_j.add((round(pt[0], 10), round(pt[1], 10)))
            shared = coords_i & coords_j
            if len(shared) >= 2:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def spatial_weights_knn(
    features: Sequence[Record],
    k: int = 4,
    *,
    geometry_column: str = "geometry",
) -> list[list[int]]:
    """Build k-nearest-neighbour spatial weights.

    Parameters
    ----------
    features : sequence of records
    k : int
        Number of neighbours. Default 4.
    geometry_column : str

    Returns
    -------
    list[list[int]]
        ``result[i]`` is the list of k nearest-neighbour indices for feature *i*.
    """
    centroids = [geometry_centroid(f[geometry_column]) for f in features]
    n = len(centroids)
    adj: list[list[int]] = []
    for i in range(n):
        dists = [(j, math.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])) for j in range(n) if j != i]
        dists.sort(key=lambda t: t[1])
        adj.append([j for j, _ in dists[:k]])
    return adj


__all__ = [
    "adaptive_kernel_density",
    "average_nearest_neighbor",
    "catchment_analysis",
    "central_feature",
    "cluster_outlier_analysis",
    "directional_mean",
    "gearys_c",
    "geographically_weighted_summary",
    "getis_ord_g",
    "hot_spot_analysis",
    "idw_interpolation",
    "kernel_density",
    "kriging_interpolation",
    "local_morans_i",
    "location_allocation",
    "mean_center",
    "median_center",
    "morans_i",
    "multi_criteria_decision_analysis",
    "neighborhood_statistics",
    "od_matrix_optimization",
    "point_pattern_analysis",
    "route_alternative_scoring",
    "scenario_comparison_engine",
    "semivariogram",
    "spatial_lag",
    "spatial_outliers",
    "spatial_regression",
    "spatial_weights_knn",
    "spatial_weights_matrix",
    "spatial_weights_queen",
    "spatial_weights_rook",
    "spline_interpolation",
    "standard_deviational_ellipse",
    "standard_distance",
    "suitability_model",
    "territory_design",
    "trend_surface",
    "zonal_statistics_by_class",
    # G5 additions
    "join_count_statistic",
    "nearest_neighbor_index",
    "head_tail_breaks",
    "fisher_jenks",
]


# ---------------------------------------------------------------------------
# G5 additions — spatial statistics
# ---------------------------------------------------------------------------

def join_count_statistic(frame: Any, spatial_weights: Any | None = None, *,
                         attribute: str = "class", permutations: int = 99) -> dict[str, float]:
    """Compute the Join Count statistic for a binary attribute.

    The Join Count statistic tests whether like-valued neighbouring polygons
    cluster spatially.  This is a pure-Python implementation for BB, BW, WW
    counts.

    Args:
        frame: A :class:`~geoprompt.GeoPromptFrame` with geometry and *attribute*.
        spatial_weights: Optional pre-computed spatial weights object with
            a ``neighbors`` mapping ``{id: [id, ...]}``; if ``None``, Queen
            contiguity is estimated from bounding-box overlap.
        attribute: Column name containing binary (0/1 or bool) class labels.
        permutations: Number of permutations for pseudo-p-value estimation.

    Returns:
        Dict with keys ``BB``, ``BW``, ``WW``, ``p_value_BB``.
    """
    import random, math
    rows = list(frame)
    n = len(rows)
    if n < 2:
        return {"BB": 0.0, "BW": 0.0, "WW": 0.0, "p_value_BB": 1.0}

    labels = [int(bool(r.get(attribute, 0))) for r in rows]

    # Build simple Queen neighbours from bounding-box overlap if no weights given
    if spatial_weights is None:
        def _bbox(geom: dict) -> tuple[float, float, float, float]:
            coords: list[tuple[float, float]] = []
            def _extract(c: Any) -> None:
                if isinstance(c[0], (int, float)):
                    coords.append((float(c[0]), float(c[1])))
                else:
                    for sub in c:
                        _extract(sub)
            _extract(geom.get("coordinates", [(0, 0)]))
            if not coords:
                return (0.0, 0.0, 0.0, 0.0)
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            return (min(xs), min(ys), max(xs), max(ys))

        geom_col = getattr(frame, "geometry_column", "geometry")
        bboxes = [_bbox(r.get(geom_col) or {"coordinates": [(0, 0)]}) for r in rows]

        def _overlaps(a: tuple, b: tuple) -> bool:
            return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

        neighbours: dict[int, list[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if _overlaps(bboxes[i], bboxes[j]):
                    neighbours[i].append(j)
                    neighbours[j].append(i)
    else:
        neighbours = spatial_weights.neighbors

    def _counts(labs: list[int]) -> tuple[float, float, float]:
        bb = bw = ww = 0
        for i, nbs in neighbours.items():
            for j in nbs:
                if j <= i:
                    continue
                a, b = labs[i], labs[j]
                if a == 1 and b == 1:
                    bb += 1
                elif a == 0 and b == 0:
                    ww += 1
                else:
                    bw += 1
        return float(bb), float(bw), float(ww)

    obs_bb, obs_bw, obs_ww = _counts(labels)

    # Permutation test
    count_bb_gte = 0
    shuffled = labels[:]
    for _ in range(permutations):
        random.shuffle(shuffled)
        perm_bb, _, _ = _counts(shuffled)
        if perm_bb >= obs_bb:
            count_bb_gte += 1
    p_val = (count_bb_gte + 1) / (permutations + 1)
    return {"BB": obs_bb, "BW": obs_bw, "WW": obs_ww, "p_value_BB": p_val}


def nearest_neighbor_index(frame: Any) -> dict[str, float]:
    """Compute the Average Nearest Neighbor Index (NNI) for point features.

    The NNI (Clark-Evans index) compares the observed mean nearest-neighbour
    distance to the expected distance under a Poisson process.

    Args:
        frame: A :class:`~geoprompt.GeoPromptFrame` with Point geometries.

    Returns:
        Dict with keys ``observed_mean``, ``expected_mean``, ``nni``,
        ``z_score``, and ``p_value``.
    """
    import math

    geom_col = getattr(frame, "geometry_column", "geometry")
    rows = list(frame)
    points = []
    for r in rows:
        geom = r.get(geom_col) or {}
        if geom.get("type") == "Point":
            c = geom.get("coordinates", (0.0, 0.0))
            points.append((float(c[0]), float(c[1])))

    n = len(points)
    if n < 2:
        return {"observed_mean": 0.0, "expected_mean": 0.0, "nni": 1.0, "z_score": 0.0, "p_value": 1.0}

    # Study area — bounding box area
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    area = max((max(xs) - min(xs)) * (max(ys) - min(ys)), 1e-12)

    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    nn_dists = []
    for i, p in enumerate(points):
        min_d = min(_dist(p, q) for j, q in enumerate(points) if j != i)
        nn_dists.append(min_d)

    obs_mean = sum(nn_dists) / n
    density = n / area
    exp_mean = 1.0 / (2.0 * math.sqrt(density))
    nni = obs_mean / exp_mean if exp_mean > 0 else 1.0
    se = math.sqrt((4 - math.pi) / (4 * math.pi * n * density + 1e-30))
    z = (obs_mean - exp_mean) / (se + 1e-30)
    # Two-tailed p (rough normal approximation)
    p = 2 * (1 - 0.5 * math.erfc(-abs(z) / math.sqrt(2)))
    return {"observed_mean": obs_mean, "expected_mean": exp_mean, "nni": nni, "z_score": z, "p_value": p}


def head_tail_breaks(values: list[float]) -> list[float]:
    """Compute class breaks using the Head/Tail classification scheme.

    Head/Tail Breaks iteratively splits data at the arithmetic mean,
    producing a hierarchy that is well-suited for heavy-tailed distributions.

    Args:
        values: A list of numeric values to classify.

    Returns:
        A sorted list of break points (including min and max).
    """
    if not values:
        return []
    arr = sorted(float(v) for v in values)
    breaks = [arr[0]]

    def _split(data: list[float]) -> None:
        if len(data) < 2:
            return
        mean = sum(data) / len(data)
        head = [v for v in data if v > mean]
        breaks.append(mean)
        if 1 <= len(head) < len(data):
            _split(head)

    _split(arr)
    breaks.append(arr[-1])
    return sorted(set(breaks))


def fisher_jenks(values: list[float], k: int = 5) -> list[float]:
    """Compute natural (Fisher-Jenks) classification breaks.

    Uses the Jenks optimization algorithm to minimise within-class variance.
    This is an O(n²k) implementation suitable for datasets up to ~50,000
    observations; for larger datasets use the ``jenkspy`` package.

    Args:
        values: Numeric values to classify.
        k: Number of classes.

    Returns:
        A sorted list of ``k + 1`` break points (including min and max).
    """
    # Try jenkspy first for speed
    try:
        import jenkspy  # type: ignore[import]
        return jenkspy.jenks_breaks(values, nb_class=k)
    except ImportError:
        pass

    arr = sorted(float(v) for v in values)
    n = len(arr)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)

    # Variance table initialisation (Jenks O(n²k))
    import math
    INF = float("inf")

    # matrices: lower_class_limits and variance_combinations
    LC = [[0] * (k + 1) for _ in range(n + 1)]
    VC = [[INF] * (k + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        LC[i][1] = 1
        VC[i][1] = 0.0

    for q in range(2, k + 1):
        for i in range(q, n + 1):
            sx = sx2 = 0.0
            for m in range(i, q - 1, -1):
                sx += arr[m - 1]
                sx2 += arr[m - 1] ** 2
                count = i - m + 1
                variance = sx2 - sx ** 2 / count
                j = m - 1
                if j != 0:
                    v = variance + VC[j][q - 1]
                    if v < VC[i][q]:
                        LC[i][q] = m
                        VC[i][q] = v
                else:
                    LC[i][q] = m
                    VC[i][q] = variance

    # Backtrack to find breaks
    breaks = [arr[-1]]
    klass = n
    for q in range(k, 0, -1):
        idx = LC[klass][q] - 1
        breaks.append(arr[idx] if idx >= 0 else arr[0])
        klass = LC[klass][q] - 1
    breaks.append(arr[0])
    return sorted(set(breaks))
