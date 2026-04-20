"""Remaining A4 spatial-analysis helpers and public wrappers for GeoPrompt."""

from __future__ import annotations

import math
from heapq import heappop, heappush
from typing import Any, Iterable, Sequence

from .environmental import (
    demand_from_land_use,
    flood_inundation,
    gaussian_plume,
    habitat_suitability_index,
    landslide_susceptibility,
    noise_propagation,
    pipe_sizing,
    rusle_erosion,
    scs_curve_number_runoff,
    sea_level_rise_inundation,
    solar_radiation_surface,
    sun_shadow_map,
    voltage_drop,
    wildfire_spread,
    wind_exposure_index,
)
from .spacetime import (
    build_turn_features,
    calibrate_lrs_route,
    exploratory_regression,
    multi_distance_cluster_analysis,
    pycnophylactic_interpolation,
    retail_trade_area,
    skater_regionalization,
    space_time_cube,
    space_time_pattern_mining,
    time_series_clustering,
)
from .spatial_analysis import dbscan_spatial
from .stats import idw_interpolation, kernel_density, kriging_interpolation


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _grid_mean(grid: Sequence[Sequence[float]]) -> float:
    vals = [float(v) for row in grid for v in row]
    return _mean(vals)


def _nearest_neighbours(points: Sequence[Sequence[float]], idx: int, k: int) -> list[int]:
    pairs = sorted(((_euclidean(points[idx], points[j]), j) for j in range(len(points)) if j != idx), key=lambda item: item[0])
    return [j for _, j in pairs[:k]]


def _dijkstra(
    graph: dict[Any, dict[Any, float]],
    start: Any,
    end: Any | None = None,
    *,
    blocked_edges: set[tuple[Any, Any]] | None = None,
) -> tuple[dict[Any, float], dict[Any, Any | None]]:
    blocked = blocked_edges or set()
    dist: dict[Any, float] = {start: 0.0}
    prev: dict[Any, Any | None] = {start: None}
    heap: list[tuple[float, Any]] = [(0.0, start)]
    while heap:
        cost, node = heappop(heap)
        if cost > dist.get(node, float("inf")):
            continue
        if end is not None and node == end:
            break
        for nxt, weight in graph.get(node, {}).items():
            if (node, nxt) in blocked or (nxt, node) in blocked:
                continue
            new_cost = cost + float(weight)
            if new_cost < dist.get(nxt, float("inf")):
                dist[nxt] = new_cost
                prev[nxt] = node
                heappush(heap, (new_cost, nxt))
    return dist, prev


def _reconstruct_path(prev: dict[Any, Any | None], end: Any) -> list[Any]:
    if end not in prev:
        return []
    path = [end]
    node = end
    while prev.get(node) is not None:
        node = prev[node]
        path.append(node)
    path.reverse()
    return path


def multi_scale_gwr(
    points: Sequence[Sequence[float]],
    y: Sequence[float],
    x_matrix: Sequence[Sequence[float]],
    *,
    bandwidths: Sequence[float],
) -> dict[str, Any]:
    """Approximate MGWR by fitting local weighted summaries for each bandwidth."""
    models: list[dict[str, Any]] = []
    n = len(points)
    if not points or not y or not x_matrix:
        return {"models": models}
    num_features = len(x_matrix[0])
    y_mean = _mean(y)
    ss_tot = sum((v - y_mean) ** 2 for v in y) or 1.0
    for bandwidth in bandwidths:
        local_predictions: list[float] = []
        local_coefficients: list[list[float]] = []
        for i in range(n):
            weights = []
            for j in range(n):
                d = _euclidean(points[i], points[j])
                w = math.exp(-(d ** 2) / max(2 * bandwidth ** 2, 1e-9))
                weights.append(w)
            coeffs = []
            for feature_idx in range(num_features):
                numer = sum(weights[j] * x_matrix[j][feature_idx] * y[j] for j in range(n))
                denom = sum(weights[j] * x_matrix[j][feature_idx] ** 2 for j in range(n)) or 1.0
                coeffs.append(numer / denom)
            prediction = sum(coeffs[k] * x_matrix[i][k] for k in range(num_features)) / max(num_features, 1)
            local_coefficients.append([round(c, 4) for c in coeffs])
            local_predictions.append(prediction)
        ss_res = sum((y[i] - local_predictions[i]) ** 2 for i in range(n))
        models.append({
            "bandwidth": bandwidth,
            "local_coefficients": local_coefficients,
            "local_predictions": [round(v, 4) for v in local_predictions],
            "r2": round(1.0 - ss_res / ss_tot, 4),
        })
    return {"models": models}


def local_bivariate_relationships(
    values_a: Sequence[float],
    values_b: Sequence[float],
    points: Sequence[Sequence[float]],
    *,
    k: int = 4,
) -> list[dict[str, Any]]:
    """Classify local bivariate relationships using z-scores and neighbour lags."""
    if not values_a or not values_b:
        return []
    mu_a, sd_a = _mean(values_a), _std(values_a) or 1.0
    mu_b, sd_b = _mean(values_b), _std(values_b) or 1.0
    out = []
    for i in range(len(points)):
        nb = _nearest_neighbours(points, i, min(k, len(points) - 1)) if len(points) > 1 else []
        lag_b = _mean([values_b[j] for j in nb]) if nb else values_b[i]
        za = (values_a[i] - mu_a) / sd_a
        zb = (lag_b - mu_b) / sd_b
        if za >= 0 and zb >= 0:
            rel = "high-high"
        elif za < 0 and zb < 0:
            rel = "low-low"
        elif za >= 0 and zb < 0:
            rel = "high-low"
        else:
            rel = "low-high"
        out.append({"index": i, "z_a": round(za, 4), "z_b": round(zb, 4), "relationship": rel})
    return out


def space_time_pattern_mining(
    events: Sequence[dict[str, Any]],
    x_field: str = "x",
    y_field: str = "y",
    t_field: str = "t",
    *,
    spatial_threshold: float = 1.0,
    temporal_threshold: float = 1.0,
) -> list[list[int]]:
    """Mine space-time clusters with a small-sample fallback."""
    from .spacetime import space_time_pattern_mining as _space_time_pattern_mining

    clusters = _space_time_pattern_mining(
        events,
        x_field=x_field,
        y_field=y_field,
        t_field=t_field,
        spatial_threshold=spatial_threshold,
        temporal_threshold=temporal_threshold,
    )
    if clusters:
        return clusters

    linked: list[int] = []
    for i in range(len(events)):
        for j in range(len(events)):
            if i == j:
                continue
            sd = _euclidean(
                (events[i][x_field], events[i][y_field]),
                (events[j][x_field], events[j][y_field]),
            )
            td = abs(float(events[i][t_field]) - float(events[j][t_field]))
            if sd <= spatial_threshold and td <= temporal_threshold:
                linked.extend([i, j])
    return [sorted(set(linked))] if linked else []


def emerging_hot_spot_analysis(cube: dict[int, dict[int, dict[int, int]]], temporal_bins: int, spatial_bins: int) -> dict[tuple[int, int], str]:
    """Classify hot-spot emergence from a space-time cube."""
    from .spacetime import emerging_hot_spot

    return emerging_hot_spot(cube, temporal_bins=temporal_bins, spatial_bins=spatial_bins)


def spatial_mann_kendall_analysis(series_by_location: dict[Any, Sequence[float]]) -> dict[Any, dict[str, Any]]:
    """Run Mann-Kendall trend analysis for each spatial unit."""
    from .environmental import mann_kendall_trend

    return {key: mann_kendall_trend(series) for key, series in series_by_location.items()}


def change_point_detection_spatial(series_by_location: dict[Any, Sequence[float]]) -> dict[Any, dict[str, Any]]:
    """Detect spatial change points using a mean-shift heuristic."""
    results: dict[Any, dict[str, Any]] = {}
    for key, series in series_by_location.items():
        if len(series) < 4:
            results[key] = {"change_points": [], "scores": []}
            continue
        best_idx = 1
        best_score = -1.0
        scores = []
        for i in range(1, len(series) - 1):
            left = _mean(series[:i])
            right = _mean(series[i:])
            score = abs(right - left)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_idx = i
        threshold = max(_std(series), 0.25)
        results[key] = {"change_points": [best_idx] if best_score >= threshold else [], "scores": scores, "max_statistic": best_score}
    return results


def spatially_constrained_multivariate_clustering(
    adjacency: dict[int, list[int]],
    attributes: dict[int, Sequence[float]],
    *,
    num_regions: int = 5,
) -> dict[int, int]:
    """Cluster multivariate attributes while respecting adjacency."""
    return skater_regionalization(adjacency, attributes, num_regions=num_regions)


def hdbscan_spatial_clustering(
    points: Sequence[Sequence[float]],
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
) -> dict[str, Any]:
    """Approximate HDBSCAN via adaptive DBSCAN neighborhood sizing."""
    if not points:
        return {"labels": [], "n_clusters": 0, "eps": 0.0}
    ms = min_samples or min_cluster_size
    neighbour_dists = []
    for i in range(len(points)):
        dists = sorted(_euclidean(points[i], points[j]) for j in range(len(points)) if j != i)
        idx = min(len(dists) - 1, max(0, ms - 1)) if dists else 0
        neighbour_dists.append(dists[idx] if dists else 0.0)
    eps = _mean(neighbour_dists) * 1.1 if neighbour_dists else 0.0
    result = dbscan_spatial(points, eps=eps or 0.5, min_samples=ms)
    result["eps"] = round(eps or 0.5, 4)
    return result


def regionalization_max_p(
    adjacency: dict[int, list[int]],
    values: dict[int, float],
    *,
    min_pop: float = 0.0,
    populations: dict[int, float] | None = None,
) -> dict[int, int]:
    """Public wrapper for max-p regionalization."""
    from .spacetime import max_p_regionalization

    return max_p_regionalization(adjacency, values, min_pop=min_pop, populations=populations)


def azp_regionalization(
    adjacency: dict[int, list[int]],
    attributes: dict[int, Sequence[float]],
    *,
    num_regions: int = 5,
) -> dict[int, int]:
    """Simplified AZP regionalization starting from SKATER-like seeds."""
    labels = skater_regionalization(adjacency, attributes, num_regions=num_regions)
    return labels


def weighted_overlay_raster(rasters: Sequence[Sequence[Sequence[float]]], weights: Sequence[float]) -> list[list[float]]:
    """Combine multiple rasters using weighted overlay."""
    if not rasters:
        return []
    rows = len(rasters[0])
    cols = len(rasters[0][0]) if rows else 0
    total_w = sum(weights) or 1.0
    out = [[0.0] * cols for _ in range(rows)]
    for raster, weight in zip(rasters, weights):
        for r in range(rows):
            for c in range(cols):
                out[r][c] += float(raster[r][c]) * weight / total_w
    return [[round(v, 4) for v in row] for row in out]


def universal_kriging(
    known_points: Sequence[tuple[float, float]],
    known_values: Sequence[float],
    query_points: Sequence[tuple[float, float]],
) -> list[float]:
    """Universal kriging approximation with a linear drift term."""
    base = kriging_interpolation(known_points, known_values, query_points)
    return [round(entry["value"] + 0.01 * (qx + qy), 4) for entry, (qx, qy) in zip(base, query_points)]


def co_kriging(
    known_points: Sequence[tuple[float, float]],
    primary_values: Sequence[float],
    secondary_values: Sequence[float],
    query_points: Sequence[tuple[float, float]],
) -> list[float]:
    """Simple co-kriging style blend of primary kriging and secondary trend."""
    primary = universal_kriging(known_points, primary_values, query_points)
    secondary = [v or 0.0 for v in idw_interpolation(known_points, secondary_values, query_points)]
    return [round(0.7 * p + 0.3 * s, 4) for p, s in zip(primary, secondary)]


def empirical_bayesian_kriging(
    known_points: Sequence[tuple[float, float]],
    known_values: Sequence[float],
    query_points: Sequence[tuple[float, float]],
) -> dict[str, list[float]]:
    """Approximate EBK by averaging multiple kriging runs with perturbed values."""
    predictions = [0.0] * len(query_points)
    variances = [0.0] * len(query_points)
    for shift in (-0.05, 0.0, 0.05):
        shifted = [v * (1.0 + shift) for v in known_values]
        result = kriging_interpolation(known_points, shifted, query_points)
        for i, entry in enumerate(result):
            predictions[i] += entry["value"] / 3.0
            variances[i] += entry["variance"] / 3.0
    return {"predictions": [round(v, 4) for v in predictions], "variance": [round(v, 4) for v in variances]}


def areal_interpolation_dasymetric(
    source_values: Sequence[float],
    source_areas: Sequence[float],
    overlap_areas: Sequence[Sequence[float]],
) -> list[float]:
    """Public wrapper for dasymetric areal interpolation."""
    from .spacetime import areal_interpolation

    return areal_interpolation(source_values, source_areas, overlap_areas)


def diffusion_interpolation_with_barriers(
    known_points: Sequence[tuple[float, float]],
    known_values: Sequence[float],
    query_points: Sequence[tuple[float, float]],
    *,
    barriers: Sequence[Sequence[int]] | None = None,
) -> list[float]:
    """Diffuse interpolation while damping predictions through barrier cells."""
    values = [v or 0.0 for v in idw_interpolation(known_points, known_values, query_points)]
    barrier_penalty = 0.85 if barriers and any(any(row) for row in barriers) else 1.0
    return [round(v * barrier_penalty, 4) for v in values]


def kernel_interpolation_with_barriers(
    known_points: Sequence[tuple[float, float]],
    known_values: Sequence[float],
    *,
    cell_size: float = 1.0,
    bandwidth: float = 1.0,
    barriers: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Kernel interpolation surface with optional barrier attenuation."""
    surface = kernel_density(known_points, weights=known_values, bandwidth=bandwidth, cell_size=cell_size)
    if barriers:
        rows = min(len(surface["grid"]), len(barriers))
        cols = min(len(surface["grid"][0]), len(barriers[0])) if rows else 0
        for r in range(rows):
            for c in range(cols):
                if barriers[r][c]:
                    surface["grid"][r][c] *= 0.25
    return surface


def prediction_standard_error_surface(
    known_points: Sequence[tuple[float, float]],
    known_values: Sequence[float],
    *,
    cell_size: float = 1.0,
) -> dict[str, Any]:
    """Create a simple prediction standard-error raster from sample spacing."""
    if not known_points:
        return {"grid": [], "rows": 0, "cols": 0}
    xs = [p[0] for p in known_points]
    ys = [p[1] for p in known_points]
    bounds = (min(xs), min(ys), max(xs) + cell_size, max(ys) + cell_size)
    min_x, min_y, max_x, max_y = bounds
    cols = max(1, int(math.ceil((max_x - min_x) / cell_size)))
    rows = max(1, int(math.ceil((max_y - min_y) / cell_size)))
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            qx = min_x + (c + 0.5) * cell_size
            qy = min_y + (r + 0.5) * cell_size
            nearest = min(_euclidean((qx, qy), p) for p in known_points)
            row.append(round(nearest, 4))
        grid.append(row)
    return {"grid": grid, "rows": rows, "cols": cols, "cell_size": cell_size, "bounds": bounds}


def drive_time_polygon(graph: dict[Any, dict[Any, float]], start: Any, *, max_cost: float) -> dict[str, Any]:
    """Approximate drive-time polygon as the set of reachable nodes within a cost threshold."""
    dist, _ = _dijkstra(graph, start)
    reachable = {node: round(cost, 4) for node, cost in dist.items() if cost <= max_cost}
    return {"mode": "drive", "start": start, "max_cost": max_cost, "reachable_nodes": sorted(reachable), "travel_costs": reachable}


def walk_time_polygon(graph: dict[Any, dict[Any, float]], start: Any, *, max_cost: float) -> dict[str, Any]:
    """Approximate walk-time polygon as reachable nodes within a walking threshold."""
    result = drive_time_polygon(graph, start, max_cost=max_cost)
    result["mode"] = "walk"
    return result


def retail_trade_area_analysis(store_location: tuple[float, float], customer_locations: Sequence[tuple[float, float]]) -> dict[str, float]:
    """Public wrapper for retail trade-area radii."""
    return retail_trade_area(store_location, customer_locations)


def route_with_barriers(
    graph: dict[Any, dict[Any, float]],
    start: Any,
    end: Any,
    *,
    barriers: Iterable[tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Shortest path routing with blocked edges."""
    dist, prev = _dijkstra(graph, start, end, blocked_edges=set(barriers or []))
    return {"path": _reconstruct_path(prev, end), "cost": dist.get(end, float("inf"))}


def route_with_time_windows(
    graph: dict[Any, dict[Any, float]],
    start: Any,
    end: Any,
    time_windows: dict[Any, tuple[float, float]],
) -> dict[str, Any]:
    """Route and report whether arrivals satisfy node time windows."""
    dist, prev = _dijkstra(graph, start, end)
    path = _reconstruct_path(prev, end)
    ok = True
    cumulative = 0.0
    for i in range(len(path)):
        node = path[i]
        if i > 0:
            cumulative += float(graph[path[i - 1]][node])
        window = time_windows.get(node)
        if window is not None and not (window[0] <= cumulative <= window[1]):
            ok = False
    return {"path": path, "cost": dist.get(end, float("inf")), "arrives_within_window": ok}


def multi_modal_routing(
    mode_graphs: dict[str, dict[Any, dict[Any, float]]],
    start: Any,
    end: Any,
    *,
    preferred_mode: str | None = None,
) -> dict[str, Any]:
    """Choose a route across multiple modal networks."""
    if preferred_mode and preferred_mode in mode_graphs:
        result = route_with_barriers(mode_graphs[preferred_mode], start, end)
        result["mode"] = preferred_mode
        return result
    best_mode = None
    best_result = {"path": [], "cost": float("inf")}
    for mode, graph in mode_graphs.items():
        result = route_with_barriers(graph, start, end)
        if result["cost"] < best_result["cost"]:
            best_mode = mode
            best_result = result
    best_result["mode"] = best_mode
    return best_result


def demand_estimation_from_land_use(land_use_areas: dict[str, float], demand_rates: dict[str, float] | None = None) -> dict[str, Any]:
    """Public wrapper for land-use demand estimation."""
    return demand_from_land_use(land_use_areas, demand_rates=demand_rates)


def load_flow_analysis(loads: dict[Any, float] | Sequence[dict[str, Any]], *, losses: float = 0.0) -> dict[str, Any]:
    """Estimate aggregate load flow for water or electric networks."""
    if isinstance(loads, dict):
        values = [float(v) for v in loads.values()]
    else:
        values = [float(item.get("load", item.get("current_amps", 0.0))) for item in loads]
    total = sum(values)
    delivered = total * (1.0 - losses)
    return {"total_load": round(total, 4), "delivered_load": round(delivered, 4), "losses": losses}


def pipe_sizing_from_flow_velocity(flow_rate_m3s: float, target_velocity_ms: float = 1.5) -> dict[str, float]:
    """Public wrapper for pipe sizing."""
    return pipe_sizing(flow_rate_m3s, target_velocity_ms)


def water_age_analysis(pipes: Sequence[dict[str, Any]], *, source_age_hours: float = 0.0) -> list[dict[str, Any]]:
    """Estimate water age along a sequence of pipes."""
    age = source_age_hours
    out = []
    for pipe in pipes:
        velocity = max(float(pipe.get("velocity_ms", 1.0)), 1e-6)
        travel_hours = float(pipe.get("length_m", 0.0)) / velocity / 3600.0
        age += travel_hours
        out.append({**pipe, "water_age_hours": round(age, 4)})
    return out


def chlorine_decay_analysis(age_results: Sequence[dict[str, Any]], *, initial_mg_l: float = 1.0, decay_constant: float = 0.1) -> list[dict[str, Any]]:
    """Apply first-order chlorine decay to water-age results."""
    out = []
    for item in age_results:
        age = float(item.get("water_age_hours", 0.0))
        residual = initial_mg_l * math.exp(-decay_constant * age)
        out.append({**item, "chlorine_mg_l": round(residual, 4)})
    return out


def short_circuit_analysis(voltage_v: float, impedance_ohms: float) -> dict[str, float]:
    """Compute short-circuit fault current from source voltage and impedance."""
    fault_current = voltage_v / max(impedance_ohms, 1e-9)
    return {"fault_current_a": round(fault_current, 4), "impedance_ohms": impedance_ohms}


def voltage_drop_calculation(current_amps: float, length_m: float, conductor_resistance_per_km: float, *, voltage_supply: float = 240.0, phases: int = 1) -> dict[str, float]:
    """Public wrapper for voltage-drop calculation."""
    return voltage_drop(current_amps, length_m, conductor_resistance_per_km, voltage_supply=voltage_supply, phases=phases)


def transformer_loading_analysis(loads_kva: Sequence[float], *, rated_kva: float) -> dict[str, Any]:
    """Summarise transformer loading against its rated capacity."""
    pcts = [100.0 * float(v) / rated_kva if rated_kva else 0.0 for v in loads_kva]
    return {"loadings_pct": [round(v, 4) for v in pcts], "max_loading_pct": round(max(pcts or [0.0]), 4), "overloaded": any(v > 100 for v in pcts)}


def fault_location_analysis(sensor_currents: Sequence[float]) -> dict[str, Any]:
    """Locate the most likely fault from current anomalies."""
    if not sensor_currents:
        return {"fault_index": None, "severity": 0.0}
    idx = max(range(len(sensor_currents)), key=lambda i: sensor_currents[i])
    return {"fault_index": idx, "severity": float(sensor_currents[idx])}


def protection_coordination_analysis(downstream_trip_times: Sequence[float], upstream_trip_times: Sequence[float]) -> dict[str, Any]:
    """Check that upstream protection delays exceed downstream clearing times."""
    margins = [u - d for d, u in zip(downstream_trip_times, upstream_trip_times)]
    return {"coordinated": all(m > 0 for m in margins), "minimum_margin": round(min(margins or [0.0]), 4)}


def turn_feature_class(edges: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Public wrapper for turn-feature generation."""
    return build_turn_features(edges)


def location_referencing_system(route: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Build an LRS measure set from route vertices."""
    measures = [0.0]
    for i in range(1, len(route)):
        measures.append(measures[-1] + _euclidean(route[i - 1], route[i]))
    return {"route": list(route), "measures": [round(m, 4) for m in measures], "total_length": round(measures[-1] if measures else 0.0, 4)}


def calibrate_lrs_routes(route: Sequence[tuple[float, float]], calibration_points: Sequence[tuple[float, float, float]]) -> list[float]:
    """Public wrapper for LRS calibration."""
    return calibrate_lrs_route(route, calibration_points)


def lrs_event_overlay_tool(route_measures: Sequence[float], events: Sequence[tuple[float, float]]) -> list[tuple[int, int]]:
    """Public wrapper for LRS event overlay."""
    from .spacetime import lrs_event_overlay

    return lrs_event_overlay(route_measures, events)


def _lookup_geocode(query: str, locators: Sequence[dict[str, Any]], preferred_type: str | None = None) -> dict[str, Any]:
    for item in locators:
        if preferred_type is not None and item.get("type") != preferred_type:
            continue
        if str(item.get("name", "")).lower() == query.lower():
            return {
                "matched": True,
                "match_type": item.get("type"),
                "x": item.get("x"),
                "y": item.get("y"),
                "name": item.get("name"),
            }
    return {"matched": False, "match_type": preferred_type, "x": None, "y": None, "name": query}


def parcel_centroid_geocoding(query: str, locators: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Geocode a parcel-centroid style locator."""
    return _lookup_geocode(query, locators, preferred_type="parcel")


def rooftop_level_geocoding(query: str, locators: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Geocode a rooftop-style locator."""
    return _lookup_geocode(query, locators, preferred_type="rooftop")


def geocoding_composite_locator(query: str, locators: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Search multiple locator types in sequence."""
    for kind in ("rooftop", "parcel", None):
        result = _lookup_geocode(query, locators, preferred_type=kind)
        if result["matched"]:
            return result
    return {"matched": False, "name": query}


def _single_observer_viewshed(dem: Sequence[Sequence[float]], observer: tuple[int, int]) -> list[list[int]]:
    rows = len(dem)
    cols = len(dem[0]) if rows else 0
    orow, ocol = observer
    if not (0 <= orow < rows and 0 <= ocol < cols):
        return [[0] * cols for _ in range(rows)]
    observer_elev = float(dem[orow][ocol])
    visible = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            d = math.hypot(r - orow, c - ocol)
            if d == 0:
                visible[r][c] = 1
            else:
                target = float(dem[r][c])
                threshold = observer_elev - 0.15 * d
                visible[r][c] = 1 if target >= threshold else 0
    return visible


def viewshed_multiple_observers(dem: Sequence[Sequence[float]], observers: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Combine viewsheds from multiple observers."""
    rows = len(dem)
    cols = len(dem[0]) if rows else 0
    out = [[0] * cols for _ in range(rows)]
    for observer in observers:
        view = _single_observer_viewshed(dem, observer)
        for r in range(rows):
            for c in range(cols):
                out[r][c] = max(out[r][c], view[r][c])
    return out


def observer_points_analysis(dem: Sequence[Sequence[float]], observers: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
    """Summarise visible-cell counts for multiple observer points."""
    results = []
    for observer in observers:
        view = _single_observer_viewshed(dem, observer)
        count = sum(sum(row) for row in view)
        results.append({"observer": observer, "visible_cells": count})
    return results


def visibility_frequency_surface(dem: Sequence[Sequence[float]], observers: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Count the number of observers that can see each cell."""
    rows = len(dem)
    cols = len(dem[0]) if rows else 0
    freq = [[0] * cols for _ in range(rows)]
    for observer in observers:
        view = _single_observer_viewshed(dem, observer)
        for r in range(rows):
            for c in range(cols):
                freq[r][c] += view[r][c]
    return freq


def sun_shadow_volume(elevation_grid: Sequence[Sequence[float]], *, sun_altitude: float, sun_azimuth: float, cell_size: float = 1.0) -> dict[str, Any]:
    """Compute a simple sun-shadow volume summary from terrain."""
    grid = sun_shadow_map(elevation_grid, sun_altitude, sun_azimuth, cell_size=cell_size)
    shadow_cells = sum(sum(row) for row in grid)
    return {"grid": grid, "shadow_cells": shadow_cells, "shadow_fraction": shadow_cells / max(1, len(grid) * len(grid[0]) if grid else 1)}


def solar_radiation_surface_analysis(slope_grid: Sequence[Sequence[float]], aspect_grid: Sequence[Sequence[float]], *, latitude: float, day_of_year: int = 172) -> list[list[float]]:
    """Public wrapper for solar-radiation surface analysis."""
    return solar_radiation_surface(slope_grid, aspect_grid, latitude, day_of_year=day_of_year)


def wind_exposure_surface(elevation_grid: Sequence[Sequence[float]], *, search_radius: int = 3) -> list[list[float]]:
    """Public wrapper for wind-exposure surface analysis."""
    return wind_exposure_index(elevation_grid, search_radius=search_radius)


def noise_propagation_model(source: tuple[float, float], source_power_db: float, receptor_points: Sequence[tuple[float, float]], *, ground_absorption: float = 0.5, frequency_hz: float = 500.0) -> list[float]:
    """Public wrapper for noise propagation."""
    return noise_propagation(source, source_power_db, receptor_points, ground_absorption=ground_absorption, frequency_hz=frequency_hz)


def air_quality_dispersion_model(source: tuple[float, float], emission_rate: float, wind_speed: float, wind_direction: float, receptor_points: Sequence[tuple[float, float]], *, stack_height: float = 0.0, stability_class: str = "D") -> list[float]:
    """Public wrapper for Gaussian plume dispersion."""
    return gaussian_plume(source, emission_rate, wind_speed, wind_direction, receptor_points, stack_height=stack_height, stability_class=stability_class)


def flood_inundation_from_dem(dem: Sequence[Sequence[float]], *, water_level: float, nodata: float = -9999.0) -> list[list[int]]:
    """Public wrapper for DEM flood inundation."""
    return flood_inundation(dem, water_level, nodata=nodata)


def sea_level_rise_inundation_analysis(dem: Sequence[Sequence[float]], sea_level_rise: float, *, current_sea_level: float = 0.0, nodata: float = -9999.0) -> list[list[int]]:
    """Public wrapper for sea-level-rise inundation."""
    return sea_level_rise_inundation(dem, sea_level_rise, current_sea_level=current_sea_level, nodata=nodata)


def rainfall_runoff_model(rainfall: float, curve_number: float, *, initial_abstraction_ratio: float = 0.2) -> float:
    """Public wrapper for SCS-CN rainfall-runoff modelling."""
    return scs_curve_number_runoff(rainfall, curve_number, initial_abstraction_ratio=initial_abstraction_ratio)


def erosion_risk_model_rusle(r_factor: float, k_factor: float, ls_factor: float, c_factor: float, p_factor: float) -> float:
    """Public wrapper for RUSLE erosion risk."""
    return rusle_erosion(r_factor, k_factor, ls_factor, c_factor, p_factor)


def landslide_susceptibility_model(factors: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    """Public wrapper for landslide susceptibility analysis."""
    return landslide_susceptibility(factors, weights)


def wildfire_spread_model(fuel_grid: Sequence[Sequence[float]], ignition: tuple[int, int], *, spread_threshold: float = 0.5, max_steps: int = 100) -> list[list[int]]:
    """Public wrapper for wildfire spread modelling."""
    return wildfire_spread(fuel_grid, ignition, spread_threshold=spread_threshold, max_steps=max_steps)


def habitat_suitability_model(factors: dict[str, Sequence[float]], weights: dict[str, float] | None = None) -> list[float]:
    """Public wrapper for habitat suitability modelling."""
    return habitat_suitability_index(factors, weights=weights)


def species_distribution_model_maxent(records: Sequence[dict[str, Any]], predictor_fields: Sequence[str]) -> dict[str, Any]:
    """Lightweight MaxEnt-style habitat suitability scoring from predictor fields."""
    if not records or not predictor_fields:
        return {"scores": [], "importance": {}}
    field_means = {field: _mean([float(r.get(field, 0.0)) for r in records]) for field in predictor_fields}
    importance = {field: abs(val) for field, val in field_means.items()}
    total = sum(importance.values()) or 1.0
    importance = {field: round(val / total, 4) for field, val in importance.items()}
    scores = []
    for rec in records:
        vals = [float(rec.get(field, 0.0)) for field in predictor_fields]
        normalised = []
        for value, field in zip(vals, predictor_fields):
            denom = field_means[field] or 1.0
            normalised.append(min(max(value / denom, 0.0), 2.0) / 2.0)
        scores.append(round(_mean(normalised), 4))
    return {"scores": scores, "importance": importance}


def circuit_theory_connectivity(resistance_grid: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Estimate circuit-theory connectivity from a resistance surface."""
    vals = [float(v) for row in resistance_grid for v in row]
    mean_resistance = _mean(vals) or 1.0
    connectivity = 1.0 / mean_resistance
    return {"connectivity_score": round(connectivity, 4), "mean_resistance": round(mean_resistance, 4)}


def graph_theory_connectivity(graph: dict[str, Any]) -> dict[str, Any]:
    """Compute basic graph-theory connectivity metrics."""
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    degrees = {node: 0 for node in nodes}
    for u, v in edges:
        degrees[u] = degrees.get(u, 0) + 1
        degrees[v] = degrees.get(v, 0) + 1
    edge_connectivity = min(degrees.values()) if degrees else 0
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "edge_connectivity": edge_connectivity,
        "density": round((2 * len(edges)) / max(1, len(nodes) * max(len(nodes) - 1, 1)), 4),
    }


__all__ = [
    "air_quality_dispersion_model",
    "areal_interpolation_dasymetric",
    "azp_regionalization",
    "change_point_detection_spatial",
    "chlorine_decay_analysis",
    "circuit_theory_connectivity",
    "co_kriging",
    "demand_estimation_from_land_use",
    "drive_time_polygon",
    "emerging_hot_spot_analysis",
    "empirical_bayesian_kriging",
    "erosion_risk_model_rusle",
    "exploratory_regression",
    "fault_location_analysis",
    "flood_inundation_from_dem",
    "geocoding_composite_locator",
    "graph_theory_connectivity",
    "habitat_suitability_model",
    "hdbscan_spatial_clustering",
    "kernel_interpolation_with_barriers",
    "landslide_susceptibility_model",
    "load_flow_analysis",
    "local_bivariate_relationships",
    "location_referencing_system",
    "lrs_event_overlay_tool",
    "multi_distance_cluster_analysis",
    "multi_modal_routing",
    "multi_scale_gwr",
    "noise_propagation_model",
    "observer_points_analysis",
    "parcel_centroid_geocoding",
    "pipe_sizing_from_flow_velocity",
    "prediction_standard_error_surface",
    "protection_coordination_analysis",
    "pycnophylactic_interpolation",
    "rainfall_runoff_model",
    "regionalization_max_p",
    "retail_trade_area_analysis",
    "rooftop_level_geocoding",
    "route_with_barriers",
    "route_with_time_windows",
    "sea_level_rise_inundation_analysis",
    "skater_regionalization",
    "solar_radiation_surface_analysis",
    "space_time_cube",
    "space_time_pattern_mining",
    "spatial_mann_kendall_analysis",
    "spatially_constrained_multivariate_clustering",
    "species_distribution_model_maxent",
    "short_circuit_analysis",
    "sun_shadow_volume",
    "time_series_clustering",
    "transformer_loading_analysis",
    "turn_feature_class",
    "universal_kriging",
    "viewshed_multiple_observers",
    "visibility_frequency_surface",
    "voltage_drop_calculation",
    "walk_time_polygon",
    "water_age_analysis",
    "weighted_overlay_raster",
    "wildfire_spread_model",
    "wind_exposure_surface",
    "calibrate_lrs_routes",
    "diffusion_interpolation_with_barriers",
]
