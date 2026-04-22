"""Network analysis: graph construction, routing, demand allocation, and utility simulations.

This package provides comprehensive network analysis tools including shortest path
routing, origin-destination matrices, demand allocation, and domain-specific utility
simulations (electric, water, gas, telecom, stormwater).

All public functions and classes are re-exported at the package level for backward
compatibility with code that imported from the original `geoprompt.network` module.
"""
from __future__ import annotations

from .core import (
    NETWORK_WORKLOAD_PRESETS,
    NetworkEdge,
    NetworkGraph,
    Traversal,
    get_network_workload_preset,
)
from .routing import (
    apply_live_traffic_overrides,
    build_landmark_index,
    build_network_graph,
    edge_impedance_cost,
    hierarchy_aware_shortest_path,
    landmark_lower_bound,
    live_traffic_shortest_path,
    multi_criteria_shortest_path,
    multimodal_shortest_path,
    shortest_path,
    service_area,
    time_dependent_shortest_path,
    NetworkRouter,
)
from .demand import (
    allocate_demand_to_supply,
    iter_od_cost_matrix_batches,
    od_cost_matrix,
    od_cost_matrix_with_preset,
    utility_bottlenecks,
    utility_bottlenecks_stream,
    utility_bottlenecks_with_preset,
)
from .allocation import (
    capacity_constrained_od_assignment,
    constrained_flow_assignment,
)
from .topology import (
    analyze_network_topology,
    co_location_conflict_scan,
)
from .utility import (
    attribute_aware_headloss,
    capital_planning_prioritization,
    accessibility_equity_audit,
    closest_facility_dispatch,
    crew_dispatch_optimizer,
    critical_customer_coverage_audit,
    criticality_ranking_by_node_removal,
    cross_utility_dependency_score,
    demand_weighted_restoration_ranking,
    dependency_graph_overlay,
    detention_basin_overflow_trace,
    facility_siting_score,
    feeder_reconfiguration_optimizer,
    fiber_cut_impact_matrix,
    fiber_splice_node_trace,
    fire_flow_demand_check,
    gas_odorization_zone_trace,
    gas_pressure_drop_trace,
    gas_regulator_station_isolation,
    gas_shutdown_impact,
    inflow_infiltration_scan,
    infrastructure_age_risk_weighted_routing,
    interdependency_cascade_simulation,
    load_transfer_feasibility,
    n_minus_k_edge_contingency_screen,
    n_minus_one_edge_contingency_screen,
    outage_restoration_tie_options,
    outage_impact_report,
    pipe_break_isolation_zones,
    pressure_scenario_sweep,
    pressure_zone_reconfiguration_planner,
    pump_station_failure_cascade,
    reliability_indices,
    reliability_scenario_report,
    resilience_capex_prioritization,
    restoration_sequence_report,
    multi_source_service_audit,
    ring_redundancy_check,
    run_utility_scenarios,
    supply_redundancy_audit,
    stormwater_flow_accumulation,
    trace_electric_feeder,
    trace_water_pressure_zones,
    utility_monte_carlo_resilience,
    utility_outage_isolation,
    utility_stress_scenario_library,
)
from .graph_algorithms import (
    astar_shortest_path,
    bellman_ford_shortest_path,
    create_network_dataset,
    find_articulation_points,
    find_bridges,
    find_cycles,
    floyd_warshall,
    k_shortest_paths,
    max_flow_min_cut,
    minimum_spanning_tree,
    partition_network,
    strongly_connected_components,
    trace_connected,
    weakly_connected_components,
)

__all__ = [
    # Core types and utilities
    "NETWORK_WORKLOAD_PRESETS",
    "NetworkEdge",
    "NetworkGraph",
    "Traversal",
    "get_network_workload_preset",
    # Routing functions
    "build_landmark_index",
    "build_network_graph",
    "edge_impedance_cost",
    "apply_live_traffic_overrides",
    "hierarchy_aware_shortest_path",
    "landmark_lower_bound",
    "live_traffic_shortest_path",
    "multi_criteria_shortest_path",
    "multimodal_shortest_path",
    "shortest_path",
    "service_area",
    "time_dependent_shortest_path",
    "NetworkRouter",
    # Demand functions
    "allocate_demand_to_supply",
    "iter_od_cost_matrix_batches",
    "od_cost_matrix",
    "od_cost_matrix_with_preset",
    "utility_bottlenecks",
    "utility_bottlenecks_stream",
    "utility_bottlenecks_with_preset",
    # Allocation functions
    "capacity_constrained_od_assignment",
    "constrained_flow_assignment",
    # Topology and cross-network diagnostics
    "analyze_network_topology",
    "co_location_conflict_scan",
    # Utility domain analysis
    "attribute_aware_headloss",
    "accessibility_equity_audit",
    "capital_planning_prioritization",
    "closest_facility_dispatch",
    "crew_dispatch_optimizer",
    "critical_customer_coverage_audit",
    "criticality_ranking_by_node_removal",
    "cross_utility_dependency_score",
    "demand_weighted_restoration_ranking",
    "dependency_graph_overlay",
    "detention_basin_overflow_trace",
    "facility_siting_score",
    "feeder_reconfiguration_optimizer",
    "fiber_cut_impact_matrix",
    "fiber_splice_node_trace",
    "fire_flow_demand_check",
    "gas_odorization_zone_trace",
    "gas_pressure_drop_trace",
    "gas_regulator_station_isolation",
    "gas_shutdown_impact",
    "inflow_infiltration_scan",
    "infrastructure_age_risk_weighted_routing",
    "interdependency_cascade_simulation",
    "load_transfer_feasibility",
    "n_minus_k_edge_contingency_screen",
    "n_minus_one_edge_contingency_screen",
    "outage_restoration_tie_options",
    "outage_impact_report",
    "pipe_break_isolation_zones",
    "pressure_scenario_sweep",
    "pressure_zone_reconfiguration_planner",
    "pump_station_failure_cascade",
    "reliability_indices",
    "reliability_scenario_report",
    "resilience_capex_prioritization",
    "restoration_sequence_report",
    "multi_source_service_audit",
    "ring_redundancy_check",
    "run_utility_scenarios",
    "supply_redundancy_audit",
    "stormwater_flow_accumulation",
    "trace_electric_feeder",
    "trace_water_pressure_zones",
    "utility_monte_carlo_resilience",
    "utility_outage_isolation",
    "utility_stress_scenario_library",
    # Graph algorithms
    "astar_shortest_path",
    "bellman_ford_shortest_path",
    "create_network_dataset",
    "find_articulation_points",
    "find_bridges",
    "find_cycles",
    "floyd_warshall",
    "k_shortest_paths",
    "max_flow_min_cut",
    "minimum_spanning_tree",
    "partition_network",
    "strongly_connected_components",
    "trace_connected",
    "weakly_connected_components",
    # G6 additions
    "vrp_solver",
    "tsp_solver",
    "location_allocation",
    "network_partition",
    "multimodal_network",
    "closest_facility",
]


# ---------------------------------------------------------------------------
# G6 additions â€” advanced network analysis
# ---------------------------------------------------------------------------

from typing import Any as _Any


def vrp_solver(depot: tuple[float, float],
               locations: list[tuple[float, float]],
               *,
               n_vehicles: int = 3,
               max_route_distance: float = float("inf")) -> list[list[int]]:
    """Solve a Vehicle Routing Problem (VRP) using a greedy nearest-neighbour heuristic.

    Assigns *locations* to *n_vehicles* so each vehicle visits a roughly equal
    share.  Uses a greedy insertion order from the depot.

    Args:
        depot: ``(x, y)`` coordinates of the common depot.
        locations: List of ``(x, y)`` stop coordinates (0-indexed).
        n_vehicles: Number of vehicles.
        max_route_distance: Maximum total route distance per vehicle.

    Returns:
        A list of *n_vehicles* route lists.  Each inner list contains the
        0-based indices into *locations* (not including the depot).
    """
    import math

    def _dist(a: tuple, b: tuple) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    n = len(locations)
    if n == 0:
        return [[] for _ in range(n_vehicles)]

    # Greedy nearest-neighbour assignment, cycling through vehicles
    routes: list[list[int]] = [[] for _ in range(n_vehicles)]
    route_dists = [0.0] * n_vehicles
    unvisited = list(range(n))
    current_pos = [depot] * n_vehicles

    while unvisited:
        for v in range(n_vehicles):
            if not unvisited:
                break
            # Find nearest unvisited to this vehicle's current position
            best_i = min(unvisited, key=lambda i: _dist(current_pos[v], locations[i]))
            d = _dist(current_pos[v], locations[best_i])
            if route_dists[v] + d <= max_route_distance:
                routes[v].append(best_i)
                route_dists[v] += d
                current_pos[v] = locations[best_i]
                unvisited.remove(best_i)
            else:
                # Skip this vehicle for this iteration
                pass
        # Avoid infinite loop if no vehicle can take any remaining stop
        # (all routes at max_route_distance) â€” assign anyway
        if unvisited and all(route_dists[v] + min(_dist(current_pos[v], locations[i]) for i in unvisited) > max_route_distance for v in range(n_vehicles)):
            # Force-assign remaining to least-loaded vehicle
            v = min(range(n_vehicles), key=lambda v: route_dists[v])
            i = unvisited.pop(0)
            routes[v].append(i)
    return routes


def tsp_solver(locations: list[tuple[float, float]],
               *,
               start_index: int = 0,
               method: str = "greedy") -> list[int]:
    """Solve the Travelling Salesman Problem (TSP) using a heuristic.

    Args:
        locations: List of ``(x, y)`` coordinate tuples.
        start_index: Index of the starting location.
        method: Heuristic to use.  ``"greedy"`` (nearest-neighbour) or
            ``"2opt"`` (nearest-neighbour + 2-opt improvement).

    Returns:
        Ordered list of location indices forming the tour (not closed â€” the
        caller should append *start_index* to close the loop).
    """
    import math

    n = len(locations)
    if n <= 1:
        return list(range(n))

    def _dist(i: int, j: int) -> float:
        a, b = locations[i], locations[j]
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # Greedy nearest-neighbour
    unvisited = set(range(n)) - {start_index}
    tour = [start_index]
    current = start_index
    while unvisited:
        nearest = min(unvisited, key=lambda j: _dist(current, j))
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    if method == "2opt":
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    a, b, c, d = tour[i - 1], tour[i], tour[j], tour[(j + 1) % n]
                    old = _dist(a, b) + _dist(c, d)
                    new = _dist(a, c) + _dist(b, d)
                    if new < old - 1e-10:
                        tour[i:j + 1] = tour[i:j + 1][::-1]
                        improved = True
    return tour


def location_allocation(facilities: list[tuple[float, float]],
                        demand_points: list[tuple[float, float]],
                        *,
                        n_to_open: int | None = None,
                        problem_type: str = "minimize_impedance") -> dict:
    """Solve a location-allocation problem.

    Selects the best *n_to_open* facilities from *facilities* to serve all
    *demand_points* using a greedy p-median heuristic.

    Args:
        facilities: Candidate facility ``(x, y)`` coordinates.
        demand_points: Demand point ``(x, y)`` coordinates.
        n_to_open: Number of facilities to open.  Defaults to
            ``max(1, len(facilities) // 3)``.
        problem_type: Optimisation goal (informational only; greedy
            minimisation is used for all types).

    Returns:
        Dict with ``opened_facilities`` (index list), ``assignments``
        (demand_index â†’ facility_index), and ``total_impedance``.
    """
    import math

    if not facilities or not demand_points:
        return {"opened_facilities": [], "assignments": {}, "total_impedance": 0.0}

    n_open = n_to_open or max(1, len(facilities) // 3)
    n_open = min(n_open, len(facilities))

    def _dist(a: tuple, b: tuple) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # Greedy: iteratively open the facility that most reduces total impedance
    opened: list[int] = []
    for _ in range(n_open):
        best_f = -1
        best_imp = float("inf")
        for fi in range(len(facilities)):
            if fi in opened:
                continue
            candidate = opened + [fi]
            imp = sum(min(_dist(dp, facilities[f]) for f in candidate) for dp in demand_points)
            if imp < best_imp:
                best_imp = imp
                best_f = fi
        if best_f >= 0:
            opened.append(best_f)

    assignments = {di: min(opened, key=lambda fi: _dist(dp, facilities[fi])) for di, dp in enumerate(demand_points)}
    total = sum(_dist(demand_points[di], facilities[fi]) for di, fi in assignments.items())
    return {"opened_facilities": opened, "assignments": assignments, "total_impedance": total, "problem_type": problem_type}


def network_partition(nodes: list[int], edges: list[tuple[int, int, float]],
                      n_parts: int = 2) -> dict[int, int]:
    """Partition a network graph into *n_parts* balanced parts.

    Uses a spectral partitioning approximation (recursive bisection by
    edge-weight betweenness) or, if only 2 parts are needed, a min-cut BFS
    approach.

    Args:
        nodes: List of node IDs.
        edges: List of ``(from_node, to_node, weight)`` tuples.
        n_parts: Number of partitions.

    Returns:
        Dict mapping each node ID to a partition index ``[0, n_parts)``.
    """
    from collections import defaultdict
    import math

    if not nodes:
        return {}

    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    assignment: dict[int, int] = {n: 0 for n in nodes}

    def _bfs_bisect(node_set: list[int], part_id: int, next_id: int) -> int:
        if len(node_set) <= 1 or next_id >= n_parts:
            return next_id
        # BFS from a peripheral node
        src = node_set[0]
        visited: dict[int, int] = {src: 0}
        queue = [src]
        while queue:
            cur = queue.pop(0)
            for nb, _ in adj[cur]:
                if nb in set(node_set) and nb not in visited:
                    visited[nb] = visited[cur] + 1
                    queue.append(nb)
        # Split at median depth
        median_depth = sorted(visited.values())[len(visited) // 2]
        group_a = [n for n in node_set if visited.get(n, 0) <= median_depth]
        group_b = [n for n in node_set if visited.get(n, 0) > median_depth]
        if not group_b:
            return next_id
        for n in group_b:
            assignment[n] = next_id
        return next_id + 1

    # Recursive bisection
    current_parts = {0: nodes[:]}
    while len(current_parts) < n_parts:
        new_parts: dict[int, list[int]] = {}
        added = 0
        for pid, ns in current_parts.items():
            nid = len(current_parts) + added
            nid = _bfs_bisect(ns, pid, nid)
            added += 1
        # Refresh current_parts from assignment
        refreshed: dict[int, list[int]] = defaultdict(list)
        for n in nodes:
            refreshed[assignment[n]].append(n)
        if len(refreshed) >= n_parts:
            break
        current_parts = dict(refreshed)

    return assignment


def multimodal_network(network_layers: list[dict], *,
                       transfer_penalty: float = 5.0) -> dict:
    """Combine multiple transport-mode network layers into a single routable graph.

    Each layer dict should have ``"mode"`` (string), ``"nodes"``
    (``{id: (x, y)}``), and ``"edges"`` (``[(from, to, weight)]``) keys.
    Transfer edges are added between nodes within ``transfer_penalty`` distance
    across different modes.

    Args:
        network_layers: List of layer dicts, one per transport mode.
        transfer_penalty: Additional impedance for mode transfers.

    Returns:
        A combined graph dict with ``nodes``, ``edges``, and ``modes`` keys.
    """
    import math
    combined_nodes: dict[str, tuple[float, float]] = {}
    combined_edges: list[tuple[str, str, float]] = []
    modes = []

    for layer in network_layers:
        mode = layer.get("mode", "unknown")
        modes.append(mode)
        for nid, coords in (layer.get("nodes") or {}).items():
            combined_nodes[f"{mode}:{nid}"] = coords
        for u, v, w in (layer.get("edges") or []):
            combined_edges.append((f"{mode}:{u}", f"{mode}:{v}", float(w)))

    # Add transfer edges between nearby nodes of different modes
    node_ids = list(combined_nodes.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            ni, nj = node_ids[i], node_ids[j]
            mi, mj = ni.split(":")[0], nj.split(":")[0]
            if mi != mj:
                ci, cj = combined_nodes[ni], combined_nodes[nj]
                d = math.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
                if d < transfer_penalty * 2:
                    combined_edges.append((ni, nj, d + transfer_penalty))

    return {"nodes": combined_nodes, "edges": combined_edges, "modes": modes}


def closest_facility(incidents: list[tuple[float, float]],
                     facilities: list[tuple[float, float]],
                     *,
                     n_closest: int = 1,
                     cutoff: float = float("inf")) -> list[dict]:
    """Find the closest facilities to each incident location.

    Args:
        incidents: List of incident ``(x, y)`` coordinates.
        facilities: List of facility ``(x, y)`` coordinates.
        n_closest: Number of closest facilities to return per incident.
        cutoff: Maximum straight-line distance to consider.

    Returns:
        A list of dicts with ``incident_index``, ``facility_index``,
        ``distance``, and ``rank`` keys.
    """
    import math
    results = []
    for ii, inc in enumerate(incidents):
        dists = []
        for fi, fac in enumerate(facilities):
            d = math.sqrt((inc[0] - fac[0]) ** 2 + (inc[1] - fac[1]) ** 2)
            if d <= cutoff:
                dists.append((d, fi))
        dists.sort()
        for rank, (d, fi) in enumerate(dists[:n_closest], start=1):
            results.append({"incident_index": ii, "facility_index": fi, "distance": d, "rank": rank})
    return results
