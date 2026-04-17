"""Utility-domain network workflows built on core routing primitives."""

from __future__ import annotations

import math
from collections import deque
from itertools import combinations
from typing import Any, cast

from ..equations import utility_headloss_hazen_williams, utility_service_deficit
from .core import NetworkEdge, NetworkGraph
from .demand import utility_bottlenecks
from .routing import build_network_graph, multi_criteria_shortest_path, service_area, shortest_path


def _edge_blocked(edge: NetworkEdge, failed_edges: set[str], respect_open_devices: bool = False) -> bool:
    edge_id = str(edge.get("edge_id", ""))
    if edge_id in failed_edges:
        return True
    if not respect_open_devices:
        return False
    state = str(edge.get("state", "")).lower()
    return state in {"open", "normally_open"}


def _reachable_nodes(
    graph: NetworkGraph,
    source_nodes: list[str],
    failed_edges: list[str] | None = None,
    respect_open_devices: bool = False,
) -> set[str]:
    failed = set(failed_edges or [])
    if respect_open_devices:
        for edge in graph.edge_attributes.values():
            if _edge_blocked(edge, set(), respect_open_devices=True):
                failed.add(str(edge.get("edge_id", "")))

    rows = service_area(
        graph,
        origins=source_nodes,
        max_cost=math.inf,
        blocked_edges=sorted(failed),
    )
    return {str(row["node"]) for row in rows}


def _edge_list(graph: NetworkGraph) -> list[NetworkEdge]:
    return [cast(NetworkEdge, dict(edge)) for edge in graph.edge_attributes.values()]


def _graph_with_added_edges(graph: NetworkGraph, add_edges: list[dict[str, object]]) -> NetworkGraph:
    merged = _edge_list(graph)
    for idx, edge in enumerate(add_edges):
        normalized = dict(edge)
        normalized.setdefault("edge_id", f"project-edge-{idx}")
        merged.append(cast(NetworkEdge, normalized))
    return build_network_graph(merged, directed=graph.directed)


def _edge_headloss(edge: NetworkEdge) -> float:
    if any(field in edge for field in {"length", "flow", "diameter", "roughness", "roughness_coefficient"}):
        length = max(0.0, float(edge.get("length", edge.get("cost", 0.0))))
        flow = max(0.0, float(edge.get("flow", edge.get("load", 0.0))))
        diameter = max(1e-9, float(edge.get("diameter", 1.0)))
        roughness = max(1e-9, float(edge.get("roughness_coefficient", edge.get("roughness", 130.0))))
        return utility_headloss_hazen_williams(length, flow, diameter, roughness)
    return max(0.0, float(edge.get("cost", 0.0)))


def trace_electric_feeder(graph: NetworkGraph, source_nodes: list[str]) -> list[dict[str, Any]]:
    energized = _reachable_nodes(graph, source_nodes, respect_open_devices=True)
    rows: list[dict[str, Any]] = []
    for node in sorted(graph.adjacency.keys()):
        rows.append({"node": node, "energized": node in energized})
    return rows


def utility_outage_isolation(
    graph: NetworkGraph,
    source_nodes: list[str],
    failed_edges: list[str],
) -> dict[str, Any]:
    energized = _reachable_nodes(graph, source_nodes, failed_edges=failed_edges, respect_open_devices=True)
    all_nodes = set(graph.adjacency.keys())
    deenergized = sorted(all_nodes - energized)
    return {
        "energized_nodes": sorted(energized),
        "deenergized_nodes": deenergized,
        "deenergized_count": len(deenergized),
        "failed_edge_count": len(failed_edges),
    }


def run_utility_scenarios(
    graph: NetworkGraph,
    source_nodes: list[str],
    outage_edges: list[str] | None = None,
    restoration_edges: list[str] | None = None,
) -> dict[str, Any]:
    outage_edges = outage_edges or []
    restoration_edges = restoration_edges or []
    baseline = utility_outage_isolation(graph, source_nodes, failed_edges=[])
    outage = utility_outage_isolation(graph, source_nodes, failed_edges=outage_edges)
    restored_failures = sorted(set(outage_edges) - set(restoration_edges))
    restoration = utility_outage_isolation(graph, source_nodes, failed_edges=restored_failures)
    return {"baseline": baseline, "outage": outage, "restoration": restoration}


def load_transfer_feasibility(
    graph: NetworkGraph,
    feeder_a_source: str,
    feeder_b_source: str,
    tie_edge: NetworkEdge,
) -> dict[str, Any]:
    tie_capacity = float(tie_edge.get("capacity", math.inf))
    tie_load = float(tie_edge.get("load", 0.0))
    spare = tie_capacity - tie_load if math.isfinite(tie_capacity) else math.inf
    path = shortest_path(graph, feeder_a_source, feeder_b_source)
    return {
        "feasible": bool(path["reachable"]) and spare > 0,
        "available_transfer_capacity": max(0.0, spare) if math.isfinite(spare) else math.inf,
        "path_cost": float(path["total_cost"]),
    }


def n_minus_one_edge_contingency_screen(
    graph: NetworkGraph,
    source_nodes: list[str],
    demand_by_node: dict[str, float] | None = None,
    critical_nodes: list[str] | None = None,
    candidate_edge_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not source_nodes:
        raise ValueError("source_nodes must not be empty")
    demand_by_node = demand_by_node or {}
    critical_nodes = critical_nodes or []
    baseline = _reachable_nodes(graph, source_nodes)
    candidates = candidate_edge_ids or sorted(graph.edge_attributes.keys())

    rows: list[dict[str, Any]] = []
    for edge_id in candidates:
        energized = _reachable_nodes(graph, source_nodes, failed_edges=[edge_id])
        lost_nodes = sorted(baseline - energized)
        lost_demand = sum(float(demand_by_node.get(node, 0.0)) for node in lost_nodes)
        lost_critical = [node for node in critical_nodes if node in lost_nodes]
        rows.append(
            {
                "cut_edge_id": edge_id,
                "lost_node_count": len(lost_nodes),
                "lost_demand": lost_demand,
                "lost_critical_count": len(lost_critical),
                "lost_nodes": lost_nodes,
            }
        )

    rows.sort(key=lambda item: (-float(item["lost_demand"]), -int(item["lost_critical_count"]), str(item["cut_edge_id"])))
    return rows


def outage_restoration_tie_options(
    graph: NetworkGraph,
    source_nodes: list[str],
    affected_nodes: list[str],
    tie_edge_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not source_nodes:
        raise ValueError("source_nodes must not be empty")

    if tie_edge_ids is None:
        detected: list[str] = []
        for edge_id, edge in graph.edge_attributes.items():
            state = str(edge.get("state", "")).lower()
            device_type = str(edge.get("device_type", "")).lower()
            if state in {"open", "normally_open"} or device_type in {"tie", "switch"}:
                detected.append(edge_id)
        tie_edge_ids = sorted(detected)

    rows: list[dict[str, Any]] = []
    for tie_edge_id in tie_edge_ids:
        failed = [eid for eid, edge in graph.edge_attributes.items() if _edge_blocked(edge, set(), True)]
        failed = [eid for eid in failed if eid != tie_edge_id]
        energized = _reachable_nodes(graph, source_nodes, failed_edges=failed)
        restored = [node for node in affected_nodes if node in energized]
        rows.append(
            {
                "tie_edge_id": tie_edge_id,
                "restored_target_count": len(restored),
                "restored_nodes": sorted(restored),
            }
        )

    rows.sort(key=lambda item: (-int(item["restored_target_count"]), str(item["tie_edge_id"])))
    return rows


def n_minus_k_edge_contingency_screen(
    graph: NetworkGraph,
    source_nodes: list[str],
    k: int,
    candidate_edge_ids: list[str] | None = None,
    max_combinations: int | None = None,
    prefilter_with_n_minus_one: bool = False,
) -> list[dict[str, Any]]:
    if k <= 0:
        raise ValueError("k must be >= 1")
    candidates = candidate_edge_ids or sorted(graph.edge_attributes.keys())
    if prefilter_with_n_minus_one:
        ranked = n_minus_one_edge_contingency_screen(graph, source_nodes, candidate_edge_ids=candidates)
        candidates = [str(row["cut_edge_id"]) for row in ranked]

    baseline = _reachable_nodes(graph, source_nodes)
    rows: list[dict[str, Any]] = []
    for idx, combo in enumerate(combinations(candidates, k)):
        if max_combinations is not None and idx >= max_combinations:
            break
        energized = _reachable_nodes(graph, source_nodes, failed_edges=list(combo))
        rows.append(
            {
                "k": k,
                "cut_edge_ids": list(combo),
                "lost_node_count": len(baseline - energized),
            }
        )

    rows.sort(key=lambda item: (-int(item["lost_node_count"]), str(item["cut_edge_ids"])))
    return rows


def crew_dispatch_optimizer(
    graph: NetworkGraph,
    source_nodes: list[str],
    failed_edges: list[str],
    repair_time_by_edge: dict[str, float] | None = None,
    demand_by_node: dict[str, float] | None = None,
    critical_nodes: list[str] | None = None,
) -> dict[str, Any]:
    if not failed_edges:
        return {"total_repairs_planned": 0, "repair_plan": []}

    repair_time_by_edge = repair_time_by_edge or {}
    demand_by_node = demand_by_node or {}
    critical_nodes = critical_nodes or []

    unresolved = set(failed_edges)
    plan: list[dict[str, Any]] = []
    while unresolved:
        best_edge = None
        best_score = -math.inf
        for edge_id in sorted(unresolved):
            after_repair_failed = sorted(unresolved - {edge_id})
            outage = utility_outage_isolation(graph, source_nodes, failed_edges=after_repair_failed)
            lost_nodes = set(outage["deenergized_nodes"])
            served_demand = sum(float(demand_by_node.get(node, 0.0)) for node in graph.adjacency if node not in lost_nodes)
            protected_critical = sum(1 for node in critical_nodes if node not in lost_nodes)
            repair_time = max(0.1, float(repair_time_by_edge.get(edge_id, 1.0)))
            score = (served_demand + 1000.0 * protected_critical) / repair_time
            if score > best_score:
                best_score = score
                best_edge = edge_id

        assert best_edge is not None
        unresolved.remove(best_edge)
        plan.append({"repair_edge_id": best_edge, "priority_score": best_score})

    return {"total_repairs_planned": len(plan), "repair_plan": plan}


def pressure_zone_reconfiguration_planner(
    graph: NetworkGraph,
    source_nodes: list[str],
    max_headloss: float = 10.0,
) -> list[dict[str, Any]]:
    baseline = trace_water_pressure_zones(graph, source_nodes, max_headloss=max_headloss)
    baseline_count = sum(1 for row in baseline if bool(row["within_pressure_zone"]))

    candidate_edges = [
        edge_id
        for edge_id, edge in graph.edge_attributes.items()
        if str(edge.get("device_type", "")).lower() in {"valve", "switch", "tie"}
    ]

    rows: list[dict[str, Any]] = []
    for edge_id in candidate_edges:
        edge = graph.edge_attributes[edge_id]
        state = str(edge.get("state", "open")).lower()
        currently_open = state in {"open", "normally_open", "closed_for_maintenance"}
        simulated_failed = [] if currently_open else [edge_id]
        simulated = _reachable_nodes(graph, source_nodes, failed_edges=simulated_failed)
        delta = len(simulated) - baseline_count
        rows.append(
            {
                "action_edge_id": edge_id,
                "proposed_action": "close" if currently_open else "open",
                "delta_within_pressure_band": delta,
            }
        )

    rows.sort(key=lambda item: (-int(item["delta_within_pressure_band"]), str(item["action_edge_id"])))
    return rows


def pump_station_failure_cascade(
    graph: NetworkGraph,
    pump_nodes: list[str],
    initial_failed_pumps: list[str],
    dependency_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    del graph  # graph reserved for future hydraulic coupling
    dependency_map = dependency_map or {}
    failed: set[str] = set(initial_failed_pumps)
    queue: deque[str] = deque(initial_failed_pumps)

    while queue:
        pump = queue.popleft()
        for dependent in dependency_map.get(pump, []):
            if dependent in pump_nodes and dependent not in failed:
                failed.add(dependent)
                queue.append(dependent)

    return {
        "final_failed_count": len(failed),
        "final_failed_pumps": sorted(failed),
    }


def multi_source_service_audit(
    graph: NetworkGraph,
    source_nodes: list[str],
    demand_by_node: dict[str, float] | None = None,
    source_capacity_by_node: dict[str, float] | None = None,
    critical_nodes: list[str] | None = None,
) -> dict[str, Any]:
    if not source_nodes:
        raise ValueError("source_nodes must not be empty")

    demand_by_node = demand_by_node or {}
    source_capacity_by_node = source_capacity_by_node or {}
    critical = set(critical_nodes or [])
    unique_sources = sorted(dict.fromkeys(source_nodes))
    for source in unique_sources:
        if source not in graph.adjacency:
            raise KeyError(source)

    source_state = {
        source: {
            "assigned_demand": 0.0,
            "assigned_nodes": [],
            "capacity": max(0.0, float(source_capacity_by_node.get(source, math.inf))),
        }
        for source in unique_sources
    }

    assignments: list[dict[str, Any]] = []
    ordered_nodes = sorted(graph.adjacency.keys(), key=lambda node: (-float(demand_by_node.get(node, 0.0)), str(node)))
    for node in ordered_nodes:
        demand = float(demand_by_node.get(node, 0.0))
        options: list[tuple[bool, float, float, str]] = []
        cost_lookup: list[tuple[str, float]] = []
        for source in unique_sources:
            result = shortest_path(graph, source, node)
            if not bool(result["reachable"]):
                continue
            cost = float(result["total_cost"])
            cost_lookup.append((source, cost))
            capacity = source_state[source]["capacity"]
            projected_load = float(source_state[source]["assigned_demand"]) + demand
            load_ratio = 0.0 if not math.isfinite(capacity) or capacity == 0.0 else projected_load / capacity
            would_overload = math.isfinite(capacity) and projected_load > capacity
            options.append((would_overload, load_ratio, cost, source))

        cost_lookup.sort(key=lambda item: (item[1], item[0]))
        assigned_source = None
        assigned_cost = None
        backup_source = cost_lookup[1][0] if len(cost_lookup) >= 2 else None
        backup_cost = cost_lookup[1][1] if len(cost_lookup) >= 2 else None
        if options:
            options.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            _, _, assigned_cost, assigned_source = options[0]
            source_state[assigned_source]["assigned_demand"] = float(source_state[assigned_source]["assigned_demand"]) + demand
            cast(list[str], source_state[assigned_source]["assigned_nodes"]).append(node)

        assignments.append(
            {
                "node": node,
                "assigned_source": assigned_source,
                "assigned_cost": assigned_cost,
                "backup_source": backup_source,
                "backup_cost": backup_cost,
                "is_critical": node in critical,
                "demand": demand,
                "reachable_source_count": len(cost_lookup),
            }
        )

    source_summary: list[dict[str, Any]] = []
    for source in unique_sources:
        assigned_demand = float(source_state[source]["assigned_demand"])
        capacity = float(source_state[source]["capacity"])
        overloaded = math.isfinite(capacity) and assigned_demand > capacity
        spare_capacity = None if not math.isfinite(capacity) else capacity - assigned_demand
        source_summary.append(
            {
                "source_node": source,
                "assigned_node_count": len(cast(list[str], source_state[source]["assigned_nodes"])),
                "assigned_nodes": sorted(cast(list[str], source_state[source]["assigned_nodes"])),
                "assigned_demand": assigned_demand,
                "capacity": None if not math.isfinite(capacity) else capacity,
                "spare_capacity": spare_capacity,
                "utilization_ratio": None if not math.isfinite(capacity) or capacity == 0.0 else assigned_demand / capacity,
                "overloaded": overloaded,
            }
        )

    source_summary.sort(key=lambda item: (-float(item["assigned_demand"]), str(item["source_node"])))
    assignments.sort(key=lambda item: (str(item["node"])))
    return {
        "source_count": len(unique_sources),
        "overloaded_source_count": sum(1 for row in source_summary if bool(row["overloaded"])),
        "unserved_node_count": sum(1 for row in assignments if row["assigned_source"] is None),
        "source_summary": source_summary,
        "node_assignments": assignments,
    }


def supply_redundancy_audit(
    graph: NetworkGraph,
    source_nodes: list[str],
    demand_by_node: dict[str, float] | None = None,
    critical_nodes: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not source_nodes:
        raise ValueError("source_nodes must not be empty")

    demand_by_node = demand_by_node or {}
    critical = set(critical_nodes or [])
    unique_sources = sorted(dict.fromkeys(source_nodes))
    for source in unique_sources:
        if source not in graph.adjacency:
            raise KeyError(source)

    rows: list[dict[str, Any]] = []
    for node in sorted(graph.adjacency.keys()):
        costs: list[tuple[str, float]] = []
        for source in unique_sources:
            result = shortest_path(graph, source, node)
            if bool(result["reachable"]):
                costs.append((source, float(result["total_cost"])))
        costs.sort(key=lambda item: (item[1], item[0]))

        path_count = len(costs)
        best_source = costs[0][0] if costs else None
        primary_cost = costs[0][1] if costs else math.inf
        backup_cost = costs[1][1] if path_count >= 2 else None
        if path_count >= 2:
            tier = "high" if (backup_cost is not None and (backup_cost - primary_cost) <= 1.0) or path_count >= 3 else "medium"
        else:
            tier = "low"

        demand = float(demand_by_node.get(node, 0.0))
        rows.append(
            {
                "node": node,
                "best_source": best_source,
                "source_path_count": path_count,
                "primary_cost": None if primary_cost is math.inf else primary_cost,
                "backup_cost": backup_cost,
                "single_source_dependency": path_count <= 1,
                "is_critical": node in critical,
                "demand": demand,
                "resilience_tier": tier,
                "priority_score": (2.0 if node in critical else 1.0) * (1.0 + demand) / max(path_count, 1),
            }
        )

    rows.sort(
        key=lambda item: (
            not bool(item["single_source_dependency"]),
            not bool(item["is_critical"]),
            -float(item["priority_score"]),
            str(item["node"]),
        )
    )
    return rows


def restoration_sequence_report(
    graph: NetworkGraph,
    source_nodes: list[str],
    failed_edges: list[str],
    repair_time_by_edge: dict[str, float] | None = None,
    demand_by_node: dict[str, float] | None = None,
    critical_nodes: list[str] | None = None,
) -> dict[str, Any]:
    demand_by_node = demand_by_node or {}
    critical_nodes = critical_nodes or []
    baseline_outage = utility_outage_isolation(graph, source_nodes, failed_edges=failed_edges)
    if not failed_edges:
        return {
            "total_steps": 0,
            "baseline_deenergized_nodes": list(baseline_outage["deenergized_nodes"]),
            "final_deenergized_nodes": list(baseline_outage["deenergized_nodes"]),
            "stages": [],
        }

    optimization = crew_dispatch_optimizer(
        graph,
        source_nodes=source_nodes,
        failed_edges=failed_edges,
        repair_time_by_edge=repair_time_by_edge,
        demand_by_node=demand_by_node,
        critical_nodes=critical_nodes,
    )

    unresolved = set(failed_edges)
    previously_served = set(graph.adjacency.keys()) - set(baseline_outage["deenergized_nodes"])
    stages: list[dict[str, Any]] = []
    final_deenergized_nodes = list(baseline_outage["deenergized_nodes"])
    cumulative_restored_demand = 0.0

    for step_index, step in enumerate(optimization["repair_plan"], start=1):
        edge_id = str(step["repair_edge_id"])
        unresolved.discard(edge_id)
        outage = utility_outage_isolation(graph, source_nodes, failed_edges=sorted(unresolved))
        served_nodes = set(graph.adjacency.keys()) - set(outage["deenergized_nodes"])
        restored_nodes = sorted(served_nodes - previously_served)
        previously_served = served_nodes
        final_deenergized_nodes = list(outage["deenergized_nodes"])
        restored_demand = sum(float(demand_by_node.get(node, 0.0)) for node in restored_nodes)
        cumulative_restored_demand += restored_demand
        stages.append(
            {
                "step": step_index,
                "repair_edge_id": edge_id,
                "priority_score": float(step["priority_score"]),
                "restored_node_count": len(restored_nodes),
                "restored_nodes": restored_nodes,
                "restored_demand": restored_demand,
                "cumulative_restored_demand": cumulative_restored_demand,
                "served_node_count": len(served_nodes),
                "remaining_outage_count": int(outage["deenergized_count"]),
                "protected_critical_count": sum(1 for node in critical_nodes if node in served_nodes),
            }
        )

    return {
        "total_steps": len(stages),
        "baseline_deenergized_nodes": list(baseline_outage["deenergized_nodes"]),
        "final_deenergized_nodes": final_deenergized_nodes,
        "stages": stages,
    }


def feeder_reconfiguration_optimizer(
    graph: NetworkGraph,
    source_nodes: list[str],
    tie_edge_ids: list[str],
    demand_by_node: dict[str, float] | None = None,
    max_switch_actions: int = 1,
) -> dict[str, Any]:
    demand_by_node = demand_by_node or {}
    if not tie_edge_ids:
        return {"switch_actions": []}

    candidates = outage_restoration_tie_options(
        graph,
        source_nodes=source_nodes,
        affected_nodes=list(graph.adjacency.keys()),
        tie_edge_ids=tie_edge_ids,
    )

    actions: list[dict[str, Any]] = []
    for row in candidates[: max(0, max_switch_actions)]:
        restored_nodes = list(row["restored_nodes"])
        restored_demand = sum(float(demand_by_node.get(node, 0.0)) for node in restored_nodes)
        actions.append(
            {
                "tie_edge_id": row["tie_edge_id"],
                "restored_target_count": row["restored_target_count"],
                "restored_demand": restored_demand,
            }
        )

    actions.sort(key=lambda item: (-float(item["restored_demand"]), str(item["tie_edge_id"])))
    return {"switch_actions": actions}


def resilience_capex_prioritization(
    graph: NetworkGraph,
    source_nodes: list[str],
    demand_by_node: dict[str, float],
    project_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_rows = n_minus_one_edge_contingency_screen(graph, source_nodes, demand_by_node=demand_by_node)
    baseline_risk = sum(float(row["lost_demand"]) for row in baseline_rows)

    rows: list[dict[str, Any]] = []
    for candidate in project_candidates:
        project_graph = _graph_with_added_edges(graph, list(candidate.get("add_edges", [])))
        upgraded_rows = n_minus_one_edge_contingency_screen(project_graph, source_nodes, demand_by_node=demand_by_node)
        upgraded_risk = sum(float(row["lost_demand"]) for row in upgraded_rows)
        avoided = max(0.0, baseline_risk - upgraded_risk)
        capex = max(1.0, float(candidate.get("capex_cost", 1.0)))
        rows.append(
            {
                "project_id": str(candidate.get("project_id", "unknown")),
                "capex_cost": capex,
                "risk_avoided": avoided,
                "benefit_cost_ratio": avoided / capex,
            }
        )

    rows.sort(key=lambda item: (-float(item["benefit_cost_ratio"]), -float(item["risk_avoided"]), str(item["project_id"])))
    return rows


def trace_water_pressure_zones(
    graph: NetworkGraph,
    source_nodes: list[str] | None = None,
    max_headloss: float | None = None,
    *,
    source_node: str | None = None,
    supply_head: float | None = None,
    min_residual_pressure: float = 0.0,
) -> list[dict[str, Any]]:
    if source_nodes is None:
        if source_node is None:
            raise ValueError("source_nodes or source_node must be provided")
        source_nodes = [source_node]
    if not source_nodes:
        raise ValueError("source_nodes must not be empty")
    if min_residual_pressure < 0:
        raise ValueError("min_residual_pressure must be zero or greater")
    if max_headloss is None:
        if supply_head is None:
            raise ValueError("max_headloss or supply_head must be provided")
        max_headloss = max(0.0, float(supply_head) - float(min_residual_pressure))
    if max_headloss < 0:
        raise ValueError("max_headloss must be zero or greater")

    base_pressure = float(supply_head) if supply_head is not None else float(max_headloss)
    rows = service_area(graph, source_nodes, max_cost=math.inf)
    results: list[dict[str, Any]] = []
    for row in rows:
        node = str(row["node"])
        assigned_source = str(row.get("assigned_origin", source_nodes[0]))
        path = shortest_path(graph, assigned_source, node)
        path_edges = list(path.get("path_edges", []))
        if path_edges:
            headloss = sum(_edge_headloss(graph.edge_attributes[edge_id]) for edge_id in path_edges)
        else:
            headloss = float(row["cost"])
        residual_pressure = max(0.0, base_pressure - headloss)
        results.append(
            {
                "node": node,
                "assigned_source": assigned_source,
                "headloss": headloss,
                "residual_pressure": residual_pressure,
                "pressure_deficit": utility_service_deficit(base_pressure, residual_pressure),
                "within_pressure_zone": headloss <= max_headloss,
            }
        )
    return results


def pipe_break_isolation_zones(graph: NetworkGraph, break_edge_id: str) -> dict[str, Any]:
    edge = graph.edge_attributes.get(break_edge_id, {})
    from_node = str(edge.get("from_node", ""))
    to_node = str(edge.get("to_node", ""))
    reachable_from_upstream = _reachable_nodes(graph, [from_node], failed_edges=[break_edge_id]) if from_node else set()
    reachable_from_downstream = _reachable_nodes(graph, [to_node], failed_edges=[break_edge_id]) if to_node else set()
    affected = sorted(set(graph.adjacency) - reachable_from_upstream.intersection(reachable_from_downstream))
    return {
        "break_edge_id": break_edge_id,
        "affected_nodes": affected,
        "isolation_boundary_nodes": sorted({from_node, to_node} - {""}),
    }


def fire_flow_demand_check(
    graph: NetworkGraph,
    hydrant_nodes: list[str],
    demand_gpm: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hydrant in hydrant_nodes:
        available = 0.0
        for edge in graph.edge_attributes.values():
            if hydrant in {str(edge.get("from_node", "")), str(edge.get("to_node", ""))}:
                capacity = float(edge.get("capacity", 0.0))
                flow = float(edge.get("flow", 0.0))
                available += max(0.0, capacity - flow)
        deficit = max(0.0, float(demand_gpm) - available)
        rows.append(
            {
                "hydrant_node": hydrant,
                "available_gpm": available,
                "available_margin_gpm": available - float(demand_gpm),
                "demand_gpm": float(demand_gpm),
                "deficit_gpm": deficit,
                "service_deficit": utility_service_deficit(float(demand_gpm), available),
                "adequate_for_fire_flow": deficit <= 0.0,
            }
        )
    return rows


def gas_pressure_drop_trace(
    graph: NetworkGraph,
    source_node: str,
    inlet_pressure: float,
    drop_per_cost: float = 1.0,
    minimum_required_pressure: float = 0.0,
) -> dict[str, Any]:
    rows = service_area(graph, [source_node], max_cost=math.inf)
    profile: list[dict[str, Any]] = []
    for row in rows:
        cost = float(row["cost"])
        pressure = max(0.0, float(inlet_pressure) - cost * drop_per_cost)
        profile.append(
            {
                "node": row["node"],
                "pressure": pressure,
                "pressure_deficit": utility_service_deficit(float(minimum_required_pressure), pressure),
            }
        )
    min_pressure = min((float(item["pressure"]) for item in profile), default=0.0)
    return {
        "source_node": source_node,
        "zone_node_count": len(profile),
        "min_pressure": min_pressure,
        "max_pressure": max((float(item["pressure"]) for item in profile), default=0.0),
        "pressure_profile": profile,
    }


def outage_impact_report(
    graph: NetworkGraph,
    source_nodes: list[str],
    failed_edges: list[str],
    demand_by_node: dict[str, float] | None = None,
    customer_count_by_node: dict[str, int] | None = None,
    critical_nodes: list[str] | None = None,
    outage_hours: float = 1.0,
    cost_per_customer_hour: float = 8.0,
    cost_per_demand_unit: float = 3.0,
) -> dict[str, Any]:
    if outage_hours < 0:
        raise ValueError("outage_hours must be zero or greater")

    demand_by_node = demand_by_node or {}
    customer_count_by_node = customer_count_by_node or {}
    critical = set(critical_nodes or [])

    outage = utility_outage_isolation(graph, source_nodes, failed_edges=failed_edges)
    impacted_nodes = list(outage["deenergized_nodes"])
    impacted_demand = sum(float(demand_by_node.get(node, 0.0)) for node in impacted_nodes)
    impacted_customer_count = sum(int(customer_count_by_node.get(node, 1)) for node in impacted_nodes)
    critical_impacted = sorted(node for node in impacted_nodes if node in critical)
    customer_hours = impacted_customer_count * float(outage_hours)
    demand_hours = impacted_demand * float(outage_hours)
    estimated_cost = customer_hours * float(cost_per_customer_hour) + demand_hours * float(cost_per_demand_unit)
    severity_score = impacted_demand + (impacted_customer_count / 10.0) + (len(critical_impacted) * 100.0)
    if len(critical_impacted) >= 2 or severity_score >= 200.0:
        severity_tier = "extreme"
    elif len(critical_impacted) >= 1 or severity_score >= 80.0:
        severity_tier = "high"
    elif severity_score >= 20.0:
        severity_tier = "medium"
    else:
        severity_tier = "low"

    return {
        "impacted_node_count": len(impacted_nodes),
        "impacted_nodes": impacted_nodes,
        "impacted_customer_count": impacted_customer_count,
        "impacted_demand": impacted_demand,
        "critical_node_count": len(critical_impacted),
        "critical_impacted_nodes": critical_impacted,
        "outage_hours": float(outage_hours),
        "customer_hours_interrupted": customer_hours,
        "demand_hours_interrupted": demand_hours,
        "estimated_cost": estimated_cost,
        "severity_score": severity_score,
        "severity_tier": severity_tier,
        "failed_edge_count": len(failed_edges),
    }


def reliability_indices(
    outage_events: list[dict[str, Any]],
    total_customers: int,
    period_hours: float = 8760.0,
) -> dict[str, Any]:
    """Compute standard utility reliability indices from outage events.

    Each outage event can be a dict returned by ``outage_impact_report`` or any
    mapping containing ``impacted_customer_count`` plus either
    ``customer_hours_interrupted`` or ``outage_hours``.
    """
    if total_customers <= 0:
        raise ValueError("total_customers must be greater than zero")
    if period_hours <= 0:
        raise ValueError("period_hours must be greater than zero")

    events = outage_events or []
    customer_interruptions = 0.0
    customer_hours_interrupted = 0.0

    for event in events:
        impacted = max(0.0, float(event.get("impacted_customer_count", 0)))
        customer_interruptions += impacted
        if "customer_hours_interrupted" in event:
            customer_hours_interrupted += max(0.0, float(event.get("customer_hours_interrupted", 0.0)))
        else:
            duration = max(0.0, float(event.get("outage_hours", 1.0)))
            customer_hours_interrupted += impacted * duration

    saifi = customer_interruptions / float(total_customers)
    saidi = customer_hours_interrupted / float(total_customers)
    caidi = saidi / saifi if saifi > 0 else 0.0
    asai = 1.0 - (customer_hours_interrupted / (float(total_customers) * float(period_hours)))
    asai = max(0.0, min(1.0, asai))
    asui = 1.0 - asai

    return {
        "event_count": len(events),
        "customers_served": int(total_customers),
        "customer_interruptions": round(customer_interruptions, 4),
        "customer_hours_interrupted": round(customer_hours_interrupted, 4),
        "SAIFI": round(saifi, 6),
        "SAIDI": round(saidi, 6),
        "CAIDI": round(caidi, 6),
        "ASAI": round(asai, 8),
        "ASUI": round(asui, 8),
    }


def reliability_scenario_report(
    scenarios: dict[str, list[dict[str, Any]]],
    total_customers: int,
    period_hours: float = 8760.0,
    baseline: str | None = None,
) -> list[dict[str, Any]]:
    """Compare reliability indices across named outage scenarios."""
    rows: list[dict[str, Any]] = []
    baseline_metrics: dict[str, Any] | None = None

    if baseline is not None and baseline in scenarios:
        baseline_metrics = reliability_indices(scenarios[baseline], total_customers, period_hours=period_hours)

    for name, events in scenarios.items():
        metrics = reliability_indices(events, total_customers, period_hours=period_hours)
        row = {"scenario": name, **metrics}
        if baseline_metrics is not None:
            row["delta_SAIDI"] = round(float(metrics["SAIDI"]) - float(baseline_metrics["SAIDI"]), 6)
            row["delta_SAIFI"] = round(float(metrics["SAIFI"]) - float(baseline_metrics["SAIFI"]), 6)
            row["delta_ASAI"] = round(float(metrics["ASAI"]) - float(baseline_metrics["ASAI"]), 8)
        rows.append(row)

    rows.sort(key=lambda item: (-float(item["ASAI"]), float(item["SAIDI"]), str(item["scenario"])))
    return rows


def gas_shutdown_impact(
    graph: NetworkGraph,
    source_nodes: list[str],
    shutdown_edges: list[str],
) -> dict[str, Any]:
    outage = utility_outage_isolation(graph, source_nodes, failed_edges=shutdown_edges)
    return {
        "impacted_count": int(outage["deenergized_count"]),
        "impacted_nodes": list(outage["deenergized_nodes"]),
    }


def gas_odorization_zone_trace(graph: NetworkGraph, odorizer_nodes: list[str]) -> list[dict[str, Any]]:
    zone_map: dict[str, set[str]] = {}
    for odorizer in odorizer_nodes:
        zone_map[odorizer] = _reachable_nodes(graph, [odorizer])

    rows: list[dict[str, Any]] = []
    for odorizer in odorizer_nodes:
        current = zone_map[odorizer]
        overlap = set()
        for other, nodes in zone_map.items():
            if other == odorizer:
                continue
            overlap.update(current.intersection(nodes))
        rows.append(
            {
                "odorizer_node": odorizer,
                "zone_node_count": len(current),
                "overlap_node_count": len(overlap),
            }
        )
    return rows


def gas_regulator_station_isolation(graph: NetworkGraph, regulator_node: str) -> dict[str, Any]:
    neighbors = sorted({step.to_node for step in graph.adjacency.get(regulator_node, [])})
    alternate_available = False
    for left, right in combinations(neighbors, 2):
        blocked = [
            edge_id
            for edge_id, edge in graph.edge_attributes.items()
            if regulator_node in {str(edge.get("from_node", "")), str(edge.get("to_node", ""))}
        ]
        try:
            path = shortest_path(graph, left, right, blocked_edges=blocked)
            if bool(path["reachable"]):
                alternate_available = True
                break
        except KeyError:
            continue

    return {
        "regulator_node": regulator_node,
        "neighbor_count": len(neighbors),
        "alternate_path_available": alternate_available,
    }


def interdependency_cascade_simulation(
    primary_graph: NetworkGraph,
    dependent_graph: NetworkGraph,
    dependency_map: dict[str, list[str]],
    initial_failed_nodes: list[str],
) -> dict[str, Any]:
    del primary_graph
    failed_primary = set(initial_failed_nodes)
    failed_dependent: set[str] = set()

    frontier = deque(initial_failed_nodes)
    while frontier:
        primary_node = frontier.popleft()
        for dependent_node in dependency_map.get(primary_node, []):
            if dependent_node not in failed_dependent and dependent_node in dependent_graph.adjacency:
                failed_dependent.add(dependent_node)

    return {
        "failed_primary_nodes": sorted(failed_primary),
        "failed_dependent_nodes": sorted(failed_dependent),
    }


def infrastructure_age_risk_weighted_routing(graph: NetworkGraph, origin: str, destination: str) -> dict[str, Any]:
    weights = {
        "base": 1.0,
        "failure_risk": 1.5,
        "condition_penalty": 1.0,
    }

    adjusted_edges: dict[str, float] = {}
    for edge_id, edge in graph.edge_attributes.items():
        base_cost = float(edge.get("cost", edge.get("length", 1.0)))
        age = float(edge.get("age_years", edge.get("age", 0.0)))
        design_life = max(1.0, float(edge.get("design_life_years", edge.get("design_life", 50.0))))
        risk_ratio = min(3.0, age / design_life)
        adjusted_edges[edge_id] = base_cost * (1.0 + 0.5 * risk_ratio)

    base_path = shortest_path(graph, origin, destination)
    risk_path = shortest_path(graph, origin, destination, edge_cost_overrides=adjusted_edges)
    if not bool(risk_path["reachable"]):
        return {"path_found": False, "risk_premium": math.inf, "result": risk_path}

    base_cost = float(base_path["total_cost"]) if bool(base_path["reachable"]) else 0.0
    premium = max(0.0, float(risk_path["total_cost"]) - base_cost)
    return {
        "path_found": True,
        "risk_premium": premium,
        "weights": weights,
        "result": risk_path,
    }


def critical_customer_coverage_audit(
    graph: NetworkGraph,
    critical_customer_nodes: list[str],
    supply_nodes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for customer in critical_customer_nodes:
        single_points: list[str] = []
        for edge_id in graph.edge_attributes:
            outage = utility_outage_isolation(graph, supply_nodes, failed_edges=[edge_id])
            if customer in outage["deenergized_nodes"]:
                single_points.append(edge_id)
        rows.append(
            {
                "critical_customer_node": customer,
                "single_points_of_failure_edges": sorted(single_points),
            }
        )
    return rows


def stormwater_flow_accumulation(graph: NetworkGraph, runoff_by_node: dict[str, float]) -> list[dict[str, Any]]:
    accumulation: dict[str, float] = {node: float(runoff_by_node.get(node, 0.0)) for node in graph.adjacency}
    indegree: dict[str, int] = {node: 0 for node in graph.adjacency}
    for node, steps in graph.adjacency.items():
        for step in steps:
            indegree[step.to_node] = indegree.get(step.to_node, 0) + 1

    queue = deque([node for node, deg in indegree.items() if deg == 0])
    processed = 0
    while queue:
        node = queue.popleft()
        processed += 1
        for step in graph.adjacency.get(node, []):
            accumulation[step.to_node] = accumulation.get(step.to_node, 0.0) + accumulation.get(node, 0.0)
            indegree[step.to_node] = indegree.get(step.to_node, 0) - 1
            if indegree[step.to_node] == 0:
                queue.append(step.to_node)

    if processed < len(graph.adjacency):
        # Cycle-safe fallback for non-DAG networks.
        for _ in range(2):
            for node, steps in graph.adjacency.items():
                for step in steps:
                    accumulation[step.to_node] = max(accumulation.get(step.to_node, 0.0), accumulation.get(node, 0.0))

    rows = [{"node": node, "accumulated_flow": flow} for node, flow in accumulation.items()]
    rows.sort(key=lambda item: str(item["node"]))
    return rows


def detention_basin_overflow_trace(
    graph: NetworkGraph,
    basin_node: str,
    basin_capacity: float,
    inflow: float,
) -> dict[str, Any]:
    del graph
    overflow = max(0.0, float(inflow) - float(basin_capacity))
    return {
        "basin_node": basin_node,
        "overflow_occurring": overflow > 0,
        "overflow_volume": overflow,
    }


def inflow_infiltration_scan(graph: NetworkGraph, infiltration_threshold_ratio: float = 1.25) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edge_id, edge in graph.edge_attributes.items():
        observed = float(edge.get("observed_flow", edge.get("flow", 0.0)))
        dry = max(1e-9, float(edge.get("dry_weather_flow", observed if observed > 0 else 1.0)))
        ratio = observed / dry
        rows.append(
            {
                "edge_id": edge_id,
                "observed_flow": observed,
                "dry_weather_flow": dry,
                "infiltration_ratio": ratio,
                "flagged": ratio >= infiltration_threshold_ratio,
            }
        )
    return rows


def fiber_splice_node_trace(
    graph: NetworkGraph,
    splice_node: str,
    circuit_endpoints: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    if circuit_endpoints:
        traversing = 0
        for origin, destination in circuit_endpoints:
            result = shortest_path(graph, origin, destination)
            if splice_node in result.get("path_nodes", []):
                traversing += 1
    else:
        traversing = len(graph.adjacency.get(splice_node, []))

    return {"splice_node": splice_node, "circuits_traversing_splice": traversing}


def ring_redundancy_check(graph: NetworkGraph, ring_nodes: list[str], hub_node: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in ring_nodes:
        if node not in graph.adjacency:
            rows.append({"ring_node": node, "has_redundancy": False})
            continue

        incoming = graph.adjacency.get(node, [])
        successful_paths = 0
        checked_edges: set[str] = set()
        for step in incoming:
            if step.edge_id in checked_edges:
                continue
            checked_edges.add(step.edge_id)
            result = shortest_path(graph, node, hub_node, blocked_edges=[step.edge_id])
            if bool(result["reachable"]):
                successful_paths += 1

        rows.append({"ring_node": node, "has_redundancy": successful_paths >= 1 and len(checked_edges) >= 2})

    return rows


def fiber_cut_impact_matrix(
    graph: NetworkGraph,
    cut_candidate_edges: list[str],
    circuit_endpoints: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    baseline: dict[tuple[str, str], bool] = {}
    for origin, destination in circuit_endpoints:
        baseline[(origin, destination)] = bool(shortest_path(graph, origin, destination)["reachable"])

    rows: list[dict[str, Any]] = []
    for cut_edge in cut_candidate_edges:
        impacted = 0
        for origin, destination in circuit_endpoints:
            result = shortest_path(graph, origin, destination, blocked_edges=[cut_edge])
            reachable = bool(result["reachable"])
            if baseline[(origin, destination)] and not reachable:
                impacted += 1
        rows.append({"cut_edge_id": cut_edge, "circuits_impacted": impacted})

    rows.sort(key=lambda item: (-int(item["circuits_impacted"]), str(item["cut_edge_id"])))
    return rows


def criticality_ranking_by_node_removal(
    graph: NetworkGraph,
    source_nodes: list[str] | None = None,
    demand_by_node: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    if source_nodes is None:
        source_nodes = [sorted(graph.adjacency.keys())[0]] if graph.adjacency else []
    demand_by_node = demand_by_node or {node: 1.0 for node in graph.adjacency}

    rows: list[dict[str, Any]] = []
    for node in graph.adjacency:
        failed_edges = [
            edge_id
            for edge_id, edge in graph.edge_attributes.items()
            if node in {str(edge.get("from_node", "")), str(edge.get("to_node", ""))}
        ]
        outage = utility_outage_isolation(graph, source_nodes, failed_edges=failed_edges)
        lost_demand = sum(float(demand_by_node.get(n, 0.0)) for n in outage["deenergized_nodes"])
        rows.append({"node": node, "lost_demand": lost_demand})

    rows.sort(key=lambda item: (-float(item["lost_demand"]), str(item["node"])))
    return rows


def demand_weighted_restoration_ranking(
    graph: NetworkGraph,
    source_nodes: list[str],
    failed_edges: list[str],
    demand_by_node: dict[str, float] | None = None,
    customer_count_by_node: dict[str, int] | None = None,
    critical_nodes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank failed edges by the demand-weighted benefit of repairing each one.

    For each failed edge, computes how many nodes, customers, demand, and
    critical nodes would be restored if that single edge were repaired first.
    Returns rows sorted by ``benefit_score`` descending.
    """
    if not failed_edges:
        return []
    demand_by_node = demand_by_node or {}
    customer_count_by_node = customer_count_by_node or {}
    critical = set(critical_nodes or [])

    baseline_reachable = _reachable_nodes(graph, source_nodes, failed_edges=failed_edges)

    rows: list[dict[str, Any]] = []
    for edge_id in failed_edges:
        remaining_failed = [e for e in failed_edges if e != edge_id]
        after_reachable = _reachable_nodes(graph, source_nodes, failed_edges=remaining_failed)
        restored_nodes = sorted(after_reachable - baseline_reachable)
        restored_demand = sum(float(demand_by_node.get(n, 0.0)) for n in restored_nodes)
        restored_customers = sum(int(customer_count_by_node.get(n, 1)) for n in restored_nodes)
        restored_critical = [n for n in restored_nodes if n in critical]
        benefit = restored_demand + (restored_customers / 10.0) + (len(restored_critical) * 100.0)
        rows.append({
            "edge_id": edge_id,
            "restored_node_count": len(restored_nodes),
            "restored_demand": restored_demand,
            "restored_customer_count": restored_customers,
            "restored_critical_count": len(restored_critical),
            "benefit_score": benefit,
        })
    rows.sort(key=lambda item: (-float(item["benefit_score"]), str(item["edge_id"])))
    return rows


def cross_utility_dependency_score(
    primary_graph: NetworkGraph,
    dependent_graph: NetworkGraph,
    dependency_map: dict[str, list[str]],
    demand_by_node: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score how strongly a dependent network relies on a primary network.

    Uses ``dependency_map`` (primary-node → list of dependent-nodes) to quantify
    cascade exposure.  Returns per-primary-node scores and an overall dependency
    index (0–1 scale).
    """
    demand_by_node = demand_by_node or {}
    total_dependent_demand = sum(float(demand_by_node.get(n, 1.0)) for n in dependent_graph.adjacency)
    if total_dependent_demand == 0:
        total_dependent_demand = max(len(dependent_graph.adjacency), 1)

    node_scores: list[dict[str, Any]] = []
    cascade_demand = 0.0
    for primary_node, dep_nodes in dependency_map.items():
        valid_deps = [n for n in dep_nodes if n in dependent_graph.adjacency]
        node_demand = sum(float(demand_by_node.get(n, 1.0)) for n in valid_deps)
        cascade_demand += node_demand
        node_scores.append({
            "primary_node": primary_node,
            "dependent_node_count": len(valid_deps),
            "dependent_demand": node_demand,
        })
    node_scores.sort(key=lambda item: (-float(item["dependent_demand"]), str(item["primary_node"])))
    dependency_index = min(1.0, cascade_demand / total_dependent_demand) if total_dependent_demand else 0.0
    return {
        "dependency_index": round(dependency_index, 4),
        "total_cascade_demand": cascade_demand,
        "total_dependent_demand": total_dependent_demand,
        "node_scores": node_scores,
    }


def attribute_aware_headloss(
    graph: NetworkGraph,
    source_node: str,
    inlet_pressure: float,
    roughness_column: str = "roughness_coefficient",
    diameter_column: str = "diameter",
    flow_column: str = "flow",
    length_column: str = "length",
) -> dict[str, Any]:
    """Trace pressure through a water network using per-edge Hazen-Williams attributes.

    Unlike the simpler ``trace_water_pressure_zones`` which uses a flat cost
    proxy, this function reads per-edge ``roughness``, ``diameter``, ``flow``,
    and ``length`` attributes to compute Hazen-Williams headloss on each edge.
    """
    visited: dict[str, float] = {source_node: float(inlet_pressure)}
    queue = deque([source_node])
    while queue:
        current = queue.popleft()
        pressure = visited[current]
        for step in graph.adjacency.get(current, []):
            if step.to_node in visited:
                continue
            edge = graph.edge_attributes.get(step.edge_id, {})
            length = max(0.0, float(edge.get(length_column, edge.get("cost", 0.0))))
            flow = max(0.0, float(edge.get(flow_column, 0.0)))
            diameter = max(1e-9, float(edge.get(diameter_column, 1.0)))
            roughness = max(1e-9, float(edge.get(roughness_column, edge.get("roughness", 130.0))))
            loss = utility_headloss_hazen_williams(length, flow, diameter, roughness)
            downstream = max(0.0, pressure - loss)
            visited[step.to_node] = downstream
            queue.append(step.to_node)

    profile = [{"node": n, "pressure": p} for n, p in sorted(visited.items())]
    pressures = [p for p in visited.values()]
    return {
        "source_node": source_node,
        "inlet_pressure": inlet_pressure,
        "zone_node_count": len(visited),
        "min_pressure": min(pressures) if pressures else 0.0,
        "max_pressure": max(pressures) if pressures else 0.0,
        "mean_pressure": sum(pressures) / len(pressures) if pressures else 0.0,
        "pressure_profile": profile,
    }


# --- Facility siting and capital planning ---


def facility_siting_score(
    graph: NetworkGraph,
    candidate_nodes: list[str],
    demand_nodes: list[str],
    *,
    demand_weights: dict[str, float] | None = None,
    max_cost: float = math.inf,
) -> list[dict[str, Any]]:
    """Score candidate facility locations by weighted demand coverage.

    For each candidate the function computes a service-area tree and
    accumulates the demand it can reach.

    Args:
        graph: Network graph.
        candidate_nodes: Node ids that are potential facility sites.
        demand_nodes: Node ids representing demand locations.
        demand_weights: Per-node demand weight (defaults to 1 per node).
        max_cost: Maximum cost (distance/time) for service reach.

    Returns:
        List of dicts sorted best-first with keys ``node``,
        ``reachable_demand_nodes``, ``total_weighted_demand``,
        ``mean_cost``.
    """
    weights = demand_weights or {}
    demand_set = set(demand_nodes)
    results: list[dict[str, Any]] = []
    for cand in candidate_nodes:
        sa = service_area(graph, origins=[cand], max_cost=max_cost)
        sa_map = {str(r["node"]): float(r["cost"]) for r in sa}
        reached = [n for n in demand_nodes if n in sa_map]
        total_w = sum(weights.get(n, 1.0) for n in reached)
        costs = [sa_map[n] for n in reached]
        results.append({
            "node": cand,
            "reachable_demand_nodes": len(reached),
            "total_demand_nodes": len(demand_set),
            "coverage_ratio": len(reached) / max(len(demand_set), 1),
            "total_weighted_demand": total_w,
            "mean_cost": (sum(costs) / len(costs)) if costs else math.inf,
        })
    results.sort(key=lambda r: (-r["total_weighted_demand"], r["mean_cost"]))
    return results


def capital_planning_prioritization(
    projects: list[dict[str, Any]],
    *,
    benefit_column: str = "benefit",
    cost_column: str = "cost",
    risk_column: str | None = None,
    budget: float | None = None,
) -> list[dict[str, Any]]:
    """Rank and prioritize capital investment projects.

    Each project dict should contain at least the *benefit_column* and
    *cost_column* fields.  Projects are ranked by benefit-cost ratio and
    optionally filtered to fit within *budget* using a greedy knapsack.

    Args:
        projects: List of project dicts.
        benefit_column: Column with estimated benefit.
        cost_column: Column with estimated cost.
        risk_column: Optional column with a risk score (0-1, higher = riskier).
        budget: If given, select projects greedily up to this total cost.

    Returns:
        Ranked list of project dicts with added ``bcr`` (benefit-cost
        ratio), ``risk_adjusted_bcr``, ``cumulative_cost``, and
        ``selected`` fields.
    """
    scored: list[dict[str, Any]] = []
    for proj in projects:
        cost = max(float(proj.get(cost_column, 0)), 1e-9)
        benefit = float(proj.get(benefit_column, 0))
        risk = float(proj.get(risk_column, 0)) if risk_column and risk_column in proj else 0.0
        bcr = benefit / cost
        risk_adj = bcr * (1.0 - risk)
        scored.append({
            **proj,
            "bcr": round(bcr, 4),
            "risk_adjusted_bcr": round(risk_adj, 4),
        })

    scored.sort(key=lambda p: -p["risk_adjusted_bcr"])

    cumulative = 0.0
    for proj in scored:
        cumulative += float(proj.get(cost_column, 0))
        proj["cumulative_cost"] = round(cumulative, 2)
        if budget is not None:
            proj["selected"] = cumulative <= budget
        else:
            proj["selected"] = True

    return scored


def pressure_scenario_sweep(
    graph: NetworkGraph,
    source_node: str,
    inlet_pressures: list[float],
    *,
    low_pressure_threshold: float = 20.0,
) -> list[dict[str, Any]]:
    """Run multiple pressure traces at different inlet pressures.

    Args:
        graph: Network graph.
        source_node: Pressure source node.
        inlet_pressures: List of inlet pressure values to test.
        low_pressure_threshold: Threshold below which a node is low-pressure.

    Returns:
        List of scenario result dicts with keys ``inlet_pressure``,
        ``min_pressure``, ``mean_pressure``, ``low_pressure_count``,
        ``zone_node_count``.
    """
    results: list[dict[str, Any]] = []
    for ip in inlet_pressures:
        trace = attribute_aware_headloss(graph, source_node, ip)
        profile = trace.get("pressure_profile", [])
        low_count = sum(1 for p in profile if float(p.get("pressure", 0)) < low_pressure_threshold)
        results.append({
            "inlet_pressure": ip,
            "min_pressure": trace["min_pressure"],
            "max_pressure": trace["max_pressure"],
            "mean_pressure": trace["mean_pressure"],
            "zone_node_count": trace["zone_node_count"],
            "low_pressure_count": low_count,
        })
    return results


def dependency_graph_overlay(
    graphs: dict[str, NetworkGraph],
    shared_nodes: list[str],
) -> list[dict[str, Any]]:
    """Identify cross-utility dependencies at shared infrastructure nodes.

    Args:
        graphs: Mapping of utility name to network graph (e.g.
            ``{"water": water_graph, "electric": elec_graph}``).
        shared_nodes: Node ids that appear in multiple utility graphs.

    Returns:
        List of dicts per shared node showing which utilities it connects.
    """
    results: list[dict[str, Any]] = []
    for node in shared_nodes:
        connected: list[str] = []
        for name, g in graphs.items():
            if node in g.adjacency:
                connected.append(name)
        results.append({
            "node": node,
            "connected_utilities": connected,
            "utility_count": len(connected),
            "is_cross_dependency": len(connected) > 1,
        })
    return results
