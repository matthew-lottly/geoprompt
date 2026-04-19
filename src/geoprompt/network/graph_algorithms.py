"""Extended graph algorithms for network analysis.

Covers A*, Floyd-Warshall, minimum spanning tree, bridge/cut-edge detection,
articulation point detection, strongly/weakly connected components, and
cycle detection.  All algorithms operate on the existing NetworkGraph structure.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from typing import Any, Callable, Sequence

from .core import NetworkGraph, Traversal


# ---------------------------------------------------------------------------
# A* shortest path  (item 558)
# ---------------------------------------------------------------------------

def astar_shortest_path(
    graph: NetworkGraph,
    origin: str,
    destination: str,
    heuristic: Callable[[str, str], float] | None = None,
    max_cost: float | None = None,
    blocked_edges: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Find the shortest path using A* search.

    *heuristic* is a callable(node, destination) → estimated cost. If not
    supplied, Dijkstra behaviour is used (heuristic = 0).

    Returns a dict with keys: path, edges, total_cost, nodes_explored.
    """
    if origin not in graph.adjacency:
        raise ValueError(f"origin node not in graph: {origin!r}")
    if destination not in graph.adjacency:
        raise ValueError(f"destination node not in graph: {destination!r}")

    if heuristic is None:
        heuristic = lambda _a, _b: 0.0  # noqa: E731

    blocked = set(blocked_edges or [])
    # (f_cost, g_cost, counter, node)
    counter = 0
    open_set: list[tuple[float, float, int, str]] = [(heuristic(origin, destination), 0.0, counter, origin)]
    g_scores: dict[str, float] = {origin: 0.0}
    came_from: dict[str, tuple[str, str]] = {}  # node → (prev_node, edge_id)
    closed: set[str] = set()
    nodes_explored = 0

    while open_set:
        f_cost, g_cost, _cnt, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        nodes_explored += 1

        if current == destination:
            # Reconstruct path
            path = [destination]
            edges = []
            node = destination
            while node in came_from:
                prev, eid = came_from[node]
                path.append(prev)
                edges.append(eid)
                node = prev
            path.reverse()
            edges.reverse()
            return {
                "path": path,
                "edges": edges,
                "total_cost": g_cost,
                "nodes_explored": nodes_explored,
            }

        for traversal in graph.adjacency.get(current, []):
            if traversal.edge_id in blocked:
                continue
            neighbor = traversal.to_node
            if neighbor in closed:
                continue
            tentative_g = g_cost + traversal.cost
            if max_cost is not None and tentative_g > max_cost:
                continue
            if tentative_g < g_scores.get(neighbor, float("inf")):
                g_scores[neighbor] = tentative_g
                came_from[neighbor] = (current, traversal.edge_id)
                f = tentative_g + heuristic(neighbor, destination)
                counter += 1
                heapq.heappush(open_set, (f, tentative_g, counter, neighbor))

    return {
        "path": [],
        "edges": [],
        "total_cost": float("inf"),
        "nodes_explored": nodes_explored,
    }


# ---------------------------------------------------------------------------
# Bellman-Ford shortest path  (item 559)
# ---------------------------------------------------------------------------

def bellman_ford_shortest_path(
    graph: NetworkGraph,
    origin: str,
) -> dict[str, Any]:
    """Find shortest paths from *origin* to all reachable nodes using Bellman-Ford.

    Handles negative edge weights.  Detects negative-weight cycles.

    Returns a dict with keys: distances, predecessors, negative_cycle.
    """
    nodes = list(graph.nodes)
    dist: dict[str, float] = {n: float("inf") for n in nodes}
    pred: dict[str, str | None] = {n: None for n in nodes}
    pred_edge: dict[str, str | None] = {n: None for n in nodes}
    dist[origin] = 0.0

    # Collect all edges
    edges: list[tuple[str, str, float, str]] = []
    for node, traversals in graph.adjacency.items():
        for t in traversals:
            edges.append((node, t.to_node, t.cost, t.edge_id))

    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, w, eid in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                pred_edge[v] = eid
                updated = True
        if not updated:
            break

    # Check for negative cycles
    negative_cycle = False
    for u, v, w, _eid in edges:
        if dist[u] + w < dist[v]:
            negative_cycle = True
            break

    return {
        "distances": {k: v for k, v in dist.items() if v < float("inf")},
        "predecessors": {k: v for k, v in pred.items() if v is not None},
        "predecessor_edges": {k: v for k, v in pred_edge.items() if v is not None},
        "negative_cycle": negative_cycle,
    }


# ---------------------------------------------------------------------------
# All-pairs shortest path — Floyd-Warshall  (item 560)
# ---------------------------------------------------------------------------

def floyd_warshall(graph: NetworkGraph) -> dict[str, dict[str, float]]:
    """Compute all-pairs shortest path distances using Floyd-Warshall.

    Returns a nested dict: dist[u][v] = cost.
    """
    nodes = sorted(graph.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    INF = float("inf")
    dist = [[INF] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0.0

    for node, traversals in graph.adjacency.items():
        u = idx[node]
        for t in traversals:
            v = idx.get(t.to_node)
            if v is not None:
                dist[u][v] = min(dist[u][v], t.cost)

    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                candidate = dist[i][k] + dist[k][j]
                if candidate < dist[i][j]:
                    dist[i][j] = candidate

    result: dict[str, dict[str, float]] = {}
    for i, u in enumerate(nodes):
        row: dict[str, float] = {}
        for j, v in enumerate(nodes):
            if dist[i][j] < INF:
                row[v] = dist[i][j]
        result[u] = row
    return result


# ---------------------------------------------------------------------------
# K-shortest paths — Yen's algorithm  (item 561)
# ---------------------------------------------------------------------------

def k_shortest_paths(
    graph: NetworkGraph,
    origin: str,
    destination: str,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Find the *k* shortest paths using Yen's algorithm.

    Returns a list of dicts with keys: path, edges, total_cost.
    """
    # First path via A*
    first = astar_shortest_path(graph, origin, destination)
    if not first["path"]:
        return []

    A: list[dict[str, Any]] = [first]
    B: list[tuple[float, list[str], list[str]]] = []  # candidates

    for i in range(1, k):
        prev_path = A[-1]["path"]
        for j in range(len(prev_path) - 1):
            spur_node = prev_path[j]
            root_path = prev_path[: j + 1]

            removed_edges: set[str] = set()
            for a_path in A:
                if a_path["path"][: j + 1] == root_path and len(a_path["edges"]) > j:
                    removed_edges.add(a_path["edges"][j])

            spur_result = astar_shortest_path(
                graph, spur_node, destination, blocked_edges=list(removed_edges)
            )
            if spur_result["path"]:
                full_path = root_path[:-1] + spur_result["path"]
                # Reconstruct edges for root portion
                root_edges = prev_path and A[-1]["edges"][:j] or []
                full_edges = root_edges + spur_result["edges"]
                root_cost = sum(
                    t.cost
                    for node in root_path[:-1]
                    for t in graph.adjacency.get(node, [])
                    if t.to_node == root_path[root_path.index(node) + 1]
                ) if j > 0 else 0.0
                total_cost = root_cost + spur_result["total_cost"]
                candidate = (total_cost, full_path, full_edges)
                if candidate not in B:
                    heapq.heappush(B, candidate)

        if not B:
            break

        cost, path, edges = heapq.heappop(B)
        A.append({"path": path, "edges": edges, "total_cost": cost})

    return A


# ---------------------------------------------------------------------------
# Minimum spanning tree — Kruskal's algorithm  (item 563)
# ---------------------------------------------------------------------------

def minimum_spanning_tree(graph: NetworkGraph) -> list[dict[str, Any]]:
    """Compute the minimum spanning tree using Kruskal's algorithm.

    Returns a list of edge dicts with: edge_id, from_node, to_node, cost.
    """
    edges: list[tuple[float, str, str, str]] = []
    seen_edges: set[str] = set()
    for node, traversals in graph.adjacency.items():
        for t in traversals:
            if t.edge_id not in seen_edges:
                edges.append((t.cost, t.from_node, t.to_node, t.edge_id))
                seen_edges.add(t.edge_id)

    edges.sort()

    # Union-Find
    parent: dict[str, str] = {}
    rank: dict[str, int] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank.get(ra, 0) < rank.get(rb, 0):
            ra, rb = rb, ra
        parent[rb] = ra
        if rank.get(ra, 0) == rank.get(rb, 0):
            rank[ra] = rank.get(ra, 0) + 1
        return True

    mst: list[dict[str, Any]] = []
    for cost, u, v, eid in edges:
        if union(u, v):
            mst.append({"edge_id": eid, "from_node": u, "to_node": v, "cost": cost})

    return mst


# ---------------------------------------------------------------------------
# Network flow — max-flow / min-cut  (item 562)
# ---------------------------------------------------------------------------

def max_flow_min_cut(
    graph: NetworkGraph,
    source: str,
    sink: str,
    capacity_field: str = "capacity",
) -> dict[str, Any]:
    """Compute maximum flow and minimum cut using Edmonds-Karp (BFS-based Ford-Fulkerson).

    Edge capacities are read from *capacity_field* in edge_attributes.
    Returns dict with: max_flow, flow_edges, min_cut_nodes.
    """
    # Build residual capacity graph
    cap: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    flow_map: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for node, traversals in graph.adjacency.items():
        for t in traversals:
            attrs = graph.edge_attributes.get(t.edge_id, {})
            c = float(attrs.get(capacity_field, 1.0))
            cap[t.from_node][t.to_node] += c

    def bfs() -> list[str] | None:
        visited = {source}
        queue: deque[list[str]] = deque([[source]])
        while queue:
            path = queue.popleft()
            u = path[-1]
            if u == sink:
                return path
            for v in cap[u]:
                residual = cap[u][v] - flow_map[u][v]
                if v not in visited and residual > 0:
                    visited.add(v)
                    queue.append(path + [v])
        return None

    total_flow = 0.0
    while True:
        path = bfs()
        if path is None:
            break
        # Find bottleneck
        bottleneck = float("inf")
        for i in range(len(path) - 1):
            residual = cap[path[i]][path[i + 1]] - flow_map[path[i]][path[i + 1]]
            bottleneck = min(bottleneck, residual)
        for i in range(len(path) - 1):
            flow_map[path[i]][path[i + 1]] += bottleneck
            flow_map[path[i + 1]][path[i]] -= bottleneck
        total_flow += bottleneck

    # Find min cut: BFS from source in residual
    visited: set[str] = set()
    queue_mc: deque[str] = deque([source])
    visited.add(source)
    while queue_mc:
        u = queue_mc.popleft()
        for v in cap[u]:
            if v not in visited and cap[u][v] - flow_map[u][v] > 0:
                visited.add(v)
                queue_mc.append(v)

    return {
        "max_flow": total_flow,
        "flow_edges": {u: {v: f for v, f in inner.items() if f > 0} for u, inner in flow_map.items() if any(f > 0 for f in inner.values())},
        "min_cut_source_side": sorted(visited),
    }


# ---------------------------------------------------------------------------
# Bridge / cut-edge detection  (item 591)
# ---------------------------------------------------------------------------

def find_bridges(graph: NetworkGraph) -> list[dict[str, Any]]:
    """Detect all bridge edges (cut edges) in the graph.

    A bridge is an edge whose removal disconnects the graph.
    Returns a list of dicts with: edge_id, from_node, to_node.
    """
    disc: dict[str, int] = {}
    low: dict[str, int] = {}
    timer = [0]
    bridges: list[dict[str, Any]] = []

    def dfs(u: str, parent_edge: str | None) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for t in graph.adjacency.get(u, []):
            v = t.to_node
            if v not in disc:
                dfs(v, t.edge_id)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append({"edge_id": t.edge_id, "from_node": u, "to_node": v})
            elif t.edge_id != parent_edge:
                low[u] = min(low[u], disc[v])

    for node in graph.nodes:
        if node not in disc:
            dfs(node, None)

    return bridges


# ---------------------------------------------------------------------------
# Articulation point detection  (item 592)
# ---------------------------------------------------------------------------

def find_articulation_points(graph: NetworkGraph) -> list[str]:
    """Detect all articulation points (cut vertices) in the graph.

    An articulation point is a node whose removal disconnects the graph.
    """
    disc: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    timer = [0]
    ap: set[str] = set()

    def dfs(u: str) -> None:
        children = 0
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for t in graph.adjacency.get(u, []):
            v = t.to_node
            if v not in disc:
                children += 1
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent.get(u) is None and children > 1:
                    ap.add(u)
                if parent.get(u) is not None and low[v] >= disc[u]:
                    ap.add(u)
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for node in graph.nodes:
        if node not in disc:
            parent[node] = None
            dfs(node)

    return sorted(ap)


# ---------------------------------------------------------------------------
# Strongly connected components — Tarjan's  (item 593)
# ---------------------------------------------------------------------------

def strongly_connected_components(graph: NetworkGraph) -> list[list[str]]:
    """Find all strongly connected components using Tarjan's algorithm.

    Returns a list of node lists, one per SCC.
    """
    index_counter = [0]
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        indices[v] = lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for t in graph.adjacency.get(v, []):
            w = t.to_node
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            sccs.append(sorted(scc))

    for node in sorted(graph.nodes):
        if node not in indices:
            strongconnect(node)

    return sccs


# ---------------------------------------------------------------------------
# Weakly connected components  (item 594)
# ---------------------------------------------------------------------------

def weakly_connected_components(graph: NetworkGraph) -> list[list[str]]:
    """Find all weakly connected components (ignoring edge direction).

    Returns a list of node lists, one per component.
    """
    # Build undirected adjacency
    undirected: dict[str, set[str]] = defaultdict(set)
    for node, traversals in graph.adjacency.items():
        for t in traversals:
            undirected[node].add(t.to_node)
            undirected[t.to_node].add(node)

    visited: set[str] = set()
    components: list[list[str]] = []

    for node in sorted(graph.nodes):
        if node in visited:
            continue
        component: list[str] = []
        queue: deque[str] = deque([node])
        visited.add(node)
        while queue:
            u = queue.popleft()
            component.append(u)
            for v in undirected[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        components.append(sorted(component))

    return components


# ---------------------------------------------------------------------------
# Cycle detection  (item 600)
# ---------------------------------------------------------------------------

def find_cycles(graph: NetworkGraph, max_length: int | None = None) -> list[list[str]]:
    """Detect cycles in the graph using DFS.

    Returns a list of cycles, each represented as a list of node IDs.
    *max_length* limits the maximum cycle length to return.
    """
    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: dict[str, int] = {}  # node → index in current path
    path: list[str] = []

    def dfs(u: str) -> None:
        visited.add(u)
        rec_stack[u] = len(path)
        path.append(u)

        for t in graph.adjacency.get(u, []):
            v = t.to_node
            if v in rec_stack:
                cycle = path[rec_stack[v]:] + [v]
                if max_length is None or len(cycle) - 1 <= max_length:
                    cycles.append(cycle)
            elif v not in visited:
                dfs(v)

        path.pop()
        del rec_stack[u]

    for node in sorted(graph.nodes):
        if node not in visited:
            dfs(node)

    return cycles


# ---------------------------------------------------------------------------
# Network partitioning  (item 564)
# ---------------------------------------------------------------------------

def partition_network(graph: NetworkGraph, num_parts: int = 2) -> list[list[str]]:
    """Simple graph partitioning by iterative bisection of weakly connected components.

    Uses the BFS-level-set method to split the largest component until
    *num_parts* partitions are obtained.
    """
    components = weakly_connected_components(graph)
    partitions = [c for c in components]

    # Build undirected adjacency for BFS splitting
    undirected: dict[str, set[str]] = defaultdict(set)
    for node, traversals in graph.adjacency.items():
        for t in traversals:
            undirected[node].add(t.to_node)
            undirected[t.to_node].add(node)

    while len(partitions) < num_parts:
        # Find largest partition to split
        partitions.sort(key=len, reverse=True)
        largest = partitions.pop(0)
        if len(largest) < 2:
            partitions.append(largest)
            break

        # BFS from first node, split at midpoint
        start = largest[0]
        order: list[str] = []
        visited_set: set[str] = set()
        queue: deque[str] = deque([start])
        visited_set.add(start)
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in undirected.get(u, set()):
                if v not in visited_set and v in set(largest):
                    visited_set.add(v)
                    queue.append(v)
        mid = len(order) // 2
        partitions.append(sorted(order[:mid]))
        partitions.append(sorted(order[mid:]))

    return partitions


# ---------------------------------------------------------------------------
# Trace connected features  (item 598)
# ---------------------------------------------------------------------------

def trace_connected(
    graph: NetworkGraph,
    start_node: str,
    max_depth: int | None = None,
    blocked_edges: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Trace all features connected to *start_node* via BFS.

    Returns dict with: nodes, edges, depths.
    """
    blocked = set(blocked_edges or [])
    visited_nodes: dict[str, int] = {start_node: 0}
    visited_edges: list[str] = []
    queue: deque[tuple[str, int]] = deque([(start_node, 0)])

    while queue:
        u, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for t in graph.adjacency.get(u, []):
            if t.edge_id in blocked:
                continue
            if t.to_node not in visited_nodes:
                visited_nodes[t.to_node] = depth + 1
                visited_edges.append(t.edge_id)
                queue.append((t.to_node, depth + 1))

    return {
        "nodes": sorted(visited_nodes.keys()),
        "edges": visited_edges,
        "depths": visited_nodes,
    }


# ---------------------------------------------------------------------------
# Network dataset creation helper  (item 602)
# ---------------------------------------------------------------------------

def create_network_dataset(
    edges: Sequence[dict[str, Any]],
    *,
    from_field: str = "from_node",
    to_field: str = "to_node",
    cost_field: str = "cost",
    directed: bool = False,
) -> NetworkGraph:
    """Create a NetworkGraph from a list of edge dictionaries.

    This is a convenience factory that accepts raw dicts rather than
    typed NetworkEdge objects.
    """
    from .routing import build_network_graph
    typed_edges = []
    for i, e in enumerate(edges):
        edge: dict[str, Any] = {
            "from_node": str(e[from_field]),
            "to_node": str(e[to_field]),
            "cost": float(e.get(cost_field, 1.0)),
            "edge_id": str(e.get("edge_id", f"edge_{i}")),
        }
        for k, v in e.items():
            if k not in (from_field, to_field, cost_field, "edge_id"):
                edge[k] = v
        typed_edges.append(edge)
    return build_network_graph(typed_edges, directed=directed, cost_field="cost")


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
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
]
