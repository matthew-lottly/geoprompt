"""Landscape ecology metrics and indices.

Pure-Python implementations of landscape-level, class-level, and patch-level
metrics covering roadmap items 637-650 from A4 (Spatial Analysis).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connected_components(grid: Sequence[int], rows: int, cols: int) -> list[list[tuple[int, int]]]:
    """Find connected components (4-connected) of same-value cells."""
    visited = [[False] * cols for _ in range(rows)]
    components: list[list[tuple[int, int]]] = []

    def _flood(r: int, c: int, val: int) -> list[tuple[int, int]]:
        stack = [(r, c)]
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if visited[cr][cc]:
                continue
            if grid[cr * cols + cc] != val:
                continue
            visited[cr][cc] = True
            cells.append((cr, cc))
            stack.extend([(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)])
        return cells

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c]:
                val = grid[r * cols + c]
                component = _flood(r, c, val)
                if component:
                    components.append(component)
    return components


def _perimeter_cells(patch: list[tuple[int, int]], rows: int, cols: int, grid: Sequence[int], val: int) -> int:
    """Count edge segments of a patch (4-connected perimeter)."""
    patch_set = set(patch)
    perim = 0
    for r, c in patch:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in patch_set:
                perim += 1
    return perim


# ---------------------------------------------------------------------------
# Patch-level metrics (639, 641-642)
# ---------------------------------------------------------------------------

def patch_areas(
    grid: Sequence[int],
    rows: int,
    cols: int,
    *,
    cell_area: float = 1.0,
) -> list[dict[str, Any]]:
    """Compute area of each patch (connected component).

    Returns list of dicts with 'class', 'area', 'n_cells'.
    """
    components = _connected_components(grid, rows, cols)
    results = []
    for comp in components:
        val = grid[comp[0][0] * cols + comp[0][1]]
        results.append({
            "class": val,
            "area": len(comp) * cell_area,
            "n_cells": len(comp),
        })
    return results


def shape_index(
    patch_cells: list[tuple[int, int]],
    rows: int,
    cols: int,
    grid: Sequence[int],
    *,
    cell_size: float = 1.0,
) -> float:
    """Shape index for a single patch (perimeter / min perimeter for area)."""
    area = len(patch_cells) * cell_size ** 2
    val = grid[patch_cells[0][0] * cols + patch_cells[0][1]]
    perim = _perimeter_cells(patch_cells, rows, cols, grid, val) * cell_size
    min_perim = 4 * math.sqrt(area)  # square with same area
    return perim / min_perim if min_perim > 0 else 0.0


def fractal_dimension(
    patches: Sequence[dict[str, float]],
) -> float:
    """Estimate fractal dimension from patch area-perimeter relationship.

    *patches*: list of dicts with 'area' and 'perimeter'.
    Returns the slope of ln(perimeter) vs ln(area), doubled.
    """
    if len(patches) < 2:
        return 1.0
    ln_a = [math.log(max(p["area"], 1e-15)) for p in patches]
    ln_p = [math.log(max(p["perimeter"], 1e-15)) for p in patches]
    n = len(patches)
    mean_a = sum(ln_a) / n
    mean_p = sum(ln_p) / n
    num = sum((ln_a[i] - mean_a) * (ln_p[i] - mean_p) for i in range(n))
    den = sum((ln_a[i] - mean_a) ** 2 for i in range(n))
    slope = num / den if den != 0 else 0.0
    return 2.0 * slope if slope != 0 else 1.0


# ---------------------------------------------------------------------------
# Class-level metrics (640, 646)
# ---------------------------------------------------------------------------

def edge_density(
    grid: Sequence[int],
    rows: int,
    cols: int,
    *,
    cell_size: float = 1.0,
) -> dict[int, float]:
    """Edge density per class: total edge length / total landscape area."""
    total_area = rows * cols * cell_size ** 2
    edge_counts: dict[int, int] = Counter()
    for r in range(rows):
        for c in range(cols):
            val = grid[r * cols + c]
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if nr < rows and nc < cols:
                    nval = grid[nr * cols + nc]
                    if nval != val:
                        edge_counts[val] += 1
                        edge_counts[nval] += 1
    return {cls: (count * cell_size) / total_area for cls, count in edge_counts.items()}


def core_area(
    grid: Sequence[int],
    rows: int,
    cols: int,
    *,
    edge_depth: int = 1,
    cell_area: float = 1.0,
) -> dict[int, float]:
    """Core area per class (cells not within *edge_depth* of a different class)."""
    is_core = [[True] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            val = grid[r * cols + c]
            core = True
            for dr in range(-edge_depth, edge_depth + 1):
                for dc in range(-edge_depth, edge_depth + 1):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        core = False
                        break
                    if grid[nr * cols + nc] != val:
                        core = False
                        break
                if not core:
                    break
            is_core[r][c] = core
    core: dict[int, float] = Counter()
    for r in range(rows):
        for c in range(cols):
            if is_core[r][c]:
                core[grid[r * cols + c]] += cell_area
    return dict(core)


# ---------------------------------------------------------------------------
# Landscape-level metrics (643-645)
# ---------------------------------------------------------------------------

def contagion_index(
    grid: Sequence[int],
    rows: int,
    cols: int,
) -> float:
    """Contagion (interspersion) index for entire landscape.

    Returns value between 0 (maximum dispersion) and 100 (maximum clumping).
    """
    adjacencies: dict[tuple[int, int], int] = Counter()
    total = 0
    for r in range(rows):
        for c in range(cols):
            val = grid[r * cols + c]
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nval = grid[nr * cols + nc]
                    adjacencies[(val, nval)] += 1
                    total += 1
    if total == 0:
        return 0.0
    classes = sorted(set(grid))
    m = len(classes)
    if m < 2:
        return 100.0
    # Proportion of each class
    n_total = rows * cols
    pi = {c: sum(1 for v in grid if v == c) / n_total for c in classes}
    # Build adjacency matrix proportions
    entropy = 0.0
    for ci in classes:
        row_total = sum(adjacencies.get((ci, cj), 0) for cj in classes)
        if row_total == 0:
            continue
        for cj in classes:
            gij = adjacencies.get((ci, cj), 0) / total if total > 0 else 0
            if gij > 0:
                entropy += gij * math.log(gij)
    max_entropy = 2 * math.log(m) if m > 0 else 1.0
    return (1 + entropy / max_entropy) * 100 if max_entropy != 0 else 0.0


def landscape_diversity_shannon(
    grid: Sequence[int],
) -> float:
    """Shannon diversity index for the landscape."""
    total = len(grid)
    if total == 0:
        return 0.0
    counts = Counter(grid)
    H = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            H -= p * math.log(p)
    return H


def landscape_diversity_simpson(
    grid: Sequence[int],
) -> float:
    """Simpson diversity index (1 - D) for the landscape."""
    total = len(grid)
    if total == 0:
        return 0.0
    counts = Counter(grid)
    D = sum((c / total) ** 2 for c in counts.values())
    return 1 - D


# ---------------------------------------------------------------------------
# Connectivity metrics (645, 647-650)
# ---------------------------------------------------------------------------

def connectivity_index(
    patches: Sequence[dict[str, Any]],
    threshold: float,
) -> float:
    """Simple connectivity index based on patch proximity.

    *patches*: list of dicts with 'centroid' (x, y) and 'area'.
    Returns fraction of patch pairs within threshold distance.
    """
    n = len(patches)
    if n < 2:
        return 1.0
    connected = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            ci = patches[i]["centroid"]
            cj = patches[j]["centroid"]
            d = math.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
            total += 1
            if d <= threshold:
                connected += 1
    return connected / total if total > 0 else 0.0


def corridor_analysis(
    cost_grid: Sequence[float],
    rows: int,
    cols: int,
    source_cells: Sequence[tuple[int, int]],
    target_cells: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Identify least-cost corridor between source and target cell sets.

    Uses simple BFS-like cost accumulation on 4-connected grid.
    Returns accumulated cost grids and corridor values.
    """
    INF = float("inf")

    def _cost_surface(seeds: Sequence[tuple[int, int]]) -> list[float]:
        acc = [INF] * (rows * cols)
        queue: list[tuple[float, int, int]] = []
        for r, c in seeds:
            acc[r * cols + c] = cost_grid[r * cols + c]
            queue.append((acc[r * cols + c], r, c))
        queue.sort()
        while queue:
            cost, r, c = queue.pop(0)
            if cost > acc[r * cols + c]:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    new_cost = cost + cost_grid[nr * cols + nc]
                    if new_cost < acc[nr * cols + nc]:
                        acc[nr * cols + nc] = new_cost
                        queue.append((new_cost, nr, nc))
                        queue.sort()
            pass
        return acc

    source_acc = _cost_surface(source_cells)
    target_acc = _cost_surface(target_cells)
    corridor = [source_acc[i] + target_acc[i] for i in range(rows * cols)]
    min_corridor = min(corridor)
    return {
        "source_accumulation": source_acc,
        "target_accumulation": target_acc,
        "corridor": corridor,
        "min_cost": min_corridor,
    }


def graph_connectivity_metric(
    adjacency: dict[int, list[int]],
) -> dict[str, Any]:
    """Compute graph-theory connectivity metrics for a patch network.

    Returns number of components, gamma index, alpha index.
    """
    nodes = set(adjacency.keys())
    for nbrs in adjacency.values():
        nodes.update(nbrs)
    n = len(nodes)
    edges = set()
    for u, nbrs in adjacency.items():
        for v in nbrs:
            edges.add((min(u, v), max(u, v)))
    e = len(edges)

    # Components
    visited: set[int] = set()
    components = 0
    for node in nodes:
        if node not in visited:
            components += 1
            stack = [node]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                stack.extend(adjacency.get(cur, []))

    # Gamma = e / max_edges (planar: 3(n-2))
    max_edges = 3 * (n - 2) if n > 2 else 1
    gamma = e / max_edges if max_edges > 0 else 0.0
    # Alpha = circuits / max_circuits
    circuits = e - n + components
    max_circuits = 2 * n - 5 if n > 2 else 1
    alpha = circuits / max_circuits if max_circuits > 0 else 0.0

    return {
        "nodes": n,
        "edges": e,
        "components": components,
        "gamma_index": gamma,
        "alpha_index": alpha,
    }
