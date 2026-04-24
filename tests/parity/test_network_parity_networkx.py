from __future__ import annotations

import math

import pytest

from geoprompt.network.routing import build_network_graph, shortest_path
from .fixtures import tiny_network_edges


def test_shortest_path_cost_matches_networkx_reference() -> None:
    networkx = pytest.importorskip("networkx")

    edges = tiny_network_edges()
    graph = build_network_graph(edges, directed=False)
    result = shortest_path(graph, "A", "C")

    nxg = networkx.Graph()
    for edge in edges:
        nxg.add_edge(edge["from_node"], edge["to_node"], weight=float(edge["cost"]))
    expected_cost = networkx.shortest_path_length(nxg, "A", "C", weight="weight")
    expected_path_nodes = networkx.shortest_path(nxg, "A", "C", weight="weight")

    assert result["reachable"] is True
    assert math.isclose(float(result["total_cost"]), float(expected_cost), abs_tol=1e-12)
    assert [str(node) for node in result["path_nodes"]] == [str(node) for node in expected_path_nodes]
    assert int(result["hop_count"]) == max(len(expected_path_nodes) - 1, 0)
    assert [str(edge_id) for edge_id in result["path_edges"]] == ["e1", "e2"]
