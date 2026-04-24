from __future__ import annotations


def tiny_raster() -> dict[str, object]:
    return {
        "data": [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        "transform": (0.0, 3.0, 1.0, 1.0),
    }


def tiny_network_edges() -> list[dict[str, object]]:
    return [
        {"edge_id": "e1", "from_node": "A", "to_node": "B", "cost": 1.0},
        {"edge_id": "e2", "from_node": "B", "to_node": "C", "cost": 2.0},
        {"edge_id": "e3", "from_node": "A", "to_node": "C", "cost": 5.0},
    ]
