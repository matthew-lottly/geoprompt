import geoprompt as gp


def test_g1_transforms_preserve_extra_ordinates():
    geom = {
        "type": "LineString",
        "coordinates": [(0.0, 0.0, 5.0, 10.0), (1.0, 1.0, 6.0, 11.0)],
    }

    translated = gp.translate_geometry(geom, dx=2.0, dy=3.0)
    rotated = gp.rotate_geometry(geom, angle_degrees=90)
    affine_2d = gp.affine_transform(geom, [1.0, 0.0, 0.0, 1.0, 4.0, 5.0])
    affine_3d = gp.affine_transform(
        geom,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0, 6.0],
    )

    assert translated["coordinates"][0][2:] == (5.0, 10.0)
    assert rotated["coordinates"][0][2:] == (5.0, 10.0)
    assert affine_2d["coordinates"][0][2:] == (5.0, 10.0)
    assert affine_3d["coordinates"][0][3:] == (10.0,)


def test_g6_hierarchy_and_live_traffic_routing_helpers():
    graph = gp.build_network_graph(
        [
            {"edge_id": "direct", "from_node": "A", "to_node": "D", "cost": 2.0, "hierarchy": "local"},
            {"edge_id": "ab", "from_node": "A", "to_node": "B", "cost": 1.0, "hierarchy": "highway"},
            {"edge_id": "bd", "from_node": "B", "to_node": "D", "cost": 1.0, "hierarchy": "highway"},
        ],
        directed=True,
    )

    hierarchy_route = gp.hierarchy_aware_shortest_path(
        graph,
        "A",
        "D",
        preferred_factor=0.4,
        local_factor=2.0,
    )
    traffic_route = gp.live_traffic_shortest_path(
        graph,
        "A",
        "D",
        traffic_feed={"direct": {"speed_factor": 5.0}},
    )

    assert hierarchy_route["path_edges"] == ["ab", "bd"]
    assert traffic_route["path_edges"] == ["ab", "bd"]
    assert traffic_route["cost_mode"] == "live_traffic"


def test_g6_multimodal_respects_curb_approach():
    graph = gp.build_network_graph(
        [
            {"edge_id": "ab", "from_node": "A", "to_node": "B", "cost": 1.0, "mode": "road"},
            {"edge_id": "bd", "from_node": "B", "to_node": "D", "cost": 1.0, "mode": "road", "approach_side": "left"},
            {"edge_id": "ac", "from_node": "A", "to_node": "C", "cost": 1.0, "mode": "road"},
            {"edge_id": "cd", "from_node": "C", "to_node": "D", "cost": 1.0, "mode": "road", "approach_side": "right"},
        ],
        directed=True,
    )

    route = gp.multimodal_shortest_path(graph, "A", "D", curb_approach="right")

    assert route["reachable"] is True
    assert route["path_edges"] == ["ac", "cd"]
    assert route["curb_approach"] == "right"


def test_g7_raster_completion_helpers():
    grid = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    focal = gp.focal_statistics(grid, size=3, statistic="mean")
    block = gp.block_statistics(grid, block_rows=2, block_cols=2, statistic="mean")
    filtered = gp.majority_filter([[1, 1, 2], [1, 2, 2], [2, 2, 2]], size=3)
    allocation = gp.cost_allocation([(0, 0), (2, 2)], [[1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 1.0]])
    snapped = gp.snap_pour_points({"data": [[1, 2, 3], [4, 9, 5], [2, 3, 4]]}, [(0, 0)], search_radius=1)
    links = gp.stream_link([[1], [1], [1]], [[4], [4], [0]])

    assert round(focal[1][1], 6) == 5.0
    assert round(block[0][0], 6) == 3.0
    assert filtered[1][1] == 2
    assert allocation[0][0] == 0
    assert allocation[2][2] == 1
    assert snapped == [(1, 1)]
    assert links[0][0] == links[1][0]


def test_g8_topology_conflict_resolution_drops_duplicates():
    frame = gp.GeoPromptFrame.from_records(
        [
            {"id": 1, "geometry": {"type": "Point", "coordinates": (0.0, 0.0)}},
            {"id": 2, "geometry": {"type": "Point", "coordinates": (0.0, 0.0)}},
        ],
        geometry="geometry",
    )

    result = gp.resolve_topology_conflicts(frame)
    repaired = result["repaired_frame"]

    assert result["removed_duplicates"] == 1
    assert len(repaired.to_records()) == 1