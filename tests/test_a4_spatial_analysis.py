import geoprompt as gp


class TestA4SpatialAnalysis:
    def test_spacetime_clustering_and_regression_helpers(self):
        points = [(0.0, 0.0), (0.2, 0.1), (5.0, 5.0), (5.2, 5.1)]
        values = [1.0, 1.5, 7.5, 8.0]
        x_matrix = [[0.5, 1.0], [0.7, 1.1], [2.5, 3.0], [2.7, 3.1]]
        events = [
            {"x": 0.0, "y": 0.0, "t": 1},
            {"x": 0.1, "y": 0.2, "t": 2},
            {"x": 5.0, "y": 5.0, "t": 4},
            {"x": 5.1, "y": 5.1, "t": 5},
        ]
        adjacency = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        attrs = {0: [1.0, 1.1], 1: [1.2, 1.0], 2: [4.5, 4.8], 3: [4.7, 4.9]}

        multi = gp.multi_distance_cluster_analysis(points, [0.5, 2.0])
        mgwr = gp.multi_scale_gwr(points, values, x_matrix, bandwidths=[1.0, 3.0])
        reg = gp.exploratory_regression(values, x_matrix)
        bivariate = gp.local_bivariate_relationships(values, [2.0, 2.5, 8.0, 8.5], points, k=2)
        cube = gp.space_time_cube(events, spatial_bins=2, temporal_bins=2)
        patterns = gp.space_time_pattern_mining(events, spatial_threshold=1.0, temporal_threshold=2.0)
        emerging = gp.emerging_hot_spot_analysis(cube["cube"], temporal_bins=2, spatial_bins=2)
        trend = gp.spatial_mann_kendall_analysis({"north": [1, 2, 3, 4, 5], "south": [5, 4, 3, 2, 1]})
        change = gp.change_point_detection_spatial({"north": [1, 1, 1, 6, 6]})
        ts = gp.time_series_clustering({0: [1, 1, 1], 1: [1, 2, 1], 2: [9, 9, 8], 3: [8, 9, 9]}, k=2)
        constrained = gp.spatially_constrained_multivariate_clustering(adjacency, attrs, num_regions=2)
        hdb = gp.hdbscan_spatial_clustering(points, min_cluster_size=2)
        max_p = gp.regionalization_max_p(adjacency, {0: 1, 1: 2, 2: 8, 3: 9}, min_pop=2, populations={0: 1, 1: 1, 2: 1, 3: 1})
        skater = gp.skater_regionalization(adjacency, attrs, num_regions=2)
        azp = gp.azp_regionalization(adjacency, attrs, num_regions=2)

        assert len(multi) == 2 and "K" in multi[0]
        assert len(mgwr["models"]) == 2
        assert reg and "adj_r2" in reg[0]
        assert len(bivariate) == 4 and "relationship" in bivariate[0]
        assert cube["total_events"] == 4 and patterns
        assert emerging and set(trend) == {"north", "south"}
        assert change["north"]["change_points"]
        assert len(ts) == 4 and len(constrained) == 4 and len(hdb["labels"]) == 4
        assert len(max_p) == 4 and len(skater) == 4 and len(azp) == 4

    def test_interpolation_and_overlay_helpers(self):
        raster_a = [[1, 2], [3, 4]]
        raster_b = [[4, 3], [2, 1]]
        known_points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        known_values = [1.0, 2.0, 3.0]
        secondary_values = [10.0, 12.0, 14.0]
        queries = [(0.5, 0.5), (0.8, 0.2)]
        barriers = [[0, 1], [0, 0]]

        overlay = gp.weighted_overlay_raster([raster_a, raster_b], [0.6, 0.4])
        universal = gp.universal_kriging(known_points, known_values, queries)
        co = gp.co_kriging(known_points, known_values, secondary_values, queries)
        ebk = gp.empirical_bayesian_kriging(known_points, known_values, queries)
        areal = gp.areal_interpolation_dasymetric([100, 50], [10, 5], [[4, 6], [1, 4]])
        diffusion = gp.diffusion_interpolation_with_barriers(known_points, known_values, queries, barriers=barriers)
        kernel = gp.kernel_interpolation_with_barriers(known_points, known_values, cell_size=0.5, bandwidth=1.0, barriers=barriers)
        stderr = gp.prediction_standard_error_surface(known_points, known_values, cell_size=0.5)
        pycno = gp.pycnophylactic_interpolation([10, 20], [[[1, 0], [1, 0]], [[0, 1], [0, 1]]], iterations=4)

        assert overlay[0][0] > 0
        assert len(universal) == 2 and len(co) == 2
        assert len(ebk["predictions"]) == 2 and len(ebk["variance"]) == 2
        assert len(areal) == 2 and len(diffusion) == 2
        assert kernel["rows"] >= 1 and stderr["rows"] >= 1
        assert len(pycno) == 2 and len(pycno[0]) == 2

    def test_routing_service_and_utility_helpers(self):
        graph = {
            "A": {"B": 5, "C": 8},
            "B": {"C": 3, "D": 4},
            "C": {"D": 2},
            "D": {},
        }
        land_use = {"residential": 10000, "commercial": 5000}
        electrical_segments = [
            {"id": "L1", "current_amps": 120, "capacity_amps": 200, "length_m": 100, "resistance_per_km": 0.12},
            {"id": "L2", "current_amps": 180, "capacity_amps": 220, "length_m": 150, "resistance_per_km": 0.10},
        ]
        pipes = [
            {"id": "P1", "length_m": 100, "velocity_ms": 1.0},
            {"id": "P2", "length_m": 150, "velocity_ms": 0.8},
        ]

        drive = gp.drive_time_polygon(graph, "A", max_cost=9)
        walk = gp.walk_time_polygon(graph, "A", max_cost=8)
        trade = gp.retail_trade_area_analysis((0, 0), [(1, 1), (2, 2), (3, 3)])
        route_barriers = gp.route_with_barriers(graph, "A", "D", barriers={("B", "C")})
        route_windows = gp.route_with_time_windows(graph, "A", "D", {"D": (0, 15)})
        multimodal = gp.multi_modal_routing({"walk": graph, "drive": graph}, "A", "D", preferred_mode="drive")
        demand = gp.demand_estimation_from_land_use(land_use)
        load = gp.load_flow_analysis({"N1": 100, "N2": 80}, losses=0.05)
        pipe = gp.pipe_sizing_from_flow_velocity(0.4, 1.2)
        age = gp.water_age_analysis(pipes, source_age_hours=0.0)
        chlorine = gp.chlorine_decay_analysis(age, initial_mg_l=2.0, decay_constant=0.1)
        short_circuit = gp.short_circuit_analysis(11_000, 0.25)
        voltage = gp.voltage_drop_calculation(80, 120, 0.15)
        transformer = gp.transformer_loading_analysis([50, 75, 90], rated_kva=100)
        fault = gp.fault_location_analysis([0.2, 0.4, 3.2, 0.3])
        protection = gp.protection_coordination_analysis([0.2, 0.5, 1.0], [0.4, 0.8, 1.5])

        assert "reachable_nodes" in drive and "reachable_nodes" in walk
        assert "p50_radius" in trade
        assert route_barriers["path"][0] == "A" and route_windows["arrives_within_window"]
        assert multimodal["mode"] == "drive"
        assert demand["total_demand"] > 0 and load["total_load"] > 0
        assert pipe["diameter_m"] > 0 and len(age) == 2 and len(chlorine) == 2
        assert short_circuit["fault_current_a"] > 0 and voltage["voltage_drop_pct"] >= 0
        assert transformer["max_loading_pct"] > 0 and protection["coordinated"]
        assert fault["fault_index"] == 2

    def test_lrs_geocoding_visibility_environment_and_connectivity(self):
        edges = [
            {"edge_id": 1, "from_node": "A", "to_node": "B"},
            {"edge_id": 2, "from_node": "B", "to_node": "C"},
            {"edge_id": 3, "from_node": "C", "to_node": "D"},
        ]
        route = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        cal_points = [(0.0, 0.0, 0.0), (2.0, 0.0, 2.0)]
        locators = [
            {"type": "parcel", "name": "A", "x": 1.0, "y": 2.0},
            {"type": "rooftop", "name": "B", "x": 3.0, "y": 4.0},
        ]
        dem = [[1, 2, 3], [2, 3, 4], [1, 2, 3]]

        turns = gp.turn_feature_class(edges)
        lrs = gp.location_referencing_system(route)
        calibrated = gp.calibrate_lrs_routes(route, cal_points)
        overlay = gp.lrs_event_overlay_tool(calibrated, [(0.5, 1.5)])
        parcel = gp.parcel_centroid_geocoding("A", locators)
        rooftop = gp.rooftop_level_geocoding("B", locators)
        composite = gp.geocoding_composite_locator("B", locators)
        viewshed = gp.viewshed_multiple_observers(dem, [(0, 0), (2, 2)])
        observers = gp.observer_points_analysis(dem, [(0, 0), (1, 1)])
        freq = gp.visibility_frequency_surface(dem, [(0, 0), (2, 2)])
        shadow = gp.sun_shadow_volume(dem, sun_altitude=35, sun_azimuth=135)
        solar = gp.solar_radiation_surface_analysis([[10, 15], [20, 25]], [[90, 135], [180, 225]], latitude=40)
        wind = gp.wind_exposure_surface([[10, 11], [9, 12]])
        noise = gp.noise_propagation_model((0, 0), 90, [(10, 0), (20, 0)])
        air = gp.air_quality_dispersion_model((0, 0), 5.0, 3.0, 90, [(100, 0), (200, 0)])
        flood = gp.flood_inundation_from_dem([[0.0, 1.0], [2.0, 3.0]], water_level=1.5)
        slr = gp.sea_level_rise_inundation_analysis([[0.0, 1.0], [2.0, 3.0]], 1.0)
        runoff = gp.rainfall_runoff_model(4.0, 75)
        erosion = gp.erosion_risk_model_rusle(100, 0.3, 1.2, 0.4, 0.9)
        landslide = gp.landslide_susceptibility_model([[0.1, 0.2], [0.5, 0.8]], [0.6, 0.4])
        wildfire = gp.wildfire_spread_model([[0.9, 0.8], [0.2, 0.9]], (0, 0))
        habitat = gp.habitat_suitability_model({"water": [0.8, 0.6], "cover": [0.9, 0.7]})
        species = gp.species_distribution_model_maxent([{"temp": 20, "rain": 100}, {"temp": 22, "rain": 90}], ["temp", "rain"])
        circuit = gp.circuit_theory_connectivity([[1, 2], [2, 1]])
        graph_conn = gp.graph_theory_connectivity({"nodes": ["A", "B", "C"], "edges": [("A", "B"), ("B", "C")]})

        assert turns and len(lrs["measures"]) == 3 and overlay
        assert parcel["match_type"] == "parcel" and rooftop["match_type"] == "rooftop" and composite["matched"]
        assert len(viewshed) == 3 and len(freq) == 3 and len(observers) == 2
        assert shadow["shadow_cells"] >= 0 and len(solar) == 2 and len(wind) == 2
        assert len(noise) == 2 and len(air) == 2
        assert flood[0][0] == 1 and slr[0][0] == 1 and runoff >= 0
        assert erosion > 0 and len(landslide) == 2 and len(wildfire) == 2
        assert len(habitat) == 2 and len(species["scores"]) == 2
        assert circuit["connectivity_score"] > 0 and graph_conn["edge_connectivity"] >= 1
