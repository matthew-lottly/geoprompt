"""Tests for A9 domain-specific modules (items 1251-1500)."""
import math
import pytest
import geoprompt as gp


# ── Water / Sewer ──────────────────────────────────────────────────

class TestWaterSewer:
    def test_epanet_roundtrip(self):
        inp = "[JUNCTIONS]\nJ1\t100\t0.5\n[RESERVOIRS]\nR1\t120\n[PIPES]\nP1\tR1\tJ1\t500\t200\t130\n[END]"
        net = gp.epanet_inp_read(inp)
        assert len(net["junctions"]) == 1
        assert net["junctions"][0]["id"] == "J1"
        text = gp.epanet_inp_write(net)
        assert "[JUNCTIONS]" in text
        net2 = gp.epanet_inp_read(text)
        assert net2["junctions"][0]["elevation"] == net["junctions"][0]["elevation"]

    def test_water_distribution_solve(self):
        net = {"reservoirs": [{"id": "R1", "head": 100}],
               "junctions": [{"id": "J1", "elevation": 80}],
               "pipes": [{"id": "P1", "node1": "R1", "node2": "J1",
                           "length": 500, "diameter": 200, "roughness": 130}]}
        sol = gp.water_distribution_solve(net)
        assert "heads" in sol and "flows" in sol

    def test_swmm_roundtrip(self):
        inp = "[SUBCATCHMENTS]\nS1\tRG1\tJ1\t10\n[JUNCTIONS]\nJ1\t5\n[CONDUITS]\nC1\tJ1\tO1\t200\n[END]"
        model = gp.swmm_inp_read(inp)
        assert len(model["subcatchments"]) == 1
        text = gp.swmm_inp_write(model)
        assert "[SUBCATCHMENTS]" in text

    def test_sewer_model_solve(self):
        conduits = [{"name": "C1", "diameter": 0.6, "slope": 0.005, "flow": 0.05}]
        r = gp.sewer_model_solve(conduits)
        assert r[0]["capacity_m3s"] > 0
        assert isinstance(r[0]["surcharge"], bool)

    def test_combined_sewer_overflow(self):
        r = gp.combined_sewer_overflow(0.5, 0.3, 0.4)
        assert r["cso_active"] is True
        assert r["overflow_m3s"] > 0


# ── Electric ───────────────────────────────────────────────────────

class TestElectric:
    def test_cim_xml_roundtrip(self):
        xml = gp.cim_xml_write({"buses": [{"id": "B1", "name": "Bus1"}],
                                "lines": [{"id": "L1", "length": 10}]})
        assert "ConnectivityNode" in xml
        model = gp.cim_xml_read(xml)
        assert len(model["buses"]) == 1

    def test_load_flow(self):
        buses = [{"id": "B1", "generation": 100, "load": 0},
                 {"id": "B2", "generation": 0, "load": 80}]
        branches = [{"id": "BR1", "from": "B1", "to": "B2", "reactance": 0.1}]
        r = gp.electric_load_flow(buses, branches)
        assert "branch_flows" in r

    def test_fault_analysis(self):
        buses = [{"id": "B1"}, {"id": "B2"}]
        branches = [{"id": "BR1", "from": "B1", "to": "B2", "resistance": 0.01, "reactance": 0.1}]
        r = gp.electric_fault_analysis(buses, branches, "B1")
        assert r["fault_current_pu"] > 0

    def test_protection_coordination(self):
        devices = [{"id": "D1", "pickup_current": 100, "fault_current": 1000, "time_dial": 1.0},
                   {"id": "D2", "pickup_current": 200, "fault_current": 1000, "time_dial": 2.0}]
        r = gp.electric_protection_coordination(devices)
        assert r[0]["sequence"] == 1

    def test_hosting_capacity(self):
        r = gp.electric_hosting_capacity(10000, 5000)
        assert r["hosting_capacity_kw"] >= 0
        assert r["binding_constraint"] in ("thermal", "voltage")


# ── Gas / Telecom / Traffic ────────────────────────────────────────

class TestGasTelecomTraffic:
    def test_gas_roundtrip(self):
        text = "NODES\nN1 | 350\nPIPES\nP1 | N1 | N2 | 1000"
        m = gp.gas_model_read(text)
        assert len(m["nodes"]) == 1
        out = gp.gas_model_write(m)
        assert "N1" in out

    def test_gas_leak_heatmap(self):
        pts = [(0, 0), (1, 1), (2, 2)]
        r = gp.gas_leak_heatmap(pts, grid_size=5, bandwidth=2)
        assert r["max_density"] > 0

    def test_bandwidth_allocation(self):
        users = [{"id": "U1", "demand_mhz": 10}, {"id": "U2", "demand_mhz": 20}]
        r = gp.bandwidth_allocation(100, users)
        total = sum(u["allocated_mhz"] for u in r)
        assert abs(total - 100) < 0.01

    def test_traffic_ue(self):
        links = [{"id": "L1", "free_flow_time": 10, "capacity": 1000}]
        ods = [{"route_links": ["L1"], "demand": 500}]
        r = gp.traffic_assignment_ue(links, ods, iterations=5)
        assert r[0]["flow"] > 0

    def test_traffic_so(self):
        links = [{"id": "L1", "free_flow_time": 10, "capacity": 1000}]
        ods = [{"route_links": ["L1"], "demand": 500}]
        r = gp.traffic_assignment_so(links, ods, iterations=5)
        assert r[0]["flow"] > 0

    def test_signal_timing(self):
        phases = [{"critical_ratio": 0.3}, {"critical_ratio": 0.3}]
        r = gp.traffic_signal_timing(phases)
        assert "cycle_s" in r


# ── Environmental Screening ───────────────────────────────────────

class TestEnvironmentalScreening:
    def test_wind_study(self):
        m = [{"direction": 90, "speed": 5}, {"direction": 180, "speed": 10}]
        r = gp.wind_study(m)
        assert r["mean_speed"] > 0

    def test_view_corridor(self):
        r = gp.view_corridor_analysis((0, 0, 10), (100, 0, 10), [])
        assert r["visible"] is True
        r2 = gp.view_corridor_analysis((0, 0, 5), (100, 0, 5),
                                       [{"x": 50, "y": 0, "height": 20}])
        assert r2["visible"] is False

    def test_ei_screening(self):
        r = gp.environmental_impact_screening(
            {"x": 0, "y": 0, "name": "P1"},
            [{"x": 100, "y": 0, "name": "Wetland", "type": "wetland", "radius": 50}])
        assert r["requires_eia"] is True

    def test_ej_screening(self):
        r = gp.environmental_justice_screening(
            [{"minority_pct": 0.6, "poverty_pct": 0.1}])
        assert r[0]["ej_community"] is True


# ── Crime / Health ─────────────────────────────────────────────────

class TestCrimeHealth:
    def test_crime_prediction(self):
        inc = [{"x": i, "y": i} for i in range(5)]
        r = gp.crime_pattern_prediction(inc, grid_size=5)
        assert r["max_density"] > 0

    def test_repeat_offender(self):
        r = gp.repeat_offender_spatial(
            [{"x": 0, "y": 0}], [{"x": 1, "y": 1}, {"x": 2, "y": 2}], link_radius_m=500)
        assert r[0]["linked_incidents"] == 2

    def test_cluster_detection(self):
        cases = [{"x": 0, "y": 0}] * 10
        pop = [{"id": "Z1", "x": 0, "y": 0, "population": 50},
               {"id": "Z2", "x": 5000, "y": 5000, "population": 50}]
        r = gp.public_health_cluster_detection(cases, pop)
        assert len(r) > 0

    def test_epidemiological(self):
        r = gp.epidemiological_mapping(
            [{"x": 0, "y": 0}], [{"x": 0, "y": 0, "population": 1000, "radius": 100}])
        assert r[0]["incidence_per_100k"] > 0

    def test_contact_tracing(self):
        r = gp.contact_tracing_spatial(
            {"x": 0, "y": 0},
            [{"x": 0.5, "y": 0, "duration_min": 20}])
        assert r[0]["risk"] == "high"

    def test_exposure_assessment(self):
        r = gp.environmental_exposure_assessment(
            [{"x": 0, "y": 0}], [{"x": 10, "y": 0, "intensity": 5}])
        assert r[0]["exposure"] > 0

    def test_soil_contamination(self):
        samples = [(0, 0, 10), (100, 0, 20), (0, 100, 15)]
        r = gp.soil_contamination_interpolation(samples, (0, 0, 100, 100), grid_size=5)
        assert len(r["grid"]) == 5


# ── Hydrology / Hazards ───────────────────────────────────────────

class TestHydrologyHazards:
    def test_wellhead_protection(self):
        r = gp.wellhead_protection_area({"x": 0, "y": 0})
        assert len(r) == 3
        assert r[0]["radius_m"] > 0

    def test_wetland_delineation(self):
        r = gp.wetland_delineation([{"hydric_soil": True, "hydrophytic_vegetation": True,
                                     "wetland_hydrology": True}])
        assert r[0]["is_wetland"] is True

    def test_coastal_erosion(self):
        r = gp.coastal_erosion_model([{"beach_slope": 0.02, "closure_depth_m": 8}])
        assert r[0]["retreat_m"] > 0

    def test_tsunami_zone(self):
        r = gp.tsunami_inundation_zone([{"land_slope": 0.01}])
        assert r[0]["runup_m"] > 0

    def test_earthquake_intensity(self):
        r = gp.earthquake_shaking_intensity((35, -118), 7.0, [(35.1, -118.1)])
        assert r[0]["mmi"] > 0

    def test_liquefaction(self):
        r = gp.liquefaction_susceptibility([{"soil_type": "sand", "groundwater_depth_m": 2, "spt_n": 5}])
        assert r[0]["category"] == "high"

    def test_hurricane_surge(self):
        r = gp.hurricane_surge_zone([{"elevation_m": 2}], category=3)
        assert r[0]["inundated"] is True

    def test_tornado_tracks(self):
        r = gp.tornado_track_analysis([{"length_km": 10, "width_m": 200, "ef_rating": 3}])
        assert r["count"] == 1

    def test_drought(self):
        r = gp.drought_severity_map([{"precipitation_mm": 20, "potential_et_mm": 80}])
        assert "drought" in r[0]["category"]

    def test_avalanche(self):
        r = gp.avalanche_danger_zone([{"slope_deg": 35, "elevation_drop_m": 500}])
        assert r[0]["start_zone"] is True

    def test_fema_flood(self):
        r = gp.fema_flood_zone_analysis(
            [{"x": 0, "y": 0}],
            [{"x": 0, "y": 0, "radius": 100, "zone": "AE"}])
        assert r[0]["flood_zone"] == "AE"

    def test_insurance_risk(self):
        r = gp.insurance_risk_scoring([{"flood_risk": 0.8, "fire_risk": 0.6}])
        assert r[0]["risk_score"] > 0


# ── Agriculture / Forestry ─────────────────────────────────────────

class TestAgricultureForestry:
    def test_field_boundary(self):
        grid = [[0.2, 0.2, 0.8], [0.2, 0.2, 0.8], [0.2, 0.2, 0.8]]
        r = gp.agriculture_field_boundary(grid)
        assert r["boundary_pixel_count"] >= 0

    def test_crop_classification(self):
        r = gp.crop_type_classification([{"ndvi": 0.7, "ndwi": -0.05}])
        assert r[0]["crop_type"] != "unknown"

    def test_yield_estimation(self):
        r = gp.yield_estimation([{"mean_ndvi": 0.7, "area_ha": 10}])
        assert r[0]["total_yield_t"] > 0

    def test_variable_rate(self):
        r = gp.variable_rate_application([{"soil_nitrogen_kg_ha": 50}])
        assert r[0]["application_rate_kg_ha"] > 0

    def test_irrigation(self):
        r = gp.irrigation_scheduling([{"soil_moisture_mm": 10}])
        assert r[0]["irrigate_now"] is True

    def test_soil_sampling(self):
        r = gp.soil_sampling_design((0, 0, 100, 100), n_samples=9)
        assert len(r) == 9

    def test_precision_zones(self):
        data = [{"yield": i} for i in range(9)]
        r = gp.precision_agriculture_zones(data)
        zones = {d["management_zone"] for d in r}
        assert len(zones) > 1

    def test_forest_inventory(self):
        r = gp.forest_inventory([{"basal_area_m2_ha": 25, "stems_per_ha": 800}])
        assert r["n_plots"] == 1

    def test_timber_volume(self):
        r = gp.timber_volume_estimation([{"dbh_cm": 30, "height_m": 20}])
        assert r[0]["volume_m3"] > 0

    def test_deforestation(self):
        before = [[0.8, 0.8], [0.8, 0.8]]
        after = [[0.2, 0.2], [0.8, 0.8]]
        r = gp.deforestation_monitoring(before, after)
        assert r["deforested_pixels"] == 2

    def test_urban_tree_canopy(self):
        r = gp.urban_tree_canopy([{"area_m2": 1000, "canopy_m2": 300}])
        assert r[0]["canopy_pct"] == 30.0


# ── Urban / Utilities ──────────────────────────────────────────────

class TestUrbanUtilities:
    def test_cool_roof(self):
        r = gp.cool_roof_benefit([{"roof_area_m2": 200}])
        assert r[0]["energy_savings_kwh_yr"] > 0

    def test_energy_efficiency(self):
        r = gp.energy_efficiency_scoring([{"eui_kwh_m2": 100}])
        assert r[0]["grade"] in ("A", "B", "C", "D", "F")

    def test_renewable_siting(self):
        r = gp.renewable_energy_siting(
            [{"solar_irradiance": 5, "grid_distance": 3, "slope": 2, "land_cost": 4}])
        assert r[0]["score"] > 0

    def test_transmission_routing(self):
        r = gp.transmission_line_routing((0, 0), (10, 10), [])
        assert r["segments"] > 0

    def test_utility_pole_inventory(self):
        r = gp.utility_pole_inventory([{"material": "wood", "age_years": 15}])
        assert r["total_poles"] == 1

    def test_joint_trench(self):
        r = gp.joint_trench_design([{"diameter_m": 0.1, "depth_m": 1.2}])
        assert r["trench_width_m"] > 0

    def test_duct_bank(self):
        r = gp.duct_bank_capacity({"ducts": 4, "duct_diameter_mm": 100},
                                   [{"diameter_mm": 30, "installed": True}])
        assert r["available"] == 3


# ── Planning / Transportation ──────────────────────────────────────

class TestPlanning:
    def test_tax_valuation(self):
        r = gp.tax_parcel_valuation([{"land_value_per_m2": 100, "area_m2": 500}])
        assert r[0]["assessed_value"] > 0

    def test_property_tax_equity(self):
        r = gp.property_tax_equity([
            {"assessed_value": 100, "market_value": 120},
            {"assessed_value": 90, "market_value": 100}])
        assert r["cod"] >= 0

    def test_complete_streets(self):
        r = gp.complete_streets_analysis([{"has_sidewalk": True, "has_bike_lane": True,
                                           "has_transit": True, "speed_limit_kmh": 30}])
        assert r[0]["complete_streets_score"] > 0

    def test_lts(self):
        r = gp.level_of_traffic_stress([{"speed_limit_kmh": 30, "lanes": 2, "has_bike_lane": True}])
        assert r[0]["lts"] == 1

    def test_speed_study(self):
        r = gp.speed_study_analysis([40, 45, 50, 55, 60, 65])
        assert r["p85_kmh"] > 0

    def test_parking(self):
        r = gp.parking_inventory([{"spaces": 100, "occupied": 80}])
        assert r["occupancy_pct"] == 80.0

    def test_park_access(self):
        r = gp.park_access_analysis(
            [{"x": 0, "y": 0, "population": 100}],
            [{"x": 0, "y": 100}], walk_distance_m=200)
        assert r["access_pct"] > 0

    def test_dark_sky(self):
        r = gp.dark_sky_analysis([{"x": 10, "y": 10, "lumens": 5000}], (0, 0))
        assert r["bortle_class"] >= 1


# ── Emergency / Logistics ──────────────────────────────────────────

class TestEmergencyLogistics:
    def test_e911_validation(self):
        r = gp.e911_msag_validation(
            [{"street": "MAIN ST", "number": 100}],
            [{"street": "MAIN ST", "low": 1, "high": 200}])
        assert r[0]["msag_valid"] is True

    def test_ng911_schema(self):
        r = gp.ng911_gis_data_model()
        assert "RoadCenterline" in r

    def test_sar_coverage(self):
        r = gp.search_and_rescue_coverage(
            [{"area_km2": 10}], [{"members": 8, "hours": 8}])
        assert r["coverage_pct"] > 0

    def test_radiation_dispersion(self):
        r = gp.radiation_dispersion_model({})
        assert len(r) > 0
        assert r[0]["concentration_bq_m3"] > 0

    def test_pandemic_zones(self):
        r = gp.pandemic_response_zones([{"cases_per_100k": 500}])
        assert r[0]["risk_level"] == "critical"

    def test_vaccine_distribution(self):
        r = gp.vaccine_distribution_optimisation(
            [{"population": 1000}, {"population": 2000}], 3000)
        assert sum(s["allocation"] for s in r) <= 3000

    def test_supply_chain(self):
        r = gp.supply_chain_spatial_analysis(
            [{"id": "W1", "x": 0, "y": 0}],
            [{"x": 10, "y": 10}])
        assert r[0]["assigned_facility"] == "W1"

    def test_warehouse_location(self):
        r = gp.warehouse_location_optimisation(
            [{"id": "C1", "x": 0, "y": 0}, {"id": "C2", "x": 100, "y": 100}],
            [{"x": 5, "y": 5, "demand": 100}])
        assert r[0]["weighted_distance"] < r[1]["weighted_distance"]

    def test_drone_flight(self):
        r = gp.drone_flight_path([(0, 0, 50), (100, 0, 80), (200, 0, 150)])
        assert r["waypoints"][-1]["z"] == 120  # clamped


# ── Maritime ───────────────────────────────────────────────────────

class TestMaritime:
    def test_ais_track(self):
        pos = [{"lat": 0, "lon": 0, "sog_knots": 10},
               {"lat": 0.01, "lon": 0.01, "sog_knots": 12}]
        r = gp.ais_track_analysis(pos)
        assert r["total_distance_nm"] > 0

    def test_oil_spill(self):
        r = gp.oil_spill_trajectory((0, 0), hours=5)
        assert len(r) == 6

    def test_coral_health(self):
        r = gp.coral_reef_health([{"coral_cover_pct": 40, "bleaching_pct": 5}])
        assert r["health_rating"] == "good"

    def test_tidal_datum(self):
        r = gp.tidal_datum_mapping([{"high_water": [1.5, 1.7], "low_water": [-0.5, -0.3]}])
        assert r[0]["tidal_range"] > 0

    def test_beach_nourishment(self):
        r = gp.beach_nourishment_design(1000)
        assert r["total_volume_m3"] > 0


# ── Mining / Energy ────────────────────────────────────────────────

class TestMiningEnergy:
    def test_borehole(self):
        r = gp.borehole_mapping([{"depth_m": 100}, {"depth_m": 200}])
        assert r["mean_depth_m"] == 150

    def test_mine_plan(self):
        blocks = [{"grade": 1.0, "tonnes": 1000}, {"grade": 0.1, "tonnes": 500}]
        r = gp.mine_plan_pit_shell(blocks)
        assert r["ore_blocks"] == 1

    def test_tailings_dam(self):
        r = gp.tailings_dam_monitoring([{"pore_pressure_kpa": 250}])
        assert r[0]["alert"] is True

    def test_renewable_capacity(self):
        r = gp.renewable_energy_capacity([{"area_m2": 10000}])
        assert r[0]["annual_kwh"] > 0

    def test_solar_irradiance(self):
        r = gp.solar_irradiance_map([{"latitude": 35}])
        assert r[0]["irradiance_kwh_m2_yr"] > 0

    def test_ev_charging(self):
        r = gp.ev_charging_station_siting(
            [{"x": 0, "y": 0}],
            [{"x": 1, "y": 1, "ev_registrations": 100}])
        assert r[0]["demand_served"] == 100

    def test_geothermal(self):
        r = gp.geothermal_resource_map([{"geothermal_gradient_c_km": 60, "heat_flow_mw_m2": 80}])
        assert r[0]["category"] == "high"


# ── Smart City / 3D ───────────────────────────────────────────────

class TestSmartCity3D:
    def test_sensor_placement(self):
        cands = [{"x": 0, "y": 0}, {"x": 50, "y": 50}]
        targets = [{"x": 5, "y": 5}, {"x": 45, "y": 45}]
        r = gp.smart_city_sensor_placement(cands, targets, sensor_range_m=50)
        assert len(r) > 0

    def test_iot_coverage(self):
        r = gp.iot_device_coverage([{"x": 50, "y": 50}], (0, 0, 100, 100), range_m=60)
        assert r["coverage_pct"] > 0

    def test_wifi_coverage(self):
        r = gp.wifi_coverage_mapping([{"x": 50, "y": 50, "tx_power_dbm": 20}],
                                     grid_bounds=(0, 0, 100, 100), grid_size=5)
        assert len(r["grid"]) == 5

    def test_satellite_track(self):
        r = gp.satellite_ground_track(n_points=10)
        assert len(r) == 10

    def test_radar_coverage(self):
        r = gp.radar_coverage_footprint({"x": 0, "y": 0})
        assert r["coverage_area_km2"] > 0

    def test_link_budget(self):
        r = gp.link_budget_calculation(30, 10, 5, 100)
        assert "link_viable" in r

    def test_indoor_spatial(self):
        r = gp.indoor_spatial_analysis({"rooms": [{"type": "office", "area_m2": 25}]})
        assert r["room_count"] == 1

    def test_bim_to_gis(self):
        r = gp.bim_to_gis_conversion([{"local_x": 10, "local_y": 20, "ifc_type": "Wall"}])
        assert r[0]["geometry"]["coordinates"] == [10, 20]

    def test_citygml(self):
        xml = gp.citygml_model([{"id": "B1", "height_m": 15, "x": 1, "y": 2}])
        assert "Building" in xml

    def test_digital_twin(self):
        r = gp.digital_twin_layer(
            [{"id": "A1", "x": 0, "y": 0}],
            [{"asset_id": "A1", "temperature": 25}])
        assert r[0]["reading_count"] == 1

    def test_reality_capture(self):
        pts = [{"x": 0.1, "y": 0.1, "z": 0.1}, {"x": 0.2, "y": 0.2, "z": 0.2}]
        r = gp.reality_capture_integration(pts)
        assert r["voxel_count"] >= 1

    def test_ar_vr_scene(self):
        r = gp.ar_vr_georeferenced_scene(
            {"x": 0, "y": 0, "z": 0},
            [{"name": "tree", "model": "tree.glb", "x": 5, "y": 0, "z": 0}])
        assert r["objects"][0]["offset_x"] == 5

    def test_game_engine_export(self):
        hm = [[1, 2], [3, 4]]
        r = gp.game_engine_terrain_export(hm)
        assert r["rows"] == 2
        assert r["format"] == "R16"


# ── Route Optimisation ─────────────────────────────────────────────

class TestRouteOptimisation:
    def test_snow_plough(self):
        roads = [{"x": 10, "y": 0}, {"x": 0, "y": 10}]
        r = gp.snow_plough_route_optimisation(roads)
        assert len(r) == 2

    def test_refuse_collection(self):
        stops = [{"x": i, "y": 0, "weight_kg": 3000} for i in range(5)]
        r = gp.refuse_collection_route(stops, capacity_kg=10000)
        assert len(r) >= 1
        assert sum(len(t) for t in r) == 5

    def test_meter_reading(self):
        meters = [{"x": 10, "y": 0}, {"x": 5, "y": 5}]
        r = gp.meter_reading_route(meters)
        assert r[0]["sequence"] == 1
