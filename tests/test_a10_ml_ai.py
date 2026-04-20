"""Regression tests for A10 machine learning and AI integration helpers."""

import geoprompt as gp


class TestA10MachineLearningAI:
    def test_feature_engineering_raster_pixels(self):
        raster = [[1, 2], [3, 4]]
        pts = [{"x": 0, "y": 0}, {"x": 1, "y": 1}]
        out = gp.feature_engineering_raster_pixels(pts, raster)
        assert out[0]["pixel_value"] == 1
        assert out[1]["pixel_value"] == 4

    def test_spatial_cross_validation_buffer(self):
        pts = [{"x": 0, "y": 0, "target": 1}, {"x": 10, "y": 0, "target": 2}, {"x": 20, "y": 0, "target": 3}]
        out = gp.spatial_cross_validation_buffer(pts, target_key="target", buffer_distance=5)
        assert out["folds"] >= 1
        assert "mean_error" in out

    def test_spatial_cross_validation_leave_one_out(self):
        pts = [{"x": 0, "y": 0, "target": 1}, {"x": 5, "y": 0, "target": 2}, {"x": 10, "y": 0, "target": 3}]
        out = gp.spatial_cross_validation_leave_one_out(pts, target_key="target")
        assert out["n"] == 3
        assert out["mae"] >= 0

    def test_spatial_resampling(self):
        rows = [{"target": 1}] * 2 + [{"target": 0}] * 6
        over = gp.spatial_resampling(rows, target_key="target", strategy="oversample")
        under = gp.spatial_resampling(rows, target_key="target", strategy="undersample")
        assert len(over) >= len(rows)
        assert len(under) <= len(rows)

    def test_build_sklearn_pipeline(self):
        pipe = gp.build_sklearn_pipeline(["scale", "rf"])
        assert pipe["steps"] == ["scale", "rf"]

    def test_xgboost_lightgbm_integration(self):
        out = gp.xgboost_lightgbm_integration()
        assert "xgboost" in out and "lightgbm" in out

    def test_classic_models(self):
        rows = [{"f1": 1, "f2": 2, "target": 0}, {"f1": 2, "f2": 3, "target": 1}, {"f1": 3, "f2": 5, "target": 1}]
        rf = gp.random_forest_spatial_prediction(rows, feature_keys=["f1", "f2"], target_key="target")
        gb = gp.gradient_boosted_spatial_prediction(rows, feature_keys=["f1", "f2"], target_key="target")
        svm = gp.svm_spatial_classification(rows, feature_keys=["f1", "f2"], target_key="target")
        logit = gp.logistic_regression_spatial_features(rows, feature_keys=["f1", "f2"], target_key="target")
        assert len(rf) == len(rows)
        assert len(gb) == len(rows)
        assert len(svm) == len(rows)
        assert len(logit) == len(rows)

    def test_neural_and_graph_models(self):
        seqs = [[1, 2, 3], [2, 3, 4]]
        nn = gp.neural_network_integration(seqs)
        gnn = gp.graph_neural_network_prediction({"nodes": [1, 2], "edges": [(1, 2)]})
        cnn = gp.convolutional_neural_network_on_rasters([[1, 2], [3, 4]])
        rnn = gp.recurrent_neural_network_spatial_time_series(seqs)
        tr = gp.transformer_model_spatial_sequences(seqs)
        assert nn["backend"] in {"heuristic", "pytorch", "tensorflow"}
        assert gnn["node_count"] == 2
        assert "embedding_dim" in cnn
        assert rnn["sequence_count"] == 2
        assert tr["attention_heads"] >= 1

    def test_automl_and_hyperparameter_search(self):
        rows = [{"x": 1, "y": 2, "target": 0}, {"x": 2, "y": 3, "target": 1}]
        auto = gp.automl_spatial_workflow(rows, target_key="target")
        opt = gp.optuna_hyperparameter_search({"depth": [2, 4], "lr": [0.1, 0.01]})
        assert "best_model" in auto
        assert "best_params" in opt

    def test_interpretability(self):
        rows = [{"f1": 1, "f2": 2}, {"f1": 2, "f2": 1}]
        shap = gp.shap_spatial_interpretability(rows, feature_keys=["f1", "f2"])
        lime = gp.lime_spatial_interpretability(rows[0], feature_keys=["f1", "f2"])
        pdp = gp.partial_dependence_spatial(rows, feature_key="f1")
        fim = gp.feature_importance_map(rows, feature_keys=["f1", "f2"])
        conf = gp.prediction_confidence_map([0.1, 0.5, 0.9])
        assert len(shap["shap_values"]) == 2
        assert len(lime["explanations"]) == 2
        assert len(pdp["points"]) > 0
        assert fim[0]["importance"] >= 0
        assert conf[0]["confidence"] == 0.9

    def test_anomaly_and_changepoint(self):
        obs = [{"t": i, "value": v, "x": 0, "y": 0} for i, v in enumerate([1, 1, 1, 10, 1])]
        anom = gp.temporal_spatial_anomaly_detection(obs, value_key="value")
        cp = gp.spatial_time_series_changepoint_detection(obs, value_key="value")
        assert len(anom) >= 1
        assert cp["changepoints"]

    def test_vision_and_rag(self):
        image = [[0, 0, 2], [0, 3, 4], [1, 0, 0]]
        det = gp.object_detection_aerial_imagery(image, threshold=2)
        seg = gp.semantic_segmentation_aerial_imagery(image, threshold=1)
        inst = gp.instance_segmentation_aerial_imagery(image, threshold=2)
        pano = gp.panoptic_segmentation_aerial_imagery(image, threshold=1)
        cap = gp.image_caption_scene(image)
        vlm = gp.vision_language_model_integration(image, prompt="describe roads and rooftops")
        sam = gp.segment_anything_integration(image, seed_points=[(1, 1)])
        emb = gp.generate_embeddings(["river", "road"])
        sims = gp.embedding_similarity_search(emb[0], emb)
        rag = gp.retrieval_augmented_generation_spatial("What is near the river?", [{"text": "River crosses downtown."}])
        assert len(det["detections"]) >= 1
        assert seg["classes"]["foreground"] >= 1
        assert inst["instance_count"] >= 1
        assert pano["segments"] >= 1
        assert isinstance(cap["caption"], str)
        assert "response" in vlm
        assert sam["mask_pixels"] >= 1
        assert len(sims) == 2
        assert "answer" in rag

    def test_language_geo_helpers(self):
        text = "Flooding near Denver and Boulder is improving."
        ner = gp.named_entity_recognition_place_names(text)
        top = gp.toponym_resolution(["Denver"])
        geo = gp.geoparsing_extract_locations(text)
        sent = gp.sentiment_location_analysis([{"text": "good roads in Denver", "location": "Denver"}])
        social = gp.social_media_geotagging([{"text": "Checking in from Denver"}])
        assert "Denver" in ner["places"]
        assert top[0]["name"] == "Denver"
        assert len(geo["locations"]) >= 1
        assert sent[0]["sentiment_score"] > 0
        assert social[0]["geotag"] is not None

    def test_trajectory_helpers(self):
        traj = [{"x": 0, "y": 0, "t": 0}, {"x": 10, "y": 0, "t": 1}, {"x": 10, "y": 0, "t": 5}, {"x": 30, "y": 0, "t": 6}]
        move = gp.movement_pattern_classification(traj)
        seg = gp.trajectory_segmentation(traj)
        clus = gp.trajectory_clustering([traj, traj])
        pred = gp.trajectory_prediction(traj, steps=2)
        act = gp.activity_space_estimation(traj)
        hw = gp.home_work_location_inference([{"x": 0, "y": 0, "hour": 23}, {"x": 10, "y": 10, "hour": 10}])
        od = gp.od_matrix_from_gps_traces([traj])
        assert move["mode"] in {"move", "stop-move", "stop"}
        assert len(seg["segments"]) >= 1
        assert len(clus["clusters"]) == 2
        assert len(pred["predicted_points"]) == 2
        assert act["bbox"][0] == 0
        assert "home" in hw and "work" in hw
        assert od[0]["count"] == 1

    def test_probabilistic_and_privacy(self):
        rows = [{"x": 0, "y": 0, "value": 1}, {"x": 1, "y": 1, "value": 2}]
        bayes = gp.bayesian_spatial_model_bridge(rows)
        gpreg = gp.gaussian_process_spatial_regression(rows, target_key="value")
        bo = gp.bayesian_optimisation_spatial_sampling([(0, 0), (1, 1), (2, 2)])
        active = gp.active_learning_labelling(rows)
        semi = gp.semi_supervised_spatial_classification(rows, label_key="value")
        fed = gp.federated_learning_spatial_data([[{"value": 1}], [{"value": 3}]])
        priv = gp.differential_privacy_spatial_data(rows, epsilon=1.0)
        synth = gp.synthetic_spatial_data_generation(5)
        aug = gp.spatial_data_augmentation([{"x": 1, "y": 2}])
        assert bayes["posterior_mean"] >= 0
        assert len(gpreg) == 2
        assert bo["selected_point"] is not None
        assert len(active["priority_samples"]) >= 1
        assert len(semi) == 2
        assert fed["site_count"] == 2
        assert len(priv) == 2
        assert len(synth) == 5
        assert len(aug) >= 1

    def test_model_ops_and_forecasts(self):
        reg = gp.model_registry_mlflow("demo")
        serving = gp.model_serving_predict_endpoint([{"f1": 1}])
        drift = gp.model_monitoring_drift_detection([1, 1, 1], [1, 3, 5])
        ab = gp.ab_model_comparison_spatial_predictions([0.7, 0.8], [0.6, 0.9], [1, 1])
        uq = gp.uncertainty_quantification_conformal_prediction([0.4, 0.8])
        ens = gp.ensemble_model_spatial_boosting([[0.2, 0.8], [0.4, 0.6]])
        stk = gp.stacking_spatial_models([[0.1, 0.9], [0.2, 0.8]])
        val = gp.spatial_forecast_validation([1, 2], [1.2, 1.8])
        diff = gp.time_series_forecast_spatial_diffusion([{"region": "A", "value": 10}], steps=2)
        now = gp.nowcasting_short_term_spatial_prediction([{"value": 5}, {"value": 7}])
        assert reg["model_name"] == "demo"
        assert len(serving["predictions"]) == 1
        assert drift["drift_detected"] is True
        assert ab["winner"] in {"A", "B", "tie"}
        assert uq["intervals"]
        assert len(ens["ensemble_prediction"]) == 2
        assert len(stk["stacked_prediction"]) == 2
        assert val["mae"] >= 0
        assert len(diff["forecast"]) == 2
        assert "prediction" in now

    def test_simulation_and_optimisation(self):
        abm = gp.agent_based_model_spatial_output_analysis([{"x": 0, "y": 0}, {"x": 1, "y": 1}])
        sdm = gp.system_dynamics_spatial_mapping({"stock": 100}, [{"id": "A"}, {"id": "B"}])
        ca = gp.cellular_automata_land_use_change([[0, 0], [1, 0]], steps=2)
        gol = gp.game_of_life_spatial_model([[0, 1], [1, 1]], steps=1)
        ga = gp.genetic_algorithm_spatial_optimisation([(0, 0), (1, 1), (2, 2)])
        sa = gp.simulated_annealing_spatial_optimisation([(0, 0), (1, 1), (2, 2)])
        pso = gp.particle_swarm_spatial_optimisation([(0, 0), (1, 1)])
        ant = gp.ant_colony_optimisation_routing([(0, 0), (1, 1), (2, 2)])
        mo = gp.multi_objective_spatial_optimisation([{"cost": 2, "score": 8}, {"cost": 5, "score": 9}])
        cp = gp.constraint_programming_spatial([{"id": "A", "required": True}])
        ip = gp.integer_programming_facility_location([{"id": "F1", "cost": 5}, {"id": "F2", "cost": 3}], budget=4)
        lp = gp.linear_programming_allocation([{"id": "A", "demand": 2}, {"id": "B", "demand": 1}], capacity=6)
        flow = gp.network_optimisation_min_cost_flow([{"from": "S", "to": "T", "cost": 2, "capacity": 5}], demand=4)
        rob = gp.robust_optimisation_under_uncertainty([{"option": "A", "cost": 5, "uncertainty": 0.1}])
        assert abm["agent_count"] == 2
        assert len(sdm) == 2
        assert len(ca["grid"]) == 2
        assert len(gol["grid"]) == 2
        assert ga["best_score"] >= 0
        assert sa["best_score"] >= 0
        assert pso["best_position"] is not None
        assert len(ant["route"]) == 3
        assert mo["pareto_front"]
        assert cp["feasible"] is True
        assert ip["selected"] == ["F2"]
        assert sum(x["allocated"] for x in lp) == 6
        assert flow["total_flow"] == 4
        assert rob["best_option"] == "A"

    def test_sampling_and_imaging(self):
        lhs = gp.latin_hypercube_sampling(5, dimensions=2)
        sob = gp.sensitivity_analysis_sobol([0.2, 0.3, 0.5])
        mor = gp.morris_screening([{"name": "a", "effect": 0.5}])
        sur = gp.surrogate_model_spatial_simulation([{"x": 0, "y": 0, "value": 1}])
        dem = gp.digital_elevation_model_fusion([[[1, 2], [3, 4]], [[2, 3], [4, 5]]])
        sr = gp.image_super_resolution([[1, 2], [3, 4]], scale=2)
        imp = gp.spatial_missing_data_imputation([{"value": None}, {"value": 2}, {"value": 4}], key="value")
        down = gp.downscaling_coarse_to_fine([[10, 20]], scale=2)
        up = gp.upscaling_fine_to_coarse([[1, 2], [3, 4]], factor=2)
        match = gp.feature_matching_image_to_image([[1, 2], [3, 4]], [[1, 2], [0, 4]])
        assert len(lhs) == 5
        assert sob["total_index"] == 1.0
        assert mor[0]["rank"] == 1
        assert sur["training_points"] == 1
        assert dem[0][0] == 1.5
        assert len(sr) == 4
        assert imp[0]["value"] == 3
        assert len(down) == 2
        assert up[0][0] == 2.5
        assert match["match_score"] > 0

    def test_point_cloud_and_carto_ml(self):
        pts = [{"x": 0, "y": 0, "z": 1}, {"x": 1, "y": 1, "z": 10}, {"x": 2, "y": 2, "z": 20}]
        cls = gp.point_cloud_classification_ml(pts)
        sem = gp.point_cloud_semantic_segmentation(pts)
        inst = gp.point_cloud_instance_segmentation(pts)
        obj = gp.point_cloud_3d_object_detection(pts)
        bld = gp.building_reconstruction_from_point_cloud(pts)
        tree = gp.tree_detection_from_point_cloud(pts)
        line = gp.power_line_detection_from_point_cloud(pts)
        road = gp.road_surface_classification_from_point_cloud(pts)
        terr = gp.terrain_classification_from_point_cloud(pts)
        scene = gp.scene_understanding_indoor_outdoor([[0, 1], [2, 3]])
        depth = gp.depth_estimation_monocular_imagery([[1, 2], [3, 4]])
        pose = gp.pose_estimation_georeferenced_cameras([{"x": 0, "y": 0, "z": 10}])
        mvs = gp.multi_view_stereo_reconstruction([[[1, 2], [3, 4]], [[1, 2], [3, 5]]])
        nerf = gp.nerf_spatial_scenes([{"x": 0, "y": 0, "z": 0}])
        gs = gp.gaussian_splatting_integration([{"x": 0, "y": 0, "z": 0, "r": 1}])
        cbr = gp.coordinate_based_neural_representation([(0, 0, 1), (1, 1, 2)])
        gatt = gp.spatial_graph_attention_network({"nodes": [1, 2, 3], "edges": [(1, 2), (2, 3)]})
        diff = gp.spatial_diffusion_model_generative([[0, 1], [1, 0]])
        sty = gp.style_transfer_for_maps([[0, 1], [1, 0]], style="pastel")
        gen = gp.map_generalisation_ml([{"scale": 5000, "geometry": "line"}])
        lab = gp.automated_cartographic_labelling_ml([{"name": "Main"}])
        lay = gp.layout_optimisation_ml([{"id": "map"}, {"id": "legend"}])
        qa = gp.spatial_data_quality_assessment_ml([{"completeness": 0.9, "consistency": 0.8}])
        assert len(cls) == 3
        assert sem["classes"]
        assert inst["instances"] >= 1
        assert obj["objects_detected"] >= 1
        assert bld["building_count"] >= 1
        assert tree["tree_count"] >= 0
        assert line["line_count"] >= 0
        assert road["surface_type"] in {"rough", "smooth"}
        assert terr["terrain_class"] in {"flat", "hilly", "mountainous"}
        assert scene["scene_type"] in {"indoor", "outdoor"}
        assert len(depth) == 2
        assert pose[0]["yaw"] == 0
        assert mvs["points"] >= 1
        assert nerf["samples"] == 1
        assert gs["splats"] == 1
        assert cbr["sample_count"] == 2
        assert gatt["attention_edges"] == 2
        assert diff["generated_pixels"] == 4
        assert sty["style"] == "pastel"
        assert gen[0]["generalised"] is True
        assert lab[0]["label"] == "Main"
        assert lay[0]["position"] == 1
        assert qa["quality_score"] > 0

    def test_conflation_and_impact(self):
        dedup = gp.address_deduplication([{"address": "1 Main St"}, {"address": "1 MAIN ST"}])
        er = gp.entity_resolution_across_datasets([{"id": 1, "name": "A"}], [{"id": 99, "name": "A"}])
        schema = gp.schema_matching_between_datasets(["name", "addr"], ["name", "address"])
        onto = gp.ontology_mapping([{"feature_type": "school"}])
        feat = gp.feature_alignment_conflation([{"id": "r1"}], [{"id": "r1"}])
        road = gp.road_conflation([{"id": 1}], [{"id": 2}])
        bld = gp.building_conflation([{"id": 1}], [{"id": 2}])
        harm = gp.boundary_harmonisation([{"id": 1, "length": 10}], tolerance=2)
        chg = gp.change_detection_multitemporal_ml([[0, 1], [1, 1]], [[1, 1], [1, 0]])
        dmg = gp.damage_assessment_from_imagery([[0, 3], [4, 0]], threshold=2)
        hum = gp.humanitarian_mapping_ml_assisted([{"priority": 0.9}])
        pop = gp.population_estimation_from_imagery([[1, 2], [3, 4]])
        pov = gp.poverty_mapping_satellite_imagery([[0, 1], [1, 0]])
        lights = gp.nighttime_lights_analysis([[1, 2], [3, 4]])
        lst = gp.land_surface_temperature_from_satellite([[10, 20], [30, 40]])
        assert dedup["duplicates_removed"] == 1
        assert er[0]["matched"] is True
        assert schema[0]["left"] == "name"
        assert onto[0]["concept"] == "education"
        assert feat["matched_count"] == 1
        assert road["matched_segments"] == 1
        assert bld["matched_buildings"] == 1
        assert harm["adjusted_boundaries"] == 1
        assert chg["changed_pixels"] == 2
        assert dmg["damaged_pixels"] == 2
        assert hum[0]["suggested"] is True
        assert pop["estimated_population"] > 0
        assert pov["poverty_index"] >= 0
        assert lights["mean_lights"] == 2.5
        assert len(lst) == 2
