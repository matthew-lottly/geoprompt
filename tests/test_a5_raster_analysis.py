"""Regression tests for remaining A5 raster and image analysis items."""

from pathlib import Path

import geoprompt as gp


class TestA5RasterAnalysis:
    def test_raster_math_and_pansharpen(self):
        out = gp.raster_math_advanced([[1, 4], [9, 16]], operation="sqrt")
        pan = gp.pansharpen_raster([[10, 12], [14, 16]], [[2, 3], [4, 5]])
        assert out[0][1] == 2.0
        assert len(pan) == 2

    def test_deep_learning_raster_helpers(self):
        image = [[0, 2], [3, 1]]
        pix = gp.deep_learning_pixel_classification(image)
        det = gp.deep_learning_object_detection(image, threshold=2)
        chg = gp.deep_learning_change_detection([[0, 1], [1, 1]], [[1, 1], [1, 0]])
        samples = gp.training_sample_creation(image, labels={(0, 0): "water", (1, 1): "urban"})
        assert pix[0][1] in {"low", "medium", "high"}
        assert len(det["objects"]) == 2
        assert chg["changed_pixels"] == 2
        assert len(samples) == 2

    def test_solar_and_terrain_indices(self):
        pts = gp.points_solar_radiation([(0, 0, 10), (1, 1, 20)])
        area = gp.area_solar_radiation([[1, 2], [3, 4]])
        hli = gp.heat_load_index([[1, 2], [3, 4]])
        sti = gp.sediment_transport_index([[1, 2], [3, 4]])
        assert pts[0]["solar_radiation"] > 0
        assert area["mean_radiation"] > 0
        assert hli["mean_index"] > 0
        assert sti["mean_index"] > 0

    def test_spectral_workflows(self):
        bands = {"red": [[1, 2], [3, 4]], "nir": [[5, 6], [7, 8]], "green": [[2, 2], [2, 2]]}
        tasseled = gp.tasseled_cap_transformation(bands)
        mnf = gp.minimum_noise_fraction([[1, 2], [3, 4]])
        unmixed = gp.spectral_unmixing({"soil": [0.8, 0.2], "veg": [0.2, 0.8]}, [0.5, 0.5])
        anom = gp.spectral_anomaly_detection([[1, 2], [20, 2]])
        assert "brightness" in tasseled
        assert mnf["component_count"] >= 1
        assert abs(sum(unmixed["fractions"].values()) - 1.0) < 0.001
        assert anom["anomaly_pixels"] >= 1

    def test_time_series_and_crop_products(self, tmp_path: Path):
        out = gp.time_series_animation_export([[[1]], [[2]], [[3]]], tmp_path / "anim.gif")
        phen = gp.phenology_metric_extraction([0.1, 0.2, 0.8, 0.7, 0.3])
        gdd = gp.growing_degree_days_accumulation([10, 12, 20], base_temp=5)
        crop = gp.crop_mask_generation([[0.2, 0.7], [0.8, 0.1]], threshold=0.5)
        land = gp.land_cover_map([[0.1, 0.4], [0.7, 0.9]])
        imperv = gp.impervious_surface_map([[0.2, 0.8], [0.9, 0.1]])
        assert Path(out).exists()
        assert phen["peak_index"] == 2
        assert gdd["accumulated_gdd"] > 0
        assert crop["crop_pixels"] == 2
        assert land[0][0] in {"water", "bare", "vegetation", "urban"}
        assert imperv["impervious_pixels"] == 2

    def test_imagery_feature_extraction(self):
        heights = gp.tree_canopy_height_model([[10, 12], [14, 16]], [[2, 3], [4, 5]])
        buildings = gp.building_footprint_extraction([[0, 2], [3, 0]], threshold=2)
        roads = gp.road_extraction_from_imagery([[0, 1], [1, 1]], threshold=1)
        water = gp.water_body_extraction([[0.1, 0.8], [0.9, 0.2]], threshold=0.7)
        shadow = gp.shadow_detection_and_removal([[0, 5], [1, 6]], threshold=1)
        assert heights[0][0] == 8
        assert buildings["building_pixels"] == 2
        assert roads["road_pixels"] == 3
        assert water["water_pixels"] == 2
        assert shadow["shadow_pixels"] == 2

    def test_image_corrections(self):
        cloud = gp.cloud_masking([[0.2, 0.95], [0.5, 0.99]], threshold=0.9)
        atm = gp.atmospheric_correction([[10, 20], [30, 40]], haze=2)
        radio = gp.radiometric_calibration([[10, 20], [30, 40]], gain=0.1, bias=1)
        ortho = gp.orthorectification([[1, 2], [3, 4]])
        geo = gp.georeferencing_from_gcps([{"pixel": (0, 0), "map": (100, 200)}])
        reg = gp.image_registration_feature_matching([[1, 2], [3, 4]], [[1, 2], [3, 5]])
        assert cloud["cloud_pixels"] == 2
        assert atm[0][0] == 8
        assert radio[0][0] == 2.0
        assert ortho["orthorectified"] is True
        assert geo["gcp_count"] == 1
        assert reg["match_score"] > 0

    def test_photogrammetry_and_lidar(self):
        dem = gp.stereo_pair_dem_extraction([[1, 2], [3, 4]], [[2, 3], [4, 5]])
        block = gp.photogrammetric_block_adjustment([{"id": 1}, {"id": 2}])
        pts = [{"x": 0, "y": 0, "z": 1, "intensity": 10}, {"x": 1, "y": 1, "z": 15, "intensity": 20}, {"x": 2, "y": 2, "z": 30, "intensity": 30}]
        ground = gp.lidar_ground_classification(pts)
        filt = gp.lidar_noise_filter(pts + [{"x": 1000, "y": 1000, "z": 999}])
        chm = gp.lidar_canopy_height_model([{"x": 0, "y": 0, "z": 20, "class": "canopy"}], [{"x": 0, "y": 0, "z": 5, "class": "ground"}])
        bld = gp.lidar_building_extraction(pts)
        power = gp.lidar_power_line_extraction(pts)
        norm = gp.lidar_intensity_normalisation(pts)
        thin = gp.lidar_point_thinning(pts, keep_every=2)
        assert len(dem) == 2
        assert block["camera_count"] == 2
        assert any(p["class"] == "ground" for p in ground)
        assert len(filt) < len(pts) + 1
        assert chm[0]["height"] == 15
        assert bld["building_points"] >= 1
        assert power["powerline_points"] >= 1
        assert max(p["intensity_norm"] for p in norm) == 1.0
        assert len(thin) == 2

    def test_lidar_surfaces_and_drone_products(self):
        pts = [{"x": 0, "y": 0, "z": 1, "class": "ground", "band_red": 1, "band_nir": 2, "temp": 20}, {"x": 1, "y": 1, "z": 5, "class": "canopy", "band_red": 2, "band_nir": 4, "temp": 22}]
        dem = gp.lidar_dem_creation_ground_returns(pts, grid_size=2)
        dsm = gp.lidar_dsm_creation_first_returns(pts, grid_size=2)
        density = gp.lidar_point_density_raster(pts, grid_size=2)
        mosaic = gp.drone_imagery_mosaic([[[1, 2]], [[3, 4]]])
        sfm = gp.structure_from_motion_point_cloud([{"image_id": 1}, {"image_id": 2}, {"image_id": 3}])
        idx = gp.multispectral_drone_index_maps(pts)
        thermal = gp.thermal_imagery_analysis([{"temp": 20}, {"temp": 22}])
        sarf = gp.sar_speckle_filter([[1, 2], [3, 4]])
        sarc = gp.sar_coherence_interferometry([[1, 2], [3, 4]], [[1, 1], [4, 4]])
        assert dem["rows"] == 2
        assert dsm["rows"] == 2
        assert density["max_density"] >= 1
        assert len(mosaic) == 1
        assert sfm["point_count"] >= 3
        assert "ndvi" in idx
        assert thermal["mean_temperature"] == 21.0
        assert len(sarf) == 2
        assert sarc["mean_coherence"] >= 0
