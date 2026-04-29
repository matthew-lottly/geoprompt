"""J8.90 – Non-regression tests for fixed tier-metadata mismatches.

These tests ensure that specific tier assignments that were previously incorrect
(or at risk of regression) remain correct and internally consistent.

Background: Audit round 10 found 41 tier-metadata symbol mismatches. The
fixes are locked here as non-regression assertions.
"""
from __future__ import annotations

import pytest

from geoprompt._tier_metadata import (
    TIER_BETA,
    TIER_EXPERIMENTAL,
    TIER_METADATA,
    TIER_SIMULATION,
    TIER_STABLE,
    get_tier,
    warn_if_non_stable,
)


# ---------------------------------------------------------------------------
# J8.90.1 – Critical stable-tier symbols must remain stable
# ---------------------------------------------------------------------------


class TestStableTierRegressions:
    """Symbols declared stable must not regress to a lower tier."""

    @pytest.mark.parametrize(
        "key",
        [
            "frame.GeoPromptFrame",
            "geometry.hausdorff_distance",
            "geometry.frechet_distance",
            "geometry.clip_by_rect",
            "geometry.minimum_rotated_rectangle",
            "geometry.minimum_bounding_circle",
            "geometry.representative_point",
            "geometry.get_coordinates",
            "stats.moran_i",
            "stats.geary_c",
            "stats.ripley_k",
            "stats.clark_evans",
            "stats.idw_interpolation",
            "network.vrp_solver",
            "network.tsp_solver",
            "network.location_allocation",
            "network.closest_facility",
            "network.service_area",
            "network.network_partition",
            "raster.contour_generation",
            "raster.flow_direction_raster",
            "raster.raster_slope_aspect",
            "raster.raster_hillshade",
            "raster.raster_watershed",
            "raster.zonal_summary",
            "raster.raster_cost_distance",
            "raster.raster_least_cost_path",
            "io.read_geojson",
            "io.write_geojson",
            "io.read_shapefile",
            "io.write_shapefile",
            "geoprocessing.ModelBuilder",
            "geoprocessing.get_environment",
            "geoprocessing.set_environment",
        ],
    )
    def test_symbol_is_stable(self, key: str) -> None:
        assert TIER_METADATA.get(key) == TIER_STABLE, (
            f"Tier regression: {key!r} should be STABLE but is "
            f"{TIER_METADATA.get(key)!r}"
        )


# ---------------------------------------------------------------------------
# J8.90.2 – Simulation-only tier symbols must not be promoted to stable
# ---------------------------------------------------------------------------


class TestSimulationTierRegressions:
    """Simulation-only symbols must not silently become stable."""

    @pytest.mark.parametrize(
        "key",
        [
            "ml.gradient_boosted_spatial_prediction",
            "ml.svm_spatial_classification",
            "ml.convolutional_neural_network_on_rasters",
            "ml.graph_neural_network_prediction",
            "ml.neural_network_integration",
            "ml.recurrent_neural_network_spatial_time_series",
            "ml.transformer_model_spatial_sequences",
        ],
    )
    def test_symbol_is_simulation(self, key: str) -> None:
        assert TIER_METADATA.get(key) == TIER_SIMULATION, (
            f"Tier regression: {key!r} should be SIMULATION but is "
            f"{TIER_METADATA.get(key)!r}"
        )


# ---------------------------------------------------------------------------
# J8.90.3 – Beta tier assignments (non-regression for corrected mismatches)
# ---------------------------------------------------------------------------


class TestBetaTierRegressions:
    """Symbols that were corrected to BETA must not regress to STABLE."""

    @pytest.mark.parametrize(
        "key",
        [
            "geometry.concave_hull",
            "geometry.affine_transform",
            "stats.natural_neighbor_interpolation",
            "stats.variogram_fit",
            "stats.gwr",
            "network.multimodal_network",
            "raster.terrain_ruggedness_index",
            "raster.topographic_wetness_index",
            "io.read_cloud_json",
            "io.write_cloud_json",
            "io.read_dxf",
            "ml.spatial_cross_validation",
            "geoprocessing.ToolChain",
            "geoprocessing.batch_process",
            "standards.ogc_api_features_implementation",
            "standards.ogc_api_processes_implementation",
            "standards.ogc_api_records_implementation",
            "standards.ogc_api_tiles_implementation",
            "standards.ogc_api_maps_implementation",
            "standards.ogc_wfs_client",
            "standards.ogc_wms_client",
            "performance.gpu_accelerated_distance_matrix",
        ],
    )
    def test_symbol_is_beta(self, key: str) -> None:
        assert TIER_METADATA.get(key) == TIER_BETA, (
            f"Tier regression: {key!r} should be BETA but is "
            f"{TIER_METADATA.get(key)!r}"
        )


# ---------------------------------------------------------------------------
# J8.90.4 – warn_if_non_stable must warn for every non-stable symbol
# ---------------------------------------------------------------------------


class TestWarnIfNonStable:
    """warn_if_non_stable must emit a warning for every beta/experimental/simulation symbol."""

    def test_beta_emits_future_warning(self) -> None:
        with pytest.warns(FutureWarning):
            warn_if_non_stable("gwr")

    def test_simulation_emits_user_warning(self) -> None:
        with pytest.warns(UserWarning):
            warn_if_non_stable("wms_capabilities_document")

    def test_stable_does_not_warn(self) -> None:
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_non_stable("moran_i")
        assert not w, f"Unexpected warnings for stable symbol: {[str(x.message) for x in w]}"


# ---------------------------------------------------------------------------
# J8.90.5 – get_tier returns correct values
# ---------------------------------------------------------------------------


class TestGetTier:
    def test_get_tier_for_stable_symbol(self) -> None:
        assert get_tier("moran_i") == TIER_STABLE

    def test_get_tier_for_beta_symbol(self) -> None:
        assert get_tier("gwr") == TIER_BETA

    def test_get_tier_for_simulation_symbol(self) -> None:
        assert get_tier("wms_capabilities_document") == TIER_SIMULATION

    def test_get_tier_returns_none_for_unknown_symbol(self) -> None:
        result = get_tier("completely_unknown_function_xyz")
        assert result is None


# ---------------------------------------------------------------------------
# J8.90.6 – TIER_METADATA keys use consistent module.function format
# ---------------------------------------------------------------------------


class TestTierMetadataKeyFormat:
    def test_all_keys_contain_dot(self) -> None:
        """Every key must be in 'module.symbol' format."""
        bad_keys = [k for k in TIER_METADATA if "." not in k]
        assert not bad_keys, f"Malformed tier metadata keys (missing '.'): {bad_keys}"

    def test_all_values_are_known_tier_strings(self) -> None:
        known = {TIER_STABLE, TIER_BETA, TIER_EXPERIMENTAL, TIER_SIMULATION}
        bad = {k: v for k, v in TIER_METADATA.items() if v not in known}
        assert not bad, f"Unknown tier values in TIER_METADATA: {bad}"

    def test_tier_metadata_is_not_empty(self) -> None:
        assert len(TIER_METADATA) >= 20, (
            f"TIER_METADATA suspiciously small ({len(TIER_METADATA)} entries)"
        )
