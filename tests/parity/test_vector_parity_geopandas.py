from __future__ import annotations

import math
import warnings

import pytest

import geoprompt as gp


def test_geopandas_roundtrip_and_bounds_parity() -> None:
    frame = gp.read_data("data/sample_features.json", crs="EPSG:4326")
    gdf = gp.to_geopandas(frame)

    bounds = frame.bounds()
    ref_bounds = [float(v) for v in gdf.total_bounds]
    assert math.isclose(bounds.min_x, ref_bounds[0], abs_tol=1e-9)
    assert math.isclose(bounds.min_y, ref_bounds[1], abs_tol=1e-9)
    assert math.isclose(bounds.max_x, ref_bounds[2], abs_tol=1e-9)
    assert math.isclose(bounds.max_y, ref_bounds[3], abs_tol=1e-9)


def test_spatial_join_pair_count_parity_with_geopandas() -> None:
    geopandas = pytest.importorskip("geopandas")

    left = gp.read_data("data/benchmark_regions.json", crs="EPSG:4326")
    right = gp.read_data("data/sample_points.json", crs="EPSG:4326")

    gp_join = left.spatial_join(right, predicate="intersects")
    gpd_left = gp.to_geopandas(left)
    gpd_right = gp.to_geopandas(right)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*np.find_common_type is deprecated.*", category=DeprecationWarning)
        gpd_join = geopandas.sjoin(gpd_left, gpd_right, predicate="intersects", how="inner")

    assert len(gp_join) == len(gpd_join)

    gp_pairs = sorted((str(row["region_id"]), str(row["site_id"])) for row in gp_join.to_pandas().to_dict(orient="records"))
    gpd_pairs = sorted((str(row["region_id"]), str(row["site_id"])) for row in gpd_join.to_dict(orient="records"))
    assert gp_pairs == gpd_pairs
