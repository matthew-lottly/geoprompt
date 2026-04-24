from __future__ import annotations

import math

import pytest

import geoprompt as gp


def test_to_crs_bounds_parity_with_geopandas_pyproj_stack() -> None:
    pytest.importorskip("pyproj")
    pytest.importorskip("geopandas")

    frame = gp.read_data("data/sample_points.json", crs="EPSG:4326")
    projected = frame.to_crs("EPSG:3857")

    gdf = gp.to_geopandas(frame).to_crs("EPSG:3857")

    observed = projected.bounds()
    expected = [float(v) for v in gdf.total_bounds]
    assert math.isclose(observed.min_x, expected[0], rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(observed.min_y, expected[1], rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(observed.max_x, expected[2], rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(observed.max_y, expected[3], rel_tol=0.0, abs_tol=1e-6)

    projected_rows = projected.to_pandas().to_dict(orient="records")
    gdf_rows = gdf.to_dict(orient="records")
    assert len(projected_rows) == len(gdf_rows)

    for lhs, rhs in zip(projected_rows, gdf_rows):
        assert str(lhs["site_id"]) == str(rhs["site_id"])
        assert str(lhs["name"]) == str(rhs["name"])

        lhs_x, lhs_y = lhs["geometry"]["coordinates"]
        rhs_x = float(rhs["geometry"].x)
        rhs_y = float(rhs["geometry"].y)
        assert math.isclose(float(lhs_x), rhs_x, rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(float(lhs_y), rhs_y, rel_tol=0.0, abs_tol=1e-6)
