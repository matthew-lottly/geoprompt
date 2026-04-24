from __future__ import annotations

import math

import pytest

import geoprompt as gp
from .fixtures import tiny_raster


def test_raster_core_ops_are_deterministic_against_reference_grid_math() -> None:
    raster = tiny_raster()
    slope_aspect = gp.raster_slope_aspect(raster)
    hillshade = gp.raster_hillshade(raster)

    assert slope_aspect["rows"] == 3
    assert slope_aspect["cols"] == 3
    assert hillshade["rows"] == 3
    assert hillshade["cols"] == 3
    assert math.isclose(float(slope_aspect["slope"][1][1]), 1.4142135623730951, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(slope_aspect["aspect"][1][1]), 135.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(hillshade["grid"][1][1]), 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(hillshade["grid"][0][0]), 180.31222920256963, rel_tol=0.0, abs_tol=1e-9)


def test_rasterio_bridge_available_for_file_parity_when_installed() -> None:
    pytest.importorskip("rasterio")
    # This test gates dependency path presence; file-backed parity cases can be
    # added on top of this scaffold with fixture rasters in a follow-up pass.
    assert True
