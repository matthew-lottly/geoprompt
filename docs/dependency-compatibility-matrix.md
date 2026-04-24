# Dependency Compatibility Matrix

This matrix tracks dependency ranges, known warning drift, and upgrade tasks so deprecations do not silently become runtime errors.

## Runtime Compatibility Ranges

| Package | Current Range | Purpose | Risk Notes |
|---|---|---|---|
| numpy | >=1.26,<3.0 | Numeric kernels / optional network extras | `np.find_common_type` deprecation impacts pandas/geopandas internals |
| pandas | >=2.2,<3.0 | Tabular operations / IO extras | Uses deprecated numpy API in some join paths (upstream) |
| geopandas | >=1.0,<2.0 | Comparator parity and geospatial frame bridges | Depends on pandas/numpy behavior for joins |
| pyproj | >=3.6,<4.0 | CRS parity and projections | Watch for CRS serialization API changes |
| shapely | >=2.0,<3.0 | Geometry parity operations | Monitor GEOS compatibility shifts |
| rasterio | >=1.3,<2.0 | Raster parity tests | GDAL ABI upgrades may require pin refresh |

## Warning Watchlist

| Warning Signature | Observed In | Status | Action |
|---|---|---|---|
| `np.find_common_type is deprecated` | GeoPandas sjoin path in parity test | Known / filtered in parity test | Remove filter once pandas/geopandas upstream no longer emits warning |
| optional import initialization failures (`matplotlib`) | parity push viz tests on some Windows envs | Mitigated | `pytest.importorskip(..., exc_type=ImportError)` enforced |

## Monitoring Tasks

1. Run `python -m pytest tests/parity -q` on each dependency upgrade PR.
2. Run `python tests/parity/check_parity_drift.py` to detect explicit benchmark mapping drift.
3. If new dependency warnings appear, either:
   - pin/upgrade dependency ranges in `pyproject.toml`, or
   - add a targeted temporary warning filter with a linked upstream issue.
4. Remove temporary warning filters during each quarterly dependency refresh.
