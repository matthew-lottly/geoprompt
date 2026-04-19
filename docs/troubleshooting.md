# Troubleshooting GeoPrompt

This page covers the most common setup and runtime issues.

## 1. Import errors for optional features

GeoPrompt keeps several features behind optional extras.

### Symptoms
- ImportError for `folium`, `geopandas`, `sqlalchemy`, `openpyxl`, or `shapely`
- mapping, database, or overlay helpers are unavailable

### Fix
Install the needed profile:

```bash
pip install geoprompt[viz]
pip install geoprompt[io]
pip install geoprompt[db]
pip install geoprompt[excel]
pip install geoprompt[overlay]
pip install geoprompt[all]
```

## 2. GeoPackage or shapefile read/write fails

### Possible causes
- `geopandas` is not installed
- native GDAL / Fiona dependencies are missing in the environment

### Fix
1. Start with `pip install geoprompt[io]`
2. If the environment still fails, create a fresh virtual environment
3. On Windows, prefer a clean venv and current pip tooling before retrying

## 3. Overlay helpers fail with Shapely errors

The frame methods for simplify, hull, difference, symmetric difference, and union use Shapely-backed paths when available.

### Fix

```bash
pip install geoprompt[overlay]
```

If you still see geometry issues, run validation and repair first:

```python
import geoprompt as gp

fixed = frame.assign(geometry=[gp.repair_geometry(g) for g in frame.geometry])
```

## 4. Map output does not render

### Check
- make sure `folium` is installed
- save the map to HTML and open it in a browser

```python
from geoprompt.viz import to_folium_map, save_map

m = to_folium_map(frame, tooltip_fields=["site_id"])
save_map(m, "outputs/map.html")
```

## 5. Database connectors fail

### PostGIS
- install the database extra: `pip install geoprompt[db]`
- make sure your SQLAlchemy connection string is valid
- confirm the query returns a geometry column

### DuckDB
- install DuckDB separately if needed
- make sure the spatial extension is available in the runtime environment

## 6. Notebooks keep connecting or appear to hang on Windows

### Symptoms
- the kernel looks like it is connecting forever
- even a tiny cell seems stuck
- the same code works fine from the terminal or test suite

### Fix
1. use the project virtual environment
2. install the notebook profile or dev profile
3. run the timed notebook executor to surface real errors quickly

```bash
pip install -e .[notebook]
python examples/notebooks/execute_notebooks_with_timeout.py --timeout 20
```

GeoPrompt now includes a small Windows startup hook that switches the event loop policy used by notebook kernels to the selector-based path, which avoids common pyzmq notebook hangs.

## 7. Tests or examples behave differently across environments

### Recommended workflow
1. create a fresh virtual environment
2. install the package in editable mode
3. install the profile you need
4. run the test suite

```bash
pip install -e .[dev]
pytest
```

## 8. Slow performance on large workloads

### Tips
- use workload presets where available
- prefer indexed spatial queries for repeated windows
- limit columns during ingest when possible
- benchmark with the included corpus before scaling to production data

## 9. Still stuck?

When reporting an issue, include:
- Python version
- OS
- install command used
- the file format being read or written
- the smallest reproducible example
