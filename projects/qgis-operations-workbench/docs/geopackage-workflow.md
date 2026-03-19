# GeoPackage Workflow

This project can export a GeoPackage bundle for direct use in QGIS.

## Command

```bash
python -m qgis_operations_workbench.workbench --export-geopackage
```

## Output

The command writes `outputs/qgis_review_bundle.gpkg` with:

- `station_review_points` as a point feature layer in EPSG:4326
- `inspection_routes` as an attribute table for route and layout review
- GeoPackage metadata tables so QGIS can load the bundle as a normal desktop GIS source

## Reviewer Use

1. Open QGIS.
2. Add the GeoPackage file from the `outputs/` folder.
3. Load `station_review_points` for map review.
4. Open `inspection_routes` to inspect route and layout definitions alongside the map.

This keeps the repository runnable without QGIS while still producing a real desktop GIS artifact.
