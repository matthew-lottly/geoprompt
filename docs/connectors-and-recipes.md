# Connectors and Recipes

## Cloud Object Storage Bridges

GeoPrompt can read remote JSON and service-style resources and works well with chunked remote ingestion patterns. For larger deployments, pair it with your preferred object-storage SDK or file-system bridge.

## DuckDB and PostGIS Examples

```python
from geoprompt.db import read_duckdb, write_duckdb

rows = [
    {"id": "a", "geometry": {"type": "Point", "coordinates": [0, 0]}},
    {"id": "b", "geometry": {"type": "Point", "coordinates": [1, 1]}},
]

write_duckdb(rows, "sample_points")
roundtrip = read_duckdb("SELECT * FROM sample_points")
print(roundtrip)
```

## Arrow and Parquet Guidance

Use Parquet and Arrow-backed flows when:
- row counts are large
- only a subset of columns is needed
- you want predicate pushdown or faster roundtrips

## Raster and Vector End-to-End Recipe

1. Read or sample a raster.
2. Clip or mask it.
3. Rasterize feature overlays when needed.
4. Run change detection, terrain, or hydrology helpers.
5. Export a report card or vectorized summary.

## Enterprise Editing Recipe

Use AuthProfile, paginated_request, and feature_service_sync together for safer service updates and auditing.
