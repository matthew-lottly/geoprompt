# Tool Roadmap

## Current Direction

Geoprompt is strongest when a tool does one of these well:

- turns a repeated GIS workflow into a compact frame method
- stays lightweight without forcing pandas-style table machinery
- composes with the current nearest, predicate, buffer, and coverage primitives
- remains easy to validate against Shapely and GeoPandas when reference behavior exists

## Recently Completed

### Foundation and Compatibility

- Python 3.9 compatibility fallback for advanced typing support
- Canonical CRS normalization so EPSG variants resolve consistently
- Multi-part geometry support across core geometry metrics and predicates
- Frame row indexing and JSON export

### IO and Interoperability

- WKT support for Point, MultiPoint, LineString, and Polygon tabular reads
- GeoJSON export support for MultiPoint, MultiLineString, and MultiPolygon
- Geometry-column reads in `read_table(...)` without requiring `x_column`/`y_column`

### Performance and Scale

- Indexed paths for nearest joins, proximity joins, spatial joins, query radius, within-distance masks, nearest neighbors, and coverage summaries
- Broader benchmark coverage for indexed-versus-direct comparisons

### Utility and Niche Workflows

- Richer water pressure tracing with residual pressure and pressure deficit outputs
- Richer fire-flow checks with service deficit and available margin outputs
- Richer gas pressure traces with minimum/maximum pressure summaries

## Large Backlog: High-Value Next Work

### 1. Catchment Competition

- Implemented through `catchment_competition(...)`
- Provides exclusive and contested target counts for overlapping provider catchments

### 2. Corridor Reach

- Implemented through `corridor_reach(...)`
- Adds route-like screening around line features without requiring full network analysis

### 3. Overlay Summaries

- Implemented through `overlay_summary(...)`
- Returns summary metrics instead of raw geometry outputs when users only need counts/share indicators

### 4. Zone Fit Scoring

- Implemented through `zone_fit_scoring(...)`
- Scores how well a site fits a zone by demand and proximity context

### 5. Multi-Scale Clustering

- Implemented through `multi_scale_clustering(...)`
- Uses a deterministic distance-threshold clustering approach for points and mixed geometries

### 6. Format Expansion

- Implemented WKT support for MultiLineString and MultiPolygon
- Implemented `write_geoparquet(...)` with enriched GeoParquet `geo` metadata
- Implemented `read_geoparquet_metadata(...)` for metadata-only reads
- Implemented `discover_layers(...)` for layer and schema discovery

### 7. Network Depth

- Implemented `attribute_aware_headloss(...)` with per-edge Hazen-Williams
- Implemented `demand_weighted_restoration_ranking(...)` for demand-weighted benefit ranking
- Implemented `cross_utility_dependency_score(...)` for cascade exposure scoring

### 8. Reporting and UX

- Implemented HTML/Markdown/CSV export on `PromptTable`
- Implemented `GeoPromptFrame.to_markdown(...)` and `GeoPromptFrame.summary()`
- Expanded cookbook examples around utility decision support

### 9. Validation and Benchmarking

- Implemented `save_benchmark_snapshot(...)` for reproducible benchmark artifacts
- Implemented larger stress corpora generators (4x scale)
- Existing parity coverage against Shapely and GeoPandas

### 10. Package Maturity

- Implemented `api-stability.md` with stable/advanced/network API classification
- Implemented `GeoPromptFrame.explode()` and `GeoPromptFrame.unary_union()` (GeoPandas parity)
- Existing: `dissolve(...)` with full aggregation, error validation at boundaries

## Design Rules For New Tools

- Prefer a frame method when the output should stay row-oriented and composable.
- Prefer summaries when users need metrics, not derived geometry.
- Add `max_distance`, `k`, or aggregation controls early if they change workflow value materially.
- Keep geometry normalization and CRS handling inside existing frame and geometry layers.
- Add comparison coverage when a clear Shapely or GeoPandas reference exists.

## Recommended Next Implementation Order

1. Catchment competition
2. Overlay summaries
3. Corridor reach
4. Zone fit scoring
5. Multi-scale clustering
6. MultiLineString and MultiPolygon WKT support
7. HTML table/report exports
8. larger stress benchmarks and published benchmark snapshots
