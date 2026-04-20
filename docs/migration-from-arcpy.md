# Migration from ArcPy

GeoPrompt is designed for lightweight, scriptable spatial workflows using plain Python objects, GeoJSON-like records, and a small frame API.

## Common translations

- ArcPy feature-class reads → use the unified data readers such as read_data and read_features.
- ArcPy table operations → use GeoPromptFrame, PromptTable, and the table helpers.
- ArcPy geoprocessing chains → use the geoprocessing and workflow helpers.
- ArcPy reporting → use the comparison, report, and resilience export helpers.

## Typical workflow shift

1. Load data with a reader from the IO surface.
2. Keep geometry in a GeoJSON-like structure.
3. Apply spatial analysis or network helpers.
4. Export to JSON, CSV, HTML, or GeoJSON.

## Practical notes

- GeoPandas and Shapely remain optional for richer interop.
- Core frame and network workflows stay usable without a heavy GIS stack.
- The migration path is usually smallest for analyst-style scripts and reporting pipelines.
