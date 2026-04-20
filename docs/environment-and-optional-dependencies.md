# Environment Variables and Optional Dependencies

## Environment variables

- GEOPROMPT_RUN_BENCHMARKS — enables benchmark regression gates.
- GEOPROMPT_RUN_GEO_IO — enables optional geospatial IO round-trip tests.
- GEOPROMPT_RUN_REMOTE_INTEGRATION — enables remote integration checks.
- GEOPROMPT_RUN_DB_INTEGRATION — enables database integration checks.

## Optional dependency groups

- dev — pytest, mypy, Ruff, notebook tooling, and development checks.
- notebook — notebook execution support.
- viz — plotting and map visualization helpers.
- io — pandas, pyarrow, geopandas interop and richer file IO.
- excel — spreadsheet helpers.
- db — database connectors.
- compare — comparison and benchmark proof tooling.
- overlay — geometry overlay helpers.
- projection — CRS and reprojection helpers.
- raster — raster workflows.
- service — API and FastAPI service support.
- all — the full optional stack.
