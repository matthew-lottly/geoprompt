# Environment Variables and Optional Dependencies

## Environment variables

- GEOPROMPT_RUN_BENCHMARKS — enables benchmark regression gates.
- GEOPROMPT_RUN_GEO_IO — enables optional geospatial IO round-trip tests.
- GEOPROMPT_RUN_REMOTE_INTEGRATION — enables remote integration checks.
- GEOPROMPT_RUN_DB_INTEGRATION — enables database integration checks.

## Optional dependency groups

- dev — pytest, mypy, Ruff, notebook tooling, and development checks.
- developer — alias for the development tooling profile.
- analyst — common analyst-facing stack for plotting, tabular IO, Excel, GeoPandas interop, CRS, and overlay-heavy workflows.
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
- full — alias for the full optional stack.

## Core install behavior

- `pip install geoprompt` keeps the base install lightweight and does not require `matplotlib` or GeoPandas.
- Plotting and chart export paths should use `geoprompt[viz]` or `geoprompt[analyst]`.

---

## Optional dependency failure-mode matrix (J5.6)

Every optional dependency is classified by the failure mode that applies when it is absent.
Use the `geoprompt.capability_status()` function or the CLI `geoprompt capability-report` to see a live snapshot.

| Capability | pip extra | Failure mode | Affected public functions |
|---|---|---|---|
| geopandas | `geoprompt[io]` | **hard-fail** | `to_geodataframe`, `from_geodataframe`, `write_shapefile`, `read_geoparquet` |
| pandas | `geoprompt[io]` | **hard-fail** | `to_dataframe`, `from_dataframe`, `read_excel`, `write_excel`, `read_feather` |
| pyarrow | `geoprompt[io]` | **hard-fail** | `to_arrow`, `from_arrow`, `write_feather`, `read_feather` |
| shapely | `geoprompt[overlay]` | **soft-fail** | `geometry_union`, `geometry_buffer`, `clip_frame` |
| pyproj | `geoprompt[projection]` | **soft-fail** | `reproject_frame`, `transform_geometry`, `crs_info` |
| fiona | `geoprompt[io]` | **soft-fail** | `read_shapefile`, `read_vector` |
| pyogrio | `geoprompt[io]` | **soft-fail** | `read_shapefile`, `read_vector` |
| rasterio | `geoprompt[raster]` | **hard-fail** | `read_raster`, `write_raster`, `raster_algebra` |
| matplotlib | `geoprompt[viz]` | **degraded** | `plot_frame`, `choropleth_map`, `histogram` |
| folium | `geoprompt[viz]` | **degraded** | `to_folium_map`, `interactive_map` |
| plotly | `geoprompt[viz]` | **degraded** | `scatter_map`, `density_plot` |
| openpyxl | `geoprompt[excel]` | **hard-fail** | `read_excel`, `write_excel` |
| sqlalchemy | `geoprompt[db]` | **hard-fail** | `read_postgis`, `write_postgis` |
| duckdb | `pip install duckdb` | **hard-fail** | `read_duckdb`, `write_duckdb` |
| fastapi | `geoprompt[service]` | **hard-fail** | `serve` |
| uvicorn | `geoprompt[service]` | **hard-fail** | `serve` |
| ezdxf | `pip install ezdxf` | **hard-fail** | `read_dxf` |
| polars | `pip install polars` | **soft-fail** | `to_polars`, `from_polars` |
| numpy | `geoprompt[network]` | **degraded** | `network_centrality`, `raster_statistics` |

**Failure mode definitions**

- **hard-fail**: Raises `DependencyError` immediately with a `pip install` hint. No useful output is produced.
- **soft-fail**: Raises `DependencyError` with a clear message; a different code path exists for environments without the dep.
- **degraded**: Continues with reduced functionality and emits a `UserWarning`. Output remains valid but may be less accurate or format-limited.

---

## CI test-environment matrix (J5.8, J5.16)

The following profiles are tested in CI to guarantee degraded-mode correctness.
Profile constants are available in `geoprompt._capabilities.CI_EXTRAS_PROFILES`.

| CI profile | Extras installed | Degraded-mode guarantee |
|---|---|---|
| `core-only` | _(none)_ | JSON/GeoJSON and plain CSV work. Shapefile, GeoParquet, Excel, and DB connectors raise `DependencyError` with a pip hint. |
| `common` | `io`, `viz`, `overlay` | Adds GeoDataFrame, Arrow, and web-map export. Raster, DB, and service still require explicit extras. |
| `analyst` | `analyst` | Full analyst stack (io + viz + overlay + excel + projection). Raster, DB, service still require extras. |
| `all` | `all` | All optional features available. No degraded-mode paths expected. |

Run `geoprompt capability-report` in any environment to see which capabilities are enabled, disabled, or degraded.

---

## Runtime pip-hint behavior (J5.7)

When an optional dependency is missing, GeoPrompt raises `DependencyError` (a subclass of `GeoPromptError`) with an actionable message:

```
DependencyError: Optional dependency 'rasterio' is required (needed by read_raster) but is not installed.
  Enables raster read/write and band algebra.
  Install with: pip install geoprompt[raster]
```

This applies to all hard-fail and soft-fail paths. Degraded paths issue a `UserWarning` instead.

---

## Adaptive chunk sizing (J5.12–J5.15)

`iter_data` and similar batch-read paths support three chunking modes.

| Mode | Behaviour |
|---|---|
| `fixed` | Always use the caller-supplied or default chunk size (50 000 rows). |
| `adaptive` | Estimate row width from a sample row or column list, then divide the memory budget (default 128 MiB). Clamped to [1 000, 500 000]. |
| `auto` | Uses `fixed` when an explicit `chunk_size` is passed; uses `adaptive` otherwise. |

Use `geoprompt.estimate_chunk_size()` to compute a `ChunkDecision` object.
The `ChunkDecision.reasoning` field explains exactly why the size was chosen so you can verify the decision is deterministic.

**Determinism guarantee**: given the same `columns`, `sample_row`, and `memory_budget_bytes`, `estimate_chunk_size` always returns the same result. No OS-level memory queries are made unless `psutil` is installed, in which case 25 % of available virtual memory is used as the budget.

**Explicit override always wins**: passing `explicit_chunk_size` bypasses all estimation logic and is never silently ignored.

