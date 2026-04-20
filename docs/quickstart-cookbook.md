# Quickstart Cookbook

This cookbook shows common workflows with practical defaults.

## 1. Read Huge CSV Point Data

```python
import geoprompt as gp

frame = gp.read_csv_points(
    "assets.csv",
    x_column="lon",
    y_column="lat",
    preset="large",
    use_columns=["asset_id", "lon", "lat", "demand"],
)
```

## 2. Stream Very Large CSV Files

```python
import geoprompt as gp

for chunk in gp.iter_csv_points(
    "assets.csv",
    x_column="lon",
    y_column="lat",
    preset="huge",
):
    # Analyze each chunk without loading the whole file in memory.
    _ = chunk.nearest_neighbors(k=1)
```

## 3. Use Progress Callbacks

```python
import geoprompt as gp

progress = []
for _chunk in gp.iter_data(
    "assets.csv",
    x_column="lon",
    y_column="lat",
    chunk_size=50000,
    progress_callback=progress.append,
):
    pass

print(progress[-1])
# {'event': 'chunk', 'path': 'assets.csv', 'chunk_index': ..., 'chunk_rows': ..., 'rows_emitted': ...}
```

## 4. Batch OD Matrix for Large Graphs

```python
from geoprompt.network import build_network_graph, od_cost_matrix_with_preset

graph = build_network_graph(edges)
rows = od_cost_matrix_with_preset(
    graph,
    origins=origin_nodes,
    destinations=destination_nodes,
    preset="large",
)
```

## 5. Stream Bottleneck Analysis

```python
from geoprompt.network import utility_bottlenecks_with_preset

rows = utility_bottlenecks_with_preset(
    graph,
    od_demands=((o, d, q) for o, d, q in huge_od_iterable),
    preset="huge",
)
```

## 6. Read/Write Geospatial Parquet

```python
import geoprompt as gp

frame = gp.read_data("assets.parquet")
gp.write_data("assets_out.parquet", frame)
```

## 7. Clean Strings, Dates, and Rank Priorities

```python
import geoprompt as gp

sites = gp.geopromptframe.from_records(records)
cleaned = (
    sites
    .str.strip("facility_name")
    .dt.year("inspection_date", new_column="inspection_year")
    .query("priority >= 2 and status == 'open'")
    .nlargest(10, "priority")
)
```

## 8. Check Raster Alignment Before Map Algebra

```python
import geoprompt as gp

report = gp.raster_alignment_report([risk_surface, outage_surface])
if report["aligned"]:
    combined = gp.raster_lazy_algebra("risk + outage", {"risk": risk_surface, "outage": outage_surface})
```

## 9. Inspect Enterprise Persistence and Index Guidance

```python
import geoprompt as gp

matrix = gp.enterprise_persistence_matrix()
index_plan = gp.index_planning_suggestions(records, candidate_fields=["asset_id", "status", "owner"])
```

## Preset Reference

- `small`: low memory and bounded default reads.
- `medium`: balanced defaults for most laptop workloads.
- `large`: full-fidelity default for bigger jobs.
- `huge`: favors throughput with mild sampling.
