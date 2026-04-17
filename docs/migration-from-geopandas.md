# Migration from GeoPandas to GeoPrompt

This guide maps common GeoPandas habits to the GeoPrompt workflow.

## Core idea

GeoPrompt is not trying to replace every part of the GeoPandas ecosystem. It is designed to cover the most common analyst tasks while adding stronger network and resilience workflows.

## Quick mental model

| GeoPandas habit | GeoPrompt equivalent |
|---|---|
| `GeoDataFrame(...)` | `GeoPromptFrame(...)` |
| `read_file(...)` | `read_data(...)`, `read_geopackage(...)`, `read_shapefile(...)` |
| `to_file(...)` | `write_data(...)`, `write_geopackage(...)`, `write_shapefile(...)` |
| `groupby(...).agg(...)` | `frame.groupby(...).agg(...)` |
| `pivot_table(...)` | `frame.pivot(...)` |
| `merge(...)` | `frame.merge(...)` |
| `fillna(...)` | `frame.fillna(...)` |
| plotting | `geoprompt.viz.quickplot(...)` or `to_folium_map(...)` |

## Example: load and summarize

```python
import geoprompt as gp

frame = gp.read_data("data/sample_features.json")
summary = frame.groupby("geometry_type").agg({"site_id": "count"})
print(summary.to_records())
```

## Example: clean fields

```python
clean = (
    frame.fillna({"capacity": 0})
         .rename_columns({"old_name": "new_name"})
         .dropna(subset=["site_id"])
)
```

## Example: map output

```python
from geoprompt.viz import to_folium_map, save_map

m = to_folium_map(frame, tooltip_fields=["site_id", "region"])
save_map(m, "outputs/sites.html")
```

## When to stay in GeoPandas

GeoPandas is still the better choice when you need:
- a very broad mature vector stack
- deeper ecosystem integrations across many external packages
- workflows centered around full pandas parity expectations

## When GeoPrompt is a better fit

GeoPrompt stands out when you want:
- a lightweight geospatial frame with analyst-friendly chaining
- built-in network and resilience analysis
- decision-support and reporting workflows
- a cleaner bridge from spatial analysis to operational scenario evaluation

## Interop is still available

If you want both tools in the same workflow:

```python
import geoprompt as gp

frame = gp.read_data("data/sample_features.json")
if gp.geopandas_available():
    gdf = gp.to_geopandas(frame)
    frame_again = gp.from_geopandas(gdf)
```
