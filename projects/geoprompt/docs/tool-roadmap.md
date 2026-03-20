# Tool Roadmap

## Current Direction

Geoprompt is strongest when a tool does one of these well:

- turns a repeated GIS workflow into a compact frame method
- stays lightweight without forcing pandas-style table machinery
- composes with the current nearest, predicate, buffer, and coverage primitives
- remains easy to validate against Shapely and GeoPandas when reference behavior exists

## Near-Term Tools

### 1. Assignment Summaries

- Implemented through `summarize_assignments(...)`
- Current scope covers per-origin assigned ids, counts, distance summaries, and target aggregation rollups
- Next extension should add contested and unassigned-target summaries

### 2. Catchment Competition

- Implemented through `catchment_competition(...)`
- Current scope covers exclusive, shared, won, and unserved target summaries inside a service radius
- Next extension should add geometry-driven competition summaries on top of buffered service areas and contested-demand rollups

### 3. Corridor Reach

- Implemented through `corridor_reach(...)`
- Current scope covers per-feature corridor matching within a distance limit, corridor distance summaries, total corridor length aggregation, Euclidean or haversine distance support, direct or network-style corridor distance, weighted direction-aware corridor scoring, and corridor-path anchor controls
- Companion diagnostics are implemented through `corridor_diagnostics(...)`
- Next extension should add corridor comparison views across alternate anchor or scoring strategies

## Mid-Term Tools

### 4. Overlay Summaries

- Implemented through `overlay_summary(...)`
- Current scope covers intersecting ids, intersection counts, overlap area, overlap length, per-feature area or length shares, grouped overlay summaries, optional right-side normalization, and overlay-group comparison helpers through `overlay_group_comparison(...)`
- Next extension should add ranked multi-group summaries and pairwise comparison tables

### 5. Zone Fit Scoring

- Implemented through `zone_fit_score(...)`
- Current scope covers containment, overlap, area similarity, access scoring, optional directional alignment, custom scoring weight control, grouped zone rankings, user-defined scoring callbacks, and best-zone assignment
- Next extension should add pairwise zone comparison outputs

### 6. Multi-Scale Clustering

- Implemented through `centroid_cluster(...)`
- Current scope covers deterministic k-means centroid-distance clustering with cluster ids, centers, distances, cluster SSE, silhouette-style quality metrics, cluster-count diagnostics through `cluster_diagnostics(...)`, and grouped cluster summaries through `summarize_clusters(...)`
- Next extension should add alternate recommendation heuristics and cluster-to-cluster comparison summaries

## Design Rules For New Tools

- Prefer a frame method when the output should stay row-oriented and composable.
- Prefer summaries when users need metrics, not derived geometry.
- Add `max_distance`, `k`, or aggregation controls early if they change workflow value materially.
- Keep geometry normalization and CRS handling inside existing frame and geometry layers.
- Add comparison coverage when a clear Shapely or GeoPandas reference exists.

## Recommended Next Implementation Order

1. Corridor comparison views across alternate anchor and score strategies
2. Pairwise zone comparison outputs
3. Ranked multi-group overlay summaries
4. Alternate cluster recommendation heuristics
5. Cluster-to-cluster comparison summaries