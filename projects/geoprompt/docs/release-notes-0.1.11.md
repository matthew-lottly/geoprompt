# Release Notes — 0.1.11

## Summary

This release closes the remaining gaps from the 0.1.9 feature pass. Corridor reach now works in haversine mode, zone fit scoring supports configurable component weights and optional directional alignment, centroid clustering now exposes quality metrics and deterministic id-based behavior, and the compare harness benchmarks the newer tools.

## Corridor Reach

`GeoPromptFrame.corridor_reach(...)` now supports both:

- `distance_method="euclidean"`
- `distance_method="haversine"`

The haversine path uses local tangent-plane segment distance so lon/lat corridor screening behaves consistently for route-style proximity checks.

## Zone Fit Scoring

`GeoPromptFrame.zone_fit_score(...)` now supports:

- `score_weights={...}` for configurable weighting of `containment`, `overlap`, `size`, `access`, and `alignment`
- `preferred_bearing=` for direction-aware scoring
- per-zone component outputs for access and alignment inside the returned zone score records

## Centroid Clustering

`GeoPromptFrame.centroid_cluster(...)` now adds:

- deterministic seed selection using `id_column`
- stable tie handling
- `cluster_size`
- `cluster_mean_distance`
- `cluster_sse`
- `cluster_silhouette`
- `cluster_sse_total`
- `cluster_silhouette_mean`

## Benchmark Coverage

The comparison harness now includes benchmark operations for:

- `centroid_cluster(...)`
- `zone_fit_score(...)`
- `corridor_reach(...)`

## Validation

- 60 tests passing
- Updated docs and roadmap
- Package version bumped to `0.1.11`
