# Release Notes — 0.1.14

## Summary

This release extends the new network engine from pathfinding into operational service coverage and allocation. Service areas can now return partial reachable edge segments, routing tools expose optional diagnostics, and facilities can allocate demand by network cost with optional capacity limits.

## Partial Service Areas

`GeoPromptFrame.service_area(...)` now supports:

- `include_partial_edges=True` for clipped reachable edge segments
- `include_diagnostics=True` for runtime counters and network reach summaries

Added output fields include:

- `coverage_start_*`
- `coverage_end_*`
- `coverage_ratio_*`
- `partial_*`
- reachable-segment and partial-edge counters when diagnostics are enabled

This makes service-area results more usable for network coverage workflows where the reachable limit falls partway through an edge.

## Routing Diagnostics

`GeoPromptFrame.shortest_path(...)` now supports:

- `include_diagnostics=True`

This exposes routing counters such as visited-node count, relaxation count, and network-node count so path-search behavior is easier to inspect during benchmarking and workflow debugging.

## Location Allocation

Added:

- `GeoPromptFrame.location_allocate(...)`

This provides network-cost-based assignment from demand features to facility features with support for:

- optional `demand_weight_column`
- optional `facility_capacity_column`
- optional `max_cost`
- facility-centered aggregate rollups
- optional diagnostics counters

The initial implementation is deterministic, cache-aware, and built directly on the reusable network graph introduced in the previous release tranche.

## Benchmark Coverage

The comparison harness now benchmarks:

- `location_allocate(...)`

The comparison report has been regenerated to include this benchmark alongside the existing network analysis coverage.

## Validation

- 81 tests passing
- package version bumped to `0.1.14`