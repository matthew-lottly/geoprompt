# Release Notes - 0.1.15

## Summary

This release opens the topology lane in earnest. Geometries can now be snapped to a shared tolerance, cleaned for duplicate vertices and short segments, split into stable line parts, and unioned into polygon faces with explicit lineage on both sides.

## Topology Snapping

Added:

- `GeoPromptFrame.snap_geometries(...)`

This method applies deterministic tolerance-based vertex snapping across mixed geometry inputs. Nearby vertices collapse onto a stable anchor point, and optional diagnostics report changed vertex counts, collapse state, and shared vertex totals.

## Topology Cleanup

Added:

- `GeoPromptFrame.clean_topology(...)`

This method builds on the snapping lane by removing duplicate consecutive vertices and pruning line or ring segments shorter than a caller-supplied threshold. Optional diagnostics expose input and output vertex counts, removed short-segment counts, and collapse behavior.

## Line Splitting

Added:

- `GeoPromptFrame.line_split(...)`

Lines can now be segmented either from explicit point or line splitters or by detecting line intersections directly. Output records include stable part identifiers, start and end fractions, splitter lineage, and optional diagnostics for self-intersections and applied splitter counts.

## Overlay Union

Added:

- `GeoPromptFrame.overlay_union(...)`

The first union pass focuses on explainable polygon overlay behavior. The method partitions combined polygon boundaries into faces, preserves which left and right source features contributed to each face, and returns explicit source-side classification plus per-face area.

## Benchmark Coverage

The comparison harness now benchmarks:

- `snap_geometries(...)`
- `clean_topology(...)`
- `line_split(...)`
- `overlay_union(...)`

The comparison report has been regenerated to include the new topology and overlay-union coverage.

## Validation

- 86 tests passing
- package version bumped to `0.1.15`