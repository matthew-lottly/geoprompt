# Release Notes — 0.1.13

## Summary

This release completes the next roadmap pass after 0.1.12. Corridor reach now supports anchor-aware network distance, zone fit scoring can accept custom callbacks, clustering has cluster-level rollups, overlay summaries have a direct comparison helper, and corridors now have a diagnostics view for served-feature analysis.

## Corridor Reach And Diagnostics

`GeoPromptFrame.corridor_reach(...)` now supports:

- `path_anchor="start" | "end" | "nearest"`
- anchor-distance-aware network scoring through `anchor_distance`
- `path_anchor_*` output fields in reach results

Added:

- `GeoPromptFrame.corridor_diagnostics(...)`

This provides per-corridor served-feature counts, best-match counts, score summaries, distance summaries, and anchor-distance summaries.

## Zone Fit Scoring

`GeoPromptFrame.zone_fit_score(...)` now supports:

- `score_callback=` for workflow-specific score adjustment after the built-in weighted component scoring step

## Cluster Summaries

Added:

- `GeoPromptFrame.summarize_clusters(...)`

This produces per-cluster member counts, member ids, dominant group summaries, cluster-center rollups, and optional aggregate statistics.

## Overlay Group Comparison

Added:

- `GeoPromptFrame.overlay_group_comparison(...)`

This exposes grouped overlap winners, runner-up groups, and gap metrics so grouped overlays can be compared directly without post-processing.

## Benchmark Coverage

The comparison harness now benchmarks:

- `summarize_clusters(...)`
- `overlay_group_comparison(...)`
- `corridor_diagnostics(...)`

## Validation

- 69 tests passing
- package version bumped to `0.1.13`