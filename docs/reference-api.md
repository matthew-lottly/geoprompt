# Reference API Guide

GeoPrompt exports a typed public API and includes a package typing marker for editor support.

## Stable Core Modules

- frame
- geometry
- io
- equations
- table
- tools
- compare
- interop

## Advanced and Growing Modules

- raster
- stats
- temporal
- topology
- workspace
- service
- enterprise
- geocoding
- cartography
- ecosystem
- ai
- data_management

## Public API Themes

### Spatial Table Workflows

- GeoPromptFrame
- GroupedGeoPromptFrame
- PromptTable

### Geometry and Overlay

- geometry metrics and predicates
- repair and validation helpers
- advanced editing and overlay tools

### Reporting and Decision Support

- scenario comparison outputs
- resilience portfolio reporting
- benchmark proof bundles
- map and dashboard exports

## Stability Notes

Use the API stability guide to distinguish mature surfaces from newer expansion modules. Deprecated names should keep a migration path before removal.

## Simulation and Deprecation Labels

Per-symbol simulation and deprecation labels are published in `docs/reference-simulation-labels.md`.
Generate or refresh this artifact with `geoprompt.quality.export_simulation_symbol_labels(...)`.

## Capability Contract

`geoprompt.capability_report()` is a stable contract for startup capability signaling.

- Returns schema fields: `schema_version`, `enabled`, `disabled`, `degraded`, `disabled_reasons`, `degraded_reasons`, `fallback_policy`, `package_version`, `optional_dependency_versions`, `checked_at_utc`.
- Use `geoprompt capability-report --format text` for human-readable output.
- Use `geoprompt capability-report --format json` for machine-readable output.
- Capability report schema changes should be treated as public contract changes and tested with `tests/test_capability_report_contract.py`.
