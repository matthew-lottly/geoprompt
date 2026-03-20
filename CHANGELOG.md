# Changelog

## 0.1.5

- Optimized the Shapely-backed overlay path by caching Shapely module loading inside `overlay.py`.
- Replaced the GeoJSON reshaping step in `geometry_to_shapely(...)` with direct Point, LineString, and Polygon construction.
- Improved `clip(...)` runtime enough to move ahead of the reference path in the current benchmark and stress corpora while keeping all comparison parity flags green.

## 0.1.4

- Added `GeoPromptFrame.buffer_join(...)` for service-area style joins driven by buffered geometries.
- Added `GeoPromptFrame.coverage_summary(...)` for per-zone counts, covered ids, and target aggregation rollups.
- Kept the recently added `buffer(...)`, `within_distance(...)`, `query_radius(...)`, and `proximity_join(...)` tools in the public API.
- Expanded tests and docs for the service-area workflow surface.

## 0.1.3

- Added `GeoPromptFrame.buffer(...)`.
- Added `GeoPromptFrame.within_distance(...)`.
- Improved `clip(...)` with prepared-geometry checks.
- Published the `0.1.3` release to PyPI.

## 0.1.2

- Added `query_radius(...)` and `proximity_join(...)`.
- Improved benchmark coverage with a generated stress corpus.

## 0.1.1

- Fixed PyPI trusted publishing configuration.

## 0.1.0

- Initial public Geoprompt package release.