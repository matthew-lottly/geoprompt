# Release Evidence and Trust Bar

GeoPrompt treats release quality as a trust milestone rather than a function-count milestone.

## Evidence required for a release

- green targeted completion regressions
- a green full repository test run
- benchmark bundle export and correctness comparisons
- API stability review against the documented public surface
- packaging smoke checks across the main extras profiles

## Recommended cadence

- run the correctness and benchmark bundle every release
- publish updated proof artifacts on a regular cadence
- review migration and quickstart docs before tagging a new version
- keep the benchmark and notebook outputs synchronized with the current stable API surface
