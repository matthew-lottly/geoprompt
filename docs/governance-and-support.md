# Governance and Support

## Deprecation Policy

GeoPrompt follows a simple migration-first policy:
- prefer adding new names before removing old ones
- emit clear deprecation warnings for renamed public helpers
- keep migration notes in docs and release notes
- avoid silent breaking changes in minor releases

## Stable vs Experimental

Core frame, geometry, and reporting surfaces are the stability baseline. Newer expansion areas such as AI, enterprise, and some raster workflows may iterate faster while keeping clear docs.

## Release Cadence

Recommended cadence:
- small parity and bugfix releases as needed
- monthly or milestone-based feature drops
- benchmark proof and smoke runs before each public release

## Support Expectations

- best support for the core spatial frame and reporting workflows
- good-faith support for documented optional extras
- issue-driven follow-up for experimental modules

## Long-Term Package Direction

GeoPrompt should stay clearly differentiated in:
- resilience screening
- scenario comparison
- utility and infrastructure decision support
- report-ready outputs for nontechnical stakeholders
