# GeoPrompt Maturity Matrix

This matrix distinguishes hardened workflows from exploratory ones.

Runtime tier labels map to this table as: stable -> Hardened, beta -> Supported optional, experimental -> Experimental, simulation_only -> Simulation-only.

| Tier | Meaning | Examples |
| --- | --- | --- |
| Hardened | Verified regularly and recommended for production-style analyst workflows | frame, geometry, network, scenario reporting |
| Supported optional | Supported when extras are installed | viz, db, raster, service, compare |
| Experimental | Useful but still evolving | advanced domain, ML, and imagery helpers |
| Simulation-only | Placeholder or integration starter, not a production backend | notification stubs, SAML metadata stub, serverless endpoint stub |

## Guidance

Use the hardened and supported optional tiers for stakeholder-facing work. Treat simulation-only helpers as scaffolding or integration starters only.

## Evidence Gates By Tier

| Tier | Required test and artifact evidence | Release expectation |
| --- | --- | --- |
| Hardened | API contract tests, docs markdown/flowchart gates, output quality gates, stale artifact checks, claims-to-tests manifest validation | Must pass on full CI matrix before release |
| Supported optional | Hardened evidence plus relevant optional profile checks (`io`, `viz`, `service`) | Must pass for advertised extras |
| Experimental | Syntax, unit smoke, and docs caveat review | Can ship with explicit caution in docs |
| Simulation-only | Schema/type checks and simulation-only docs validation | Not marketed as production capability |

## Release Promotion Rules

1. Promote a capability into Hardened only after two consecutive release cycles with passing contract tests and no breaking regressions.
2. Keep public docs aligned with real package exports by enforcing `tests/test_docs_api_contract_gate.py`.
3. Keep doc figures and chart assets in sync using `tests/test_figure_manifest_gate.py` and `docs/figures-manifest.json`.
4. Keep output artifacts current with `geoprompt docs-artifacts --check` and `tests/test_artifact_freshness_gate.py`.
5. Keep claims auditable with `claims-to-tests.json` and `tests/test_claims_to_tests_manifest.py`.

## Execution Order For Releases

1. Run core docs and API gates.
2. Rebuild and validate outputs/provenance artifacts.
3. Run full test suite and optional profile tests.
4. Regenerate benchmark history and dashboard bundles.
5. Package, sign off, and publish release artifacts.
