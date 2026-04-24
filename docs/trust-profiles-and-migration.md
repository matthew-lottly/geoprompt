# Trust Profiles, Capability Signals, and Migration

This guide defines production-safe usage, degraded-mode behavior, and migration from permissive fallback behavior to strict trust mode.

## Capability Flags by Surface

| Surface | Tier | Capability Flag | Missing Dependency Behavior | Production Guidance |
| --- | --- | --- | --- | --- |
| `frame`, `geometry`, `network`, `query`, scenario reporting | Stable | Core install (`pip install geoprompt`) | No optional dependency required for core JSON/CSV workflows | Primary production-safe path |
| `viz` module (Folium/Plotly/Matplotlib) | Beta optional | `folium`, `plotly`, `matplotlib` | Capability checks gate optional rendering paths | Use in production when extras are pinned |
| `db` connectors | Beta optional | `sqlalchemy`, optional database driver | Raises `DependencyError` when connector stack is missing | Use in production with connector smoke checks |
| `raster` helpers | Beta optional | `rasterio` | Raises `DependencyError` for unavailable raster backend | Use in production only with raster extras installed |
| `service` API | Beta optional | `fastapi`, `uvicorn` | Hard fail with dependency guidance | Use in production with auth/signature/PII gates enabled |
| `compare` workflows | Beta optional | `geopandas`, `shapely` | Explicit dependency failure instead of silent fallback | Use for release validation evidence |
| `ai`, `ml`, selected `standards` and `performance` helpers | Experimental / simulation-only | Module-specific | Explicit simulation/degraded notes in docstrings | Not a production backend without replacement implementation |

## Production-Safe vs Development Profiles

### Production-Safe Profile

- Install only required extras for the deployment: `geoprompt[io]`, `geoprompt[service]`, etc.
- Run `geoprompt capability-report` on startup and persist output in release evidence.
- Set strict fallback policy (`FallbackPolicy.ERROR`) for critical pipelines.
- Treat simulation-only helpers as non-production placeholders.
- Require passing trust tests and release evidence gates before promotion.

### Development Profile

- Use permissive fallback behavior for local exploration.
- Keep warnings visible and review all fallback warnings before merge.
- Allow optional dependency gaps in local workflows only when the behavior is documented.

## CLI Capability Report

GeoPrompt exposes the runtime optional dependency status with:

```bash
geoprompt capability-report
```

Example fields:

- `schema_version`
- `package_version`
- `checked_at_utc`
- `fallback_policy`
- `enabled` capabilities
- `disabled` capabilities
- `degraded` capabilities

Use this output in deployment-readiness and release evidence bundles.

## Trust Notes for Degraded Functions

When a function may degrade under missing optional dependencies, docs and user-facing help must include:

1. Which capability is required.
2. Whether behavior is hard-fail, soft-fail, or degraded.
3. Exact install guidance (`pip install geoprompt[...]`).
4. Expected output differences in degraded mode.
5. Whether strict mode should block execution.

## Troubleshooting Recipes

### Capability mismatch

- Symptom: `DependencyError` for optional dependency.
- Action:
  - Run `geoprompt capability-report`.
  - Install missing extras for that capability.
  - Re-run targeted smoke test.

### Fallback-policy failure

- Symptom: strict mode rejects degraded execution.
- Action:
  - Identify required capability.
  - Install dependency or re-route to core-safe workflow.
  - Keep strict mode enabled for production-critical paths.

## Migration Playbook: Permissive to Strict Trust Mode

1. Inventory all optional-capability usage via logs and feature flags.
2. Run with warnings enabled and capture all degraded paths.
3. Install missing dependencies for critical paths.
4. Enable strict mode in staging (`FallbackPolicy.ERROR`).
5. Fix all strict-mode failures before production rollout.
6. Promote to production with trust evidence attached.

## Simulation-Only Policy

Simulation-only helpers are explicitly non-production placeholders.

Required docs language for each simulation-only helper:

- "simulation-only" or "placeholder"
- clear non-production warning
- concrete remediation guidance (backend/library/service to install)

## README and Publishing Rendering Policy

- Keep Mermaid flow diagrams in README for GitHub readability.
- Ensure PUBLISHING guidance explicitly notes that PyPI may not render Mermaid.
- Provide text fallback guidance in publishing instructions when needed.
