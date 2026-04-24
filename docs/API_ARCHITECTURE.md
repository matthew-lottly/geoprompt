# GeoPrompt API Architecture

This page defines the intended module boundaries and the long-term support shape of the package.

## Stable core

The stable core includes the frame, geometry, equations, IO, and flagship network workflows. These are the surfaces that should remain dependable for analyst and operations use.

## Supported optional surfaces

Visualization, service deployment, database bridges, raster helpers, and comparison tooling are supported when their optional dependencies are installed.

## Experimental and simulation-only surfaces

Some broader domain, ML, imagery, and integration helpers are still exploratory. They are useful for experimentation, but they are not the same maturity tier as the core.

## Module boundaries

- frame and table layers should own analyst-facing data manipulation
- geometry and CRS layers should own low-level spatial correctness
- network should own routing, dispatch, resilience, and utility scenarios
- reporting and viz should own stakeholder delivery outputs
- service and enterprise layers should own operational deployment patterns

## Request and response flow

1. input enters through file I/O, service payloads, or CLI commands
2. policy and capability checks validate environment, auth, and optional dependency state
3. processing modules execute analyst, network, raster, or reporting logic
4. output is emitted as a frame, table, json artifact, markdown summary, or html briefing

## Runtime boundary table

| Boundary | Owns | Typical failure mode |
| --- | --- | --- |
| input and connectors | local files, remote URLs, databases, service payloads | schema mismatch, auth failure, remote timeout |
| policy and trust layer | capability checks, safe expression rules, service hardening | blocked expression, dependency error, auth rejection |
| compute layer | frame, geometry, network, raster, compare | validation error, execution error, unsupported optional path |
| output layer | reports, figures, html, json, markdown | stale artifact, missing provenance, inaccessible visual |

## Extension points

- new connectors should publish their dependency and degraded-mode contract
- new reporting surfaces should emit reproducible file artifacts with generation context
- new service endpoints should reuse request-id, auth, payload, and rate-limit middleware rather than introducing parallel enforcement logic

## Design rule

Prefer one clear canonical path for common workflows, then mark broader exploratory helpers honestly rather than inflating the stable surface.
