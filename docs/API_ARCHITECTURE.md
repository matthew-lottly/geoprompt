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

## Design rule

Prefer one clear canonical path for common workflows, then mark broader exploratory helpers honestly rather than inflating the stable surface.