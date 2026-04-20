# GeoPrompt Maturity Matrix

This matrix distinguishes hardened workflows from exploratory ones.

| Tier | Meaning | Examples |
| --- | --- | --- |
| Hardened | Verified regularly and recommended for production-style analyst workflows | frame, geometry, network, scenario reporting |
| Supported optional | Supported when extras are installed | viz, db, raster, service, compare |
| Experimental | Useful but still evolving | advanced domain, ML, and imagery helpers |
| Simulation-only | Placeholder or integration starter, not a production backend | notification stubs, SAML metadata stub, serverless endpoint stub |

## Guidance

Use the hardened and supported optional tiers for stakeholder-facing work. Treat simulation-only helpers as scaffolding or integration starters only.