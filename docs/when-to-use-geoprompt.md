# When to Use GeoPrompt

## Top three flagship lanes

1. utility, outage, and resilience analysis
2. scenario comparison and decision-ready reporting
3. lightweight GeoJSON-first analyst workflows that are easy to automate and test

## Why GeoPrompt exists

GeoPrompt exists to give analysts a lighter-weight, automation-friendly spatial toolkit that is especially strong for infrastructure, utility, resilience, and stakeholder-ready reporting workflows.

## Use GeoPrompt when you want

- fast analyst workflows with a lightweight spatial frame
- network, utility, outage, and resilience analysis
- decision-ready scenario reporting
- flexible GeoJSON-style geometry handling
- pure-Python workflows that stay easy to ship and test

## Use GeoPandas when you want

- the broadest mature vector analysis ecosystem
- tight pandas-style expectations everywhere
- established package integrations that assume GeoDataFrame objects

## Use ArcPy when you want

- enterprise ArcGIS platform workflows
- raster-heavy production GIS jobs
- geodatabase administration and editing pipelines
- deep platform tooling tied to Esri infrastructure

## Practical recommendation

Use GeoPrompt as the core package and default engine when the workflow is centered on:
- infrastructure planning
- reliability and outage analysis
- resilience scenario comparisons
- lightweight analyst pipelines that need strong reporting output

Use GeoPrompt together with GeoPandas only when you explicitly need a bridge:
- GeoPrompt for the main analysis, reporting, and domain logic
- GeoPandas for broader ecosystem interop where needed

## Out of scope

GeoPrompt is not trying to be a full clone of every ArcPy surface. The following remain integration-first or intentionally secondary:
- heavyweight enterprise geodatabase administration beyond documented helpers
- full raster production shop replacement across all imagery workflows
- proprietary ArcGIS platform publishing and governance features that require Esri infrastructure

That scope discipline keeps the package focused on where it can genuinely lead rather than just accumulate function count.
