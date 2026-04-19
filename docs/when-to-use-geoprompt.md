# When to Use GeoPrompt

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
