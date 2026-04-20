# Notebook and Output Gallery

This page provides rendered output references and teaching-friendly examples. The figures below were regenerated from real package data and scenario outputs during the latest quality audit.

Executed notebooks in the repository are committed with cached outputs so GitHub can render the results directly in the browser.

## Rendered Outputs

### Portfolio scorecard

![Portfolio scorecard example](../assets/portfolio-scorecard-example.svg)

Business context: use this for portfolio triage, capital prioritization, and stakeholder-ready rankings.

### Executive before and after scenario view

![Before and after scenario example](../assets/before-after-scenario-example.svg)

Business context: use this when a decision-maker needs a clear baseline-versus-candidate story with visible tradeoffs.

### Restoration storyboard

![Restoration storyboard example](../assets/restoration-storyboard-example.svg)

Business context: use this for outage, repair, restoration, and resilience sequencing walkthroughs.

## Persona gallery

### Analyst track
- Start with the quickstart cookbook and the benchmark bundle.
- Focus on repeatable scenario comparison, summary tables, and map exports.
- Recommended outputs: HTML scorecards, CSV summaries, and choropleth maps.

### Planner track
- Start with the migration and connectors recipes.
- Focus on site screening, tradeoff comparison, and map-series generation.
- Recommended outputs: briefing packs, SVG figures, and narrative summaries.

### Operations track
- Start with the network scenario recipes and deployment guide.
- Focus on dispatch, restoration, and field coordination workflows.
- Recommended outputs: outage overlays, routing tables, and service-style reports.

## Teaching Flow

1. Start with the small quickstart example.
2. Run one persona script from the examples folder.
3. Export an HTML report or SVG map.
4. Compare outputs against the benchmark proof bundle.

## Notebook-Friendly Copy/Paste

```python
import geoprompt as gp

frame = gp.read_data("data/sample_features.json")
print(frame.head())
print(frame.summary())
print(frame.geom.area())
```
