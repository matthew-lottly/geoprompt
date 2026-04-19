# Notebook and Output Gallery

This page provides rendered output references and teaching-friendly examples. The figures below were regenerated from real package data and scenario outputs during the latest quality audit.

Executed notebooks in the repository are committed with cached outputs so GitHub can render the results directly in the browser.

## Rendered Outputs

### Portfolio scorecard

![Portfolio scorecard example](../assets/portfolio-scorecard-example.svg)

### Executive before and after scenario view

![Before and after scenario example](../assets/before-after-scenario-example.svg)

### Restoration storyboard

![Restoration storyboard example](../assets/restoration-storyboard-example.svg)

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
