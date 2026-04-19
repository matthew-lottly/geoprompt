# Start Here

This page is the fastest way to get productive with GeoPrompt.

## Pick Your Path

| Persona | Best first file | Outcome |
| --- | --- | --- |
| Analyst | examples/personas/analyst_resilience_screen.py | Compare scenarios and export a short decision summary |
| Executive | examples/personas/executive_briefing_pack.py | Produce briefing-ready metrics and visuals |
| Operations | examples/personas/operations_dispatch.py | Build field routing and restoration actions |
| Developer | examples/personas/database_roundtrip_demo.py | Move data through connectors and packaging flows |

## 15-Minute Learning Path

1. Read the quickstart cookbook.
2. Run one persona example.
3. Export a map, report, or benchmark bundle.
4. Move to the deeper recipes and deployment guide.

## Copy/Paste Starter

```python
import geoprompt as gp

frame = gp.read_data("data/sample_features.json")
summary = frame.summary()
report = gp.build_scenario_report(
    baseline_metrics={"served_load": 100.0, "deficit": 0.14},
    candidate_metrics={"served_load": 118.0, "deficit": 0.06},
    higher_is_better=["served_load"],
)

gp.export_scenario_report(report, "outputs/quickstart-report.html")
print(summary)
```

## Where To Go Next

- quickstart-cookbook.md for core usage
- network-scenario-recipes.md for resilience and utility workflows
- connectors-and-recipes.md for database, cloud, and raster bridges
- deployment-guide.md for service and container rollout
- governance-and-support.md for stability and release expectations
