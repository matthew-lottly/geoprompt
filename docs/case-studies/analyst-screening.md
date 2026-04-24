# Case Study: Analyst Screening and Triage

## Problem

An analyst must review many assets quickly, rank concern areas, and export a credible package of maps and tables for follow-up.

## GeoPrompt workflow

1. load the sample or operational dataset
2. profile and rank the assets
3. generate a map, summary table, and decision note

## Representative outputs

| Output type | Why it matters |
| --- | --- |
| screening table | ranks the asset set into a review order |
| map or html view | gives the analyst a spatial sanity check before escalation |
| markdown or decision note | preserves the assumptions used during triage |

## Before and after lens

- before: many assets, weak prioritization, limited traceability
- after: ranked concern areas, explicit assumptions, and a portable artifact bundle for follow-up review

## Data provenance and runtime guidance

- prefer checked-in sample data or an explicitly versioned operational extract
- record the generation command if the output will be used outside the notebook or script session
- keep screening criteria deterministic when using the workflow as a reproducible triage baseline

## Outcome

The workflow stays lightweight and reproducible while still producing client-ready screening outputs.
