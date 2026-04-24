# Best Workflow by Problem Type

Use this page to choose the fastest GeoPrompt path for the problem in front of you.

## Persona First Routing

| Persona | Primary goal | Start here |
| --- | --- | --- |
| Analyst | Screen options quickly and defend assumptions | quickstart cookbook -> scenario report exports |
| Operations lead | Restore service and prioritize repair sequencing | network scenario recipes -> resilience summary + portfolio |
| Executive sponsor | Compare investments with confidence narrative | executive briefing + comparison summary bundles |
| Platform engineer | Automate repeatable pipelines and governance checks | deployment guide + docs-artifacts + CI gates |

## Problem selection table

| Problem type | Best starting point | Recommended outputs |
| --- | --- | --- |
| Analyst screening | quickstart cookbook + scenario comparison | HTML summary, CSV table, briefing note |
| Utility resilience | network scenario recipes | outage overlay, restoration timeline, resilience report |
| Planning and siting | migration cookbooks + map series | map book, scorecard, candidate comparison |
| Service deployment | deployment guide + CLI doctor | API health checks, smoke matrix, operations runbook |
| Developer extension work | extending-geoprompt guide | plugin module, connector starter, compatibility checklist |

## Workflow Execution Order

1. Define question and metrics: pick the decision metric before loading data.
2. Validate environment: run `geoprompt capability-report` and install required extras.
3. Build baseline and candidates: keep one baseline and at least one alternative.
4. Export decision bundle: generate JSON plus Markdown and HTML for human review.
5. Run governance checks: execute docs, API contract, and stale artifact gates.
6. Publish evidence set: include claims-to-tests mapping and benchmark dashboard outputs.

## Output Bundle Checklist

| Bundle | Why it matters | Typical files |
| --- | --- | --- |
| Scenario contract | Machine-readable reproducibility | `scenario-report.json`, `scenario-report.csv` |
| Decision narrative | Human-readable comparison | `scenario-report.md`, `geoprompt_comparison_summary.md` |
| Executive visuals | Stakeholder communication | `scenario-report.html`, `resilience-summary.html` |
| Governance evidence | Release sign-off and auditability | `claims-to-tests.json`, `outputs/provenance_manifest.json` |

## Fast decision rules

1. If your work is map and report heavy, start with the visualization and briefing helpers.
2. If your work is routing, restoration, or facility response, start with the network recipes.
3. If your work depends on external formats or databases, start with the connectors guide.
4. If your work needs customization, start with the extension templates and plugin registry.
