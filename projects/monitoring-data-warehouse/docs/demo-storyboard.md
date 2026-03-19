# Demo Storyboard

## Intended Audience

- Data engineering and database reviewers
- Software teams that want evidence of modeling discipline

## Narrative Arc

1. Start with `assets/warehouse-build-summary-live.png` and explain the dimensional model using a real generated build artifact rather than placeholder art.
2. Walk through the builder: raw staging data, dimensions, fact table, and alert mart.
3. Emphasize that this repo is about structure, repeatability, and quality checks rather than dashboards.
4. Use the builder summary and `artifacts/charts/warehouse-build-summary.png` to show row counts and validation results.

## What Reviewers Should Notice

- Clear separation between raw data, warehouse tables, and marts.
- Thoughtful modeling rather than a single flat table.
- Repeatable SQL and data-quality validation as first-class concerns.

## Strong Screens Or GIF Shots To Capture Later

- The schema and transform SQL side by side
- Builder output with counts and quality checks
- A future ERD or lineage diagram