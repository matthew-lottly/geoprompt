# Publishing Guide

## Recommended Standalone Repository Name

- monitoring-data-warehouse

## Recommended Description

- Warehouse-style modeling and validation project for monitoring observations, dimensions, facts, and alert marts.

## Suggested Topics

- data-warehouse
- sql
- duckdb
- dimensional-modeling
- data-engineering
- environmental-monitoring

## Split Steps

1. Create a new empty repository named `monitoring-data-warehouse`.
2. Copy this project folder into the new repository root.
3. Preserve `sql/`, `data/`, `src/`, `tests/`, and `pyproject.toml`.
4. Add a real warehouse artifact screenshot, ERD, or query-output image near the top of the README. Do not use placeholder illustration art.
5. Reference [docs/model-notes.md](docs/model-notes.md) for the modeling rationale.

For a local copy operation, use [../../docs/publishing/extract-monitoring-data-warehouse.ps1](../../docs/publishing/extract-monitoring-data-warehouse.ps1).

## First Public Polish Pass

- Add an ERD or dbt-style DAG diagram
- Add a section explaining migration from DuckDB to PostgreSQL or Snowflake
- Add row-count and quality-check output examples to the README