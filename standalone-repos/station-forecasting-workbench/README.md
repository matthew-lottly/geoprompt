# Station Forecasting Workbench

Data science portfolio project for model comparison, holdout evaluation, feature profiling, and station-level projection artifacts.

![Station forecasting workbench preview](assets/forecast-preview.svg)

## Snapshot

- Lane: Data science and forecasting
- Domain: Short-horizon monitoring projections
- Stack: Python, JSON fixtures, lightweight forecasting workbench
- Includes: station histories, feature profiles, model leaderboard, holdout evaluation, projections, tests

## Overview

This project frames data science as a forecasting workflow rather than just descriptive analytics. It loads small station histories, builds simple feature profiles, compares several candidate models against a holdout window, selects the best-performing station-level method, and exports a concise forecast review package.

## What It Demonstrates

- Candidate-model comparison across naive, trailing-average, drift, and linear-regression forecasts
- Holdout evaluation with a reproducible leaderboard per station
- Lightweight feature profiling for recent level, volatility, momentum, and slope
- A reviewable output artifact that looks more like an experiment workbench than a single hard-coded baseline

## Current Output

The default command writes `outputs/station_forecast_report.json` with:

- station feature profiles
- model leaderboard entries and holdout MAE by station
- selected forecast model per series
- future projections from the winning model

See [docs/architecture.md](docs/architecture.md) for the design notes.
See [docs/demo-storyboard.md](docs/demo-storyboard.md) for the reviewer walkthrough.