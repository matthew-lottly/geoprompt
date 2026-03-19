# Architecture

## Overview

This project models a lightweight forecasting workbench for station-level monitoring histories.

## Flow

1. Station histories are loaded from checked-in JSON fixtures.
2. Each series is split into training and holdout segments.
3. A feature profile is built from recent level, volatility, momentum, and slope.
4. Several candidate forecasts are generated and scored on the holdout window.
5. The best-performing model is selected and used for forward projection.