# Network Scenario Recipes

This page provides practical, fast-to-run examples for utility operations workflows.

## Top-tier network and utility analysis track

GeoPrompt’s flagship lane is utility, outage, resilience, and restoration decision support. The goal is not just to draw network lines, but to help teams answer operational questions such as:
- what fails first under disruption
- who loses service and how much demand is exposed
- which repair or switching action restores the most critical service fastest
- where dispatch, staging, and redundancy investments matter most

## Electric Feeder Outage and Restoration

Run:

```bash
python examples/network/electric_feeder_scenario.py
```

What it shows:
- Baseline energized footprint
- Outage impact after edge failure
- Restored footprint after repair

## Water Pressure and Fire Flow Screening

Run:

```bash
python examples/network/water_pressure_and_fire_flow.py
```

What it shows:
- Pressure-zone eligibility against headloss threshold
- Hydrant adequacy and deficit against fire-flow demand

## Stormwater Capacity and Overflow

Run:

```bash
python examples/network/stormwater_capacity_screen.py
```

What it shows:
- Accumulated runoff routing to downstream nodes
- Basin overflow signal and volume
- Inflow/infiltration ratio flags

## Telecom Redundancy and Cut Impact

Run:

```bash
python examples/network/telecom_redundancy_screen.py
```

What it shows:
- Ring node redundancy checks
- Circuit impact by candidate fiber cut edge

## Resilience Prioritization and Restoration Staging

Run:

```bash
python examples/network/resilience_restoration_screen.py
```

What it shows:
- Node-by-node supply redundancy tiers
- Multi-source service balancing and overload visibility
- Outage customer and demand impact scoring
- Stepwise repair staging with cumulative restored demand
- Stakeholder-ready HTML resilience summary output
- Portfolio ranking across baseline versus upgrade scenarios

## Reliability Dashboard Example

Run:

```bash
python examples/network/reliability_dashboard.py
```

What it shows:
- SAIDI, SAIFI, and ASAI comparison by scenario
- a lightweight dashboard image for stakeholder review
- a reproducible starter for resilience scorecards

## Why this beats a generic dataframe GIS workflow

A generic dataframe GIS workflow can store edges and nodes, but it usually leaves routing, outage isolation, restoration sequencing, and resilience ranking to custom one-off code. GeoPrompt already packages those operational steps into reusable, report-ready workflows.

## Calibration and validation notes

Use the included examples as deterministic baselines, then calibrate travel times, repair durations, demand, and hazard assumptions against your own utility or infrastructure records. That keeps the recipes reproducible while still supporting real operational-style scenario tuning.

## Notes

- These recipes are intentionally small and deterministic.
- They are suitable for onboarding, CI smoke checks, and reproducible demos.
- The fastest path to real-world adoption is to keep validating them against observed outage, dispatch, and recovery behavior.
