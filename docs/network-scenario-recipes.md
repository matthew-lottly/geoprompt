# Network Scenario Recipes

This page provides practical, fast-to-run examples for utility operations workflows.

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

## Notes

- These recipes are intentionally small and deterministic.
- They are suitable for onboarding, CI smoke checks, and reproducible demos.
