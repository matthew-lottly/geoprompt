# Network Scenario Recipes

This page provides practical, fast-to-run examples for utility operations workflows.

## Top-tier network and utility analysis track

GeoPrompt’s flagship lane is utility, outage, resilience, and restoration decision support. The goal is not just to draw network lines, but to help teams answer operational questions such as:

- what fails first under disruption
- who loses service and how much demand is exposed
- which repair or switching action restores the most critical service fastest
- where dispatch, staging, and redundancy investments matter most

## Electric Feeder Outage and Restoration

![Restoration timeline and unmet demand from repository sample scenario data](../assets/network-restoration-unmet-demand.svg)

Data-linked render: generated from checked-in sample scenario fixtures and intended to show the restoration timeline against the unmet demand curve with direct labels, legend, and axis units.

Scenario setup:

| Input | Representative source |
| --- | --- |
| feeder graph | checked-in example network edges and nodes |
| demand attributes | repository sample demand fields and outage weights |
| failure trigger | single edge outage applied to the baseline graph |
| restoration plan | deterministic staged repair sequence |

Run:

```bash
python examples/network/electric_feeder_scenario.py
```

What it shows:

- Baseline energized footprint
- Outage impact after edge failure
- Restored footprint after repair
- restoration timeline and unmet demand trend for decision pacing

Validation checkpoints:

1. Baseline served demand should be higher than outage served demand.
2. Each restoration step should increase cumulative restored demand or keep it flat.
3. Final restored demand should not exceed the baseline capacity assumptions.
4. HTML or markdown outputs should name the scenario and artifact generation context.

## Water Pressure and Fire Flow Screening

Run:

```bash
python examples/network/water_pressure_and_fire_flow.py
```

What it shows:

- Pressure-zone eligibility against headloss threshold
- Hydrant adequacy and deficit against fire-flow demand

Uncertainty assumptions:

- pipe roughness, demand multipliers, and supply head are deterministic in the starter recipe
- operational deployments should calibrate threshold values against observed field measurements

## Stormwater Capacity and Overflow

Run:

```bash
python examples/network/stormwater_capacity_screen.py
```

What it shows:

- Accumulated runoff routing to downstream nodes
- Basin overflow signal and volume
- Inflow/infiltration ratio flags

Recommended evidence to keep:

- overflow thresholds used in the run
- basin IDs that exceed the deterministic sample threshold
- any downstream asset categories affected by overflow propagation

## Telecom Redundancy and Cut Impact

Run:

```bash
python examples/network/telecom_redundancy_screen.py
```

What it shows:

- Ring node redundancy checks
- Circuit impact by candidate fiber cut edge

Validation checkpoints:

1. Simulated cut edges should produce a different impacted-circuit count than the intact baseline.
2. Any node marked redundant should retain at least one alternate path in the sample graph.

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

Benchmark-style template:

| Question | Artifact |
| --- | --- |
| Which assets fail first? | resilience summary report |
| Which repair restores the most demand? | restoration staging output |
| Which candidate investment improves resilience most? | resilience portfolio report |

## Reliability Dashboard Example

Run:

```bash
python examples/network/reliability_dashboard.py
```

What it shows:

- SAIDI, SAIFI, and ASAI comparison by scenario
- a lightweight dashboard image for stakeholder review
- a reproducible starter for resilience scorecards

Validation checkpoints:

1. Reliability metrics should be scenario-labeled and comparable on the same scale.
2. Dashboard outputs should include enough labels for a non-technical reviewer to interpret them without notebook context.

## Why this beats a generic dataframe GIS workflow

A generic dataframe GIS workflow can store edges and nodes, but it usually leaves routing, outage isolation, restoration sequencing, and resilience ranking to custom one-off code. GeoPrompt already packages those operational steps into reusable, report-ready workflows.

## Calibration and validation notes

Use the included examples as deterministic baselines, then calibrate travel times, repair durations, demand, and hazard assumptions against your own utility or infrastructure records. That keeps the recipes reproducible while still supporting real operational-style scenario tuning.

When multiple recipes fit the same problem, prefer the one that produces both a machine-readable report and a stakeholder-facing HTML artifact. That gives you a reproducible baseline plus a reviewable output bundle.

## Notes

- These recipes are intentionally small and deterministic.
- They are suitable for onboarding, CI smoke checks, and reproducible demos.
- The fastest path to real-world adoption is to keep validating them against observed outage, dispatch, and recovery behavior.
