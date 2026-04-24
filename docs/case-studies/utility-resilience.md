# Case Study: Utility Resilience Screening

## Problem

A utility operations team needs to compare outage scenarios, rank restoration priorities, and export an executive-ready summary without switching across several separate tools.

## GeoPrompt workflow

1. load the network and demand inputs
2. run resilience and restoration analysis
3. compare baseline and candidate scenarios
4. export HTML reports and briefing outputs

## Relevance-first visual

![Resilience risk heatmap with before and after mitigation impact bars](../../assets/resilience-risk-heatmap-mitigation.svg)

Data-linked render: generated from checked-in sample outage and restoration fixtures, with direct labels, legend cues, and axis units for quick operator interpretation.

## Scenario setup

| Element | Sample assumption |
| --- | --- |
| baseline state | normal service with deterministic demand and intact supply paths |
| disruption | one or more representative outage conditions applied to critical edges or nodes |
| mitigation candidate | repair or redundancy improvement compared against the baseline scenario |
| decision output | resilience summary plus portfolio-style ranking for follow-up |

## Risk assumptions

- demand, outage, and restoration steps are deterministic in the starter workflow so reviewers can reproduce the same ordering
- the checked-in fixtures are representative and should be recalibrated against real utility observations before operational use
- resilience tiers communicate relative screening priority, not a substitute for engineering sign-off

## Recovery metrics to review

1. impacted customers or demand after the disruption
2. cumulative restored demand after each repair stage
3. high-risk assets that remain single-source after mitigation
4. scenario ranking differences between baseline and candidate portfolios

## Outcome

This workflow emphasizes GeoPrompt's strongest lane: outage, resilience, routing, and stakeholder-ready reporting from the same package surface.
