from __future__ import annotations

import math

from geoprompt import equations as eq


def test_distance_equation_parity_against_known_reference() -> None:
    # Reference great-circle distance for one degree latitude at equator.
    distance = eq.haversine_distance((0.0, 0.0), (0.0, 1.0))
    assert math.isclose(distance, 111.19508, rel_tol=1e-6, abs_tol=1e-6)


def test_decay_and_interaction_parity_against_analytic_formulas() -> None:
    d = 2.5
    scale = 1.7
    power = 2.2
    w1 = 3.0
    w2 = 4.0
    expected_decay = 1.0 / ((1.0 + d / scale) ** power)
    assert math.isclose(eq.prompt_decay(d, scale=scale, power=power), expected_decay, abs_tol=1e-12)
    assert math.isclose(eq.prompt_influence(w1, d, scale=scale, power=power), w1 * expected_decay, abs_tol=1e-12)
    assert math.isclose(eq.prompt_interaction(w1, w2, d, scale=scale, power=power), w1 * w2 * expected_decay, abs_tol=1e-12)


def test_weighted_accessibility_parity_against_manual_rollup() -> None:
    supply = [10.0, 5.0, 2.0]
    cost = [1.0, 2.0, 4.0]
    manual = sum(s * math.exp(-0.5 * c) for s, c in zip(supply, cost))
    observed = eq.weighted_accessibility_score(
        supply,
        cost,
        decay_method="exponential",
        rate=0.5,
    )
    assert math.isclose(observed, manual, abs_tol=1e-12)
