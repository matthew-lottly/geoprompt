from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

import geoprompt as gp
from geoprompt import equations as eq
from geoprompt import tools


def test_haversine_reference_distance_one_degree_latitude() -> None:
    distance_km = eq.haversine_distance((0.0, 0.0), (0.0, 1.0))
    assert math.isclose(distance_km, 111.19508, rel_tol=1e-6, abs_tol=1e-6)


def test_decay_and_interaction_formulas_match_manual_reference() -> None:
    distance = 2.75
    scale = 1.6
    power = 1.8
    origin = 2.4
    destination = 3.1
    manual_decay = 1.0 / ((1.0 + distance / scale) ** power)

    assert math.isclose(eq.prompt_decay(distance, scale=scale, power=power), manual_decay, abs_tol=1e-12)
    assert math.isclose(eq.prompt_influence(origin, distance, scale=scale, power=power), origin * manual_decay, abs_tol=1e-12)
    assert math.isclose(
        eq.prompt_interaction(origin, destination, distance, scale=scale, power=power),
        origin * destination * manual_decay,
        abs_tol=1e-12,
    )


def test_weighted_accessibility_score_matches_manual_for_each_decay_mode() -> None:
    supply = [10.0, 7.5, 3.0]
    costs = [1.0, 2.5, 4.0]

    power_expected = sum(
        s * (1.0 / ((1.0 + c / 1.8) ** 2.1))
        for s, c in zip(supply, costs)
    )
    assert math.isclose(
        eq.weighted_accessibility_score(
            supply,
            costs,
            decay_method="power",
            scale=1.8,
            power=2.1,
        ),
        power_expected,
        abs_tol=1e-12,
    )

    exp_expected = sum(s * math.exp(-0.35 * c) for s, c in zip(supply, costs))
    assert math.isclose(
        eq.weighted_accessibility_score(
            supply,
            costs,
            decay_method="exponential",
            rate=0.35,
        ),
        exp_expected,
        abs_tol=1e-12,
    )

    gauss_expected = sum(s * math.exp(-(c**2) / (2.0 * 2.8**2)) for s, c in zip(supply, costs))
    assert math.isclose(
        eq.weighted_accessibility_score(
            supply,
            costs,
            decay_method="gaussian",
            sigma=2.8,
        ),
        gauss_expected,
        abs_tol=1e-12,
    )


def test_scenario_report_math_and_export_round_trip_are_consistent(tmp_path: Path) -> None:
    baseline = {"reliability": 0.80, "cost": 200.0, "travel_time": 15.0}
    candidate = {"reliability": 0.86, "cost": 188.0, "travel_time": 13.5}

    comparison = tools.compare_scenarios(baseline, candidate, higher_is_better=["reliability"])
    report = tools.build_scenario_report(
        baseline,
        candidate,
        baseline_name="baseline",
        candidate_name="candidate",
        higher_is_better=["reliability"],
    )

    for metric, values in comparison.items():
        report_values = report["metrics"][metric]
        assert math.isclose(float(report_values["delta"]), float(values["delta"]), abs_tol=1e-12)
        assert math.isclose(float(report_values["delta_percent"]), float(values["delta_percent"]), abs_tol=1e-12)
        assert report_values["direction"] == values["direction"]

    json_path = Path(tools.export_scenario_report(report, tmp_path / "scenario-report.json"))
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded == report

    csv_path = Path(tools.export_scenario_report(report, tmp_path / "scenario-report.csv"))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    keyed = {row["metric"]: row for row in rows}
    assert keyed["reliability"]["direction"] == "improved"
    assert keyed["cost"]["direction"] == "improved"
    assert keyed["travel_time"]["direction"] == "improved"


def test_geopandas_parity_for_bounds_and_area_when_available() -> None:
    pytest.importorskip("geopandas")

    frame = gp.read_data("data/sample_features.json", crs="EPSG:4326")
    gdf = gp.to_geopandas(frame)

    bounds = frame.bounds()
    gdf_bounds = [float(value) for value in gdf.total_bounds]
    assert math.isclose(bounds.min_x, gdf_bounds[0], abs_tol=1e-9)
    assert math.isclose(bounds.min_y, gdf_bounds[1], abs_tol=1e-9)
    assert math.isclose(bounds.max_x, gdf_bounds[2], abs_tol=1e-9)
    assert math.isclose(bounds.max_y, gdf_bounds[3], abs_tol=1e-9)

    frame_areas = [float(value) for value in frame.geometry_areas()]
    gdf_areas = [float(value) for value in gdf.geometry.area]
    assert len(frame_areas) == len(gdf_areas)
    for lhs, rhs in zip(frame_areas, gdf_areas):
        assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-12)
