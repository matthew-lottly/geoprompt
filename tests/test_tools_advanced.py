from __future__ import annotations

import csv
import json

import pytest

from geoprompt import GeoPromptFrame
from geoprompt.compare import benchmark_summary_table, export_comparison_bundle
from geoprompt.table import PromptTable
from geoprompt.equations import prompt_decay
from geoprompt.interop import geopandas_available
from geoprompt.tools import (
    batch_accessibility_table,
    batch_accessibility_scores,
    build_multi_scenario_report,
    bootstrap_confidence_interval,
    build_scenario_report,
    export_multi_scenario_report,
    export_scenario_report,
    gravity_interaction_table,
    multi_scenario_report_table,
    optimize_decay_parameters,
    rank_scenarios,
    build_resilience_summary_report,
    build_resilience_portfolio_report,
    export_resilience_portfolio_report,
    export_resilience_summary_report,
    scenario_report_table,
    service_probability_table,
    validate_numeric_series,
    vectorized_decay,
    vectorized_gravity_interaction,
    vectorized_service_probability,
)


def test_optimize_decay_parameters_refines_grid_search() -> None:
    observed_pairs = [(distance, prompt_decay(distance, scale=1.3, power=1.8)) for distance in [0.2, 0.5, 1.0, 2.0, 3.0]]
    result = optimize_decay_parameters(observed_pairs, method="power", refinement_steps=8)
    assert result["rmse"] < 1e-4
    assert result["scale"] == pytest.approx(1.3, abs=0.15)
    assert result["power"] == pytest.approx(1.8, abs=0.2)


def test_bootstrap_confidence_interval_mean() -> None:
    summary = bootstrap_confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0], iterations=200, seed=3)
    assert summary["lower"] <= summary["observed"] <= summary["upper"]
    assert summary["confidence_level"] == pytest.approx(0.95)


def test_build_scenario_report_structure() -> None:
    report = build_scenario_report(
        {"deficit": 0.2, "served": 100.0},
        {"deficit": 0.1, "served": 110.0},
        higher_is_better=["served"],
        uncertainty={"served": {"lower": 105.0, "upper": 115.0}},
        metadata={"scenario_id": "demo"},
    )
    assert report["summary"]["metric_count"] == 2
    assert "deficit" in report["summary"]["improved_metrics"]
    assert report["metadata"]["scenario_id"] == "demo"


def test_scenario_report_table_returns_tabular_rows() -> None:
    report = build_scenario_report(
        {"deficit": 0.2, "served": 100.0},
        {"deficit": 0.1, "served": 110.0},
        higher_is_better=["served"],
    )
    table = scenario_report_table(report)
    assert len(table) == 2
    assert "metric" in table.columns
    assert table.sort_values("metric").head(1)[0]["metric"] == "deficit"


def test_export_scenario_report_json_csv_markdown_html(tmp_path) -> None:
    report = build_scenario_report(
        {"deficit": 0.2, "served": 100.0},
        {"deficit": 0.1, "served": 110.0},
        higher_is_better=["served"],
        metadata={"scenario_id": "demo"},
    )

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    markdown_path = tmp_path / "report.md"
    html_path = tmp_path / "report.html"

    export_scenario_report(report, json_path)
    export_scenario_report(report, csv_path)
    export_scenario_report(report, markdown_path)
    export_scenario_report(report, html_path)

    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["metadata"]["scenario_id"] == "demo"

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["metric"] in {"deficit", "served"}
    assert "| Metric | Baseline | Candidate |" in markdown_path.read_text(encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")
    assert "Scenario Report" in html_text
    assert "<svg" in html_text


def test_multi_scenario_report_export_formats(tmp_path) -> None:
    report = build_multi_scenario_report(
        {
            "baseline": {"deficit": 0.2, "served": 100.0},
            "candidate_a": {"deficit": 0.12, "served": 108.0},
            "candidate_b": {"deficit": 0.09, "served": 112.0},
        },
        baseline_name="baseline",
        higher_is_better=["served"],
    )

    json_path = tmp_path / "multi-report.json"
    csv_path = tmp_path / "multi-report.csv"
    markdown_path = tmp_path / "multi-report.md"
    html_path = tmp_path / "multi-report.html"
    chart_path = tmp_path / "multi-report-chart.html"

    export_multi_scenario_report(report, json_path)
    export_multi_scenario_report(report, csv_path)
    export_multi_scenario_report(report, markdown_path)
    export_multi_scenario_report(report, html_path, chart_output_path=chart_path)

    assert "candidate_a" in json_path.read_text(encoding="utf-8")
    assert "scenario" in csv_path.read_text(encoding="utf-8")
    assert "Multi-Scenario Report" in markdown_path.read_text(encoding="utf-8")
    assert "Multi-Scenario Report" in html_path.read_text(encoding="utf-8")
    assert "svg" in chart_path.read_text(encoding="utf-8")


def test_multi_scenario_table_and_ranking() -> None:
    report = build_multi_scenario_report(
        {
            "baseline": {"deficit": 0.2, "served": 100.0, "cost": 10.0},
            "candidate_a": {"deficit": 0.12, "served": 108.0, "cost": 11.0},
            "candidate_b": {"deficit": 0.09, "served": 112.0, "cost": 9.5},
        },
        baseline_name="baseline",
        higher_is_better=["served"],
    )

    table = multi_scenario_report_table(report)
    ranking = rank_scenarios(report, metric_weights={"served": 2.0, "deficit": 1.5, "cost": 1.0})

    assert len(table) == 6
    assert set(table.columns) >= {"scenario", "metric", "delta_percent", "direction"}
    assert ranking.head(1)[0]["scenario"] in {"candidate_a", "candidate_b"}
    assert "weighted_score" in ranking.columns


def test_vectorized_decay_matches_scalar() -> None:
    distances = [0.0, 0.5, 1.0, 2.0]
    vectorized = vectorized_decay(distances, method="exponential", rate=0.8)
    scalar = [__import__("geoprompt.equations", fromlist=["exponential_decay"]).exponential_decay(d, rate=0.8) for d in distances]
    assert vectorized == pytest.approx(scalar)


def test_vectorized_gravity_interaction_matches_scalar() -> None:
    origins = [10.0, 15.0, 20.0]
    destinations = [5.0, 6.0, 7.0]
    costs = [1.2, 2.0, 3.5]
    vectorized = vectorized_gravity_interaction(origins, destinations, costs, gamma=1.4)

    from geoprompt.equations import gravity_interaction

    scalar = [
        gravity_interaction(origin, destination, cost, gamma=1.4)
        for origin, destination, cost in zip(origins, destinations, costs)
    ]
    assert vectorized == pytest.approx(scalar)


def test_vectorized_service_probability_matches_scalar() -> None:
    rows = [
        {"pressure": 0.8, "redundancy": 1.0},
        {"pressure": 0.5, "redundancy": 0.3},
    ]
    coefficients = {"pressure": 1.4, "redundancy": 0.7}

    vectorized = vectorized_service_probability(rows, coefficients, intercept=-0.5)

    from geoprompt.equations import logistic_service_probability

    scalar = [logistic_service_probability(row, coefficients, intercept=-0.5) for row in rows]
    assert vectorized == pytest.approx(scalar)


def test_batch_accessibility_scores_matches_scalar() -> None:
    supply_rows = [[100.0, 40.0, 10.0], [80.0, 20.0, 5.0]]
    cost_rows = [[0.5, 1.0, 2.0], [0.25, 0.75, 1.5]]
    batch = batch_accessibility_scores(supply_rows, cost_rows, decay_method="exponential", rate=0.7)

    from geoprompt.equations import weighted_accessibility_score

    scalar = [
        weighted_accessibility_score(supplies, costs, decay_method="exponential", rate=0.7)
        for supplies, costs in zip(supply_rows, cost_rows)
    ]
    assert batch == pytest.approx(scalar)


def test_prompt_table_supports_indexing_and_json_export(tmp_path) -> None:
    table = PromptTable.from_records([
        {"scenario": "a", "score": 10.0},
        {"scenario": "b", "score": 20.0},
    ])

    assert table[0]["scenario"] == "a"
    assert table[:1][0]["score"] == 10.0

    output_path = tmp_path / "scores.json"
    written = table.to_json(output_path)
    assert written.endswith("scores.json")
    assert '"scenario": "a"' in output_path.read_text(encoding="utf-8")

    html_path = tmp_path / "scores.html"
    html_written = table.to_html(html_path)
    assert html_written.endswith("scores.html")
    assert "<table>" in html_path.read_text(encoding="utf-8")


def test_batch_tables_return_prompt_tables() -> None:
    accessibility_table = batch_accessibility_table(
        [[100.0, 40.0], [80.0, 20.0]],
        [[0.5, 1.0], [0.25, 0.75]],
        decay_method="exponential",
        rate=0.7,
        row_ids=["a", "b"],
    )
    gravity_table = gravity_interaction_table(
        [10.0, 15.0],
        [5.0, 6.0],
        [1.2, 2.0],
        row_ids=["a", "b"],
    )
    service_table = service_probability_table(
        [{"pressure": 0.8}, {"pressure": 0.5}],
        {"pressure": 1.4},
        intercept=-0.5,
        row_ids=["a", "b"],
    )

    assert accessibility_table.head(1)[0]["row_id"] == "a"
    assert "gravity_interaction" in gravity_table.columns
    assert "service_probability" in service_table.columns


def test_prompt_table_join_where_and_pivot() -> None:
    left = batch_accessibility_table(
        [[100.0], [80.0], [60.0]],
        [[0.5], [0.6], [0.7]],
        row_ids=["north", "north", "south"],
    )
    right = gravity_interaction_table(
        [10.0, 15.0],
        [5.0, 6.0],
        [1.2, 2.0],
        row_ids=["north", "south"],
    )

    joined = left.join(right, on="row_id", how="left")
    filtered = joined.where(row_id="north")
    pivoted = joined.pivot(index="row_id", columns="decay_method", values="accessibility_score", agg="mean")

    assert len(filtered) == 2
    assert "gravity_interaction" in joined.columns
    assert any(column == "power" for column in pivoted.columns)


def test_prompt_table_summarize_groups_rows() -> None:
    table = batch_accessibility_table(
        [[100.0], [80.0], [60.0]],
        [[0.5], [0.6], [0.7]],
        row_ids=["north", "north", "south"],
    )
    summary = table.summarize("row_id", {"accessibility_score": "mean"}).sort_values("row_id")
    assert summary.head(1)[0]["row_id"] == "north"
    assert summary.head(1)[0]["row_count"] == 2


def test_prompt_table_groupby_agg_supports_multiple_metrics() -> None:
    table = PromptTable.from_records(
        [
            {"region": "north", "score": 10.0, "cost": 2.0},
            {"region": "north", "score": 20.0, "cost": 4.0},
            {"region": "south", "score": 15.0, "cost": 3.0},
        ]
    )

    grouped = table.groupby("region").agg({"score": ["mean", "max"], "cost": "sum"}).sort_values("region")

    assert grouped.head(1)[0]["region"] == "north"
    assert grouped.head(1)[0]["score_mean"] == pytest.approx(15.0)
    assert grouped.head(1)[0]["score_max"] == pytest.approx(20.0)
    assert grouped.head(1)[0]["cost_sum"] == pytest.approx(6.0)


def test_prompt_table_value_counts_and_describe() -> None:
    table = PromptTable.from_records(
        [
            {"region": "north", "score": 10.0},
            {"region": "north", "score": 20.0},
            {"region": "south", "score": None},
        ]
    )

    counts = table.value_counts("region").sort_values("region")
    stats = table.describe(["score"])

    assert counts.head(1)[0]["region"] == "north"
    assert counts.head(1)[0]["count"] == 2
    assert counts.head(1)[0]["share"] == pytest.approx(2 / 3)
    assert stats.head(1)[0]["column"] == "score"
    assert stats.head(1)[0]["non_null_count"] == 2
    assert stats.head(1)[0]["mean"] == pytest.approx(15.0)


def test_benchmark_summary_table_and_bundle_export(tmp_path) -> None:
    report = {
        "package": "geoprompt",
        "comparison": {"engines": ["geoprompt", "reference"], "corpus": ["demo"]},
        "summary": {
            "all_bounds_match": True,
            "all_nearest_neighbor_match": True,
            "all_bounds_query_match": True,
            "all_geometry_metrics_within_tolerance": True,
            "all_projected_bounds_match": True,
            "all_clip_matches": True,
            "all_dissolve_matches": True,
            "all_spatial_joins_match": True,
        },
        "datasets": [
            {
                "dataset": "demo",
                "feature_count": 3,
                "correctness": {
                    "bounds_match": True,
                    "nearest_neighbor_match": True,
                    "bounds_query_match": True,
                    "geometry_metrics_within_tolerance": True,
                    "projected_bounds_match": True,
                    "clip": None,
                    "dissolve": None,
                    "spatial_join": None,
                },
                "benchmarks": [
                    {"operation": "demo.geoprompt.nearest_neighbors", "median_seconds": 0.01, "min_seconds": 0.009, "max_seconds": 0.02, "repeats": 5},
                    {"operation": "demo.reference.nearest_neighbors", "median_seconds": 0.025, "min_seconds": 0.02, "max_seconds": 0.03, "repeats": 5},
                ],
            }
        ],
    }

    summary = benchmark_summary_table(report)
    row = summary.head(1)[0]
    written = export_comparison_bundle(report, tmp_path)

    assert row["dataset"] == "demo"
    assert row["operation"] == "nearest_neighbors"
    assert row["speedup_ratio"] == pytest.approx(2.5)
    assert row["winner"] == "geoprompt"
    assert (tmp_path / "geoprompt_comparison_report.json").exists()
    assert (tmp_path / "geoprompt_comparison_summary.md").exists()
    assert (tmp_path / "geoprompt_comparison_summary.html").exists()
    assert written["json"].endswith("geoprompt_comparison_report.json")


def test_build_and_export_resilience_summary_report(tmp_path) -> None:
    report = build_resilience_summary_report(
        redundancy_rows=[
            {"node": "hospital", "single_source_dependency": True, "is_critical": True, "resilience_tier": "low"},
            {"node": "pump", "single_source_dependency": False, "is_critical": False, "resilience_tier": "high"},
        ],
        outage_report={
            "impacted_node_count": 3,
            "impacted_customer_count": 120,
            "estimated_cost": 960.0,
            "severity_tier": "high",
        },
        restoration_report={
            "total_steps": 2,
            "stages": [
                {"step": 1, "repair_edge_id": "e1", "cumulative_restored_demand": 40.0},
                {"step": 2, "repair_edge_id": "e2", "cumulative_restored_demand": 100.0},
            ],
        },
        metadata={"scenario_id": "storm-demo"},
    )

    json_path = tmp_path / "resilience.json"
    md_path = tmp_path / "resilience.md"
    html_path = tmp_path / "resilience.html"

    export_resilience_summary_report(report, json_path)
    export_resilience_summary_report(report, md_path)
    export_resilience_summary_report(report, html_path)

    assert report["summary"]["critical_single_source_nodes"] == 1
    assert report["summary"]["low_resilience_nodes"] == 1
    assert report["metadata"]["scenario_id"] == "storm-demo"
    assert "Resilience Summary Report" in html_path.read_text(encoding="utf-8")
    assert "<svg" in html_path.read_text(encoding="utf-8")
    assert "critical_single_source_nodes" in json_path.read_text(encoding="utf-8")
    assert "Resilience Summary Report" in md_path.read_text(encoding="utf-8")


def test_build_and_export_resilience_portfolio_report(tmp_path) -> None:
    scenario_a = build_resilience_summary_report(
        redundancy_rows=[
            {"node": "hospital", "single_source_dependency": True, "is_critical": True, "resilience_tier": "low"},
            {"node": "pump", "single_source_dependency": False, "is_critical": False, "resilience_tier": "high"},
        ],
        outage_report={
            "impacted_node_count": 4,
            "impacted_customer_count": 150,
            "estimated_cost": 1800.0,
            "severity_tier": "high",
        },
        restoration_report={
            "total_steps": 3,
            "stages": [{"step": 1, "cumulative_restored_demand": 30.0}, {"step": 3, "cumulative_restored_demand": 110.0}],
        },
        metadata={"scenario_id": "storm_a"},
    )
    scenario_b = build_resilience_summary_report(
        redundancy_rows=[
            {"node": "hospital", "single_source_dependency": False, "is_critical": True, "resilience_tier": "high"},
            {"node": "pump", "single_source_dependency": False, "is_critical": False, "resilience_tier": "high"},
        ],
        outage_report={
            "impacted_node_count": 2,
            "impacted_customer_count": 50,
            "estimated_cost": 600.0,
            "severity_tier": "medium",
        },
        restoration_report={
            "total_steps": 2,
            "stages": [{"step": 1, "cumulative_restored_demand": 60.0}, {"step": 2, "cumulative_restored_demand": 120.0}],
        },
        metadata={"scenario_id": "storm_b"},
    )

    report = build_resilience_portfolio_report({"storm_a": scenario_a, "storm_b": scenario_b})
    assert report["summary"]["scenario_count"] == 2
    assert report["summary"]["best_scenario"] == "storm_b"
    assert len(report["scenarios"]) == 2
    assert report["scenarios"][0]["resilience_score"] >= report["scenarios"][1]["resilience_score"]

    json_path = tmp_path / "portfolio.json"
    csv_path = tmp_path / "portfolio.csv"
    md_path = tmp_path / "portfolio.md"
    html_path = tmp_path / "portfolio.html"

    export_resilience_portfolio_report(report, json_path)
    export_resilience_portfolio_report(report, csv_path)
    export_resilience_portfolio_report(report, md_path)
    export_resilience_portfolio_report(report, html_path)

    assert "best_scenario" in json_path.read_text(encoding="utf-8")
    assert "scenario_name" in csv_path.read_text(encoding="utf-8")
    assert "Resilience Portfolio Report" in md_path.read_text(encoding="utf-8")
    assert "<svg" in html_path.read_text(encoding="utf-8")


def test_validate_numeric_series_rejects_nan() -> None:
    with pytest.raises(ValueError, match="must not contain NaN"):
        validate_numeric_series([1.0, float("nan")])


def test_validate_numeric_series_bounds() -> None:
    cleaned = validate_numeric_series([1.0, 2.0, 3.0], min_value=1.0, max_value=3.0)
    assert cleaned == [1.0, 2.0, 3.0]


def test_geopandas_availability_returns_bool() -> None:
    assert isinstance(geopandas_available(), bool)


def test_interop_roundtrip_or_missing_dependency() -> None:
    frame = GeoPromptFrame.from_records(
        [
            {"site_id": "a", "geometry": {"type": "Point", "coordinates": (-111.9, 40.7)}},
            {"site_id": "b", "geometry": {"type": "Point", "coordinates": (-112.0, 40.8)}},
        ],
        crs="EPSG:4326",
    )

    if not geopandas_available():
        from geoprompt.interop import to_geopandas

        with pytest.raises(RuntimeError, match="GeoPandas support"):
            to_geopandas(frame)
        return

    from geoprompt.interop import from_geopandas, to_geopandas

    gdf = to_geopandas(frame)
    restored = from_geopandas(gdf)
    assert len(restored) == len(frame)
    assert restored.crs == frame.crs
