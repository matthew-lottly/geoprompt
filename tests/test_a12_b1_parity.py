from __future__ import annotations

from pathlib import Path

import geoprompt as gp


A12_FUNCTIONS = [
    "cython_accelerated_geometry_ops",
    "rust_accelerated_geometry_ops",
    "cpp_extension_module",
    "simd_accelerated_coordinate_transforms",
    "gpu_accelerated_point_in_polygon",
    "gpu_accelerated_distance_matrix",
    "gpu_accelerated_raster_algebra",
    "columnar_storage_engine",
    "out_of_core_processing",
    "tile_based_raster_streaming",
    "predicate_pushdown_to_storage",
    "column_pruning_on_read",
    "partition_aware_read",
    "spatial_partitioning_strip",
    "spatial_partitioning_quadtree",
    "distributed_spatial_join",
    "distributed_spatial_aggregation",
    "distributed_raster_processing",
    "distributed_routing",
    "mapreduce_style_spatial_pipeline",
    "connection_pool_for_database_layers",
    "websocket_geo_event_stream",
    "server_sent_events_progress",
    "grpc_spatial_service",
    "delta_encoding_for_incremental_updates",
    "spatial_index_warmup_cache",
    "tile_cache_vector_raster",
    "cache_busting_strategy",
    "quota_management_per_user",
    "priority_queue_for_jobs",
    "fair_scheduling_across_tenants",
    "multi_tenant_isolation",
    "workflow_versioning_hash_based",
    "profiling_report_export_html_flame_graph",
    "code_coverage_gate",
    "static_analysis_ruff_pylint",
    "security_scanning_bandit_pip_audit",
    "sbom_generation_cyclonedx",
    "array_api_standard_compliance",
    "typer_cli_framework",
    "jupyter_widget_for_map_display",
    "vscode_extension_spatial_preview",
    "searchable_api_reference",
    "telemetry_opt_in_out_ux",
    "user_survey_integration",
]

B1_FUNCTIONS = [
    "QualityAuditResult",
    "audit_public_docstrings",
    "audit_type_annotations",
    "audit_mutable_defaults",
    "audit_pathlib_usage",
    "audit_print_debugging",
    "quality_scorecard",
]


def test_a12_public_surface_presence() -> None:
    for name in A12_FUNCTIONS:
        assert hasattr(gp, name), name


def test_b1_public_surface_presence() -> None:
    for name in B1_FUNCTIONS:
        assert hasattr(gp, name), name


def test_a12_backend_and_scalability_helpers(tmp_path: Path) -> None:
    accel = gp.cython_accelerated_geometry_ops("buffer")
    assert accel["backend"] == "cython"
    assert accel["operation"] == "buffer"

    transformed = gp.simd_accelerated_coordinate_transforms([(1.0, 2.0), (3.0, 4.0)], scale=2.0)
    assert transformed["coordinates"][0] == (2.0, 4.0)

    matrix = gp.gpu_accelerated_distance_matrix([(0.0, 0.0), (3.0, 4.0)])
    assert matrix["matrix"][0][1] == 5.0

    filtered = gp.predicate_pushdown_to_storage(
        [{"id": 1, "status": "ok"}, {"id": 2, "status": "bad"}],
        {"status": "ok"},
    )
    assert filtered["rows"] == [{"id": 1, "status": "ok"}]

    pruned = gp.column_pruning_on_read(filtered["rows"], ["id"])
    assert pruned["rows"] == [{"id": 1}]

    partitions = gp.partition_aware_read(
        {"year=2025": [{"id": 1}], "year=2026": [{"id": 2}]},
        partition_keys=["year=2026"],
    )
    assert partitions["rows"] == [{"id": 2}]

    grouped = gp.distributed_spatial_aggregation(
        [{"kind": "a"}, {"kind": "a"}, {"kind": "b"}], group_by="kind", metric="count"
    )
    assert grouped["groups"]["a"] == 2

    queue = gp.priority_queue_for_jobs([
        {"id": "low", "priority": 1},
        {"id": "high", "priority": 10},
    ])
    assert queue["jobs"][0]["id"] == "high"

    fairness = gp.fair_scheduling_across_tenants([
        {"tenant": "alpha", "id": 1},
        {"tenant": "beta", "id": 2},
        {"tenant": "alpha", "id": 3},
    ])
    assert fairness["schedule"][0]["tenant"] == "alpha"
    assert fairness["schedule"][1]["tenant"] == "beta"

    delta = gp.delta_encoding_for_incremental_updates(
        [{"id": 1, "value": 10}, {"id": 2, "value": 20}],
        [{"id": 1, "value": 11}, {"id": 2, "value": 20}, {"id": 3, "value": 30}],
        key="id",
    )
    assert delta["changed_ids"] == [1, 3]

    quota = gp.quota_management_per_user("alice", limit=10, used=4)
    assert quota["remaining"] == 6

    flame = gp.profiling_report_export_html_flame_graph(tmp_path / "profile.html")
    assert flame["path"].endswith("profile.html")
    assert (tmp_path / "profile.html").exists()

    coverage = gp.code_coverage_gate(coverage=92.5, threshold=90.0)
    assert coverage["passed"] is True

    telemetry = gp.telemetry_opt_in_out_ux(False)
    assert telemetry["enabled"] is False


def test_b1_quality_audit_helpers(tmp_path: Path) -> None:
    sample = tmp_path / "sample_quality.py"
    sample.write_text(
        "def bad(value=[]):\n"
        "    return value\n\n"
        "def noisy(x):\n"
        "    print(x)\n"
    )

    assert gp.audit_public_docstrings(sample)["missing"] == ["bad", "noisy"]
    assert gp.audit_type_annotations(sample)["missing"] == ["bad", "noisy"]
    assert gp.audit_mutable_defaults(sample)["issues"] == ["bad"]
    assert gp.audit_print_debugging(sample)["issues"] == ["noisy"]

    repo_root = Path(gp.__file__).resolve().parent
    scorecard = gp.quality_scorecard([
        repo_root / "performance.py",
        repo_root / "quality.py",
        repo_root / "__init__.py",
    ])
    assert scorecard["total_issues"] == 0
