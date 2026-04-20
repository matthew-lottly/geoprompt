"""Tests for A8 Geoprocessing Framework (1101-1250) items."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest

import geoprompt
from geoprompt.geoprocessing import (
    ConnectionPool,
    DiskIOProfiler,
    GeometryEngineAdapter,
    MemoryProfiler,
    MetricsExporter,
    ModelBuilder,
    ModelVariable,
    Observable,
    RequestTracer,
    UndoRedoManager,
    _LICENSE_REGISTRY,
    _PLUGIN_VERSIONS,
    _TELEMETRY_ENABLED,
    _TELEMETRY_LOG,
    check_license,
    check_plugin_updates,
    clear_telemetry_log,
    cloud_deployment_template,
    column_level_redact,
    conda_recipe_meta_yaml,
    consume_license,
    dask_map,
    decrypt_field,
    enable_telemetry,
    encrypt_field,
    fuzz_string,
    generate_changelog_from_git,
    generate_migration_guide,
    generate_precommit_config,
    generate_random_parcels,
    generate_random_raster,
    generate_random_streets,
    generate_tutorial_skeleton,
    git_commit_workflow,
    golden_file_compare,
    grafana_dashboard_json,
    ide_completion_data,
    kubernetes_helm_values,
    lint_recipe,
    log_aggregation_config,
    mypy_plugin_stub,
    notify_email_stub,
    notify_slack_stub,
    oauth2_authorization_url,
    oauth2_token_exchange_payload,
    pyodide_loader_script,
    r_command,
    r_sf_read,
    r_terra_read,
    record_telemetry,
    register_license,
    register_plugin_version,
    resolve_plugin_dependencies,
    row_level_filter,
    serverless_endpoint_stub,
    singularity_def,
    sso_saml_metadata_stub,
    telemetry_enabled,
    tile_extent,
    tile_process,
)


# ---- 1107. License check ----

class TestLicensing:
    def setup_method(self):
        _LICENSE_REGISTRY.clear()

    def test_register_and_check(self):
        register_license("my_tool", "KEY123", max_uses=10)
        info = check_license("my_tool")
        assert info["valid"]
        assert info["remaining"] == 10

    def test_consume(self):
        register_license("my_tool", "KEY123", max_uses=2)
        assert consume_license("my_tool")
        assert consume_license("my_tool")
        assert not consume_license("my_tool")

    def test_unknown_tool(self):
        assert not check_license("nonexistent")["valid"]


# ---- 1109-1115. Model Builder ----

class TestModelBuilder:
    def test_basic_steps(self):
        mb = ModelBuilder("TestModel")
        mb.add_step("double", lambda x: x * 2)
        mb.add_step("add_one", lambda x: x + 1)
        results = mb.run(5)
        assert len(results) == 2
        assert results[0].output == 10
        assert results[1].output == 11

    def test_branch_if_true(self):
        mb = ModelBuilder()
        mb.add_branch(
            condition=lambda x: x > 5,
            if_func=lambda x: x * 10,
            else_func=lambda x: x * -1,
        )
        results = mb.run(10)
        assert results[0].output == 100

    def test_branch_if_false(self):
        mb = ModelBuilder()
        mb.add_branch(
            condition=lambda x: x > 5,
            if_func=lambda x: x * 10,
            else_func=lambda x: x * -1,
        )
        results = mb.run(3)
        assert results[0].output == -3

    def test_for_each(self):
        mb = ModelBuilder()
        mb.add_for_each(
            items_func=lambda x: x,
            body_func=lambda item: item ** 2,
            collect_key="squares",
        )
        results = mb.run([1, 2, 3, 4])
        assert results[0].output == [1, 4, 9, 16]
        assert mb.get_collected("squares") == [1, 4, 9, 16]

    def test_while_loop(self):
        mb = ModelBuilder()
        mb.add_while(
            condition=lambda x: x < 100,
            body_func=lambda x: x * 2,
        )
        results = mb.run(1)
        assert results[0].output == 128

    def test_collect_values(self):
        mb = ModelBuilder()
        mb.add_collect("snapshots", lambda x: x)
        mb.add_step("inc", lambda x: x + 1)
        mb.add_collect("snapshots", lambda x: x)
        results = mb.run(10)
        assert mb.get_collected("snapshots") == [10, 11]

    def test_variables(self):
        mb = ModelBuilder()
        mb.set_variable("name", "World")
        assert mb.get_variable("name") == "World"
        assert mb.substitute("Hello %name%!") == "Hello World!"

    def test_export_to_python(self):
        mb = ModelBuilder("ExportTest")
        mb.add_step("load", lambda: None)
        mb.add_branch(lambda x: True, lambda x: x, lambda x: x)
        script = mb.export_to_python()
        assert "def run_model" in script
        assert "ExportTest" in script
        assert "Branch" in script


# ---- 1114. ModelVariable ----

def test_model_variable():
    v = ModelVariable("output_dir", "/tmp/data")
    assert v.substitute("save to %output_dir%/result.csv") == "save to /tmp/data/result.csv"
    assert "output_dir" in repr(v)


# ---- 1120. Dask map ----

def test_dask_map_fallback():
    result = dask_map(lambda x: x ** 2, [1, 2, 3], use_dask=False)
    assert list(result) == [1, 4, 9]


# ---- 1122. Tile-based processing ----

def test_tile_extent():
    tiles = tile_extent((0, 0, 10, 10), 5)
    assert len(tiles) == 4
    assert tiles[0] == (0, 0, 5, 5)

def test_tile_process():
    results = tile_process(lambda t: (t[2] - t[0]) * (t[3] - t[1]), (0, 0, 10, 10), 5)
    assert all(r == 25.0 for r in results)


# ---- 1132. Undo/Redo ----

class TestUndoRedo:
    def test_basic_undo_redo(self):
        mgr = UndoRedoManager()
        mgr.push("edit1", {"value": 1})
        mgr.push("edit2", {"value": 2})
        assert mgr.can_undo()
        label, snap = mgr.undo()
        assert label == "edit2"
        assert snap == {"value": 2}
        assert mgr.can_redo()
        label2, snap2 = mgr.redo()
        assert label2 == "edit2"

    def test_history(self):
        mgr = UndoRedoManager()
        mgr.push("a", 1)
        mgr.push("b", 2)
        assert mgr.history() == ["a", "b"]

    def test_empty_undo(self):
        mgr = UndoRedoManager()
        assert mgr.undo() is None
        assert mgr.redo() is None


# ---- 1143. Jupyter magics ----

def test_register_jupyter_magics_no_ipython():
    # Should return False when IPython not available in test env
    # (or True if running inside IPython — both are acceptable)
    result = geoprompt.register_jupyter_magics()
    assert isinstance(result, bool)


# ---- 1149. R-spatial bridge ----

def test_r_command():
    cmd = r_command('print("hello")')
    assert cmd[0] == "Rscript"
    assert "-e" in cmd

def test_r_sf_read():
    cmd = r_sf_read("data.gpkg", layer="my_layer")
    assert "st_read" in cmd[-1]
    assert "my_layer" in cmd[-1]

def test_r_terra_read():
    cmd = r_terra_read("raster.tif")
    assert "terra" in cmd[-1]


# ---- 1153-1155. Plugin versioning ----

class TestPluginVersioning:
    def setup_method(self):
        _PLUGIN_VERSIONS.clear()

    def test_register_and_resolve(self):
        register_plugin_version("core", "1.0.0")
        register_plugin_version("ext", "0.5.0", dependencies={"core": "1.0.0"})
        result = resolve_plugin_dependencies("ext")
        assert result["resolved"]

    def test_missing_dependency(self):
        register_plugin_version("ext", "0.5.0", dependencies={"missing_dep": "1.0.0"})
        result = resolve_plugin_dependencies("ext")
        assert not result["resolved"]
        assert len(result["missing"]) == 1

    def test_check_updates(self):
        register_plugin_version("core", "1.0.0")
        info = check_plugin_updates("core", available_version="2.0.0")
        assert info["update_available"]
        info2 = check_plugin_updates("core", available_version="0.5.0")
        assert not info2["update_available"]


# ---- 1160. Geometry engine adapter ----

def test_geometry_engine_adapter():
    engine = GeometryEngineAdapter("test_engine")
    engine.register_op("buffer", lambda geom, dist: {"buffered": True})
    assert engine.has_op("buffer")
    assert not engine.has_op("clip")
    result = engine.execute("buffer", {}, 10)
    assert result == {"buffered": True}
    assert "buffer" in engine.list_ops()
    with pytest.raises(NotImplementedError):
        engine.execute("clip", {})


# ---- 1165. Connection pool ----

def test_connection_pool():
    counter = {"n": 0}
    def factory():
        counter["n"] += 1
        return f"conn_{counter['n']}"

    pool = ConnectionPool(factory, max_size=2)
    c1 = pool.acquire()
    c2 = pool.acquire()
    assert pool.available == 0
    with pytest.raises(RuntimeError):
        pool.acquire()
    pool.release(c1)
    assert pool.available == 1
    c3 = pool.acquire()
    assert c3 == c1


# ---- 1168. Telemetry ----

class TestTelemetry:
    def setup_method(self):
        clear_telemetry_log()
        enable_telemetry(False)

    def test_opt_in(self):
        record_telemetry("test_event")
        assert len(geoprompt.get_telemetry_log()) == 0
        enable_telemetry(True)
        assert telemetry_enabled()
        record_telemetry("test_event", tool="buffer")
        log = geoprompt.get_telemetry_log()
        assert len(log) == 1
        assert log[0]["tool"] == "buffer"

    def teardown_method(self):
        enable_telemetry(False)
        clear_telemetry_log()


# ---- 1170. Memory profiler ----

def test_memory_profiler():
    mp = MemoryProfiler()
    mp.start()
    snap = mp.snapshot("test")
    assert "current_mb" in snap
    assert "peak_mb" in snap
    report = mp.report()
    assert len(report) == 1
    mp.stop()


# ---- 1171. Disk I/O profiler ----

def test_disk_io_profiler():
    dp = DiskIOProfiler()
    dp.record_read("a.csv", 1024, 0.01)
    dp.record_write("b.csv", 2048, 0.02)
    s = dp.summary()
    assert s["total_reads"] == 1
    assert s["total_writes"] == 1
    assert s["bytes_read"] == 1024
    dp.clear()
    assert dp.summary()["total_reads"] == 0


# ---- 1174. Golden-file test framework ----

def test_golden_file_compare(tmp_path):
    golden = tmp_path / "golden.txt"
    golden.write_text("hello world")
    result = golden_file_compare("hello world", str(golden))
    assert result["match"]
    result2 = golden_file_compare("different", str(golden))
    assert not result2["match"]

def test_golden_file_update(tmp_path):
    golden = tmp_path / "new_golden.txt"
    result = golden_file_compare("new content", str(golden), update=True)
    assert result["match"]
    assert result["updated"]
    assert golden.read_text() == "new content"


# ---- 1177. Fuzz testing ----

def test_fuzz_string():
    variants = fuzz_string("POINT(1 2)", n=20, seed=42)
    assert len(variants) == 20
    assert variants[0] == "POINT(1 2)"  # first is always original
    assert any(v != "POINT(1 2)" for v in variants[1:])


# ---- 1179. Parcel generator ----

def test_generate_random_parcels():
    parcels = generate_random_parcels(10, seed=42)
    assert len(parcels) == 10
    assert all(p["type"] == "Feature" for p in parcels)
    assert all(p["geometry"]["type"] == "Polygon" for p in parcels)
    props = parcels[0]["properties"]
    assert "parcel_id" in props
    assert "address" in props
    assert "zone" in props
    assert "assessed_value" in props


# ---- 1181. Raster generator ----

def test_generate_random_raster():
    raster = generate_random_raster(50, 50, bands=3, seed=42)
    assert raster["rows"] == 50
    assert raster["cols"] == 50
    assert raster["bands"] == 3
    assert len(raster["data"]) == 3
    assert len(raster["data"][0]) == 50


# ---- 1182. Street generator ----

def test_generate_random_streets():
    streets = generate_random_streets(20, seed=42)
    assert len(streets) == 20
    assert all(s["geometry"]["type"] == "LineString" for s in streets)
    props = streets[0]["properties"]
    assert "osm_id" in props
    assert "highway" in props
    assert "maxspeed" in props


# ---- 1185. Pre-commit config ----

def test_generate_precommit_config():
    cfg = generate_precommit_config()
    assert "repos:" in cfg
    assert "ruff" in cfg
    assert "trailing-whitespace" in cfg


# ---- 1186. Recipe linter ----

def test_lint_recipe_valid():
    result = lint_recipe({"name": "test", "steps": ["read", "process"], "persona": "analyst"})
    assert result["valid"]

def test_lint_recipe_invalid():
    result = lint_recipe({"steps": []})
    assert not result["valid"]
    assert any("name" in issue for issue in result["issues"])


# ---- 1188. Mypy plugin stub ----

def test_mypy_plugin_stub():
    stub = mypy_plugin_stub()
    assert "class GeoPromptPlugin" in stub
    assert "def plugin" in stub


# ---- 1189. IDE completion data ----

def test_ide_completion_data():
    data = ide_completion_data()
    assert data["module"] == "geoprompt"
    assert len(data["completions"]) > 0
    first = data["completions"][0]
    assert "name" in first and "kind" in first


# ---- 1192. Tutorial skeleton ----

def test_generate_tutorial_skeleton():
    script = generate_tutorial_skeleton("My Tutorial", steps=["Load", "Analyse"])
    assert "My Tutorial" in script
    assert "import geoprompt" in script
    assert "Step 1: Load" in script


# ---- 1193. Changelog generator ----

def test_generate_changelog_from_git():
    # May work or fail depending on git availability; just check it returns a string
    result = generate_changelog_from_git(".", max_entries=5)
    assert isinstance(result, str)
    assert "Changelog" in result


# ---- 1194. Migration guide ----

def test_generate_migration_guide():
    guide = generate_migration_guide("1.0", "2.0", changes=[
        {"category": "API", "before": "old_func()", "after": "new_func()", "note": "Renamed"},
    ])
    assert "1.0" in guide
    assert "2.0" in guide
    assert "old_func" in guide


# ---- 1216. Async tool execution ----

def test_async_tool_execute():
    async def _run():
        result = await geoprompt.async_tool_execute(lambda x: x + 1, 5)
        return result

    result = asyncio.run(_run())
    assert result.succeeded()
    assert result.output == 6


# ---- 1218. Observable ----

def test_observable():
    obs = Observable()
    received = []
    obs.subscribe(lambda v: received.append(v))
    obs.emit("hello")
    obs.emit("world")
    assert received == ["hello", "world"]
    assert obs.subscriber_count == 1


# ---- 1219-1221. Notification stubs ----

def test_notify_email_stub():
    result = notify_email_stub("user@example.com", "Done", "Tool finished.")
    assert result["to"] == "user@example.com"

def test_notify_slack_stub():
    result = notify_slack_stub("https://hooks.slack.com/test", "Done!")
    assert result["payload"]["text"] == "Done!"


# ---- 1231. Request tracer ----

def test_request_tracer():
    tracer = RequestTracer()
    span = tracer.start_span("my_operation")
    assert "span_id" in span
    tracer.end_span(span)
    assert span["duration_ms"] >= 0
    assert len(tracer.get_traces()) == 1
    tracer.clear()
    assert len(tracer.get_traces()) == 0


# ---- 1233. Log aggregation config ----

def test_log_aggregation_config():
    cfg = log_aggregation_config("elk")
    assert cfg["type"] == "logstash"
    cfg2 = log_aggregation_config("datadog")
    assert cfg2["type"] == "datadog-agent"


# ---- 1234. Metrics exporter ----

def test_metrics_exporter():
    mx = MetricsExporter()
    mx.increment("tool_calls", 5)
    mx.gauge("active_jobs", 3)
    mx.observe("duration_seconds", 1.5)
    mx.observe("duration_seconds", 2.0)
    text = mx.export_text()
    assert "tool_calls 5" in text
    assert "active_jobs 3" in text
    assert "duration_seconds_count 2" in text
    mx.reset()
    assert mx.export_text() == ""


# ---- 1235. Grafana dashboard ----

def test_grafana_dashboard():
    dash = grafana_dashboard_json("Test Dashboard")
    assert dash["dashboard"]["title"] == "Test Dashboard"
    assert len(dash["dashboard"]["panels"]) == 3


# ---- 1238. Row-level security ----

def test_row_level_filter():
    records = [
        {"name": "A", "owner": "alice"},
        {"name": "B", "owner": "bob"},
        {"name": "C", "owner": "alice"},
    ]
    filtered = row_level_filter(records, "alice")
    assert len(filtered) == 2
    admin = row_level_filter(records, "charlie", user_roles=["admin"])
    assert len(admin) == 3


# ---- 1239. Column-level security ----

def test_column_level_redact():
    records = [{"name": "A", "ssn": "123", "city": "NYC"}]
    result = column_level_redact(records, ["name", "city"])
    assert "ssn" not in result[0]
    assert result[0]["name"] == "A"


# ---- 1240. Field encryption ----

def test_encrypt_decrypt_field():
    rec = {"name": "Alice", "secret": "TopSecret"}
    encrypted = encrypt_field(rec, "secret", "mykey")
    assert encrypted["secret"] != "TopSecret"
    decrypted = decrypt_field(encrypted, "secret", "mykey")
    assert decrypted["secret"] == "TopSecret"


# ---- 1243. OAuth2 helpers ----

def test_oauth2_authorization_url():
    url = oauth2_authorization_url(
        "https://auth.example.com/authorize",
        "client123",
        "https://app.example.com/callback",
    )
    assert "client123" in url
    assert "response_type=code" in url

def test_oauth2_token_exchange():
    payload = oauth2_token_exchange_payload(
        "https://auth.example.com/token",
        "client123",
        "secret",
        "auth_code_here",
        "https://app.example.com/callback",
    )
    assert payload["data"]["grant_type"] == "authorization_code"


# ---- 1244. SSO SAML stub ----

def test_sso_saml_metadata():
    xml = sso_saml_metadata_stub("https://sp.example.com", "https://sp.example.com/acs")
    assert "EntityDescriptor" in xml
    assert "sp.example.com" in xml


# ---- 1207. Conda recipe ----

def test_conda_recipe():
    yaml = conda_recipe_meta_yaml("geoprompt", "0.2.0")
    assert "geoprompt" in yaml
    assert "0.2.0" in yaml

# ---- 1209. Singularity ----

def test_singularity_def():
    deffile = singularity_def()
    assert "Bootstrap: docker" in deffile
    assert "pip install geoprompt" in deffile


# ---- 1210. Pyodide ----

def test_pyodide_loader():
    html = pyodide_loader_script()
    assert "pyodide" in html.lower()
    assert "geoprompt" in html


# ---- 1211-1213. Cloud templates ----

def test_cloud_deployment_templates():
    aws = cloud_deployment_template("aws_lambda")
    assert aws["provider"] == "aws"
    azure = cloud_deployment_template("azure_functions")
    assert azure["provider"] == "azure"
    gcp = cloud_deployment_template("gcp_cloud_run")
    assert gcp["provider"] == "gcp"


# ---- 1214. Kubernetes Helm ----

def test_kubernetes_helm_values():
    vals = kubernetes_helm_values(replicas=3)
    assert vals["replicaCount"] == 3
    assert vals["service"]["port"] == 8080


# ---- 1215. Serverless endpoint ----

def test_serverless_endpoint_stub():
    code = serverless_endpoint_stub()
    assert "def handler" in code
    assert "geoprompt" in code
