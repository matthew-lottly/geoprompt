from __future__ import annotations

import time
from pathlib import Path

import pytest

import geoprompt as gp


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "geom",
    [
        {"type": "Point", "coordinates": [1.0, 2.0]},
        {"type": "LineString", "coordinates": [(0.0, 0.0), (1.0, 1.0)]},
        {"type": "Polygon", "coordinates": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]]},
    ],
)
def test_b2_parametrized_geometry_coverage(geom) -> None:
    assert gp.geometry_type(geom) in {"Point", "LineString", "Polygon"}
    assert gp.geometry_bounds(geom) is not None


def test_b2_windows_unicode_sqlite_and_reexports(tmp_path: Path) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                "properties": {"naïve_name": "café", "city": "Montréal"},
            }
        ],
    }
    json_path = tmp_path / "nested" / "unicode data.json"
    gp.write_json(json_path, payload)
    frame = gp.read_data(str(json_path).replace("/", "\\"))
    assert frame[0]["naïve_name"] == "café"

    database = tmp_path / "assets.sqlite"
    rows = [{"site_id": "α", "category": "β", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]
    written = gp.write_spatialite(rows, "assets", str(database))
    loaded = gp.read_spatialite("SELECT * FROM assets", str(database))
    assert written == 1
    assert loaded[0]["site_id"] == "α"

    exported_names = ["write_spatialite", "read_spatialite", "quality_scorecard", "telemetry_opt_in_out_ux"]
    for name in exported_names:
        assert hasattr(gp, name), name


def test_b3_b4_docs_and_ci_assets_exist() -> None:
    required_paths = [
        ROOT / "docs" / "migration-from-arcpy.md",
        ROOT / "docs" / "environment-and-optional-dependencies.md",
        ROOT / "docs" / "video-walkthroughs.md",
        ROOT / ".github" / "dependabot.yml",
        ROOT / ".github" / "workflows" / "security-quality.yml",
        ROOT / "tox.ini",
        ROOT / "noxfile.py",
        ROOT / "MANIFEST.in",
    ]
    for path in required_paths:
        assert path.exists(), path

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "migration-from-arcpy" in readme
    assert "environment-and-optional-dependencies" in readme
    assert "video-walkthroughs" in readme

    ci = (ROOT / ".github" / "workflows" / "geoprompt-ci.yml").read_text(encoding="utf-8")
    assert '"3.10"' in ci
    assert "execute_notebooks_with_timeout.py" in ci
    assert "ubuntu-latest" in ci and "windows-latest" in ci and "macos-latest" in ci


def test_b5_runtime_guards_and_batch_io(tmp_path: Path) -> None:
    token = gp.CancellationToken()
    ok = gp.run_with_timeout_guard(lambda: "ok", timeout_seconds=0.25, cancel_token=token)
    assert ok["completed"] is True
    assert ok["result"] == "ok"

    token.cancel()
    cancelled = gp.run_with_timeout_guard(lambda: time.sleep(0.01), timeout_seconds=0.25, cancel_token=token)
    assert cancelled["cancelled"] is True

    guard = gp.guard_dataset_size(50, max_rows=100)
    assert guard["allowed"] is True
    with pytest.raises(MemoryError, match="cannot process"):
        gp.guard_dataset_size(101, max_rows=100, raise_on_exceed=True)

    profile = gp.profile_top_hot_functions([("sum", lambda: sum(range(1000)))])
    assert profile["top"][0]["label"] == "sum"

    batch_path = tmp_path / "records.jsonl"
    batch = gp.batch_write_json_records([{"id": 1}, {"id": 2}], batch_path, batch_size=1)
    assert batch["count"] == 2
    assert batch_path.exists()
