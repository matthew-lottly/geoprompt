from __future__ import annotations

import json
from pathlib import Path


def _sample_feature_collection() -> dict:
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [
            {
                "type": "Feature",
                "id": "a",
                "properties": {"site_id": "a", "value": 10},
                "geometry": {"type": "Point", "coordinates": [-111.9, 40.7]},
            }
        ],
    }


def test_read_service_url_geojson(tmp_path: Path):
    from geoprompt.io import read_service_url

    path = tmp_path / "service.geojson"
    path.write_text(json.dumps(_sample_feature_collection()), encoding="utf-8")

    frame = read_service_url(path.as_uri())
    assert len(frame) == 1
    assert frame[0]["site_id"] == "a"
    assert frame.crs == "EPSG:4326"


def test_read_service_url_arcgis_feature_response(tmp_path: Path):
    from geoprompt.io import read_service_url

    payload = {
        "spatialReference": {"wkid": 4326},
        "features": [
            {
                "attributes": {"site_id": "asset-1", "status": "ok"},
                "geometry": {"x": -111.95, "y": 40.75},
            }
        ],
    }
    path = tmp_path / "arcgis.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    frame = read_service_url(path.as_uri())
    assert len(frame) == 1
    assert frame[0]["status"] == "ok"
    assert frame[0]["geometry"]["type"] == "Point"


def test_plot_scenario_dashboard_creates_figure(tmp_path: Path):
    from geoprompt.viz import plot_scenario_dashboard

    fig = plot_scenario_dashboard(
        baseline_metrics={"served_load": 100.0, "deficit": 20.0},
        candidate_metrics={"served_load": 120.0, "deficit": 10.0},
        title="Scenario Comparison",
        output_path=tmp_path / "dashboard.png",
    )

    assert fig is not None
    assert (tmp_path / "dashboard.png").exists()


def test_cli_info_prints_summary(capsys):
    from geoprompt.cli import main

    main(["info"])
    captured = capsys.readouterr()
    assert "geoprompt" in captured.out.lower()
    assert "install" in captured.out.lower()
