from pathlib import Path

from spatial_data_api.demo_assets import export_monitoring_status_map, load_monitoring_sites


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_monitoring_sites() -> None:
    sites = load_monitoring_sites(PROJECT_ROOT / "src" / "spatial_data_api" / "data" / "sample_features.geojson")

    assert len(sites) == 3
    assert sites[0].feature_id == "station-001"
    assert {site.status for site in sites} == {"normal", "alert", "offline"}


def test_export_monitoring_status_map(tmp_path: Path) -> None:
    output_path = export_monitoring_status_map(
        output_dir=tmp_path,
        input_path=PROJECT_ROOT / "src" / "spatial_data_api" / "data" / "sample_features.geojson",
    )

    assert output_path.exists()
    assert output_path.name == "monitoring-status-footprint.png"