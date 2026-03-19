from pathlib import Path

from station_forecasting_workbench.workbench import build_forecast_report, export_forecast_report, load_histories


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_histories() -> None:
    histories = load_histories(PROJECT_ROOT / "data" / "forecast_histories.json")

    assert len(histories) == 3
    assert histories[1]["stationId"] == "station-central-flow-010"


def test_build_forecast_report() -> None:
    report = build_forecast_report(PROJECT_ROOT / "data" / "forecast_histories.json")

    assert report["summary"]["seriesCount"] == 3
    assert report["summary"]["forecastHorizon"] == 2
    assert report["summary"]["projectionHorizon"] == 3
    assert report["summary"]["averageWinningMae"] < 0.2
    assert report["forecasts"][0]["trainingWindow"] == 6
    assert report["forecasts"][0]["featureProfile"]["trailingMean3"] == 13.03
    assert report["forecasts"][0]["selectedModel"] in {"drift", "linear_regression", "last_value", "trailing_average_3"}
    assert len(report["forecasts"][0]["modelLeaderboard"]) == 4
    assert report["forecasts"][0]["modelLeaderboard"][0]["holdoutMae"] <= report["forecasts"][0]["modelLeaderboard"][1]["holdoutMae"]
    assert len(report["forecasts"][0]["projection"]) == 3


def test_export_forecast_report(tmp_path: Path) -> None:
    output_path = export_forecast_report(tmp_path, report_name="Forecast Review")

    assert output_path.exists()
    assert "Forecast Review" in output_path.read_text(encoding="utf-8")