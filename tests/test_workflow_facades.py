from __future__ import annotations

from pathlib import Path

from geoprompt.frame import GeoPromptFrame
from geoprompt.workflow import DataPipeline, IOConfig, JoinConfig, ReportBuilder, ReportConfig, ScenarioRunner


def test_data_pipeline_and_fluent_configs(monkeypatch) -> None:
    base = GeoPromptFrame.from_records([
        {"id": "a", "value": 2, "geometry": {"type": "Point", "coordinates": [0, 0]}},
        {"id": "b", "value": 0, "geometry": {"type": "Point", "coordinates": [1, 1]}},
    ])

    monkeypatch.setattr("geoprompt.workflow.read_data", lambda *args, **kwargs: base)

    cfg = IOConfig(source="unused").with_where(None).with_geometry("geometry")
    joined = (
        DataPipeline()
        .load("unused", config=cfg)
        .filter("value > 1")
        .join(base, config=JoinConfig().with_predicate("intersects").with_how("inner"))
        .frame()
    )

    assert len(joined) >= 1


def test_scenario_runner_collects_named_results() -> None:
    runner = ScenarioRunner().add("double", lambda x: x * 2, x=4).add("sum", lambda a, b: a + b, a=2, b=3)
    result = runner.run()
    assert result["double"] == 8
    assert result["sum"] == 5


def test_report_builder_writes_selected_format(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    cfg = ReportConfig().with_title("Run Summary").with_format("markdown").with_output(out)
    text = ReportBuilder(cfg).build({"passed": True, "rows": 2})

    assert "Run Summary" in text
    assert out.exists()
