from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from geoprompt.tools import build_scenario_report, export_scenario_report


def test_scenario_report_json_csv_contract_parity(tmp_path: Path) -> None:
    report = build_scenario_report(
        {"cost": 100.0, "risk": 0.4},
        {"cost": 90.0, "risk": 0.3},
        higher_is_better=[],
    )

    json_path = Path(export_scenario_report(report, tmp_path / "scenario.json"))
    csv_path = Path(export_scenario_report(report, tmp_path / "scenario.csv"))

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["metrics"]["cost"]["direction"] == "improved"
    assert loaded["metrics"]["risk"]["direction"] == "improved"
    assert math.isclose(float(loaded["metrics"]["cost"]["delta"]), -10.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(loaded["metrics"]["cost"]["delta_percent"]), -10.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(loaded["metrics"]["risk"]["delta"]), -0.1, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(loaded["metrics"]["risk"]["delta_percent"]), -25.0, rel_tol=0.0, abs_tol=1e-12)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    keys = {row["metric"] for row in rows}
    assert keys == {"cost", "risk"}

    csv_by_metric = {row["metric"]: row for row in rows}
    for metric in ("cost", "risk"):
        csv_row = csv_by_metric[metric]
        json_row = loaded["metrics"][metric]
        assert math.isclose(float(csv_row["baseline"]), float(json_row["baseline"]), rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(float(csv_row["candidate"]), float(json_row["candidate"]), rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(float(csv_row["delta"]), float(json_row["delta"]), rel_tol=0.0, abs_tol=1e-12)
        assert str(csv_row["direction"]) == str(json_row["direction"])
