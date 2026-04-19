from __future__ import annotations

from pathlib import Path

import geoprompt as gp


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = gp.read_data(root / "data" / "sample_features.json")
    report = gp.build_scenario_report(
        baseline_metrics={"served_load": 100.0, "deficit": 0.14, "travel_time": 18.0},
        candidate_metrics={"served_load": 118.0, "deficit": 0.06, "travel_time": 12.0},
        higher_is_better=["served_load"],
    )
    out = gp.export_scenario_report(report, output_dir / "analyst-resilience-screen.html")
    print("Analyst workflow complete")
    print(frame.summary())
    print(out)


if __name__ == "__main__":
    main()
