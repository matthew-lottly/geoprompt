from __future__ import annotations

from pathlib import Path

import geoprompt as gp


def main() -> None:
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "executive-briefing.html"

    html = gp.comparison_view(
        [
            {"name": "Baseline", "served_load": 100.0, "deficit": 14.0},
            {"name": "Upgrade", "served_load": 118.0, "deficit": 6.0},
            {"name": "Worst Case", "served_load": 82.0, "deficit": 21.0},
        ],
        title="Executive Briefing Pack",
    )
    report = gp.prompt_to_report(
        {"decision": "Proceed with upgrade", "coverage_gain_pct": 18.0, "deficit_drop_pct": 57.0},
        title="Executive Summary",
        audience="executive",
    )
    with open(destination, "w", encoding="utf-8") as f:
        f.write(html + "\n<hr/>\n" + report)
    print(f"Executive briefing bundle written to {destination}")


if __name__ == "__main__":
    main()
