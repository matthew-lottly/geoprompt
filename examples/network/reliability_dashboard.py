from __future__ import annotations

from pathlib import Path

from geoprompt.network import reliability_scenario_report
from geoprompt.viz import plot_scenario_dashboard


OUTPUT_DIR = Path("outputs")


def main() -> None:
    baseline_events = [
        {"impacted_customer_count": 180, "outage_hours": 2.5},
        {"impacted_customer_count": 75, "outage_hours": 1.0},
    ]
    hardened_events = [
        {"impacted_customer_count": 90, "outage_hours": 1.0},
        {"impacted_customer_count": 20, "outage_hours": 0.5},
    ]

    rows = reliability_scenario_report(
        {
            "baseline": baseline_events,
            "hardened": hardened_events,
        },
        total_customers=5000,
        baseline="baseline",
    )
    by_name = {row["scenario"]: row for row in rows}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_scenario_dashboard(
        baseline_metrics={
            "SAIDI": by_name["baseline"]["SAIDI"],
            "SAIFI": by_name["baseline"]["SAIFI"],
            "ASAI": by_name["baseline"]["ASAI"],
        },
        candidate_metrics={
            "SAIDI": by_name["hardened"]["SAIDI"],
            "SAIFI": by_name["hardened"]["SAIFI"],
            "ASAI": by_name["hardened"]["ASAI"],
        },
        higher_is_better=["ASAI"],
        title="Reliability improvement dashboard",
        output_path=OUTPUT_DIR / "reliability-dashboard.png",
    )
    print("Wrote dashboard to", OUTPUT_DIR / "reliability-dashboard.png")
    try:
        fig.clf()
    except Exception:
        pass


if __name__ == "__main__":
    main()
