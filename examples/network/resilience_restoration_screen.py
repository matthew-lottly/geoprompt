from __future__ import annotations

from pathlib import Path

from geoprompt.network import (
    build_network_graph,
    multi_source_service_audit,
    outage_impact_report,
    restoration_sequence_report,
    supply_redundancy_audit,
)
from geoprompt.tools import (
    build_resilience_portfolio_report,
    build_resilience_summary_report,
    export_resilience_portfolio_report,
    export_resilience_summary_report,
)


"""Small utility resilience demo.

Shows how to rank single-source dependency risk, quantify outage impact,
and stage restoration repairs with a stakeholder-ready HTML summary.
"""


def main() -> None:
    graph = build_network_graph(
        [
            {"edge_id": "e1", "from_node": "SRC", "to_node": "A", "cost": 1.0},
            {"edge_id": "e2", "from_node": "A", "to_node": "B", "cost": 1.0},
            {"edge_id": "e3", "from_node": "B", "to_node": "C", "cost": 1.0},
            {"edge_id": "tie", "from_node": "SRC", "to_node": "C", "cost": 2.0, "device_type": "tie", "state": "normally_open"},
        ],
        directed=False,
    )

    demand_by_node = {"A": 20.0, "B": 50.0, "C": 10.0}
    redundancy = supply_redundancy_audit(
        graph,
        source_nodes=["SRC", "C"],
        demand_by_node=demand_by_node,
        critical_nodes=["B"],
    )
    service_audit = multi_source_service_audit(
        graph,
        source_nodes=["SRC", "C"],
        demand_by_node=demand_by_node,
        source_capacity_by_node={"SRC": 40.0, "C": 60.0},
        critical_nodes=["B"],
    )
    outage = outage_impact_report(
        graph,
        source_nodes=["SRC"],
        failed_edges=["e1", "e2"],
        demand_by_node=demand_by_node,
        customer_count_by_node={"A": 80, "B": 120, "C": 40},
        critical_nodes=["B"],
        outage_hours=2.0,
    )
    restoration = restoration_sequence_report(
        graph,
        source_nodes=["SRC"],
        failed_edges=["e1", "e2"],
        demand_by_node=demand_by_node,
        critical_nodes=["B"],
    )

    report = build_resilience_summary_report(
        redundancy,
        outage_report=outage,
        restoration_report=restoration,
        metadata={"scenario_id": "resilience-demo"},
    )
    portfolio = build_resilience_portfolio_report(
        {
            "baseline": report,
            "upgrade-plan": build_resilience_summary_report(
                [
                    {**row, "single_source_dependency": False, "resilience_tier": "high" if row.get("node") == "B" else row.get("resilience_tier")}
                    for row in redundancy
                ],
                outage_report={**outage, "impacted_customer_count": 90, "estimated_cost": 720.0, "severity_tier": "medium"},
                restoration_report={
                    **restoration,
                    "total_steps": 1,
                    "stages": [{"step": 1, "repair_edge_id": "e1", "cumulative_restored_demand": 80.0}],
                },
                metadata={"scenario_id": "upgrade-plan"},
            ),
        }
    )
    output_path = export_resilience_summary_report(report, Path("outputs") / "resilience-summary.html")
    portfolio_path = export_resilience_portfolio_report(portfolio, Path("outputs") / "resilience-portfolio.html")

    print("Top redundancy rows:")
    for row in redundancy[:3]:
        print(row)

    print("\nService audit summary:")
    for row in service_audit["source_summary"]:
        print(row)

    print("\nOutage impact:")
    print(outage)

    print("\nRestoration stages:")
    for row in restoration["stages"]:
        print(row)

    print(f"\nStakeholder report written to: {output_path}")
    print(f"Portfolio report written to: {portfolio_path}")


if __name__ == "__main__":
    main()
