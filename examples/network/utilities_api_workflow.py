from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from geoprompt.network.routing import build_network_graph, service_area
from geoprompt.network.utility import outage_impact_report, restoration_sequence_report
from geoprompt.network import closest_facility
from geoprompt.tools import (
    build_resilience_portfolio_report,
    build_resilience_summary_report,
    export_resilience_portfolio_report,
    export_resilience_summary_report,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "private" / "api-cache"


def _fetch_json(url: str, *, timeout_seconds: float = 12.0, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = Request(url, headers=headers or {"User-Agent": "geoprompt-utilities-workflow/1.0"})
    with urlopen(req, timeout=timeout_seconds) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def _read_or_fetch_cache(name: str, url: str, *, allow_live: bool) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.json"
    if cache_path.exists() and not allow_live:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if allow_live:
        try:
            payload = _fetch_json(url)
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload
        except (URLError, TimeoutError, ValueError):
            if cache_path.exists():
                return json.loads(cache_path.read_text(encoding="utf-8"))

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    return {}


def fetch_overpass_network(*, bbox: tuple[float, float, float, float], allow_live: bool = False) -> dict[str, Any]:
    min_lat, min_lon, max_lat, max_lon = bbox
    query = (
        "[out:json];"
        f"way['highway']({min_lat},{min_lon},{max_lat},{max_lon});"
        "(._;>;);out body;"
    )
    encoded = urlencode({"data": query})
    url = f"https://overpass-api.de/api/interpreter?{encoded}"
    return _read_or_fetch_cache("overpass_network", url, allow_live=allow_live)


def fetch_census_acs_county(*, state_fips: str, county_fips: str, allow_live: bool = False) -> dict[str, Any]:
    api_key = os.getenv("CENSUS_API_KEY", "")
    params = {
        "get": "NAME,B01001_001E",
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}",
    }
    if api_key:
        params["key"] = api_key
    url = f"https://api.census.gov/data/2022/acs/acs5?{urlencode(params)}"
    payload = _read_or_fetch_cache("census_acs", url, allow_live=allow_live)
    if isinstance(payload, list) and len(payload) >= 2:
        header = payload[0]
        row = payload[1]
        return dict(zip(header, row))
    return payload if isinstance(payload, dict) else {}


def fetch_noaa_forecast(*, latitude: float, longitude: float, allow_live: bool = False) -> dict[str, Any]:
    point_url = f"https://api.weather.gov/points/{latitude},{longitude}"
    point_payload = _read_or_fetch_cache("noaa_point", point_url, allow_live=allow_live)
    forecast_url = (
        point_payload.get("properties", {}).get("forecast")
        if isinstance(point_payload, dict)
        else None
    )
    if not forecast_url:
        return point_payload if isinstance(point_payload, dict) else {}
    return _read_or_fetch_cache("noaa_forecast", forecast_url, allow_live=allow_live)


def _network_edges_from_overpass(payload: dict[str, Any]) -> list[dict[str, Any]]:
    elements = payload.get("elements", []) if isinstance(payload, dict) else []
    node_lookup: dict[int, tuple[float, float]] = {}
    way_nodes: list[list[int]] = []
    for element in elements:
        if element.get("type") == "node":
            node_lookup[int(element["id"])] = (float(element["lon"]), float(element["lat"]))
        elif element.get("type") == "way":
            nodes = [int(node_id) for node_id in element.get("nodes", [])]
            if len(nodes) > 1:
                way_nodes.append(nodes)

    edges: list[dict[str, Any]] = []
    edge_id = 1
    for node_seq in way_nodes:
        for left, right in zip(node_seq[:-1], node_seq[1:]):
            if left in node_lookup and right in node_lookup:
                lx, ly = node_lookup[left]
                rx, ry = node_lookup[right]
                cost = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
                edges.append(
                    {
                        "edge_id": f"e{edge_id}",
                        "from_node": str(left),
                        "to_node": str(right),
                        "cost": round(max(cost, 0.0001), 6),
                    }
                )
                edge_id += 1
    return edges


def _fallback_edges() -> list[dict[str, Any]]:
    return [
        {"edge_id": "e1", "from_node": "SRC", "to_node": "A", "cost": 1.0},
        {"edge_id": "e2", "from_node": "A", "to_node": "B", "cost": 1.0},
        {"edge_id": "e3", "from_node": "B", "to_node": "C", "cost": 1.0},
        {"edge_id": "e4", "from_node": "C", "to_node": "D", "cost": 1.0},
    ]


def run_simple_track(*, allow_live_api: bool = False) -> dict[str, Any]:
    overpass = fetch_overpass_network(bbox=(40.70, -111.95, 40.80, -111.80), allow_live=allow_live_api)
    census = fetch_census_acs_county(state_fips="49", county_fips="035", allow_live=allow_live_api)
    noaa = fetch_noaa_forecast(latitude=40.7608, longitude=-111.8910, allow_live=allow_live_api)

    edges = _network_edges_from_overpass(overpass)
    if not edges:
        edges = _fallback_edges()

    graph = build_network_graph(edges, directed=False)
    nodes = sorted(graph.adjacency.keys())
    incidents = [(float(i), 0.0) for i, _ in enumerate(nodes[:4])]
    facilities = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    nearest = closest_facility(incidents, facilities, n_closest=1)

    origin = nodes[:1] if nodes else ["SRC"]
    area = service_area(graph, origins=origin, max_cost=2.5)

    payload = {
        "track": "simple",
        "network_edge_count": len(edges),
        "nearest_facility_pairs": nearest,
        "service_area": area,
        "census": census,
        "noaa_period_count": len(noaa.get("properties", {}).get("periods", [])) if isinstance(noaa, dict) else 0,
    }
    return payload


def run_complex_track(*, allow_live_api: bool = False, monte_carlo_runs: int = 20) -> dict[str, Any]:
    overpass = fetch_overpass_network(bbox=(40.70, -111.95, 40.80, -111.80), allow_live=allow_live_api)
    edges = _network_edges_from_overpass(overpass)
    if not edges:
        edges = _fallback_edges()

    graph = build_network_graph(edges, directed=False)
    nodes = sorted(graph.adjacency.keys())
    if not nodes:
        return {"track": "complex", "error": "no_nodes"}

    source_nodes = [nodes[0]]
    demand_by_node = {node: float(10 + idx * 2) for idx, node in enumerate(nodes)}
    critical_nodes = nodes[1:3]

    random.seed(42)
    outage_rows: list[dict[str, Any]] = []
    edge_ids = [str(edge.get("edge_id")) for edge in edges if edge.get("edge_id")]
    for run in range(max(1, monte_carlo_runs)):
        failed = [random.choice(edge_ids)] if edge_ids else []
        outage = outage_impact_report(
            graph,
            source_nodes=source_nodes,
            failed_edges=failed,
            demand_by_node=demand_by_node,
            critical_nodes=critical_nodes,
            outage_hours=2.0,
        )
        outage_rows.append({"run": run, **outage})

    avg_cost = sum(float(row["estimated_cost"]) for row in outage_rows) / len(outage_rows)
    base_restoration = restoration_sequence_report(
        graph,
        source_nodes=source_nodes,
        failed_edges=edge_ids[:2],
        demand_by_node=demand_by_node,
        critical_nodes=critical_nodes,
    )

    baseline_summary = build_resilience_summary_report(
        [{"node": node, "single_source_dependency": True, "resilience_tier": "medium"} for node in nodes],
        outage_report={
            "impacted_node_count": int(sum(row["impacted_node_count"] for row in outage_rows) / len(outage_rows)),
            "estimated_cost": avg_cost,
            "severity_tier": "high" if avg_cost > 100 else "medium",
        },
        restoration_report=base_restoration,
        metadata={"scenario_id": "d1-utilities-baseline"},
    )
    improved_summary = build_resilience_summary_report(
        [{"node": node, "single_source_dependency": False, "resilience_tier": "high"} for node in nodes],
        outage_report={
            "impacted_node_count": max(0, int(sum(row["impacted_node_count"] for row in outage_rows) / len(outage_rows)) - 1),
            "estimated_cost": max(0.0, avg_cost * 0.7),
            "severity_tier": "medium",
        },
        restoration_report=base_restoration,
        metadata={"scenario_id": "d1-utilities-mitigation"},
    )

    portfolio = build_resilience_portfolio_report({"baseline": baseline_summary, "mitigation": improved_summary})
    return {
        "track": "complex",
        "monte_carlo_runs": len(outage_rows),
        "avg_estimated_outage_cost": round(avg_cost, 2),
        "portfolio": portfolio,
        "restoration_steps": base_restoration.get("total_steps", 0),
    }


def main() -> None:
    allow_live = os.getenv("GEOPROMPT_ALLOW_LIVE_API", "0") == "1"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    simple = run_simple_track(allow_live_api=allow_live)
    complex_payload = run_complex_track(allow_live_api=allow_live)

    simple_path = OUTPUT_DIR / "d1-utilities-simple.json"
    complex_path = OUTPUT_DIR / "d1-utilities-complex.json"
    simple_path.write_text(json.dumps(simple, indent=2), encoding="utf-8")
    complex_path.write_text(json.dumps(complex_payload, indent=2), encoding="utf-8")

    summary = build_resilience_summary_report(
        [{"node": "portfolio", "single_source_dependency": False, "resilience_tier": "high"}],
        outage_report={
            "impacted_node_count": 0,
            "estimated_cost": float(complex_payload.get("avg_estimated_outage_cost", 0.0)),
            "severity_tier": "low",
        },
        restoration_report={"total_steps": int(complex_payload.get("restoration_steps", 0)), "stages": []},
        metadata={"scenario_id": "d1-utilities-summary"},
    )
    html_path = export_resilience_summary_report(summary, OUTPUT_DIR / "d1-utilities-summary.html")
    portfolio_path = export_resilience_portfolio_report(
        complex_payload.get("portfolio", {"portfolio": []}),
        OUTPUT_DIR / "d1-utilities-portfolio.html",
    )

    print(f"Wrote {simple_path}")
    print(f"Wrote {complex_path}")
    print(f"Wrote {html_path}")
    print(f"Wrote {portfolio_path}")


if __name__ == "__main__":
    main()
