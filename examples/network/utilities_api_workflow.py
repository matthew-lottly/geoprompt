from __future__ import annotations

import csv
import json
import math
import os
import random
from heapq import heappop, heappush
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
OVERPASS_BBOX = (40.70, -111.95, 40.80, -111.80)
NOAA_POINT = (40.7608, -111.8910)
ACS_STATE_FIPS = "49"
ACS_COUNTY_FIPS = "035"


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


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _write_simple_svg_bar(path: Path, title: str, rows: list[tuple[str, float]], color: str) -> str:
    width = 880
    height = max(280, 80 + len(rows) * 40)
    pad_left = 230
    pad_top = 50
    bar_h = 22
    gap = 14
    max_value = max([value for _, value in rows], default=1.0)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; fill: #1f2937; } .title { font-size: 20px; font-weight: 700; }</style>',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="20" y="30" class="title">{title}</text>',
    ]

    for idx, (label, value) in enumerate(rows):
        y = pad_top + idx * (bar_h + gap)
        bar_w = int((value / max(max_value, 1e-9)) * (width - pad_left - 60))
        lines.append(f'<text x="18" y="{y + 15}">{label}</text>')
        lines.append(f'<rect x="{pad_left}" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{color}" opacity="0.9"/>')
        lines.append(f'<text x="{pad_left + bar_w + 8}" y="{y + 15}">{value:.2f}</text>')

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _dijkstra_service_area(edges: list[dict[str, Any]], origins: list[str], max_cost: float) -> list[dict[str, Any]]:
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for edge in edges:
        left = str(edge["from_node"])
        right = str(edge["to_node"])
        cost = float(edge["cost"])
        adjacency.setdefault(left, []).append((right, cost))
        adjacency.setdefault(right, []).append((left, cost))

    rows: list[dict[str, Any]] = []
    for origin in origins:
        pq: list[tuple[float, str]] = [(0.0, origin)]
        best: dict[str, float] = {origin: 0.0}
        while pq:
            dist, node = heappop(pq)
            if dist > best.get(node, float("inf")):
                continue
            for nxt, weight in adjacency.get(node, []):
                cand = dist + weight
                if cand < best.get(nxt, float("inf")):
                    best[nxt] = cand
                    heappush(pq, (cand, nxt))

        for node, dist in best.items():
            if dist <= max_cost:
                rows.append(
                    {
                        "node": node,
                        "assigned_origin": origin,
                        "cost": round(float(dist), 6),
                        "within_service_area": True,
                    }
                )

    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["node"])
        old = dedup.get(key)
        if old is None or float(row["cost"]) < float(old["cost"]):
            dedup[key] = row
    return sorted(dedup.values(), key=lambda item: (float(item["cost"]), str(item["node"])))


def _reference_closest_facility(incidents: list[tuple[float, float]], facilities: list[tuple[float, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for incident_idx, (ix, iy) in enumerate(incidents):
        scored: list[tuple[float, int]] = []
        for facility_idx, (fx, fy) in enumerate(facilities):
            dist = math.dist((ix, iy), (fx, fy))
            scored.append((dist, facility_idx))
        scored.sort(key=lambda item: (item[0], item[1]))
        best_dist, best_facility = scored[0]
        rows.append(
            {
                "incident_index": incident_idx,
                "facility_index": best_facility,
                "distance": round(float(best_dist), 6),
                "rank": 1,
            }
        )
    return rows


def _normalize_pairs(rows: list[dict[str, Any]]) -> list[tuple[int, int, float]]:
    return sorted(
        (int(row["incident_index"]), int(row["facility_index"]), round(float(row["distance"]), 6))
        for row in rows
    )


def _normalize_service(rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
    return sorted((str(row["node"]), round(float(row["cost"]), 6)) for row in rows)


def _service_area_geojson(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        # Deterministic pseudo-geometry for visualization if real node coordinates are not available.
        x = float(idx)
        y = float(row.get("cost", 0.0))
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "node": row.get("node"),
                    "assigned_origin": row.get("assigned_origin"),
                    "cost": float(row.get("cost", 0.0)),
                    "within_service_area": bool(row.get("within_service_area", False)),
                },
                "geometry": {"type": "Point", "coordinates": [x, y]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def fetch_utility_outage_feed(*, allow_live: bool = False) -> dict[str, Any]:
    url = os.getenv("UTILITY_OUTAGE_API_URL", "").strip()
    if not url:
        return {}
    return _read_or_fetch_cache("utility_outage_feed", url, allow_live=allow_live)


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
    overpass = fetch_overpass_network(bbox=OVERPASS_BBOX, allow_live=allow_live_api)
    census = fetch_census_acs_county(state_fips=ACS_STATE_FIPS, county_fips=ACS_COUNTY_FIPS, allow_live=allow_live_api)
    noaa = fetch_noaa_forecast(latitude=NOAA_POINT[0], longitude=NOAA_POINT[1], allow_live=allow_live_api)
    outage_feed = fetch_utility_outage_feed(allow_live=allow_live_api)

    edges = _network_edges_from_overpass(overpass)
    if not edges:
        edges = _fallback_edges()

    graph = build_network_graph(edges, directed=False)
    nodes = sorted(graph.adjacency.keys())
    incidents = [(float(i), 0.0) for i, _ in enumerate(nodes[:4])]
    facilities = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    nearest = closest_facility(incidents, facilities, n_closest=1)
    reference_nearest = _reference_closest_facility(incidents, facilities)

    origin = nodes[:1] if nodes else ["SRC"]
    area = service_area(graph, origins=origin, max_cost=2.5)
    reference_area = _dijkstra_service_area(edges, origin, max_cost=2.5)

    nearest_match = _normalize_pairs(nearest) == _normalize_pairs(reference_nearest)
    service_match = _normalize_service(area) == _normalize_service(reference_area)

    stress_rows: list[dict[str, Any]] = []
    base_load = float(census.get("B01001_001E", 50000)) if isinstance(census, dict) else 50000.0
    for idx, node in enumerate(nodes):
        stress = round((base_load / max(len(nodes), 1)) / 10000.0 + idx * 0.07, 4)
        stress_rows.append({"node": node, "stress_index": stress})

    payload = {
        "track": "simple",
        "network_edge_count": len(edges),
        "api_sources": {
            "overpass_has_elements": bool(overpass.get("elements")) if isinstance(overpass, dict) else False,
            "census_available": bool(census),
            "noaa_available": bool(noaa),
            "utility_outage_feed_available": bool(outage_feed),
        },
        "nearest_facility_pairs": nearest,
        "reference_nearest_facility_pairs": reference_nearest,
        "nearest_facility_match": nearest_match,
        "service_area": area,
        "reference_service_area": reference_area,
        "service_area_match": service_match,
        "stress_index": stress_rows,
        "census": census,
        "noaa_period_count": len(noaa.get("properties", {}).get("periods", [])) if isinstance(noaa, dict) else 0,
        "outage_feed_record_count": len(outage_feed.get("outages", [])) if isinstance(outage_feed, dict) else 0,
    }
    return payload


def run_complex_track(*, allow_live_api: bool = False, monte_carlo_runs: int = 20) -> dict[str, Any]:
    overpass = fetch_overpass_network(bbox=OVERPASS_BBOX, allow_live=allow_live_api)
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
    avg_unmet = sum(float(row["impacted_demand"]) for row in outage_rows) / len(outage_rows)
    reliability_trend = [
        {
            "run": int(row["run"]),
            "estimated_cost": float(row["estimated_cost"]),
            "severity_score": float(row["severity_score"]),
            "impacted_demand": float(row["impacted_demand"]),
            "reliability_index": round(max(0.0, 1.0 - min(float(row["severity_score"]) / 500.0, 1.0)), 4),
        }
        for row in outage_rows
    ]

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

    critical_nodes_table = [
        {
            "node": node,
            "demand": float(demand_by_node.get(node, 0.0)),
            "is_critical": node in set(critical_nodes),
        }
        for node in nodes
    ]
    scenario_rows = sorted(
        list(portfolio.get("scenarios", [])),
        key=lambda row: float(row.get("resilience_score", 0.0)),
        reverse=True,
    )
    equity_impact_rows = [
        {
            "scenario": row.get("scenario_name", "unknown"),
            "impacted_customer_count": int(row.get("impacted_customer_count", 0)),
            "estimated_cost": float(row.get("estimated_cost", 0.0)),
            "equity_gap_index": round(float(row.get("estimated_cost", 0.0)) / max(float(row.get("resilience_score", 1.0)), 1.0), 4),
        }
        for row in scenario_rows
    ]

    return {
        "track": "complex",
        "monte_carlo_runs": len(outage_rows),
        "avg_estimated_outage_cost": round(avg_cost, 2),
        "avg_unmet_demand": round(avg_unmet, 2),
        "portfolio": portfolio,
        "restoration_steps": base_restoration.get("total_steps", 0),
        "restoration_progression": list(base_restoration.get("stages", [])),
        "reliability_trend": reliability_trend,
        "critical_nodes": critical_nodes_table,
        "scenario_ranking": scenario_rows,
        "equity_impact": equity_impact_rows,
    }


def _build_executive_html(simple: dict[str, Any], complex_payload: dict[str, Any]) -> str:
    scenario_rows = complex_payload.get("scenario_ranking", [])
    scenario_html = "\n".join(
        f"<tr><td>{row.get('scenario_name')}</td><td>{float(row.get('resilience_score', 0.0)):.4f}</td><td>{float(row.get('estimated_cost', 0.0)):.2f}</td></tr>"
        for row in scenario_rows
    )
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'><head><meta charset='utf-8'><title>D1 Utilities Executive Report</title>",
            "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1f2937;} h1,h2{margin:0 0 10px;} table{border-collapse:collapse;width:100%;max-width:860px;} th,td{border:1px solid #d1d5db;padding:8px;text-align:left;} th{background:#f3f4f6;} .kpi{display:inline-block;margin-right:20px;padding:10px;background:#eef2ff;border-radius:8px;}</style>",
            "</head><body>",
            "<h1>D1 Utilities Workflow Executive Report</h1>",
            f"<div class='kpi'><strong>Network edges:</strong> {int(simple.get('network_edge_count', 0))}</div>",
            f"<div class='kpi'><strong>Avg outage cost:</strong> {float(complex_payload.get('avg_estimated_outage_cost', 0.0)):.2f}</div>",
            f"<div class='kpi'><strong>Monte Carlo runs:</strong> {int(complex_payload.get('monte_carlo_runs', 0))}</div>",
            "<h2>Simple Track Comparison</h2>",
            f"<p>Nearest-facility parity: <strong>{bool(simple.get('nearest_facility_match', False))}</strong> | Service-area parity: <strong>{bool(simple.get('service_area_match', False))}</strong></p>",
            "<h2>Scenario Ranking</h2>",
            "<table><thead><tr><th>Scenario</th><th>Resilience Score</th><th>Estimated Cost</th></tr></thead><tbody>",
            scenario_html,
            "</tbody></table>",
            "</body></html>",
        ]
    )


def generate_d1_artifacts(simple: dict[str, Any], complex_payload: dict[str, Any], output_dir: Path | None = None) -> dict[str, str]:
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    paths["simple_json"] = _write_json(out / "d1-utilities-simple.json", simple)
    paths["complex_json"] = _write_json(out / "d1-utilities-complex.json", complex_payload)

    comparison = {
        "workflow": "d1-utilities",
        "simple_track": {
            "nearest_facility_match": bool(simple.get("nearest_facility_match", False)),
            "service_area_match": bool(simple.get("service_area_match", False)),
            "reference_engine": "python_stdlib_reference",
        },
    }
    paths["comparison_json"] = _write_json(out / "d1-utilities-comparison.json", comparison)

    summary_rows = [
        {
            "metric": "network_edge_count",
            "value": int(simple.get("network_edge_count", 0)),
        },
        {
            "metric": "avg_estimated_outage_cost",
            "value": float(complex_payload.get("avg_estimated_outage_cost", 0.0)),
        },
        {
            "metric": "avg_unmet_demand",
            "value": float(complex_payload.get("avg_unmet_demand", 0.0)),
        },
        {
            "metric": "monte_carlo_runs",
            "value": int(complex_payload.get("monte_carlo_runs", 0)),
        },
    ]
    paths["summary_csv"] = _write_csv(out / "d1-utilities-summary-table.csv", summary_rows, ["metric", "value"])
    paths["critical_nodes_csv"] = _write_csv(
        out / "d1-utilities-critical-nodes.csv",
        list(complex_payload.get("critical_nodes", [])),
        ["node", "demand", "is_critical"],
    )
    paths["scenario_ranking_csv"] = _write_csv(
        out / "d1-utilities-scenario-ranking.csv",
        list(complex_payload.get("scenario_ranking", [])),
        [
            "scenario_name",
            "resilience_score",
            "severity_tier",
            "high_resilience_nodes",
            "low_resilience_nodes",
            "critical_single_source_nodes",
            "impacted_customer_count",
            "estimated_cost",
            "restoration_steps",
            "final_restored_demand",
        ],
    )
    paths["equity_impact_csv"] = _write_csv(
        out / "d1-utilities-equity-impact.csv",
        list(complex_payload.get("equity_impact", [])),
        ["scenario", "impacted_customer_count", "estimated_cost", "equity_gap_index"],
    )

    restoration_rows = [
        (f"step-{int(stage.get('step', 0))}", float(stage.get("cumulative_restored_demand", 0.0)))
        for stage in complex_payload.get("restoration_progression", [])
    ]
    if not restoration_rows:
        restoration_rows = [("step-0", 0.0)]
    paths["restoration_svg"] = _write_simple_svg_bar(
        out / "d1-utilities-restoration-progression.svg",
        "Restoration Progression",
        restoration_rows,
        color="#0f766e",
    )

    unmet_rows = [
        (f"run-{int(row.get('run', 0))}", float(row.get("impacted_demand", 0.0)))
        for row in complex_payload.get("reliability_trend", [])
    ]
    paths["unmet_demand_svg"] = _write_simple_svg_bar(
        out / "d1-utilities-unmet-demand.svg",
        "Unmet Demand Trend",
        unmet_rows,
        color="#dc2626",
    )

    reliability_rows = [
        (f"run-{int(row.get('run', 0))}", float(row.get("reliability_index", 0.0)))
        for row in complex_payload.get("reliability_trend", [])
    ]
    paths["reliability_svg"] = _write_simple_svg_bar(
        out / "d1-utilities-reliability-trend.svg",
        "Reliability Trend",
        reliability_rows,
        color="#2563eb",
    )

    geojson = _service_area_geojson(list(simple.get("service_area", [])))
    paths["service_area_geojson"] = _write_json(out / "d1-utilities-service-area-map.geojson", geojson)

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
    paths["summary_html"] = export_resilience_summary_report(summary, out / "d1-utilities-summary.html")
    paths["portfolio_html"] = export_resilience_portfolio_report(
        complex_payload.get("portfolio", {"portfolio": []}),
        out / "d1-utilities-portfolio.html",
    )

    executive_html = _build_executive_html(simple, complex_payload)
    executive_path = out / "d1-utilities-executive-report.html"
    executive_path.write_text(executive_html, encoding="utf-8")
    paths["executive_html"] = str(executive_path)
    return paths


def main() -> None:
    allow_live = os.getenv("GEOPROMPT_ALLOW_LIVE_API", "0") == "1"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    simple = run_simple_track(allow_live_api=allow_live)
    complex_payload = run_complex_track(allow_live_api=allow_live)
    artifacts = generate_d1_artifacts(simple, complex_payload, output_dir=OUTPUT_DIR)
    for _, path in artifacts.items():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
