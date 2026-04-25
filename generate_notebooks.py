r"""
Generate all geoprompt/ and geopandas/ workflow notebooks.
Run from project root: .venv-1\Scripts\python.exe generate_notebooks.py
"""
from __future__ import annotations
import json
from pathlib import Path

try:
    import nbformat
    import nbformat.v4 as nbv4
except ImportError:
    raise SystemExit("pip install nbformat first")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def code(src: str) -> dict:
    return nbv4.new_code_cell(src)

def md(src: str) -> dict:
    return nbv4.new_markdown_cell(src)

def save(cells, path: str) -> None:
    nb = nbv4.new_notebook()
    nb.cells = cells
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"  wrote {p}")


# ============================================================================
# GEOPROMPT NOTEBOOKS
# ============================================================================

# ---------------------------------------------------------------------------
# GP-D1  Utilities Workflow
# ---------------------------------------------------------------------------
GP_D1_IMPORTS = '''\
from __future__ import annotations
import json, os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from typing import cast
import matplotlib.pyplot as plt

OUTPUT_DIR = Path.cwd() / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALLOW_LIVE_API = os.getenv("GEOPROMPT_ALLOW_LIVE_API", "1") == "1"

def fetch_json(url, fallback):
    if not ALLOW_LIVE_API:
        return fallback
    try:
        req = Request(url, headers={"User-Agent": "geoprompt-notebook/2.0"})
        with urlopen(req, timeout=6) as r:
            return json.loads(r.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError):
        return fallback

def fetch_first_json(urls, validator, fallback):
    for url in urls:
        payload = fetch_json(url, None)
        if payload is not None and validator(payload):
            return payload, url, True
    return fallback, "fallback", False

import geoprompt as gp
from geoprompt import GeoPromptFrame, write_geojson
from geoprompt.network.core import NetworkEdge
from geoprompt.network.routing import build_network_graph, shortest_path, service_area
from geoprompt.tools import build_scenario_report, export_scenario_report
print("Imports OK")
'''

GP_D1_SECTION_A = '''\
grid = {
    "grid_id": "utility-grid",
    "features": [{"id": "fallback-grid"}],
}
weather = {"properties": {"forecast": "fallback"}}
forecast = {"hourly": {"temperature_2m": [0.0]}}

grid, grid_src, grid_live = fetch_first_json(
    [
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
        "https://api.github.com/repos/geopandas/geopandas",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("features") or d.get("id")),
    grid,
)
weather, wx_src, wx_live = fetch_first_json(
    [
        "https://api.weather.gov/points/41.88,-87.63",
        "https://api.weather.gov/points/40.76,-111.89",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("properties", {}).get("forecast")),
    weather,
)
forecast, fc_src, fc_live = fetch_first_json(
    [
        "https://api.open-meteo.com/v1/forecast?latitude=41.88&longitude=-87.63&hourly=temperature_2m&forecast_days=1",
        "https://api.open-meteo.com/v1/forecast?latitude=40.76&longitude=-111.89&hourly=temperature_2m&forecast_days=1",
    ],
    lambda d: isinstance(d, dict) and len(d.get("hourly", {}).get("temperature_2m", [])) > 0,
    forecast,
)

grid_count = len(grid.get("features", [])) if isinstance(grid, dict) else 0
if grid_count == 0 and isinstance(grid, dict) and grid.get("id"):
    grid_count = 1
print(f"Grid records: {grid_count} | live={grid_live} | source={grid_src}")
print(f"Weather forecast pointer exists: {bool(weather.get('properties', {}).get('forecast'))} | live={wx_live} | source={wx_src}")
print(f"Open-Meteo hourly points: {len(forecast.get('hourly', {}).get('temperature_2m', []))} | live={fc_live} | source={fc_src}")
'''

GP_D1_SECTION_B = '''\
# Node coordinates (lon, lat) for visualization.
NODE_COORDS = {
    "PLANT": (-87.70, 41.94), "SUB1": (-87.63, 41.88), "SUB2": (-87.72, 41.83),
    "SUB3": (-87.65, 41.79), "SUB4": (-87.58, 41.85), "SUB5": (-87.76, 41.91),
    "SUB6": (-87.60, 41.75),
}

edges: list[NetworkEdge] = [
    cast(NetworkEdge, {"edge_id": "e1", "from_node": "PLANT", "to_node": "SUB1", "cost": 3.0}),
    cast(NetworkEdge, {"edge_id": "e2", "from_node": "PLANT", "to_node": "SUB5", "cost": 4.0}),
    cast(NetworkEdge, {"edge_id": "e3", "from_node": "SUB1", "to_node": "SUB2", "cost": 5.0}),
    cast(NetworkEdge, {"edge_id": "e4", "from_node": "SUB1", "to_node": "SUB4", "cost": 4.0}),
    cast(NetworkEdge, {"edge_id": "e5", "from_node": "SUB2", "to_node": "SUB3", "cost": 6.0}),
    cast(NetworkEdge, {"edge_id": "e6", "from_node": "SUB4", "to_node": "SUB3", "cost": 3.0}),
    cast(NetworkEdge, {"edge_id": "e7", "from_node": "SUB3", "to_node": "SUB6", "cost": 2.0}),
    cast(NetworkEdge, {"edge_id": "e8", "from_node": "SUB5", "to_node": "SUB2", "cost": 5.0}),
]

# 1. Build network graph
graph = build_network_graph(edges, directed=False)
print("Nodes in graph:", sorted(graph.nodes))

# 2. Shortest path  PLANT -> SUB6
path_result = shortest_path(graph, "PLANT", "SUB6")
path_nodes: list[str] = path_result.get("path_nodes", [])
print(f"Shortest path PLANT->SUB6: {' -> '.join(path_nodes)}  cost={path_result.get('total_cost')}")

# 3. Service area from PLANT (max_cost=10)
area = service_area(graph, origins=["PLANT"], max_cost=10.0)
print(f"Service area (max_cost=10): {len(area)} reachable nodes")

# 4. Build GeoPromptFrame from service area nodes
node_rows = [
    {
        "node": str(r["node"]),
        "cost": float(r["cost"]),
        "geometry": {"type": "Point", "coordinates": list(NODE_COORDS.get(str(r["node"]), (-87.65, 41.85)))},
    }
    for r in area
]
nodes_frame = GeoPromptFrame(node_rows, geometry_column="geometry", crs="EPSG:4326")
print(f"GeoPromptFrame: {len(nodes_frame)} nodes")

# 5. Nearest neighbors (haversine)
neighbors = nodes_frame.nearest_neighbors(id_column="node", k=1, distance_method="haversine")
print("\\nNearest neighbor pairs (haversine km):")
for nb in neighbors:
    print(f"  {nb['origin']} -> {nb['neighbor']}  dist={nb['distance']:.4f}")

# 6. Query radius - nodes within 0.12 deg of SUB1
nearby = nodes_frame.query_radius("SUB1", max_distance=0.12, id_column="node")
print(f"\\nNodes within 0.12 deg of SUB1: {[r['node'] for r in nearby.to_records()]}")

# 7. Buffer - service zones around substations
buffered = nodes_frame.buffer(0.04)
print(f"Buffer zones created: {len(buffered)} polygons")

# 8. Query bounds - northeast quadrant
ne_nodes = nodes_frame.query_bounds(-87.72, 41.83, -87.58, 41.95)
print(f"Nodes in NE region: {[r['node'] for r in ne_nodes.to_records()]}")

# Write GeoJSON output
write_geojson(OUTPUT_DIR / "d1-gp-nodes.geojson", nodes_frame)
print("\\nFrame summary:")
print(json.dumps(nodes_frame.summary(), indent=2, default=str))

# Inline visualization
records = nodes_frame.to_records()
lons = [float(r["geometry"]["coordinates"][0]) for r in records]
lats = [float(r["geometry"]["coordinates"][1]) for r in records]
costs = [float(r["cost"]) for r in records]
labels = [str(r["node"]) for r in records]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc = axes[0].scatter(lons, lats, c=costs, cmap="RdYlGn_r", s=200, edgecolors="#333", zorder=5)
for lon, lat, lbl, cost in zip(lons, lats, labels, costs):
    axes[0].annotate(f"{lbl}\\n({cost:.0f})", (lon, lat), textcoords="offset points", xytext=(4, 4), fontsize=8)
for e in edges:
    fn, tn = str(e["from_node"]), str(e["to_node"])
    if fn in NODE_COORDS and tn in NODE_COORDS:
        axes[0].plot([NODE_COORDS[fn][0], NODE_COORDS[tn][0]], [NODE_COORDS[fn][1], NODE_COORDS[tn][1]],
                     "b-", alpha=0.3, linewidth=1.5)
if len(path_nodes) >= 2:
    for i in range(len(path_nodes)-1):
        a, b_ = path_nodes[i], path_nodes[i+1]
        if a in NODE_COORDS and b_ in NODE_COORDS:
            axes[0].plot([NODE_COORDS[a][0], NODE_COORDS[b_][0]], [NODE_COORDS[a][1], NODE_COORDS[b_][1]],
                         "r-", linewidth=2.5, zorder=4)
plt.colorbar(sc, ax=axes[0], label="Routing Cost")
axes[0].set_title("Utility Network: Service Area (red=shortest path)")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
axes[0].grid(True, alpha=0.3)

sorted_recs = sorted(records, key=lambda r: float(r["cost"]))
axes[1].barh([r["node"] for r in sorted_recs], [float(r["cost"]) for r in sorted_recs], color="#2563eb")
axes[1].set_xlabel("Routing Cost from PLANT")
axes[1].set_title("Node Reachability Costs")
axes[1].grid(True, axis="x", alpha=0.3)
plt.suptitle("D1 Utilities: GeoPromptFrame Network Analysis", fontweight="bold")
plt.tight_layout(); plt.show()
'''

GP_D1_SECTION_C = '''\
scenarios = {
    "baseline":  {"reliability": 0.88, "outage_hrs_yr": 12.0, "cost_musd": 0.0},
    "hardened":  {"reliability": 0.94, "outage_hrs_yr": 5.0,  "cost_musd": 28.0},
    "redundant": {"reliability": 0.97, "outage_hrs_yr": 2.5,  "cost_musd": 55.0},
}
report = build_scenario_report(scenarios["baseline"], scenarios["hardened"], higher_is_better=["reliability"])
report_path = export_scenario_report(report, OUTPUT_DIR / "d1-gp-scenario-report.json")
print("Scenario report:", report_path)

scenario_records = []
for name, vals in scenarios.items():
    score = round(vals["reliability"] * 0.5 + (1.0 / max(vals["outage_hrs_yr"], 0.1)) * 3 * 0.3
                  + (1.0 / max(vals["cost_musd"] + 1, 1)) * 20 * 0.2, 4)
    scenario_records.append({"scenario": name, **vals, "score": score})
scenario_records.sort(key=lambda r: -float(r["score"]))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
names = [r["scenario"] for r in scenario_records]
colors = ["#27ae60", "#e67e22", "#c0392b"]
axes[0].barh(names, [r["reliability"] for r in scenario_records], color=colors)
axes[0].set_xlabel("Reliability"); axes[0].set_title("Network Reliability"); axes[0].grid(True, axis="x", alpha=0.3)
axes[1].barh(names, [r["outage_hrs_yr"] for r in scenario_records], color=colors)
axes[1].set_xlabel("Outage Hours/Year"); axes[1].set_title("Annual Outage Hours"); axes[1].grid(True, axis="x", alpha=0.3)
axes[2].barh(names, [r["score"] for r in scenario_records], color=colors)
axes[2].set_xlabel("Composite Score"); axes[2].set_title("Scenario Score (higher=better)"); axes[2].grid(True, axis="x", alpha=0.3)
plt.suptitle("D1 Utilities: Scenario Comparison", fontweight="bold")
plt.tight_layout(); plt.show()

(OUTPUT_DIR / "d1-gp-complex.json").write_text(
    json.dumps({"scenario_ranking": scenario_records}, indent=2, default=str), encoding="utf-8"
)
print("Wrote d1-gp-complex.json")
'''


# ---------------------------------------------------------------------------
# GP-D2  Forestry Management
# ---------------------------------------------------------------------------
GP_D2_IMPORTS = '''\
from __future__ import annotations
import json, os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
import matplotlib.pyplot as plt

OUTPUT_DIR = Path.cwd() / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALLOW_LIVE_API = os.getenv("GEOPROMPT_ALLOW_LIVE_API", "1") == "1"

def fetch_json(url, fallback):
    if not ALLOW_LIVE_API:
        return fallback
    try:
        req = Request(url, headers={"User-Agent": "geoprompt-notebook/2.0"})
        with urlopen(req, timeout=6) as r:
            return json.loads(r.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError):
        return fallback

def fetch_first_json(urls, validator, fallback):
    for url in urls:
        payload = fetch_json(url, None)
        if payload is not None and validator(payload):
            return payload, url, True
    return fallback, "fallback", False

import geoprompt as gp
from geoprompt import GeoPromptFrame, write_geojson
from geoprompt.tools import build_scenario_report, export_scenario_report
print("Imports OK")
'''

GP_D2_SECTION_A = '''\
forest = {"features": [{"id": "fallback-forest"}]}
weather = {"properties": {"forecast": "fallback"}}
forecast = {"hourly": {"temperature_2m": [0.0]}}

forest, forest_src, forest_live = fetch_first_json(
    [
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
        "https://api.github.com/repos/opengeospatial/geopandas",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("features") or d.get("id")),
    forest,
)
weather, wx_src, wx_live = fetch_first_json(
    [
        "https://api.weather.gov/points/44.05,-121.31",
        "https://api.weather.gov/points/45.52,-122.67",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("properties", {}).get("forecast")),
    weather,
)
forecast, fc_src, fc_live = fetch_first_json(
    [
        "https://api.open-meteo.com/v1/forecast?latitude=44.05&longitude=-121.31&hourly=temperature_2m&forecast_days=1",
        "https://api.open-meteo.com/v1/forecast?latitude=45.52&longitude=-122.67&hourly=temperature_2m&forecast_days=1",
    ],
    lambda d: isinstance(d, dict) and len(d.get("hourly", {}).get("temperature_2m", [])) > 0,
    forecast,
)

forest_count = len(forest.get("features", [])) if isinstance(forest, dict) else 0
if forest_count == 0 and isinstance(forest, dict) and forest.get("id"):
    forest_count = 1
print(f"Forestry records: {forest_count} | live={forest_live} | source={forest_src}")
print(f"NOAA forecast exists: {bool(weather.get('properties', {}).get('forecast'))} | live={wx_live} | source={wx_src}")
print(f"Open-Meteo hourly points: {len(forecast.get('hourly', {}).get('temperature_2m', []))} | live={fc_live} | source={fc_src}")
'''

GP_D2_SECTION_B = '''\
RAW_STANDS = [
    {"stand_id": "S1", "fuel_load": 0.72, "slope": 0.35, "dist_km": 0.8,  "zone": "A",
     "geometry": {"type": "Point", "coordinates": [-121.60, 44.30]}},
    {"stand_id": "S2", "fuel_load": 0.63, "slope": 0.25, "dist_km": 1.4,  "zone": "A",
     "geometry": {"type": "Point", "coordinates": [-121.20, 44.10]}},
    {"stand_id": "S3", "fuel_load": 0.81, "slope": 0.41, "dist_km": 0.6,  "zone": "B",
     "geometry": {"type": "Point", "coordinates": [-121.00, 44.40]}},
    {"stand_id": "S4", "fuel_load": 0.77, "slope": 0.39, "dist_km": 0.9,  "zone": "B",
     "geometry": {"type": "Point", "coordinates": [-120.80, 44.20]}},
    {"stand_id": "S5", "fuel_load": 0.58, "slope": 0.21, "dist_km": 1.8,  "zone": "A",
     "geometry": {"type": "Point", "coordinates": [-121.40, 43.90]}},
    {"stand_id": "S6", "fuel_load": 0.88, "slope": 0.45, "dist_km": 0.4,  "zone": "B",
     "geometry": {"type": "Point", "coordinates": [-120.95, 44.50]}},
]
FIRE_STATIONS = [
    {"station_id": "FS1", "geometry": {"type": "Point", "coordinates": [-121.35, 44.20]}},
    {"station_id": "FS2", "geometry": {"type": "Point", "coordinates": [-120.90, 44.10]}},
]

enriched = []
for row in RAW_STANDS:
    fuel = float(row["fuel_load"]); slope = float(row["slope"]); dist = float(row["dist_km"])
    risk = round(fuel * 0.5 + slope * 0.3 + (1.0 / dist) * 0.2, 4)
    risk_tier = "HIGH" if risk > 0.75 else ("MED" if risk > 0.55 else "LOW")
    enriched.append({**row, "risk_score": risk, "risk_tier": risk_tier})
enriched.sort(key=lambda r: -r["risk_score"])
for rank, row in enumerate(enriched, 1):
    row["priority_rank"] = rank

# Build GeoPromptFrame
stands_frame = GeoPromptFrame(enriched, geometry_column="geometry", crs="EPSG:4326")
stations_frame = GeoPromptFrame(FIRE_STATIONS, geometry_column="geometry", crs="EPSG:4326")

# 1. Nearest neighbors between stands
neighbors = stands_frame.nearest_neighbors(id_column="stand_id", k=2, distance_method="haversine")
print("Nearest neighbors (top 3 pairs):")
for nb in neighbors[:3]:
    print(f"  {nb['origin']} -> {nb['neighbor']}  dist={nb['distance']:.4f}")

# 2. Query radius: stands near S3 (high-risk)
near_s3 = stands_frame.query_radius("S3", max_distance=0.5, id_column="stand_id")
print(f"\\nStands within 0.5 deg of S3: {[r['stand_id'] for r in near_s3.to_records()]}")

# 3. Buffer protection zones (0.15 degree radius)
buffers = stands_frame.buffer(0.15)
print(f"Buffer zones: {len(buffers)} polygons")

# 4. Proximity join: stands within 0.6 deg of fire stations
pj = stands_frame.proximity_join(stations_frame, max_distance=0.6, how="left")
assigned = [(r["stand_id"], r.get("station_id")) for r in pj.to_records()]
print(f"\\nProximity join results (stand -> nearest station): {assigned[:4]}")

# 5. Dissolve by risk tier: aggregate risk scores
dissolved = stands_frame.dissolve(by="risk_tier", aggregations={"risk_score": "mean", "fuel_load": "mean"})
print("\\nDissolved by risk tier:")
for row in dissolved.to_records():
    print(f"  {row['risk_tier']}: mean_risk={row.get('risk_score_mean_risk_tier',row.get('risk_score','?')):.3f}")

# 6. Query bounds: high-risk southern zone
south_stands = stands_frame.query_bounds(-121.65, 43.85, -120.75, 44.15)
print(f"\\nStands in southern zone: {[r['stand_id'] for r in south_stands.to_records()]}")

# Write GeoJSON
write_geojson(OUTPUT_DIR / "d2-gp-stands.geojson", stands_frame)
print("\\nFrame summary:")
print(json.dumps(stands_frame.summary(), indent=2, default=str))

# Inline visualization
records = stands_frame.to_records()
lons = [float(r["geometry"]["coordinates"][0]) for r in records]
lats = [float(r["geometry"]["coordinates"][1]) for r in records]
risks = [float(r["risk_score"]) for r in records]
labels = [str(r["stand_id"]) for r in records]
ranks = [int(r["priority_rank"]) for r in records]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc = axes[0].scatter(lons, lats, c=risks, cmap="YlOrRd", s=180, edgecolors="#333", zorder=5)
for lon, lat, lbl, rank in zip(lons, lats, labels, ranks):
    axes[0].annotate(f"{lbl}\\n(#{rank})", (lon, lat), textcoords="offset points", xytext=(5, 5), fontsize=8)
slons = [float(r["geometry"]["coordinates"][0]) for r in FIRE_STATIONS]
slats = [float(r["geometry"]["coordinates"][1]) for r in FIRE_STATIONS]
axes[0].scatter(slons, slats, c="blue", s=220, marker="^", zorder=6, label="Fire Station")
plt.colorbar(sc, ax=axes[0], label="Risk Score")
axes[0].set_title("Stand Risk Map (triangle=fire station)")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

sorted_recs = sorted(records, key=lambda r: -float(r["risk_score"]))
bar_colors = ["#c0392b" if r["risk_tier"]=="HIGH" else ("#e67e22" if r["risk_tier"]=="MED" else "#27ae60")
              for r in sorted_recs]
axes[1].barh([r["stand_id"] for r in sorted_recs], [float(r["risk_score"]) for r in sorted_recs], color=bar_colors)
axes[1].set_xlabel("Fire Risk Score")
axes[1].set_title("Stand Priority (red=HIGH, orange=MED, green=LOW)")
axes[1].grid(True, axis="x", alpha=0.3)
plt.suptitle("D2 Forestry: GeoPromptFrame Spatial Analysis", fontweight="bold")
plt.tight_layout(); plt.show()
'''

GP_D2_SECTION_C = '''\
scenarios = {
    "no_action":       {"risk": 0.74, "cost_musd": 0.0,  "expected_loss_musd": 240.0},
    "thinning":        {"risk": 0.51, "cost_musd": 65.0, "expected_loss_musd": 170.0},
    "prescribed_burn": {"risk": 0.43, "cost_musd": 92.0, "expected_loss_musd": 135.0},
}
report = build_scenario_report(scenarios["no_action"], scenarios["prescribed_burn"], higher_is_better=[])
report_path = export_scenario_report(report, OUTPUT_DIR / "d2-gp-scenario-report.json")
print("Scenario report:", report_path)

scenario_records = []
for name, vals in scenarios.items():
    score = round((1 - vals["risk"]) * 0.5 + (1.0 / max(vals["cost_musd"] + 1, 1)) * 10 * 0.2
                  + (1.0 / vals["expected_loss_musd"]) * 50 * 0.3, 4)
    scenario_records.append({"scenario": name, **vals, "score": score})
scenario_records.sort(key=lambda r: -float(r["score"]))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
names = [r["scenario"] for r in scenario_records]
colors = ["#27ae60", "#e67e22", "#c0392b"]
axes[0].barh(names, [r["risk"] for r in scenario_records], color=colors)
axes[0].set_xlabel("Fire Risk"); axes[0].set_title("Risk by Scenario"); axes[0].grid(True, axis="x", alpha=0.3)
axes[1].barh(names, [r["expected_loss_musd"] for r in scenario_records], color=colors)
axes[1].set_xlabel("Expected Loss ($M USD)"); axes[1].set_title("Expected Loss"); axes[1].grid(True, axis="x", alpha=0.3)
axes[2].barh(names, [r["score"] for r in scenario_records], color=colors)
axes[2].set_xlabel("Composite Score"); axes[2].set_title("Composite Score (higher=better)"); axes[2].grid(True, axis="x", alpha=0.3)
plt.suptitle("D2 Forestry: Scenario Comparison", fontweight="bold")
plt.tight_layout(); plt.show()

(OUTPUT_DIR / "d2-gp-complex.json").write_text(
    json.dumps({"scenario_ranking": scenario_records}, indent=2, default=str), encoding="utf-8"
)
print("Wrote d2-gp-complex.json")
'''


# ---------------------------------------------------------------------------
# GP-D3  Flood Analysis (raster heavy)
# ---------------------------------------------------------------------------
GP_D3_IMPORTS = '''\
from __future__ import annotations
import json, os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
import matplotlib.pyplot as plt

OUTPUT_DIR = Path.cwd() / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALLOW_LIVE_API = os.getenv("GEOPROMPT_ALLOW_LIVE_API", "1") == "1"

def fetch_json(url, fallback):
    if not ALLOW_LIVE_API:
        return fallback
    try:
        req = Request(url, headers={"User-Agent": "geoprompt-notebook/2.0"})
        with urlopen(req, timeout=6) as r:
            return json.loads(r.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError):
        return fallback

def fetch_first_json(urls, validator, fallback):
    for url in urls:
        payload = fetch_json(url, None)
        if payload is not None and validator(payload):
            return payload, url, True
    return fallback, "fallback", False

import geoprompt as gp
from geoprompt import GeoPromptFrame, write_geojson
from geoprompt.raster import raster_slope_aspect, raster_hillshade, raster_algebra, sample_raster_points
from geoprompt.tools import build_scenario_report, export_scenario_report
print("Imports OK")
'''

GP_D3_SECTION_A = '''\
flood = {"features": [{"id": "fallback-flood"}]}
weather = {"properties": {"forecast": "fallback"}}
forecast = {"hourly": {"temperature_2m": [0.0]}}

flood, flood_src, flood_live = fetch_first_json(
    [
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
        "https://api.github.com/repos/OSGeo/gdal",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("features") or d.get("id")),
    flood,
)
weather, wx_src, wx_live = fetch_first_json(
    [
        "https://api.weather.gov/points/29.76,-95.37",
        "https://api.weather.gov/points/30.27,-97.74",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("properties", {}).get("forecast")),
    weather,
)
forecast, fc_src, fc_live = fetch_first_json(
    [
        "https://api.open-meteo.com/v1/forecast?latitude=29.76&longitude=-95.37&hourly=temperature_2m&forecast_days=1",
        "https://api.open-meteo.com/v1/forecast?latitude=30.27&longitude=-97.74&hourly=temperature_2m&forecast_days=1",
    ],
    lambda d: isinstance(d, dict) and len(d.get("hourly", {}).get("temperature_2m", [])) > 0,
    forecast,
)

flood_count = len(flood.get("features", [])) if isinstance(flood, dict) else 0
if flood_count == 0 and isinstance(flood, dict) and flood.get("id"):
    flood_count = 1
print(f"Flood records: {flood_count} | live={flood_live} | source={flood_src}")
print(f"NOAA forecast exists: {bool(weather.get('properties', {}).get('forecast'))} | live={wx_live} | source={wx_src}")
print(f"Open-Meteo hourly points: {len(forecast.get('hourly', {}).get('temperature_2m', []))} | live={fc_live} | source={fc_src}")
'''

GP_D3_SECTION_B = '''\
RAW_ASSETS = [
    {"asset_id": "A1", "zone": "X",  "replacement_musd": 4.0, "elevation_m": 11.0,
     "geometry": {"type": "Point", "coordinates": [-95.42, 29.81]}},
    {"asset_id": "A2", "zone": "AE", "replacement_musd": 7.5, "elevation_m": 7.0,
     "geometry": {"type": "Point", "coordinates": [-95.35, 29.76]}},
    {"asset_id": "A3", "zone": "AE", "replacement_musd": 3.2, "elevation_m": 8.0,
     "geometry": {"type": "Point", "coordinates": [-95.29, 29.73]}},
    {"asset_id": "A4", "zone": "VE", "replacement_musd": 9.1, "elevation_m": 5.9,
     "geometry": {"type": "Point", "coordinates": [-95.26, 29.68]}},
    {"asset_id": "A5", "zone": "AE", "replacement_musd": 5.5, "elevation_m": 6.5,
     "geometry": {"type": "Point", "coordinates": [-95.38, 29.71]}},
]
ZONE_RISK = {"VE": 1.0, "AE": 0.8, "X": 0.2}

enriched = []
for row in RAW_ASSETS:
    zone_risk = ZONE_RISK.get(str(row["zone"]), 0.3)
    elev = float(row["elevation_m"])
    risk = round(zone_risk * 0.7 + (10.0 / elev) * 0.3, 4)
    loss = round(float(row["replacement_musd"]) * risk * 0.18, 4)
    enriched.append({**row, "flood_risk_score": risk, "expected_loss_musd": loss})

assets_frame = GeoPromptFrame(enriched, geometry_column="geometry", crs="EPSG:4326")

# 1. Nearest neighbors (haversine)
neighbors = assets_frame.nearest_neighbors(id_column="asset_id", k=1, distance_method="haversine")
print("Nearest neighbor pairs:")
for nb in neighbors:
    print(f"  {nb['origin']} -> {nb['neighbor']}  dist={nb['distance']:.5f}")

# 2. Query radius: assets near A2 (AE zone)
near_a2 = assets_frame.query_radius("A2", max_distance=0.15, id_column="asset_id")
print(f"\\nAssets within 0.15 deg of A2: {[r['asset_id'] for r in near_a2.to_records()]}")

# 3. Buffer: flood exposure buffers
buffers = assets_frame.buffer(0.08)
print(f"Buffer zones: {len(buffers)} polygons")

# Inline scatter map
records = assets_frame.to_records()
lons = [float(r["geometry"]["coordinates"][0]) for r in records]
lats = [float(r["geometry"]["coordinates"][1]) for r in records]
risks = [float(r["flood_risk_score"]) for r in records]
asset_ids = [str(r["asset_id"]) for r in records]
losses = [float(r["expected_loss_musd"]) for r in records]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc = axes[0].scatter(lons, lats, c=risks, cmap="Blues", s=200, edgecolors="#1d4ed8", zorder=5)
for lon, lat, aid, risk in zip(lons, lats, asset_ids, risks):
    axes[0].annotate(f"{aid}\\n({risk:.2f})", (lon, lat), textcoords="offset points", xytext=(5, 5), fontsize=9)
plt.colorbar(sc, ax=axes[0], label="Flood Risk Score")
axes[0].set_title("D3 Flood Exposure Map"); axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
axes[0].grid(True, alpha=0.3)
sorted_recs = sorted(records, key=lambda r: -float(r["flood_risk_score"]))
axes[1].barh([r["asset_id"] for r in sorted_recs], [float(r["flood_risk_score"]) for r in sorted_recs],
             color="#1d4ed8")
axes[1].set_xlabel("Flood Risk Score"); axes[1].set_title("Asset Risk Ranking")
axes[1].grid(True, axis="x", alpha=0.3)
plt.suptitle("D3 Flood: Asset Risk Analysis", fontweight="bold")
plt.tight_layout(); plt.show()

write_geojson(OUTPUT_DIR / "d3-gp-assets.geojson", assets_frame)
print("\\nFrame summary:")
print(json.dumps(assets_frame.summary(), indent=2, default=str))
'''

GP_D3_SECTION_C = '''\
# Define a 5x5 raster grid (same transform for algebra operations)
RASTER_TRANSFORM = (-95.50, 29.90, 0.05, 0.05)  # (min_x, max_y, cell_w, cell_h)

# Flood depth raster (simulated inundation depths in meters)
flood_depth_raster = {
    "data": [
        [0.0, 0.2, 0.5, 0.8, 1.2],
        [0.1, 0.4, 0.9, 1.5, 2.1],
        [0.2, 0.6, 1.2, 2.0, 2.8],
        [0.1, 0.3, 0.7, 1.3, 1.9],
        [0.0, 0.1, 0.3, 0.6, 1.0],
    ],
    "transform": RASTER_TRANSFORM,
}

# Elevation risk raster (higher elevation = lower risk)
elevation_raster = {
    "data": [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.5, 0.6, 0.7],
        [0.3, 0.4, 0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5],
    ],
    "transform": RASTER_TRANSFORM,
}

# 4. Raster slope/aspect analysis
slope_data = raster_slope_aspect(flood_depth_raster)
print(f"Slope grid center value: {slope_data['slope'][2][2]:.4f}")

# 5. Hillshade visualization
hillshade_data = raster_hillshade(flood_depth_raster)
print(f"Hillshade grid center value: {hillshade_data['grid'][2][2]:.2f}")

# 6. Raster algebra: combine flood depth + elevation risk (add)
combined_raster = raster_algebra(flood_depth_raster, elevation_raster, operation="add")
print(f"Combined raster center value: {combined_raster['data'][2][2]:.3f}")

# 7. Sample raster values at asset point locations
sampled_frame = sample_raster_points(flood_depth_raster, assets_frame, value_column="flood_depth_m")
print("\\nSampled flood depths at asset locations:")
for row in sampled_frame.to_records():
    print(f"  {row['asset_id']}: flood_depth_m={row['flood_depth_m']:.3f}")

# Scenario comparison
baseline   = {"expected_annual_loss": 18.0, "impacted_assets": 4, "response_time_hours": 8.0}
mitigation = {"expected_annual_loss": 10.5, "impacted_assets": 2, "response_time_hours": 6.0}
report = build_scenario_report(baseline, mitigation, higher_is_better=[])
report_path = export_scenario_report(report, OUTPUT_DIR / "d3-gp-scenario-report.json")
print("\\nScenario report:", report_path)

# Inline raster + scenario visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Flood depth raster
im0 = axes[0][0].imshow(flood_depth_raster["data"], cmap="Blues", origin="upper")
plt.colorbar(im0, ax=axes[0][0], label="Depth (m)"); axes[0][0].set_title("Flood Depth Raster")

# Hillshade
im1 = axes[0][1].imshow(hillshade_data["grid"], cmap="gray", origin="upper")
plt.colorbar(im1, ax=axes[0][1], label="Hillshade"); axes[0][1].set_title("Terrain Hillshade")

# Combined raster
im2 = axes[1][0].imshow(combined_raster["data"], cmap="RdYlBu_r", origin="upper")
plt.colorbar(im2, ax=axes[1][0], label="Combined Risk"); axes[1][0].set_title("Combined Risk (depth+elevation)")

# Scenario comparison
metrics = list(baseline.keys())
x = range(len(metrics))
width = 0.38
axes[1][1].bar([i - width/2 for i in x], [float(baseline[m]) for m in metrics], width=width, label="Baseline", color="#94a3b8")
axes[1][1].bar([i + width/2 for i in x], [float(mitigation[m]) for m in metrics], width=width, label="Mitigation", color="#2563eb")
axes[1][1].set_xticks(list(x)); axes[1][1].set_xticklabels(metrics, rotation=15)
axes[1][1].set_title("Baseline vs Mitigation"); axes[1][1].legend(); axes[1][1].grid(True, axis="y", alpha=0.3)

plt.suptitle("D3 Flood: Raster Analysis + Scenario Comparison", fontweight="bold")
plt.tight_layout(); plt.show()

(OUTPUT_DIR / "d3-gp-complex.json").write_text(
    json.dumps({"slope_center": slope_data["slope"][2][2], "hillshade_center": hillshade_data["grid"][2][2],
                "combined_center": combined_raster["data"][2][2]}, indent=2, default=str), encoding="utf-8"
)
print("Wrote d3-gp-complex.json")
'''


# ---------------------------------------------------------------------------
# GP-D4  Transportation Workflow
# ---------------------------------------------------------------------------
GP_D4_IMPORTS = '''\
from __future__ import annotations
import json, os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from typing import cast
import matplotlib.pyplot as plt

OUTPUT_DIR = Path.cwd() / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALLOW_LIVE_API = os.getenv("GEOPROMPT_ALLOW_LIVE_API", "1") == "1"

def fetch_json(url, fallback):
    if not ALLOW_LIVE_API:
        return fallback
    try:
        req = Request(url, headers={"User-Agent": "geoprompt-notebook/2.0"})
        with urlopen(req, timeout=6) as r:
            return json.loads(r.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError):
        return fallback

def fetch_first_json(urls, validator, fallback):
    for url in urls:
        payload = fetch_json(url, None)
        if payload is not None and validator(payload):
            return payload, url, True
    return fallback, "fallback", False

import geoprompt as gp
from geoprompt import GeoPromptFrame, write_geojson
from geoprompt.network.core import NetworkEdge
from geoprompt.network.routing import build_network_graph, shortest_path, service_area
from geoprompt.tools import build_scenario_report, export_scenario_report
print("Imports OK")
'''

GP_D4_SECTION_A = '''\
transport = {"features": [{"id": "fallback-transport"}]}
weather = {"properties": {"forecast": "fallback"}}
forecast = {"hourly": {"temperature_2m": [0.0]}}

transport, tr_src, tr_live = fetch_first_json(
    [
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
        "https://api.github.com/repos/osm-search/Nominatim",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("features") or d.get("id")),
    transport,
)
weather, wx_src, wx_live = fetch_first_json(
    [
        "https://api.weather.gov/points/40.75,-111.90",
        "https://api.weather.gov/points/34.05,-118.24",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("properties", {}).get("forecast")),
    weather,
)
forecast, fc_src, fc_live = fetch_first_json(
    [
        "https://api.open-meteo.com/v1/forecast?latitude=40.75&longitude=-111.90&hourly=temperature_2m&forecast_days=1",
        "https://api.open-meteo.com/v1/forecast?latitude=34.05&longitude=-118.24&hourly=temperature_2m&forecast_days=1",
    ],
    lambda d: isinstance(d, dict) and len(d.get("hourly", {}).get("temperature_2m", [])) > 0,
    forecast,
)

transport_count = len(transport.get("features", [])) if isinstance(transport, dict) else 0
if transport_count == 0 and isinstance(transport, dict) and transport.get("id"):
    transport_count = 1
print(f"Transport records: {transport_count} | live={tr_live} | source={tr_src}")
print(f"NOAA forecast exists: {bool(weather.get('properties', {}).get('forecast'))} | live={wx_live} | source={wx_src}")
print(f"Open-Meteo hourly points: {len(forecast.get('hourly', {}).get('temperature_2m', []))} | live={fc_live} | source={fc_src}")
'''

GP_D4_SECTION_B = '''\
# Schematic node coordinates for spatial visualization
NODE_COORDS = {
    "O": (-111.90, 40.75), "A": (-111.87, 40.75), "B": (-111.84, 40.75),
    "C": (-111.78, 40.75), "D": (-111.84, 40.72), "E": (-111.81, 40.72), "F": (-111.86, 40.78),
}

edges: list[NetworkEdge] = [
    cast(NetworkEdge, {"edge_id": "r1", "from_node": "O", "to_node": "A", "cost": 5.0}),
    cast(NetworkEdge, {"edge_id": "r2", "from_node": "A", "to_node": "B", "cost": 4.0}),
    cast(NetworkEdge, {"edge_id": "r3", "from_node": "B", "to_node": "C", "cost": 3.0}),
    cast(NetworkEdge, {"edge_id": "r4", "from_node": "O", "to_node": "C", "cost": 15.0}),
    cast(NetworkEdge, {"edge_id": "r5", "from_node": "B", "to_node": "D", "cost": 5.0}),
    cast(NetworkEdge, {"edge_id": "r6", "from_node": "D", "to_node": "E", "cost": 2.0}),
    cast(NetworkEdge, {"edge_id": "r7", "from_node": "E", "to_node": "C", "cost": 2.0}),
    cast(NetworkEdge, {"edge_id": "r8", "from_node": "O", "to_node": "F", "cost": 3.0}),
    cast(NetworkEdge, {"edge_id": "r9", "from_node": "F", "to_node": "A", "cost": 4.0}),
]

# 1. Build network graph
graph = build_network_graph(edges, directed=False)
print("Network nodes:", sorted(graph.nodes))

# 2. Shortest path O -> C
path_result = shortest_path(graph, "O", "C")
path_nodes: list[str] = path_result.get("path_nodes", [])
print(f"Shortest path O->C: {' -> '.join(path_nodes)}  cost={path_result.get('total_cost')}")

# 3. Service area from O (max_cost=10)
area = service_area(graph, origins=["O"], max_cost=10.0)
print(f"Service area (max_cost=10): {len(area)} reachable nodes")

# 4. Build GeoPromptFrame from service area
node_rows = [
    {
        "node": str(r["node"]),
        "cost": float(r["cost"]),
        "geometry": {"type": "Point", "coordinates": list(NODE_COORDS.get(str(r["node"]), (-111.84, 40.75)))},
    }
    for r in area
]
area_frame = GeoPromptFrame(node_rows, geometry_column="geometry")

# 5. Nearest neighbors
if len(area_frame) > 1:
    neighbors = area_frame.nearest_neighbors(id_column="node", k=1)
    print("\\nNearest neighbor pairs:")
    for nb in neighbors:
        print(f"  {nb['origin']} -> {nb['neighbor']}  dist={nb['distance']:.4f}")

# 6. Query radius: nodes within 0.08 deg of O
nearby = area_frame.query_radius("O", max_distance=0.08, id_column="node")
print(f"\\nNodes within 0.08 deg of O: {[r['node'] for r in nearby.to_records()]}")

# 7. Buffer: service zones around nodes
buffers = area_frame.buffer(0.03)
print(f"Buffer service zones: {len(buffers)} polygons")

# 8. Proximity join: connect incident points to nearest nodes
incident_rows = [
    {"inc_id": "I1", "geometry": {"type": "Point", "coordinates": [-111.88, 40.76]}},
    {"inc_id": "I2", "geometry": {"type": "Point", "coordinates": [-111.83, 40.73]}},
    {"inc_id": "I3", "geometry": {"type": "Point", "coordinates": [-111.79, 40.75]}},
]
incidents_frame = GeoPromptFrame(incident_rows, geometry_column="geometry")
pj = area_frame.proximity_join(incidents_frame, max_distance=0.08, how="left")
print(f"\\nProximity join (node -> incidents within 0.08 deg): {len(pj)} rows")

write_geojson(OUTPUT_DIR / "d4-gp-network.geojson", area_frame)
print("\\nFrame summary:")
print(json.dumps(area_frame.summary(), indent=2, default=str))

# Inline visualization
records = area_frame.to_records()
node_costs = {str(r["node"]): float(r["cost"]) for r in records}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for e in edges:
    fn, tn = str(e["from_node"]), str(e["to_node"])
    if fn in NODE_COORDS and tn in NODE_COORDS:
        axes[0].plot([NODE_COORDS[fn][0], NODE_COORDS[tn][0]], [NODE_COORDS[fn][1], NODE_COORDS[tn][1]],
                     "b-", alpha=0.3, linewidth=1.5)
# Highlight shortest path
for i in range(len(path_nodes)-1):
    a, b_ = path_nodes[i], path_nodes[i+1]
    if a in NODE_COORDS and b_ in NODE_COORDS:
        axes[0].plot([NODE_COORDS[a][0], NODE_COORDS[b_][0]], [NODE_COORDS[a][1], NODE_COORDS[b_][1]],
                     "r-", linewidth=2.5, zorder=4)
sc_lons = [NODE_COORDS[r["node"]][0] for r in records if r["node"] in NODE_COORDS]
sc_lats = [NODE_COORDS[r["node"]][1] for r in records if r["node"] in NODE_COORDS]
sc_costs = [float(r["cost"]) for r in records if r["node"] in NODE_COORDS]
sc_labels = [r["node"] for r in records if r["node"] in NODE_COORDS]
sc = axes[0].scatter(sc_lons, sc_lats, c=sc_costs, cmap="RdYlGn_r", s=180, edgecolors="#333", zorder=5)
for lon, lat, lbl in zip(sc_lons, sc_lats, sc_labels):
    axes[0].annotate(lbl, (lon, lat), textcoords="offset points", xytext=(4, 4), fontsize=9, fontweight="bold")
inc_lons = [float(r["geometry"]["coordinates"][0]) for r in incident_rows]
inc_lats = [float(r["geometry"]["coordinates"][1]) for r in incident_rows]
axes[0].scatter(inc_lons, inc_lats, c="red", s=120, marker="x", zorder=6, linewidths=2, label="Incidents")
plt.colorbar(sc, ax=axes[0], label="Routing Cost")
axes[0].set_title("Transport Network (red=shortest path, x=incidents)")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

sorted_recs = sorted(records, key=lambda r: float(r["cost"]))
axes[1].barh([r["node"] for r in sorted_recs], [float(r["cost"]) for r in sorted_recs], color="#0891b2")
axes[1].set_xlabel("Travel Cost from Origin O")
axes[1].set_title("Node Reachability Costs")
axes[1].grid(True, axis="x", alpha=0.3)
plt.suptitle("D4 Transportation: GeoPromptFrame Network Analysis", fontweight="bold")
plt.tight_layout(); plt.show()
'''

GP_D4_SECTION_C = '''\
scenarios = {
    "no_action":    {"throughput": 1000, "avg_delay_min": 18.0, "cost_musd": 0.0},
    "signal_opt":   {"throughput": 1280, "avg_delay_min": 11.0, "cost_musd": 15.0},
    "new_corridor": {"throughput": 1600, "avg_delay_min": 7.0,  "cost_musd": 85.0},
}
report = build_scenario_report(scenarios["no_action"], scenarios["new_corridor"],
                                higher_is_better=["throughput"])
report_path = export_scenario_report(report, OUTPUT_DIR / "d4-gp-scenario-report.json")
print("Scenario report:", report_path)

scenario_records = []
for name, vals in scenarios.items():
    score = round(vals["throughput"] / 1600 * 0.5 + (1.0 / vals["avg_delay_min"]) * 10 * 0.3
                  + (1.0 / max(vals["cost_musd"] + 1, 1)) * 20 * 0.2, 4)
    scenario_records.append({"scenario": name, **vals, "score": score})
scenario_records.sort(key=lambda r: -float(r["score"]))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
names = [r["scenario"] for r in scenario_records]
colors = ["#27ae60", "#e67e22", "#c0392b"]
axes[0].barh(names, [r["throughput"] for r in scenario_records], color=colors)
axes[0].set_xlabel("Throughput (veh/hr)"); axes[0].set_title("Throughput"); axes[0].grid(True, axis="x", alpha=0.3)
axes[1].barh(names, [r["avg_delay_min"] for r in scenario_records], color=colors)
axes[1].set_xlabel("Avg Delay (min)"); axes[1].set_title("Average Delay"); axes[1].grid(True, axis="x", alpha=0.3)
axes[2].barh(names, [r["score"] for r in scenario_records], color=colors)
axes[2].set_xlabel("Composite Score"); axes[2].set_title("Composite Score (higher=better)"); axes[2].grid(True, axis="x", alpha=0.3)
plt.suptitle("D4 Transportation: Scenario Comparison", fontweight="bold")
plt.tight_layout(); plt.show()

(OUTPUT_DIR / "d4-gp-complex.json").write_text(
    json.dumps({"scenario_ranking": scenario_records}, indent=2, default=str), encoding="utf-8"
)
print("Wrote d4-gp-complex.json")
'''


# ---------------------------------------------------------------------------
# GP-D5  Climate Workflow
# ---------------------------------------------------------------------------
GP_D5_IMPORTS = '''\
from __future__ import annotations
import json, os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
import matplotlib.pyplot as plt

OUTPUT_DIR = Path.cwd() / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALLOW_LIVE_API = os.getenv("GEOPROMPT_ALLOW_LIVE_API", "1") == "1"

def fetch_json(url, fallback):
    if not ALLOW_LIVE_API:
        return fallback
    try:
        req = Request(url, headers={"User-Agent": "geoprompt-notebook/2.0"})
        with urlopen(req, timeout=6) as r:
            return json.loads(r.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError):
        return fallback

def fetch_first_json(urls, validator, fallback):
    for url in urls:
        payload = fetch_json(url, None)
        if payload is not None and validator(payload):
            return payload, url, True
    return fallback, "fallback", False

import geoprompt as gp
from geoprompt import GeoPromptFrame, write_geojson
from geoprompt.raster import raster_algebra, raster_slope_aspect
from geoprompt.tools import build_scenario_report, export_scenario_report
print("Imports OK")
'''

GP_D5_SECTION_A = '''\
climate = {"features": [{"id": "fallback-climate"}]}
weather = {"properties": {"forecast": "fallback"}}
forecast = {"hourly": {"temperature_2m": [0.0]}}

climate, cl_src, cl_live = fetch_first_json(
    [
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
        "https://api.github.com/repos/ecmwf/cdsapi",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("features") or d.get("id")),
    climate,
)
weather, wx_src, wx_live = fetch_first_json(
    [
        "https://api.weather.gov/points/39.00,-98.00",
        "https://api.weather.gov/points/38.90,-77.03",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("properties", {}).get("forecast")),
    weather,
)
forecast, fc_src, fc_live = fetch_first_json(
    [
        "https://api.open-meteo.com/v1/forecast?latitude=39.00&longitude=-98.00&hourly=temperature_2m&forecast_days=1",
        "https://api.open-meteo.com/v1/forecast?latitude=38.90&longitude=-77.03&hourly=temperature_2m&forecast_days=1",
    ],
    lambda d: isinstance(d, dict) and len(d.get("hourly", {}).get("temperature_2m", [])) > 0,
    forecast,
)

climate_count = len(climate.get("features", [])) if isinstance(climate, dict) else 0
if climate_count == 0 and isinstance(climate, dict) and climate.get("id"):
    climate_count = 1
print(f"Climate records: {climate_count} | live={cl_live} | source={cl_src}")
print(f"NOAA forecast exists: {bool(weather.get('properties', {}).get('forecast'))} | live={wx_live} | source={wx_src}")
print(f"Open-Meteo hourly points: {len(forecast.get('hourly', {}).get('temperature_2m', []))} | live={fc_live} | source={fc_src}")
'''

GP_D5_SECTION_B = '''\
RAW_ZONES = [
    {"zone_id": "Z1", "heat_index": 0.78, "drought_idx": 0.65, "sea_level_risk": 0.30,
     "pop_density": 1200.0, "geometry": {"type": "Point", "coordinates": [-98.10, 39.20]}},
    {"zone_id": "Z2", "heat_index": 0.62, "drought_idx": 0.72, "sea_level_risk": 0.15,
     "pop_density":  850.0, "geometry": {"type": "Point", "coordinates": [-97.80, 38.90]}},
    {"zone_id": "Z3", "heat_index": 0.85, "drought_idx": 0.55, "sea_level_risk": 0.80,
     "pop_density": 3400.0, "geometry": {"type": "Point", "coordinates": [-97.50, 39.40]}},
    {"zone_id": "Z4", "heat_index": 0.71, "drought_idx": 0.80, "sea_level_risk": 0.20,
     "pop_density":  550.0, "geometry": {"type": "Point", "coordinates": [-98.40, 38.70]}},
    {"zone_id": "Z5", "heat_index": 0.90, "drought_idx": 0.45, "sea_level_risk": 0.60,
     "pop_density": 2100.0, "geometry": {"type": "Point", "coordinates": [-97.65, 39.60]}},
]
ADAPTATION_SITES = [
    {"site_id": "AS1", "capacity_mw": 120, "geometry": {"type": "Point", "coordinates": [-97.90, 39.10]}},
    {"site_id": "AS2", "capacity_mw":  90, "geometry": {"type": "Point", "coordinates": [-98.20, 39.30]}},
]

enriched = []
for row in RAW_ZONES:
    risk = round(float(row["heat_index"]) * 0.35 + float(row["drought_idx"]) * 0.35
                 + float(row["sea_level_risk"]) * 0.30, 4)
    risk_tier = "HIGH" if risk > 0.65 else ("MED" if risk > 0.45 else "LOW")
    enriched.append({**row, "composite_risk": risk, "risk_tier": risk_tier})

zones_frame = GeoPromptFrame(enriched, geometry_column="geometry", crs="EPSG:4326")
sites_frame = GeoPromptFrame(ADAPTATION_SITES, geometry_column="geometry", crs="EPSG:4326")

# 1. Nearest neighbors between climate zones
neighbors = zones_frame.nearest_neighbors(id_column="zone_id", k=1, distance_method="haversine")
print("Nearest zone pairs:")
for nb in neighbors:
    print(f"  {nb['origin']} -> {nb['neighbor']}  dist={nb['distance']:.4f}")

# 2. Query radius: zones near Z3 (highest risk)
near_z3 = zones_frame.query_radius("Z3", max_distance=0.6, id_column="zone_id")
print(f"\\nZones within 0.6 deg of Z3: {[r['zone_id'] for r in near_z3.to_records()]}")

# 3. Buffer: climate impact zones
buffers = zones_frame.buffer(0.2)
print(f"Climate buffer zones: {len(buffers)} polygons")

# 4. Proximity join: zones to adaptation sites within 0.7 deg
pj = zones_frame.proximity_join(sites_frame, max_distance=0.7, how="left")
print(f"\\nZone->site proximity assignments: {len(pj)} rows")
for row in pj.to_records()[:3]:
    print(f"  {row['zone_id']} -> {row.get('site_id', 'none')}  dist={row.get('distance_right', 'n/a')}")

# 5. Dissolve by risk tier
dissolved = zones_frame.dissolve(by="risk_tier", aggregations={"composite_risk": "mean", "pop_density": "sum"})
print("\\nDissolved by risk tier:")
for row in dissolved.to_records():
    print(f"  {row['risk_tier']}: zones dissolved")

# 6. Raster algebra: combine heat + drought rasters
RASTER_TRANSFORM = (-98.60, 39.80, 0.10, 0.10)
heat_raster = {
    "data": [[0.7, 0.8, 0.9, 0.8, 0.7], [0.6, 0.75, 0.85, 0.78, 0.65],
             [0.5, 0.65, 0.78, 0.72, 0.60], [0.45, 0.60, 0.70, 0.65, 0.55],
             [0.40, 0.55, 0.65, 0.60, 0.50]],
    "transform": RASTER_TRANSFORM,
}
drought_raster = {
    "data": [[0.5, 0.6, 0.7, 0.65, 0.55], [0.55, 0.7, 0.8, 0.75, 0.60],
             [0.60, 0.75, 0.85, 0.78, 0.65], [0.55, 0.70, 0.78, 0.72, 0.60],
             [0.50, 0.62, 0.70, 0.65, 0.55]],
    "transform": RASTER_TRANSFORM,
}
combined_risk_raster = raster_algebra(heat_raster, drought_raster, operation="add")
print(f"\\nCombined climate risk raster center: {combined_risk_raster['data'][2][2]:.3f}")

# 7. Query bounds: filter high-risk northern zone
north_zones = zones_frame.query_bounds(-98.50, 39.30, -97.40, 39.70)
print(f"Zones in northern band: {[r['zone_id'] for r in north_zones.to_records()]}")

write_geojson(OUTPUT_DIR / "d5-gp-zones.geojson", zones_frame)
print("\\nFrame summary:")
print(json.dumps(zones_frame.summary(), indent=2, default=str))

# Inline visualization
records = zones_frame.to_records()
lons = [float(r["geometry"]["coordinates"][0]) for r in records]
lats = [float(r["geometry"]["coordinates"][1]) for r in records]
risks = [float(r["composite_risk"]) for r in records]
labels = [str(r["zone_id"]) for r in records]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc = axes[0].scatter(lons, lats, c=risks, cmap="YlOrRd", s=200, edgecolors="#333", zorder=5)
for lon, lat, lbl, risk in zip(lons, lats, labels, risks):
    axes[0].annotate(f"{lbl}\\n({risk:.2f})", (lon, lat), textcoords="offset points", xytext=(5, 5), fontsize=9)
slons = [float(r["geometry"]["coordinates"][0]) for r in ADAPTATION_SITES]
slats = [float(r["geometry"]["coordinates"][1]) for r in ADAPTATION_SITES]
axes[0].scatter(slons, slats, c="blue", s=220, marker="*", zorder=6, label="Adaptation Site")
plt.colorbar(sc, ax=axes[0], label="Composite Climate Risk")
axes[0].set_title("Climate Risk Map (star=adaptation site)")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

im = axes[1].imshow(combined_risk_raster["data"], cmap="RdYlGn_r", origin="upper")
plt.colorbar(im, ax=axes[1], label="Combined Risk (heat+drought)")
axes[1].set_title("Combined Climate Risk Raster")
plt.suptitle("D5 Climate: GeoPromptFrame + Raster Analysis", fontweight="bold")
plt.tight_layout(); plt.show()
'''

GP_D5_SECTION_C = '''\
baseline   = {"annual_loss_musd": 58.0, "resilience_index": 0.44, "service_reliability": 0.82}
adaptation = {"annual_loss_musd": 32.0, "resilience_index": 0.68, "service_reliability": 0.91}
report = build_scenario_report(baseline, adaptation, higher_is_better=["resilience_index", "service_reliability"])
report_path = export_scenario_report(report, OUTPUT_DIR / "d5-gp-scenario-report.json")
print("Scenario report:", report_path)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
metrics = list(baseline.keys())
x = range(len(metrics))
width = 0.38
axes[0].bar([i - width/2 for i in x], [float(baseline[m]) for m in metrics], width=width, label="Baseline", color="#94a3b8")
axes[0].bar([i + width/2 for i in x], [float(adaptation[m]) for m in metrics], width=width, label="Adaptation", color="#2563eb")
axes[0].set_xticks(list(x)); axes[0].set_xticklabels(metrics, rotation=15)
axes[0].set_title("Baseline vs Adaptation"); axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.3)

delta = [round((adaptation[m] - baseline[m]) / abs(baseline[m]) * 100, 1) for m in metrics]
bar_colors = ["#27ae60" if d > 0 else "#c0392b" for d in delta]
axes[1].barh(metrics, delta, color=bar_colors)
axes[1].axvline(0, color="#555", linewidth=1)
axes[1].set_xlabel("% Change vs Baseline"); axes[1].set_title("Improvement (positive=better)")
axes[1].grid(True, axis="x", alpha=0.3)
plt.suptitle("D5 Climate: Adaptation Scenario Analysis", fontweight="bold")
plt.tight_layout(); plt.show()

(OUTPUT_DIR / "d5-gp-complex.json").write_text(
    json.dumps({"baseline": baseline, "adaptation": adaptation}, indent=2, default=str), encoding="utf-8"
)
print("Wrote d5-gp-complex.json")
'''


# ============================================================================
# GEOPANDAS NOTEBOOKS
# ============================================================================

GPD_IMPORTS_TEMPLATE = '''\
from __future__ import annotations
import json, os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path.cwd() / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALLOW_LIVE_API = os.getenv("GEOPROMPT_ALLOW_LIVE_API", "1") == "1"

def fetch_json(url, fallback):
    if not ALLOW_LIVE_API:
        return fallback
    try:
        req = Request(url, headers={"User-Agent": "geoprompt-notebook/2.0"})
        with urlopen(req, timeout=6) as r:
            return json.loads(r.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError):
        return fallback

def fetch_first_json(urls, validator, fallback):
    for url in urls:
        payload = fetch_json(url, None)
        if payload is not None and validator(payload):
            return payload, url, True
    return fallback, "fallback", False

import geopandas as gpd
from shapely.geometry import Point, Polygon, box
print(f"geopandas {gpd.__version__} ready")
'''

# ---------------------------------------------------------------------------
# GPD-D1  Utilities (geopandas)
# ---------------------------------------------------------------------------
GPD_D1_SECTION_A = '''\
grid = {"features": [{"id": "fallback-grid"}]}
weather = {"properties": {"forecast": "fallback"}}
forecast = {"hourly": {"temperature_2m": [0.0]}}

grid, grid_src, grid_live = fetch_first_json(
    [
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
        "https://api.github.com/repos/geopandas/geopandas",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("features") or d.get("id")),
    grid,
)
weather, wx_src, wx_live = fetch_first_json(
    [
        "https://api.weather.gov/points/41.88,-87.63",
        "https://api.weather.gov/points/40.76,-111.89",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("properties", {}).get("forecast")),
    weather,
)
forecast, fc_src, fc_live = fetch_first_json(
    [
        "https://api.open-meteo.com/v1/forecast?latitude=41.88&longitude=-87.63&hourly=temperature_2m&forecast_days=1",
        "https://api.open-meteo.com/v1/forecast?latitude=40.76&longitude=-111.89&hourly=temperature_2m&forecast_days=1",
    ],
    lambda d: isinstance(d, dict) and len(d.get("hourly", {}).get("temperature_2m", [])) > 0,
    forecast,
)

grid_count = len(grid.get("features", [])) if isinstance(grid, dict) else 0
if grid_count == 0 and isinstance(grid, dict) and grid.get("id"):
    grid_count = 1
print(f"Grid records: {grid_count} | live={grid_live} | source={grid_src}")
print(f"Weather forecast pointer exists: {bool(weather.get('properties', {}).get('forecast'))} | live={wx_live} | source={wx_src}")
print(f"Open-Meteo hourly points: {len(forecast.get('hourly', {}).get('temperature_2m', []))} | live={fc_live} | source={fc_src}")
'''

GPD_D1_SECTION_B = '''\
# Utility substation data
nodes_data = {
    "node":   ["PLANT", "SUB1", "SUB2", "SUB3", "SUB4", "SUB5", "SUB6"],
    "cost":   [0.0,     3.0,    8.0,    11.0,   7.0,    4.0,    13.0],
    "cap_mw": [500,     120,    90,      80,    110,    100,     70],
    "type":   ["plant", "sub",  "sub",   "sub",  "sub",  "sub",  "sub"],
}
lons = [-87.70, -87.63, -87.72, -87.65, -87.58, -87.76, -87.60]
lats = [ 41.94,  41.88,  41.83,  41.79,  41.85,  41.91,  41.75]

# 1. Build GeoDataFrame
gdf = gpd.GeoDataFrame(nodes_data, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
print("GeoDataFrame:")
print(gdf[["node", "cost", "cap_mw"]].to_string(index=False))

# 2. Buffer: service zones around substations
gdf_projected = gdf.to_crs("EPSG:3857")
gdf_projected["buffer_geom"] = gdf_projected.buffer(5000)  # 5 km buffer
print(f"\\nBuffer zones (5 km): {len(gdf_projected)} polygons")

# 3. Bounding box filter: cx selector for northeast quadrant
ne_gdf = gdf.cx[-87.72:-87.58, 41.83:41.95]
print(f"\\nNodes in NE quadrant: {list(ne_gdf['node'])}")

# 4. Nearest join: find closest substation to each demand point
demand_data = {"demand_id": ["D1","D2","D3"], "load_mw": [45, 70, 30]}
demand_lons = [-87.66, -87.75, -87.61]
demand_lats = [ 41.87,  41.86,  41.81]
demand_gdf = gpd.GeoDataFrame(demand_data, geometry=gpd.points_from_xy(demand_lons, demand_lats), crs="EPSG:4326")
nearest = gpd.sjoin_nearest(demand_gdf, gdf, how="left", distance_col="dist_deg")
print("\\nNearest substation assignments:")
print(nearest[["demand_id", "node", "dist_deg"]].to_string(index=False))

# 5. Spatial join: demand points within substation buffers
gdf_buf = gpd.GeoDataFrame(gdf[["node"]], geometry=gdf.to_crs("EPSG:3857").buffer(8000).to_crs("EPSG:4326"))
sj = gpd.sjoin(demand_gdf, gdf_buf[["node","geometry"]], how="left", predicate="within")
print("Demand within buffer zones:")
print(sj[["demand_id", "node"]].to_string(index=False))

# 6. Dissolve by node type: aggregate capacity
dissolved = gdf.dissolve(by="type", aggfunc={"cap_mw": "sum", "cost": "mean"})
print("\\nDissolved by type:")
print(dissolved[["cap_mw", "cost"]].to_string())

# 7. High-cost nodes filter
high_cost = gdf[gdf["cost"] > 7.0]
print(f"\\nHigh-cost nodes (cost > 7): {list(high_cost['node'])}")

# Save to file
gdf.to_file(str(OUTPUT_DIR / "d1-gpd-nodes.geojson"), driver="GeoJSON")
print("\\nWrote d1-gpd-nodes.geojson")

# Inline visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
gdf.plot(ax=axes[0], column="cost", cmap="RdYlGn_r", markersize=120, legend=True,
         legend_kwds={"label": "Routing Cost"})
for _, row in gdf.iterrows():
    axes[0].annotate(row["node"], (row.geometry.x, row.geometry.y),
                     textcoords="offset points", xytext=(4, 4), fontsize=8)
demand_gdf.plot(ax=axes[0], color="blue", markersize=80, marker="D", label="Demand")
axes[0].set_title("Utility Nodes (diamonds=demand points)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

gdf_sorted = gdf.sort_values("cost")
axes[1].barh(gdf_sorted["node"], gdf_sorted["cost"], color="#2563eb")
axes[1].set_xlabel("Routing Cost"); axes[1].set_title("Node Costs")
axes[1].grid(True, axis="x", alpha=0.3)
plt.suptitle("D1 Utilities: GeoPandas Spatial Analysis", fontweight="bold")
plt.tight_layout(); plt.show()
'''

GPD_D1_SECTION_C = '''\
scenarios = {
    "baseline":  {"reliability": 0.88, "outage_hrs_yr": 12.0, "cost_musd": 0.0},
    "hardened":  {"reliability": 0.94, "outage_hrs_yr":  5.0, "cost_musd": 28.0},
    "redundant": {"reliability": 0.97, "outage_hrs_yr":  2.5, "cost_musd": 55.0},
}
scenario_records = []
for name, vals in scenarios.items():
    score = round(vals["reliability"] * 0.5 + (1.0 / max(vals["outage_hrs_yr"], 0.1)) * 3 * 0.3
                  + (1.0 / max(vals["cost_musd"] + 1, 1)) * 20 * 0.2, 4)
    scenario_records.append({"scenario": name, **vals, "score": score})
scenario_records.sort(key=lambda r: -float(r["score"]))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
names = [r["scenario"] for r in scenario_records]
colors = ["#27ae60", "#e67e22", "#c0392b"]
axes[0].barh(names, [r["reliability"] for r in scenario_records], color=colors)
axes[0].set_xlabel("Reliability"); axes[0].set_title("Network Reliability"); axes[0].grid(True, axis="x", alpha=0.3)
axes[1].barh(names, [r["outage_hrs_yr"] for r in scenario_records], color=colors)
axes[1].set_xlabel("Outage Hours/Year"); axes[1].set_title("Annual Outages"); axes[1].grid(True, axis="x", alpha=0.3)
axes[2].barh(names, [r["score"] for r in scenario_records], color=colors)
axes[2].set_xlabel("Composite Score"); axes[2].set_title("Scenario Score"); axes[2].grid(True, axis="x", alpha=0.3)
plt.suptitle("D1 Utilities (GeoPandas): Scenario Comparison", fontweight="bold")
plt.tight_layout(); plt.show()

(OUTPUT_DIR / "d1-gpd-complex.json").write_text(
    json.dumps({"scenario_ranking": scenario_records}, indent=2, default=str), encoding="utf-8"
)
print("Wrote d1-gpd-complex.json")
for r in scenario_records:
    print(f"  {r['scenario']}: score={r['score']}")
'''


# ---------------------------------------------------------------------------
# GPD-D2  Forestry (geopandas)
# ---------------------------------------------------------------------------
GPD_D2_SECTION_A = '''\
forest = {"features": [{"id": "fallback-forest"}]}
weather = {"properties": {"forecast": "fallback"}}
forecast = {"hourly": {"temperature_2m": [0.0]}}

forest, forest_src, forest_live = fetch_first_json(
    [
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
        "https://api.github.com/repos/opengeospatial/geopandas",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("features") or d.get("id")),
    forest,
)
weather, wx_src, wx_live = fetch_first_json(
    [
        "https://api.weather.gov/points/44.05,-121.31",
        "https://api.weather.gov/points/45.52,-122.67",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("properties", {}).get("forecast")),
    weather,
)
forecast, fc_src, fc_live = fetch_first_json(
    [
        "https://api.open-meteo.com/v1/forecast?latitude=44.05&longitude=-121.31&hourly=temperature_2m&forecast_days=1",
        "https://api.open-meteo.com/v1/forecast?latitude=45.52&longitude=-122.67&hourly=temperature_2m&forecast_days=1",
    ],
    lambda d: isinstance(d, dict) and len(d.get("hourly", {}).get("temperature_2m", [])) > 0,
    forecast,
)

forest_count = len(forest.get("features", [])) if isinstance(forest, dict) else 0
if forest_count == 0 and isinstance(forest, dict) and forest.get("id"):
    forest_count = 1
print(f"Forestry records: {forest_count} | live={forest_live} | source={forest_src}")
print(f"NOAA forecast exists: {bool(weather.get('properties', {}).get('forecast'))} | live={wx_live} | source={wx_src}")
print(f"Open-Meteo hourly points: {len(forecast.get('hourly', {}).get('temperature_2m', []))} | live={fc_live} | source={fc_src}")
'''

GPD_D2_SECTION_B = '''\
stands_data = {
    "stand_id": ["S1","S2","S3","S4","S5","S6"],
    "fuel_load": [0.72, 0.63, 0.81, 0.77, 0.58, 0.88],
    "slope":     [0.35, 0.25, 0.41, 0.39, 0.21, 0.45],
    "dist_km":   [0.8,  1.4,  0.6,  0.9,  1.8,  0.4],
    "zone":      ["A",  "A",  "B",  "B",  "A",  "B"],
}
lons = [-121.60, -121.20, -121.00, -120.80, -121.40, -120.95]
lats = [  44.30,   44.10,   44.40,   44.20,   43.90,   44.50]

gdf = gpd.GeoDataFrame(stands_data, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
gdf["risk_score"] = (gdf["fuel_load"] * 0.5 + gdf["slope"] * 0.3
                     + (1.0 / gdf["dist_km"]) * 0.2).round(4)
gdf["risk_tier"] = gdf["risk_score"].apply(lambda r: "HIGH" if r > 0.75 else ("MED" if r > 0.55 else "LOW"))
gdf = gdf.sort_values("risk_score", ascending=False).reset_index(drop=True)
gdf["priority_rank"] = range(1, len(gdf) + 1)
print("Stand risk scores:")
print(gdf[["stand_id","risk_score","risk_tier","priority_rank"]].to_string(index=False))

# Fire station GeoDataFrame
stations_gdf = gpd.GeoDataFrame(
    {"station_id": ["FS1","FS2"]},
    geometry=gpd.points_from_xy([-121.35, -120.90], [44.20, 44.10]),
    crs="EPSG:4326",
)

# 1. Buffer: protection zones around stands (project to meters first)
gdf_proj = gdf.to_crs("EPSG:3857")
buf_zones = gdf_proj.buffer(8000).to_crs("EPSG:4326")
print(f"Buffer zones (8 km): {len(buf_zones)} polygons")

# 2. Nearest join: assign each stand to nearest fire station
nearest_sta = gpd.sjoin_nearest(gdf, stations_gdf, how="left", distance_col="dist_to_station")
print("\\nNearest fire station per stand:")
print(nearest_sta[["stand_id","station_id","dist_to_station"]].head(4).to_string(index=False))

# 3. Spatial join: stands within station buffer radius
sta_buf = stations_gdf.to_crs("EPSG:3857").copy()
sta_buf.geometry = sta_buf.buffer(50000).to_crs("EPSG:4326")  # 50km buffer
in_range = gpd.sjoin(gdf, sta_buf, how="inner", predicate="within")
print(f"\\nStands within 50km of any fire station: {len(in_range)}")

# 4. Dissolve: aggregate by risk tier
dissolved = gdf.dissolve(by="risk_tier", aggfunc={"risk_score": "mean", "fuel_load": "mean"})
print("\\nDissolved by risk tier:")
print(dissolved[["risk_score","fuel_load"]].to_string())

# 5. Bounding box filter: southern zone
south_gdf = gdf.cx[-121.65:-120.75, 43.85:44.15]
print(f"\\nStands in southern zone: {list(south_gdf['stand_id'])}")

# 6. High risk overlay: clip to risk zone bbox
risk_bbox = box(-121.10, 44.10, -120.75, 44.55)
risk_mask = gpd.GeoDataFrame({"id": [1]}, geometry=[risk_bbox], crs="EPSG:4326")
clipped = gpd.clip(gdf, risk_mask)
print(f"Stands clipped to high-risk bbox: {list(clipped['stand_id'])}")

# 7. To file
gdf.to_file(str(OUTPUT_DIR / "d2-gpd-stands.geojson"), driver="GeoJSON")
print("\\nWrote d2-gpd-stands.geojson")

# Inline visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
gdf.plot(ax=axes[0], column="risk_score", cmap="YlOrRd", markersize=140, legend=True,
         legend_kwds={"label": "Risk Score"})
for _, row in gdf.iterrows():
    axes[0].annotate(f"{row['stand_id']}\\n#{row['priority_rank']}", (row.geometry.x, row.geometry.y),
                     textcoords="offset points", xytext=(4,4), fontsize=8)
stations_gdf.plot(ax=axes[0], color="blue", markersize=180, marker="^", label="Fire Station")
axes[0].set_title("Stand Risk Map (triangles=fire stations)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

gdf_s = gdf.sort_values("risk_score", ascending=False)
bar_colors = ["#c0392b" if t=="HIGH" else ("#e67e22" if t=="MED" else "#27ae60") for t in gdf_s["risk_tier"]]
axes[1].barh(gdf_s["stand_id"], gdf_s["risk_score"], color=bar_colors)
axes[1].set_xlabel("Fire Risk Score"); axes[1].set_title("Stand Priority Ranking")
axes[1].grid(True, axis="x", alpha=0.3)
plt.suptitle("D2 Forestry: GeoPandas Spatial Analysis", fontweight="bold")
plt.tight_layout(); plt.show()
'''

GPD_D2_SECTION_C = '''\
scenarios = {
    "no_action":       {"risk": 0.74, "cost_musd": 0.0,  "expected_loss_musd": 240.0},
    "thinning":        {"risk": 0.51, "cost_musd": 65.0, "expected_loss_musd": 170.0},
    "prescribed_burn": {"risk": 0.43, "cost_musd": 92.0, "expected_loss_musd": 135.0},
}
scenario_records = []
for name, vals in scenarios.items():
    score = round((1 - vals["risk"]) * 0.5 + (1.0 / max(vals["cost_musd"] + 1, 1)) * 10 * 0.2
                  + (1.0 / vals["expected_loss_musd"]) * 50 * 0.3, 4)
    scenario_records.append({"scenario": name, **vals, "score": score})
scenario_records.sort(key=lambda r: -float(r["score"]))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
names = [r["scenario"] for r in scenario_records]
colors = ["#27ae60", "#e67e22", "#c0392b"]
axes[0].barh(names, [r["risk"] for r in scenario_records], color=colors)
axes[0].set_xlabel("Fire Risk"); axes[0].set_title("Risk by Scenario"); axes[0].grid(True, axis="x", alpha=0.3)
axes[1].barh(names, [r["expected_loss_musd"] for r in scenario_records], color=colors)
axes[1].set_xlabel("Expected Loss ($M)"); axes[1].set_title("Expected Loss"); axes[1].grid(True, axis="x", alpha=0.3)
axes[2].barh(names, [r["score"] for r in scenario_records], color=colors)
axes[2].set_xlabel("Composite Score"); axes[2].set_title("Scenario Score"); axes[2].grid(True, axis="x", alpha=0.3)
plt.suptitle("D2 Forestry (GeoPandas): Scenario Comparison", fontweight="bold")
plt.tight_layout(); plt.show()

(OUTPUT_DIR / "d2-gpd-complex.json").write_text(
    json.dumps({"scenario_ranking": scenario_records}, indent=2, default=str), encoding="utf-8"
)
print("Wrote d2-gpd-complex.json")
for r in scenario_records:
    print(f"  {r['scenario']}: score={r['score']}")
'''


# ---------------------------------------------------------------------------
# GPD-D3  Flood Analysis (geopandas)
# ---------------------------------------------------------------------------
GPD_D3_SECTION_A = '''\
flood = {"features": [{"id": "fallback-flood"}]}
weather = {"properties": {"forecast": "fallback"}}
forecast = {"hourly": {"temperature_2m": [0.0]}}

flood, flood_src, flood_live = fetch_first_json(
    [
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
        "https://api.github.com/repos/OSGeo/gdal",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("features") or d.get("id")),
    flood,
)
weather, wx_src, wx_live = fetch_first_json(
    [
        "https://api.weather.gov/points/29.76,-95.37",
        "https://api.weather.gov/points/30.27,-97.74",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("properties", {}).get("forecast")),
    weather,
)
forecast, fc_src, fc_live = fetch_first_json(
    [
        "https://api.open-meteo.com/v1/forecast?latitude=29.76&longitude=-95.37&hourly=temperature_2m&forecast_days=1",
        "https://api.open-meteo.com/v1/forecast?latitude=30.27&longitude=-97.74&hourly=temperature_2m&forecast_days=1",
    ],
    lambda d: isinstance(d, dict) and len(d.get("hourly", {}).get("temperature_2m", [])) > 0,
    forecast,
)

flood_count = len(flood.get("features", [])) if isinstance(flood, dict) else 0
if flood_count == 0 and isinstance(flood, dict) and flood.get("id"):
    flood_count = 1
print(f"Flood records: {flood_count} | live={flood_live} | source={flood_src}")
print(f"NOAA forecast exists: {bool(weather.get('properties', {}).get('forecast'))} | live={wx_live} | source={wx_src}")
print(f"Open-Meteo hourly points: {len(forecast.get('hourly', {}).get('temperature_2m', []))} | live={fc_live} | source={fc_src}")
'''

GPD_D3_SECTION_B = '''\
assets_data = {
    "asset_id":        ["A1",   "A2",   "A3",   "A4",   "A5"],
    "zone":            ["X",    "AE",   "AE",   "VE",   "AE"],
    "replacement_musd":[4.0,    7.5,    3.2,    9.1,    5.5],
    "elevation_m":     [11.0,   7.0,    8.0,    5.9,    6.5],
}
lons = [-95.42, -95.35, -95.29, -95.26, -95.38]
lats = [ 29.81,  29.76,  29.73,  29.68,  29.71]

ZONE_RISK = {"VE": 1.0, "AE": 0.8, "X": 0.2}
gdf = gpd.GeoDataFrame(assets_data, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
gdf["zone_risk"] = gdf["zone"].map(ZONE_RISK).fillna(0.3)
gdf["flood_risk_score"] = (gdf["zone_risk"] * 0.7 + (10.0 / gdf["elevation_m"]) * 0.3).round(4)
gdf["expected_loss_musd"] = (gdf["replacement_musd"] * gdf["flood_risk_score"] * 0.18).round(4)
print("Asset flood risk:")
print(gdf[["asset_id","zone","flood_risk_score","expected_loss_musd"]].to_string(index=False))

# Flood zone polygons
flood_zones = gpd.GeoDataFrame(
    {"zone_id": ["Z_AE", "Z_VE"],
     "zone_type": ["AE", "VE"]},
    geometry=[box(-95.40, 29.69, -95.25, 29.78), box(-95.30, 29.65, -95.22, 29.73)],
    crs="EPSG:4326",
)

# 1. Spatial join: assets within flood zones
in_zone = gpd.sjoin(gdf, flood_zones, how="left", predicate="within")
print("\\nAssets in flood zones:")
print(in_zone[["asset_id","zone_type"]].to_string(index=False))

# 2. Buffer: flood exposure buffers around assets (projected)
gdf_proj = gdf.to_crs("EPSG:3857")
gdf_proj["buffer_geom"] = gdf_proj.buffer(3000)  # 3 km
print(f"\\nExposure buffers (3km): {len(gdf_proj)}")

# 3. Nearest join: assign each asset to nearest zone centroid
flood_zone_centroids = flood_zones.copy()
flood_zone_centroids.geometry = flood_zones.centroid
nearest_zone = gpd.sjoin_nearest(gdf, flood_zone_centroids, how="left", distance_col="dist_to_zone")
print("\\nNearest flood zone per asset:")
print(nearest_zone[["asset_id","zone_id","dist_to_zone"]].to_string(index=False))

# 4. Clip assets to AE zone extent
ae_mask = flood_zones[flood_zones["zone_type"] == "AE"]
clipped = gpd.clip(gdf, ae_mask)
print(f"\\nAssets within AE zone extent: {list(clipped['asset_id'])}")

# 5. Dissolve: total exposure by zone type
in_zone_clean = in_zone.dropna(subset=["zone_type"])
if len(in_zone_clean) > 0:
    dissolved = in_zone_clean.dissolve(by="zone_type", aggfunc={"replacement_musd": "sum", "flood_risk_score": "mean"})
    print("\\nExposure dissolved by zone type:")
    print(dissolved[["replacement_musd","flood_risk_score"]].to_string())

# 6. Bounding box filter
high_risk_area = gdf.cx[-95.40:-95.25, 29.68:29.78]
print(f"\\nAssets in high-risk area: {list(high_risk_area['asset_id'])}")

# 7. Overlay: find intersection of asset buffers with flood zones
gdf_buf = gpd.GeoDataFrame(gdf, geometry=gdf.to_crs("EPSG:3857").buffer(5000).to_crs("EPSG:4326"))
overlapping = gpd.overlay(gdf_buf[["asset_id","geometry"]], flood_zones[["zone_id","geometry"]], how="intersection")
print(f"\\nBuffer-zone intersections: {len(overlapping)} areas")

gdf.to_file(str(OUTPUT_DIR / "d3-gpd-assets.geojson"), driver="GeoJSON")
print("\\nWrote d3-gpd-assets.geojson")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
gdf.plot(ax=axes[0], column="flood_risk_score", cmap="Blues", markersize=160, legend=True,
         legend_kwds={"label": "Flood Risk"})
flood_zones.boundary.plot(ax=axes[0], color="red", linewidth=2, linestyle="--", label="Flood Zones")
for _, row in gdf.iterrows():
    axes[0].annotate(row["asset_id"], (row.geometry.x, row.geometry.y),
                     textcoords="offset points", xytext=(4,4), fontsize=9)
axes[0].set_title("Asset Flood Risk (dashed=flood zones)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

gdf_s = gdf.sort_values("flood_risk_score", ascending=False)
axes[1].barh(gdf_s["asset_id"], gdf_s["flood_risk_score"], color="#1d4ed8")
axes[1].set_xlabel("Flood Risk Score"); axes[1].set_title("Asset Risk Ranking")
axes[1].grid(True, axis="x", alpha=0.3)
plt.suptitle("D3 Flood: GeoPandas Spatial Analysis", fontweight="bold")
plt.tight_layout(); plt.show()
'''

GPD_D3_SECTION_C = '''\
baseline   = {"expected_annual_loss": 18.0, "impacted_assets": 4, "response_time_hours": 8.0}
mitigation = {"expected_annual_loss": 10.5, "impacted_assets": 2, "response_time_hours": 6.0}

metrics = list(baseline.keys())
delta_pct = [round((mitigation[m] - baseline[m]) / abs(baseline[m]) * 100, 1) for m in metrics]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = range(len(metrics))
width = 0.38
axes[0].bar([i - width/2 for i in x], [baseline[m] for m in metrics], width=width, label="Baseline", color="#94a3b8")
axes[0].bar([i + width/2 for i in x], [mitigation[m] for m in metrics], width=width, label="Mitigation", color="#2563eb")
axes[0].set_xticks(list(x)); axes[0].set_xticklabels(metrics, rotation=15)
axes[0].set_title("Flood Mitigation Scenarios"); axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.3)

bar_colors = ["#27ae60" if d < 0 else "#c0392b" for d in delta_pct]
axes[1].barh(metrics, delta_pct, color=bar_colors)
axes[1].axvline(0, color="#555", linewidth=1)
axes[1].set_xlabel("% Change"); axes[1].set_title("Delta % (negative=improvement)")
axes[1].grid(True, axis="x", alpha=0.3)
plt.suptitle("D3 Flood (GeoPandas): Scenario Analysis", fontweight="bold")
plt.tight_layout(); plt.show()

(OUTPUT_DIR / "d3-gpd-complex.json").write_text(
    json.dumps({"baseline": baseline, "mitigation": mitigation, "delta_pct": dict(zip(metrics, delta_pct))},
               indent=2, default=str), encoding="utf-8"
)
print("Wrote d3-gpd-complex.json")
print("Delta % per metric:", dict(zip(metrics, delta_pct)))
'''


# ---------------------------------------------------------------------------
# GPD-D4  Transportation (geopandas)
# ---------------------------------------------------------------------------
GPD_D4_SECTION_A = '''\
transport = {"features": [{"id": "fallback-transport"}]}
weather = {"properties": {"forecast": "fallback"}}
forecast = {"hourly": {"temperature_2m": [0.0]}}

transport, tr_src, tr_live = fetch_first_json(
    [
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
        "https://api.github.com/repos/osm-search/Nominatim",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("features") or d.get("id")),
    transport,
)
weather, wx_src, wx_live = fetch_first_json(
    [
        "https://api.weather.gov/points/40.75,-111.90",
        "https://api.weather.gov/points/34.05,-118.24",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("properties", {}).get("forecast")),
    weather,
)
forecast, fc_src, fc_live = fetch_first_json(
    [
        "https://api.open-meteo.com/v1/forecast?latitude=40.75&longitude=-111.90&hourly=temperature_2m&forecast_days=1",
        "https://api.open-meteo.com/v1/forecast?latitude=34.05&longitude=-118.24&hourly=temperature_2m&forecast_days=1",
    ],
    lambda d: isinstance(d, dict) and len(d.get("hourly", {}).get("temperature_2m", [])) > 0,
    forecast,
)

transport_count = len(transport.get("features", [])) if isinstance(transport, dict) else 0
if transport_count == 0 and isinstance(transport, dict) and transport.get("id"):
    transport_count = 1
print(f"Transport records: {transport_count} | live={tr_live} | source={tr_src}")
print(f"NOAA forecast exists: {bool(weather.get('properties', {}).get('forecast'))} | live={wx_live} | source={wx_src}")
print(f"Open-Meteo hourly points: {len(forecast.get('hourly', {}).get('temperature_2m', []))} | live={fc_live} | source={fc_src}")
'''

GPD_D4_SECTION_B = '''\
nodes_data = {
    "node":        ["O",     "A",     "B",     "C",     "D",     "E",     "F"],
    "cost":        [0.0,     5.0,     9.0,     12.0,    14.0,    11.0,    3.0],
    "throughput":  [1500,    1200,    1100,    900,     800,     950,     1300],
    "node_type":   ["origin","inter","inter","dest","inter","inter","inter"],
}
lons = [-111.90, -111.87, -111.84, -111.78, -111.84, -111.81, -111.86]
lats = [  40.75,   40.75,   40.75,   40.75,   40.72,   40.72,   40.78]

gdf = gpd.GeoDataFrame(nodes_data, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
print("Transport nodes:")
print(gdf[["node","cost","throughput"]].to_string(index=False))

# Incident data
incidents_gdf = gpd.GeoDataFrame(
    {"inc_id": ["I1","I2","I3"], "severity": [3, 1, 2]},
    geometry=gpd.points_from_xy([-111.88, -111.83, -111.79], [40.76, 40.73, 40.75]),
    crs="EPSG:4326",
)

# 1. Buffer: service zones around nodes (projected)
gdf_proj = gdf.to_crs("EPSG:3857")
service_buffers = gpd.GeoDataFrame(gdf[["node","cost"]], geometry=gdf_proj.buffer(3000).to_crs("EPSG:4326"))
print(f"\\nService buffers (3km): {len(service_buffers)}")

# 2. Nearest join: assign incidents to nearest node
nearest = gpd.sjoin_nearest(incidents_gdf, gdf, how="left", distance_col="dist_to_node")
print("\\nIncidents assigned to nearest node:")
print(nearest[["inc_id","severity","node","dist_to_node"]].to_string(index=False))

# 3. Spatial join: incidents within service buffers
in_buf = gpd.sjoin(incidents_gdf, service_buffers, how="left", predicate="within")
print("Incidents within 3km service zones:")
node_col = "node" if "node" in in_buf.columns else [c for c in in_buf.columns if c.startswith("node")][0]
print(in_buf[["inc_id", node_col]].dropna().to_string(index=False))

# 4. Dissolve by node type
dissolved = gdf.dissolve(by="node_type", aggfunc={"throughput": "sum", "cost": "mean"})
print("\\nDissolved by node type:")
print(dissolved[["throughput","cost"]].to_string())

# 5. High-throughput nodes
high_tp = gdf[gdf["throughput"] > 1100]
print(f"\\nHigh-throughput nodes (>1100): {list(high_tp['node'])}")

# 6. Bounding box: central corridor
central = gdf.cx[-111.90:-111.78, 40.73:40.77]
print(f"Nodes in central corridor: {list(central['node'])}")

# 7. Overlay: buffer intersections
buf_union = gpd.GeoDataFrame(gdf[["node"]], geometry=gdf_proj.buffer(5000).to_crs("EPSG:4326"))
overlaps = gpd.overlay(buf_union.iloc[:3], buf_union.iloc[1:4], how="intersection")
print(f"\\nBuffer overlap areas: {len(overlaps)}")

gdf.to_file(str(OUTPUT_DIR / "d4-gpd-network.geojson"), driver="GeoJSON")
print("\\nWrote d4-gpd-network.geojson")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
gdf.plot(ax=axes[0], column="cost", cmap="RdYlGn_r", markersize=160, legend=True,
         legend_kwds={"label": "Routing Cost"})
for _, row in gdf.iterrows():
    axes[0].annotate(row["node"], (row.geometry.x, row.geometry.y),
                     textcoords="offset points", xytext=(4,4), fontsize=9, fontweight="bold")
incidents_gdf.plot(ax=axes[0], color="red", markersize=100, marker="x", label="Incidents")
axes[0].set_title("Transport Network (x=incidents)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

gdf_s = gdf.sort_values("cost")
axes[1].barh(gdf_s["node"], gdf_s["cost"], color="#0891b2")
axes[1].set_xlabel("Routing Cost"); axes[1].set_title("Node Reachability Costs")
axes[1].grid(True, axis="x", alpha=0.3)
plt.suptitle("D4 Transportation: GeoPandas Spatial Analysis", fontweight="bold")
plt.tight_layout(); plt.show()
'''

GPD_D4_SECTION_C = '''\
scenarios = {
    "no_action":    {"throughput": 1000, "avg_delay_min": 18.0, "cost_musd": 0.0},
    "signal_opt":   {"throughput": 1280, "avg_delay_min": 11.0, "cost_musd": 15.0},
    "new_corridor": {"throughput": 1600, "avg_delay_min":  7.0, "cost_musd": 85.0},
}
scenario_records = []
for name, vals in scenarios.items():
    score = round(vals["throughput"] / 1600 * 0.5 + (1.0 / vals["avg_delay_min"]) * 10 * 0.3
                  + (1.0 / max(vals["cost_musd"] + 1, 1)) * 20 * 0.2, 4)
    scenario_records.append({"scenario": name, **vals, "score": score})
scenario_records.sort(key=lambda r: -float(r["score"]))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
names = [r["scenario"] for r in scenario_records]
colors = ["#27ae60", "#e67e22", "#c0392b"]
axes[0].barh(names, [r["throughput"] for r in scenario_records], color=colors)
axes[0].set_xlabel("Throughput"); axes[0].set_title("Network Throughput"); axes[0].grid(True, axis="x", alpha=0.3)
axes[1].barh(names, [r["avg_delay_min"] for r in scenario_records], color=colors)
axes[1].set_xlabel("Avg Delay (min)"); axes[1].set_title("Average Delay"); axes[1].grid(True, axis="x", alpha=0.3)
axes[2].barh(names, [r["score"] for r in scenario_records], color=colors)
axes[2].set_xlabel("Composite Score"); axes[2].set_title("Scenario Score"); axes[2].grid(True, axis="x", alpha=0.3)
plt.suptitle("D4 Transportation (GeoPandas): Scenario Comparison", fontweight="bold")
plt.tight_layout(); plt.show()

(OUTPUT_DIR / "d4-gpd-complex.json").write_text(
    json.dumps({"scenario_ranking": scenario_records}, indent=2, default=str), encoding="utf-8"
)
print("Wrote d4-gpd-complex.json")
for r in scenario_records:
    print(f"  {r['scenario']}: score={r['score']}")
'''


# ---------------------------------------------------------------------------
# GPD-D5  Climate (geopandas)
# ---------------------------------------------------------------------------
GPD_D5_SECTION_A = '''\
climate = {"features": [{"id": "fallback-climate"}]}
weather = {"properties": {"forecast": "fallback"}}
forecast = {"hourly": {"temperature_2m": [0.0]}}

climate, cl_src, cl_live = fetch_first_json(
    [
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
        "https://api.github.com/repos/ecmwf/cdsapi",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("features") or d.get("id")),
    climate,
)
weather, wx_src, wx_live = fetch_first_json(
    [
        "https://api.weather.gov/points/39.00,-98.00",
        "https://api.weather.gov/points/38.90,-77.03",
    ],
    lambda d: isinstance(d, dict) and bool(d.get("properties", {}).get("forecast")),
    weather,
)
forecast, fc_src, fc_live = fetch_first_json(
    [
        "https://api.open-meteo.com/v1/forecast?latitude=39.00&longitude=-98.00&hourly=temperature_2m&forecast_days=1",
        "https://api.open-meteo.com/v1/forecast?latitude=38.90&longitude=-77.03&hourly=temperature_2m&forecast_days=1",
    ],
    lambda d: isinstance(d, dict) and len(d.get("hourly", {}).get("temperature_2m", [])) > 0,
    forecast,
)

climate_count = len(climate.get("features", [])) if isinstance(climate, dict) else 0
if climate_count == 0 and isinstance(climate, dict) and climate.get("id"):
    climate_count = 1
print(f"Climate records: {climate_count} | live={cl_live} | source={cl_src}")
print(f"NOAA forecast exists: {bool(weather.get('properties', {}).get('forecast'))} | live={wx_live} | source={wx_src}")
print(f"Open-Meteo hourly points: {len(forecast.get('hourly', {}).get('temperature_2m', []))} | live={fc_live} | source={fc_src}")
'''

GPD_D5_SECTION_B = '''\
zones_data = {
    "zone_id":    ["Z1",  "Z2",  "Z3",  "Z4",  "Z5"],
    "heat_index": [0.78,  0.62,  0.85,  0.71,  0.90],
    "drought_idx":[0.65,  0.72,  0.55,  0.80,  0.45],
    "sea_level":  [0.30,  0.15,  0.80,  0.20,  0.60],
    "pop_density":[1200,  850,   3400,  550,   2100],
}
lons = [-98.10, -97.80, -97.50, -98.40, -97.65]
lats = [ 39.20,  38.90,  39.40,  38.70,  39.60]

gdf = gpd.GeoDataFrame(zones_data, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
gdf["composite_risk"] = (gdf["heat_index"] * 0.35 + gdf["drought_idx"] * 0.35
                          + gdf["sea_level"] * 0.30).round(4)
gdf["risk_tier"] = gdf["composite_risk"].apply(lambda r: "HIGH" if r > 0.65 else ("MED" if r > 0.45 else "LOW"))
print("Climate zone risk scores:")
print(gdf[["zone_id","composite_risk","risk_tier"]].to_string(index=False))

# Adaptation sites
adapt_gdf = gpd.GeoDataFrame(
    {"site_id": ["AS1","AS2"], "capacity_mw": [120, 90]},
    geometry=gpd.points_from_xy([-97.90, -98.20], [39.10, 39.30]),
    crs="EPSG:4326",
)

# 1. Buffer: climate impact zones (50km)
gdf_proj = gdf.to_crs("EPSG:3857")
buf_zones = gdf_proj.buffer(50000).to_crs("EPSG:4326")
print(f"Climate impact buffers (50km): {len(buf_zones)}")

# 2. Nearest join: assign zones to adaptation sites
nearest = gpd.sjoin_nearest(gdf, adapt_gdf, how="left", distance_col="dist_to_site")
print("\\nNearest adaptation site per zone:")
print(nearest[["zone_id","site_id","dist_to_site"]].to_string(index=False))

# 3. Spatial join: zones within adaptation site coverage
adapt_buffers = gpd.GeoDataFrame(adapt_gdf[["site_id"]], geometry=adapt_gdf.to_crs("EPSG:3857").buffer(100000).to_crs("EPSG:4326"))
covered = gpd.sjoin(gdf, adapt_buffers, how="inner", predicate="within")
print(f"\\nZones within 100km of adaptation sites: {list(covered['zone_id'])}")

# 4. Dissolve by risk tier
dissolved = gdf.dissolve(by="risk_tier", aggfunc={"composite_risk": "mean", "pop_density": "sum"})
print("\\nDissolved by risk tier:")
print(dissolved[["composite_risk","pop_density"]].to_string())

# 5. High-risk filter
high_risk = gdf[gdf["composite_risk"] > 0.65]
print(f"\\nHigh-risk zones: {list(high_risk['zone_id'])}")

# 6. Bounding box: northern band
north = gdf.cx[-98.50:-97.40, 39.30:39.70]
print(f"Zones in northern band: {list(north['zone_id'])}")

# 7. Overlay: zone buffer intersections with risk polygon
risk_poly = gpd.GeoDataFrame({"id": [1]}, geometry=[box(-98.20, 39.00, -97.40, 39.70)], crs="EPSG:4326")
buf_gdf = gpd.GeoDataFrame(gdf[["zone_id"]], geometry=gdf_proj.buffer(30000).to_crs("EPSG:4326"))
clipped = gpd.clip(buf_gdf, risk_poly)
print(f"Buffer areas intersecting risk polygon: {len(clipped)}")

gdf.to_file(str(OUTPUT_DIR / "d5-gpd-zones.geojson"), driver="GeoJSON")
print("\\nWrote d5-gpd-zones.geojson")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
gdf.plot(ax=axes[0], column="composite_risk", cmap="YlOrRd", markersize=160, legend=True,
         legend_kwds={"label": "Composite Risk"})
for _, row in gdf.iterrows():
    axes[0].annotate(f"{row['zone_id']}\\n({row['composite_risk']:.2f})", (row.geometry.x, row.geometry.y),
                     textcoords="offset points", xytext=(4, 4), fontsize=9)
adapt_gdf.plot(ax=axes[0], color="blue", markersize=220, marker="*", label="Adaptation Site")
axes[0].set_title("Climate Risk Zones (stars=adaptation sites)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

gdf_s = gdf.sort_values("composite_risk", ascending=False)
bar_colors = ["#c0392b" if t=="HIGH" else ("#e67e22" if t=="MED" else "#27ae60") for t in gdf_s["risk_tier"]]
axes[1].barh(gdf_s["zone_id"], gdf_s["composite_risk"], color=bar_colors)
axes[1].set_xlabel("Composite Climate Risk"); axes[1].set_title("Zone Risk Ranking")
axes[1].grid(True, axis="x", alpha=0.3)
plt.suptitle("D5 Climate: GeoPandas Spatial Analysis", fontweight="bold")
plt.tight_layout(); plt.show()
'''

GPD_D5_SECTION_C = '''\
baseline   = {"annual_loss_musd": 58.0, "resilience_index": 0.44, "service_reliability": 0.82}
adaptation = {"annual_loss_musd": 32.0, "resilience_index": 0.68, "service_reliability": 0.91}

metrics = list(baseline.keys())
delta = [round((adaptation[m] - baseline[m]) / abs(baseline[m]) * 100, 1) for m in metrics]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = range(len(metrics))
width = 0.38
axes[0].bar([i-width/2 for i in x], [baseline[m] for m in metrics], width=width, label="Baseline", color="#94a3b8")
axes[0].bar([i+width/2 for i in x], [adaptation[m] for m in metrics], width=width, label="Adaptation", color="#2563eb")
axes[0].set_xticks(list(x)); axes[0].set_xticklabels(metrics, rotation=15)
axes[0].set_title("Baseline vs Adaptation"); axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.3)

bar_colors = ["#27ae60" if d > 0 else "#c0392b" for d in delta]
axes[1].barh(metrics, delta, color=bar_colors)
axes[1].axvline(0, color="#555", linewidth=1)
axes[1].set_xlabel("% Change vs Baseline"); axes[1].set_title("Improvement (positive=better)")
axes[1].grid(True, axis="x", alpha=0.3)
plt.suptitle("D5 Climate (GeoPandas): Adaptation Scenario Analysis", fontweight="bold")
plt.tight_layout(); plt.show()

(OUTPUT_DIR / "d5-gpd-complex.json").write_text(
    json.dumps({"baseline": baseline, "adaptation": adaptation}, indent=2, default=str), encoding="utf-8"
)
print("Wrote d5-gpd-complex.json")
print("Delta per metric:", dict(zip(metrics, delta)))
'''


# ============================================================================
# Assemble and save all notebooks
# ============================================================================

def make_gp_notebook(title, desc, imports_src, sec_a_src, sec_b_src, sec_c_src):
    return [
        md(f"# {title}\n\n{desc}"),
        code(imports_src),
        md("## Section A: Pull Data Sources"),
        code(sec_a_src),
        md("## Section B: Spatial Analysis"),
        code(sec_b_src),
        md("## Section C: Scenario Comparison"),
        code(sec_c_src),
    ]

def make_gpd_notebook(title, desc, sec_a_src, sec_b_src, sec_c_src):
    return [
        md(f"# {title}\n\n{desc}"),
        code(GPD_IMPORTS_TEMPLATE),
        md("## Section A: Pull Data Sources"),
        code(sec_a_src),
        md("## Section B: Spatial Analysis (GeoPandas)"),
        code(sec_b_src),
        md("## Section C: Scenario Comparison"),
        code(sec_c_src),
    ]


notebooks = {
    "examples/notebooks/geoprompt/d1_utilities_workflow.ipynb": make_gp_notebook(
        "D1 Utilities Workflow — GeoPrompt",
        "Network routing, service area analysis, and GeoPromptFrame spatial operations.",
        GP_D1_IMPORTS, GP_D1_SECTION_A, GP_D1_SECTION_B, GP_D1_SECTION_C,
    ),
    "examples/notebooks/geoprompt/d2_forestry_management_workflow.ipynb": make_gp_notebook(
        "D2 Forestry Management Workflow — GeoPrompt",
        "Stand risk scoring, proximity analysis, and forest management scenarios.",
        GP_D2_IMPORTS, GP_D2_SECTION_A, GP_D2_SECTION_B, GP_D2_SECTION_C,
    ),
    "examples/notebooks/geoprompt/d3_flood_analysis_workflow.ipynb": make_gp_notebook(
        "D3 Flood Analysis Workflow — GeoPrompt",
        "Flood exposure screening, raster terrain analysis, and mitigation scenarios.",
        GP_D3_IMPORTS, GP_D3_SECTION_A, GP_D3_SECTION_B, GP_D3_SECTION_C,
    ),
    "examples/notebooks/geoprompt/d4_transportation_workflow.ipynb": make_gp_notebook(
        "D4 Transportation Workflow — GeoPrompt",
        "Network routing, service area coverage, and mobility scenario analysis.",
        GP_D4_IMPORTS, GP_D4_SECTION_A, GP_D4_SECTION_B, GP_D4_SECTION_C,
    ),
    "examples/notebooks/geoprompt/d5_climate_workflow.ipynb": make_gp_notebook(
        "D5 Climate Risk Workflow — GeoPrompt",
        "Climate zone risk scoring, raster algebra, and adaptation scenario analysis.",
        GP_D5_IMPORTS, GP_D5_SECTION_A, GP_D5_SECTION_B, GP_D5_SECTION_C,
    ),
    "examples/notebooks/geopandas/d1_utilities_workflow.ipynb": make_gpd_notebook(
        "D1 Utilities Workflow — GeoPandas",
        "Utility network analysis using GeoPandas spatial operations.",
        GPD_D1_SECTION_A, GPD_D1_SECTION_B, GPD_D1_SECTION_C,
    ),
    "examples/notebooks/geopandas/d2_forestry_management_workflow.ipynb": make_gpd_notebook(
        "D2 Forestry Management Workflow — GeoPandas",
        "Stand risk and forest management using GeoPandas spatial operations.",
        GPD_D2_SECTION_A, GPD_D2_SECTION_B, GPD_D2_SECTION_C,
    ),
    "examples/notebooks/geopandas/d3_flood_analysis_workflow.ipynb": make_gpd_notebook(
        "D3 Flood Analysis Workflow — GeoPandas",
        "Flood exposure screening using GeoPandas spatial operations.",
        GPD_D3_SECTION_A, GPD_D3_SECTION_B, GPD_D3_SECTION_C,
    ),
    "examples/notebooks/geopandas/d4_transportation_workflow.ipynb": make_gpd_notebook(
        "D4 Transportation Workflow — GeoPandas",
        "Transport network analysis using GeoPandas spatial operations.",
        GPD_D4_SECTION_A, GPD_D4_SECTION_B, GPD_D4_SECTION_C,
    ),
    "examples/notebooks/geopandas/d5_climate_workflow.ipynb": make_gpd_notebook(
        "D5 Climate Risk Workflow — GeoPandas",
        "Climate zone risk analysis using GeoPandas spatial operations.",
        GPD_D5_SECTION_A, GPD_D5_SECTION_B, GPD_D5_SECTION_C,
    ),
}

if __name__ == "__main__":
    print("Generating notebooks...")
    for path, cells in notebooks.items():
        save(cells, path)
    print(f"\nDone — {len(notebooks)} notebooks written.")
