"""Domain-specific analysis modules for geoprompt.

Covers utility networks, environmental assessment, public safety,
agriculture, forestry, planning, logistics, maritime, mining,
energy, and smart-city domains.
"""

from __future__ import annotations

import math
import statistics
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._exceptions import failure_payload
from .quality import simulation_only

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def _mannings_flow(n: float, slope: float, radius: float, area: float) -> float:
    if n <= 0 or slope < 0:
        return 0.0
    return (1.0 / n) * area * radius ** (2 / 3) * math.sqrt(slope)


def _inverse_distance_weight(
    known: Sequence[Tuple[float, float, float]],
    target: Tuple[float, float],
    power: float = 2.0,
) -> float:
    numer = denom = 0.0
    for kx, ky, kv in known:
        d = math.hypot(kx - target[0], ky - target[1])
        if d < 1e-12:
            return kv
        w = 1.0 / d ** power
        numer += w * kv
        denom += w
    return numer / denom if denom else 0.0


def _score_features(
    features: Sequence[Dict[str, Any]],
    fields: Sequence[str],
    weights: Sequence[float],
) -> List[Dict[str, Any]]:
    if not features or not fields:
        return []
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for f in features:
        attrs = f.get("attributes", f)
        for fld in fields:
            v = float(attrs.get(fld, 0))
            mins[fld] = min(mins.get(fld, v), v)
            maxs[fld] = max(maxs.get(fld, v), v)
    results = []
    for f in features:
        score = 0.0
        attrs = f.get("attributes", f)
        for fld, w in zip(fields, weights):
            v = float(attrs.get(fld, 0))
            rng = maxs[fld] - mins[fld]
            norm = (v - mins[fld]) / rng if rng else 0.5
            score += norm * w
        results.append({**f, "score": round(score, 4)})
    return results


def _kde_grid(
    points: Sequence[Tuple[float, float]],
    grid_size: int = 50,
    bandwidth: float = 500,
) -> Dict[str, Any]:
    if not points:
        return {"grid": [], "max_density": 0}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = (xmax - xmin) / grid_size if xmax > xmin else 1
    dy = (ymax - ymin) / grid_size if ymax > ymin else 1
    grid: List[List[float]] = []
    max_d = 0.0
    for i in range(grid_size):
        row: List[float] = []
        cy = ymin + (i + 0.5) * dy
        for j in range(grid_size):
            cx = xmin + (j + 0.5) * dx
            density = sum(
                math.exp(-0.5 * ((cx - px) ** 2 + (cy - py) ** 2) / bandwidth ** 2)
                for px, py in points
            )
            row.append(round(density, 6))
            if density > max_d:
                max_d = density
        grid.append(row)
    return {"grid": grid, "max_density": round(max_d, 6),
            "extent": [xmin, ymin, xmax, ymax]}


# ===========================================================================
# WATER / SEWER / STORMWATER  (1251 – 1256, 1260)
# ===========================================================================

def epanet_inp_read(text: str) -> Dict[str, Any]:
    """Parse an EPANET INP string into a network dict."""
    sections: Dict[str, list] = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].upper()
            sections.setdefault(current, [])
        elif current and stripped and not stripped.startswith(";"):
            sections[current].append(stripped.split())
    junctions = [
        {"id": r[0], "elevation": float(r[1]), "demand": float(r[2]) if len(r) > 2 else 0}
        for r in sections.get("JUNCTIONS", [])
    ]
    pipes = [
        {"id": r[0], "node1": r[1], "node2": r[2], "length": float(r[3]),
         "diameter": float(r[4]), "roughness": float(r[5]) if len(r) > 5 else 130}
        for r in sections.get("PIPES", []) if len(r) >= 5
    ]
    tanks = [
        {"id": r[0], "elevation": float(r[1]), "init_level": float(r[2]) if len(r) > 2 else 0}
        for r in sections.get("TANKS", [])
    ]
    reservoirs = [
        {"id": r[0], "head": float(r[1]) if len(r) > 1 else 0}
        for r in sections.get("RESERVOIRS", [])
    ]
    return {"junctions": junctions, "pipes": pipes, "tanks": tanks, "reservoirs": reservoirs}


def epanet_inp_write(network: Dict[str, Any]) -> str:
    """Serialise a network dict to EPANET INP format string."""
    lines = ["[TITLE]", "Generated by geoprompt", ""]
    lines.append("[JUNCTIONS]")
    for j in network.get("junctions", []):
        lines.append(f"{j['id']}\t{j['elevation']}\t{j.get('demand', 0)}")
    lines += ["", "[RESERVOIRS]"]
    for r in network.get("reservoirs", []):
        lines.append(f"{r['id']}\t{r.get('head', 0)}")
    lines += ["", "[TANKS]"]
    for t in network.get("tanks", []):
        lines.append(f"{t['id']}\t{t['elevation']}\t{t.get('init_level', 0)}")
    lines += ["", "[PIPES]"]
    for p in network.get("pipes", []):
        lines.append(
            f"{p['id']}\t{p['node1']}\t{p['node2']}\t{p['length']}\t{p['diameter']}\t{p.get('roughness', 130)}"
        )
    lines += ["", "[END]"]
    return "\n".join(lines)


def water_distribution_solve(network: Dict[str, Any]) -> Dict[str, Any]:
    """Hazen-Williams steady-state solve for a water distribution network."""
    heads: Dict[str, float] = {}
    for r in network.get("reservoirs", []):
        heads[r["id"]] = r.get("head", 0)
    for j in network.get("junctions", []):
        heads[j["id"]] = j["elevation"] + 10
    for t in network.get("tanks", []):
        heads[t["id"]] = t["elevation"] + t.get("init_level", 0)
    flows: Dict[str, float] = {}
    for p in network.get("pipes", []):
        h1, h2 = heads.get(p["node1"], 0), heads.get(p["node2"], 0)
        dh = h1 - h2
        sign = 1 if dh >= 0 else -1
        C = p.get("roughness", 130)
        d = p["diameter"] / 1000
        L = p["length"]
        if L > 0 and d > 0 and C > 0:
            denom = 10.67 * L / (C ** 1.852 * d ** 4.87)
            Q = sign * (abs(dh) / denom) ** (1 / 1.852) if denom else 0
        else:
            Q = 0.0
        flows[p["id"]] = round(Q, 6)
    return {"heads": heads, "flows": flows}


def swmm_inp_read(text: str) -> Dict[str, Any]:
    """Parse a SWMM INP string into a model dict."""
    sections: Dict[str, list] = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].upper()
            sections.setdefault(current, [])
        elif current and stripped and not stripped.startswith(";"):
            sections[current].append(stripped.split())
    subcatchments = [
        {"name": r[0], "rain_gage": r[1] if len(r) > 1 else "",
         "outlet": r[2] if len(r) > 2 else "", "area": float(r[3]) if len(r) > 3 else 0}
        for r in sections.get("SUBCATCHMENTS", [])
    ]
    conduits = [
        {"name": r[0], "from_node": r[1] if len(r) > 1 else "",
         "to_node": r[2] if len(r) > 2 else "", "length": float(r[3]) if len(r) > 3 else 0}
        for r in sections.get("CONDUITS", [])
    ]
    junctions = [
        {"name": r[0], "elevation": float(r[1]) if len(r) > 1 else 0}
        for r in sections.get("JUNCTIONS", [])
    ]
    return {"subcatchments": subcatchments, "conduits": conduits, "junctions": junctions}


def swmm_inp_write(model: Dict[str, Any]) -> str:
    """Serialise a model dict to SWMM INP format string."""
    lines = ["[TITLE]", "Generated by geoprompt", ""]
    lines.append("[SUBCATCHMENTS]")
    for s in model.get("subcatchments", []):
        lines.append(f"{s['name']}\t{s.get('rain_gage', '')}\t{s.get('outlet', '')}\t{s.get('area', 0)}")
    lines += ["", "[JUNCTIONS]"]
    for j in model.get("junctions", []):
        lines.append(f"{j['name']}\t{j.get('elevation', 0)}")
    lines += ["", "[CONDUITS]"]
    for c in model.get("conduits", []):
        lines.append(f"{c['name']}\t{c.get('from_node', '')}\t{c.get('to_node', '')}\t{c.get('length', 0)}")
    lines += ["", "[END]"]
    return "\n".join(lines)


def sewer_model_solve(
    conduits: Sequence[Dict[str, Any]], *, manning_n: float = 0.013,
) -> List[Dict[str, Any]]:
    """Solve sewer conduit capacities using Manning's equation."""
    results = []
    for c in conduits:
        d = c.get("diameter", 0.6)
        slope = c.get("slope", 0.005)
        area = math.pi * (d / 2) ** 2 / 2
        wetted_p = math.pi * d / 2
        R = area / wetted_p if wetted_p else 0
        capacity = _mannings_flow(manning_n, slope, R, area)
        flow = c.get("flow", 0)
        results.append({
            "conduit": c.get("name", c.get("id", "")),
            "capacity_m3s": round(capacity, 4),
            "flow_m3s": flow,
            "utilisation": round(flow / capacity, 3) if capacity else 0,
            "surcharge": flow > capacity,
        })
    return results


def combined_sewer_overflow(
    pipe_capacity_m3s: float, dry_weather_flow_m3s: float,
    wet_weather_flow_m3s: float, *, storage_m3: float = 0,
) -> Dict[str, Any]:
    """Estimate combined-sewer overflow volume."""
    total = dry_weather_flow_m3s + wet_weather_flow_m3s
    excess = max(0, total - pipe_capacity_m3s)
    overflow = max(0, excess - storage_m3 / 3600)
    return {"total_flow_m3s": round(total, 4), "pipe_capacity_m3s": pipe_capacity_m3s,
            "excess_m3s": round(excess, 4), "overflow_m3s": round(overflow, 4),
            "cso_active": overflow > 0}


# ===========================================================================
# ELECTRIC DISTRIBUTION  (1261 – 1267)
# ===========================================================================

def cim_xml_read(text: str) -> Dict[str, Any]:
    """Parse a CIM/XML power model string into a dict."""
    root = ET.fromstring(text)
    buses: List[Dict[str, Any]] = []
    lines_out: List[Dict[str, Any]] = []
    for el in root:
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        rid = el.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}ID",
                            el.attrib.get("rdf:ID", ""))
        if "ConnectivityNode" in tag or "TopologicalNode" in tag:
            name_el = el.find(".//{http://iec.ch/TC57/CIM100#}name")
            buses.append({"id": rid, "name": name_el.text if name_el is not None else rid})
        elif "ACLineSegment" in tag:
            length_el = el.find(".//{http://iec.ch/TC57/CIM100#}length")
            lines_out.append({"id": rid, "length": float(length_el.text) if length_el is not None else 0})
    return {"buses": buses, "lines": lines_out}


def cim_xml_write(model: Dict[str, Any]) -> str:
    """Serialise a power model dict to minimal CIM/XML string."""
    parts = [
        '<?xml version="1.0"?>',
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        ' xmlns:cim="http://iec.ch/TC57/CIM100#">',
    ]
    for b in model.get("buses", []):
        parts.append(f'  <cim:ConnectivityNode rdf:ID="{b["id"]}">'
                     f'<cim:name>{b.get("name", b["id"])}</cim:name></cim:ConnectivityNode>')
    for ln in model.get("lines", []):
        parts.append(f'  <cim:ACLineSegment rdf:ID="{ln["id"]}">'
                     f'<cim:length>{ln.get("length", 0)}</cim:length></cim:ACLineSegment>')
    parts.append("</rdf:RDF>")
    return "\n".join(parts)


def electric_load_flow(
    buses: Sequence[Dict[str, Any]], branches: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """DC power-flow approximation (linearised)."""
    bus_map = {b["id"]: i for i, b in enumerate(buses)}
    n = len(buses)
    voltages = [b.get("voltage_pu", 1.0) for b in buses]
    angles = [0.0] * n
    for _it in range(20):
        for br in branches:
            fi, ti = bus_map.get(br["from"], 0), bus_map.get(br["to"], 0)
            x = br.get("reactance", 0.1)
            if x == 0:
                continue
            P = (angles[fi] - angles[ti]) / x
            angles[ti] += 0.1 * (buses[fi].get("generation", 0) - buses[fi].get("load", 0) - P)
    flows = []
    for br in branches:
        fi, ti = bus_map.get(br["from"], 0), bus_map.get(br["to"], 0)
        x = br.get("reactance", 0.1)
        P = (angles[fi] - angles[ti]) / x if x else 0
        flows.append({"branch": br.get("id", ""), "flow_mw": round(P, 4)})
    return {"voltages": {b["id"]: round(voltages[i], 4) for i, b in enumerate(buses)},
            "angles": {b["id"]: round(angles[i], 4) for i, b in enumerate(buses)},
            "branch_flows": flows}


def electric_fault_analysis(
    buses: Sequence[Dict[str, Any]], branches: Sequence[Dict[str, Any]], fault_bus: str,
) -> Dict[str, Any]:
    """Three-phase bolted fault current (simplified Z-bus)."""
    bus_map = {b["id"]: i for i, b in enumerate(buses)}
    n = len(buses)
    Z = [[0.0] * n for _ in range(n)]
    for br in branches:
        fi, ti = bus_map.get(br["from"], 0), bus_map.get(br["to"], 0)
        z = complex(br.get("resistance", 0.01), br.get("reactance", 0.1))
        Z[fi][fi] += abs(z)
        Z[ti][ti] += abs(z)
    fb = bus_map.get(fault_bus, 0)
    Zf = Z[fb][fb] if Z[fb][fb] != 0 else 0.01
    return {"fault_bus": fault_bus, "fault_current_pu": round(1.0 / Zf, 4),
            "fault_impedance_pu": round(Zf, 6)}


def electric_protection_coordination(
    devices: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Order protective devices by operating time for coordination."""
    timed = []
    for d in devices:
        tds = d.get("time_dial", 1.0)
        pickup = d.get("pickup_current", 100)
        fault = d.get("fault_current", 1000)
        ratio = fault / pickup if pickup else 10
        denom = ratio ** 0.02 - 1
        op_time = tds * 0.0515 / denom if denom > 0 else 999
        timed.append({**d, "operating_time_s": round(op_time, 4)})
    timed.sort(key=lambda x: x["operating_time_s"])
    for i, d in enumerate(timed):
        d["sequence"] = i + 1
    return timed


def electric_hosting_capacity(
    feeder_capacity_kw: float, existing_load_kw: float,
    existing_der_kw: float = 0, *,
    voltage_limit_pu: float = 1.05, thermal_margin: float = 0.15,
) -> Dict[str, Any]:
    """Estimate DER hosting capacity on a feeder."""
    thermal_available = feeder_capacity_kw * (1 - thermal_margin) - existing_load_kw
    voltage_limited = feeder_capacity_kw * (voltage_limit_pu - 1.0) * 10
    hosting = max(0, min(thermal_available, voltage_limited) - existing_der_kw)
    return {"hosting_capacity_kw": round(hosting, 1),
            "thermal_available_kw": round(thermal_available, 1),
            "voltage_limited_kw": round(voltage_limited, 1),
            "binding_constraint": "thermal" if thermal_available < voltage_limited else "voltage"}


# ===========================================================================
# GAS DISTRIBUTION  (1269 – 1272)
# ===========================================================================

def gas_model_read(text: str) -> Dict[str, Any]:
    """Parse a pipe-delimited gas network model."""
    nodes, pipes = [], []
    section = None
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("NODES"):
            section = "nodes"; continue
        elif s.upper().startswith("PIPES"):
            section = "pipes"; continue
        elif not s or s.startswith("#"):
            continue
        parts = s.split("|") if "|" in s else s.split()
        if section == "nodes" and len(parts) >= 2:
            nodes.append({"id": parts[0].strip(), "pressure_kpa": float(parts[1].strip())})
        elif section == "pipes" and len(parts) >= 4:
            pipes.append({"id": parts[0].strip(), "from": parts[1].strip(),
                          "to": parts[2].strip(), "length_m": float(parts[3].strip())})
    return {"nodes": nodes, "pipes": pipes}


def gas_model_write(model: Dict[str, Any]) -> str:
    """Serialise a gas network model to pipe-delimited text."""
    lines = ["NODES"]
    for n in model.get("nodes", []):
        lines.append(f"{n['id']} | {n.get('pressure_kpa', 0)}")
    lines.append("PIPES")
    for p in model.get("pipes", []):
        lines.append(f"{p['id']} | {p['from']} | {p['to']} | {p.get('length_m', 0)}")
    return "\n".join(lines)


def gas_leak_heatmap(
    leak_points: Sequence[Tuple[float, float]], *,
    grid_size: int = 50, bandwidth: float = 500,
) -> Dict[str, Any]:
    """Kernel density estimate for gas leak locations."""
    return _kde_grid(leak_points, grid_size, bandwidth)


# ===========================================================================
# TELECOM  (1276)
# ===========================================================================

def bandwidth_allocation(
    total_bandwidth_mhz: float, users: Sequence[Dict[str, Any]], *,
    strategy: str = "proportional",
) -> List[Dict[str, Any]]:
    """Allocate bandwidth to users. Strategies: equal, proportional, priority."""
    n = len(users)
    if n == 0:
        return []
    if strategy == "equal":
        share = total_bandwidth_mhz / n
        return [{**u, "allocated_mhz": round(share, 3)} for u in users]
    if strategy == "priority":
        total_p = sum(u.get("priority", 1) for u in users)
        return [{**u, "allocated_mhz": round(
            total_bandwidth_mhz * u.get("priority", 1) / total_p, 3)} for u in users]
    total_d = sum(u.get("demand_mhz", 1) for u in users) or n
    return [{**u, "allocated_mhz": round(
        total_bandwidth_mhz * u.get("demand_mhz", 1) / total_d, 3)} for u in users]


# ===========================================================================
# TRAFFIC  (1279 – 1282)
# ===========================================================================

def traffic_assignment_ue(
    links: Sequence[Dict[str, Any]], od_pairs: Sequence[Dict[str, Any]], *,
    iterations: int = 50, alpha: float = 0.15, beta: float = 4.0,
) -> List[Dict[str, Any]]:
    """User-equilibrium traffic assignment (BPR + MSA)."""
    flows = {l["id"]: 0.0 for l in links}
    times: Dict[str, float] = {}
    for it in range(iterations):
        for l in links:
            t0 = l.get("free_flow_time", 1)
            cap = l.get("capacity", 1000)
            f = flows[l["id"]]
            times[l["id"]] = t0 * (1 + alpha * (f / cap) ** beta)
        new_flows = {l["id"]: 0.0 for l in links}
        for od in od_pairs:
            for lid in od.get("route_links", [links[0]["id"]] if links else []):
                if lid in new_flows:
                    new_flows[lid] += od.get("demand", 100)
        lam = 2.0 / (it + 2)
        for lid in flows:
            flows[lid] = (1 - lam) * flows[lid] + lam * new_flows.get(lid, 0)
    return [{"link": l["id"], "flow": round(flows[l["id"]], 1),
             "travel_time": round(times.get(l["id"], 0), 2)} for l in links]


def traffic_assignment_so(
    links: Sequence[Dict[str, Any]], od_pairs: Sequence[Dict[str, Any]], *,
    iterations: int = 50, alpha: float = 0.15, beta: float = 4.0,
) -> List[Dict[str, Any]]:
    """System-optimal traffic assignment using marginal cost."""
    flows = {l["id"]: 0.0 for l in links}
    times: Dict[str, float] = {}
    for it in range(iterations):
        for l in links:
            t0 = l.get("free_flow_time", 1)
            cap = l.get("capacity", 1000)
            f = flows[l["id"]]
            bpr = t0 * (1 + alpha * (f / cap) ** beta)
            times[l["id"]] = bpr + t0 * alpha * beta * (f / cap) ** beta
        new_flows = {l["id"]: 0.0 for l in links}
        for od in od_pairs:
            for lid in od.get("route_links", [links[0]["id"]] if links else []):
                if lid in new_flows:
                    new_flows[lid] += od.get("demand", 100)
        lam = 2.0 / (it + 2)
        for lid in flows:
            flows[lid] = (1 - lam) * flows[lid] + lam * new_flows.get(lid, 0)
    return [{"link": l["id"], "flow": round(flows[l["id"]], 1),
             "marginal_cost": round(times.get(l["id"], 0), 2)} for l in links]


def traffic_microsim_bridge(
    config: Dict[str, Any], *, simulator: str = "sumo",
) -> Dict[str, Any]:
    """Bridge to a micro-simulation tool (SUMO, VISSIM) via subprocess."""
    import subprocess
    cmd = config.get("command", simulator)
    args = config.get("args", [])
    try:
        result = subprocess.run(
            [cmd] + args, capture_output=True, text=True,
            timeout=config.get("timeout", 300),
        )
        return {"returncode": result.returncode, "stdout": result.stdout[:4000],
                "stderr": result.stderr[:2000]}
    except FileNotFoundError:
        return failure_payload(
            code="MICROSIM_EXECUTABLE_NOT_FOUND",
            category="dependency",
            remediation="Install the configured simulator or provide a valid command path in config['command'].",
            error=f"{simulator} not found on PATH",
            returncode=-1,
        )
    except subprocess.TimeoutExpired:
        return failure_payload(
            code="MICROSIM_TIMEOUT",
            category="timeout",
            remediation="Increase config['timeout'] or simplify the simulation scenario inputs.",
            error="simulation timed out",
            returncode=-2,
        )


def traffic_signal_timing(
    phases: Sequence[Dict[str, Any]], *,
    lost_time_s: float = 4.0, cycle_range: Tuple[float, float] = (60, 120),
) -> Dict[str, Any]:
    """Webster's optimal signal timing."""
    Y = sum(p.get("critical_ratio", 0.2) for p in phases)
    n = len(phases)
    L = lost_time_s * n
    if Y >= 1:
        return failure_payload(
            code="SIGNAL_OVERSATURATED",
            category="validation",
            remediation="Reduce critical ratios so their sum is below 1.0 before computing timing.",
            error="oversaturated",
            sum_critical_ratios=Y,
        )
    Co = (1.5 * L + 5) / (1 - Y)
    Co = max(cycle_range[0], min(cycle_range[1], Co))
    green_total = Co - L
    greens = []
    for p in phases:
        y = p.get("critical_ratio", 0.2)
        g = green_total * y / Y if Y > 0 else green_total / n
        greens.append({**p, "green_s": round(g, 1)})
    return {"cycle_s": round(Co, 1), "total_lost_s": L, "phases": greens}


# ===========================================================================
# ENVIRONMENTAL SCREENING  (1308, 1309, 1311, 1312)
# ===========================================================================

def wind_study(
    measurements: Sequence[Dict[str, Any]], *, direction_bins: int = 16,
) -> Dict[str, Any]:
    """Basic wind-rose statistics from measurement records."""
    if not measurements:
        return {"bins": [], "mean_speed": 0}
    bin_width = 360 / direction_bins
    bins = [{"dir_from": i * bin_width, "dir_to": (i + 1) * bin_width,
             "count": 0, "speed_sum": 0.0} for i in range(direction_bins)]
    for m in measurements:
        d = m.get("direction", 0) % 360
        s = m.get("speed", 0)
        idx = int(d / bin_width) % direction_bins
        bins[idx]["count"] += 1
        bins[idx]["speed_sum"] += s
    for b in bins:
        b["mean_speed"] = round(b["speed_sum"] / b["count"], 2) if b["count"] else 0
        del b["speed_sum"]
    speeds = [m.get("speed", 0) for m in measurements]
    return {"bins": bins, "mean_speed": round(statistics.mean(speeds), 2),
            "max_speed": round(max(speeds), 2),
            "calm_pct": round(sum(1 for s in speeds if s < 0.5) / len(speeds) * 100, 1)}


def view_corridor_analysis(
    observer: Tuple[float, float, float],
    target: Tuple[float, float, float],
    obstacles: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check if line-of-sight from observer to target is blocked."""
    ox, oy, oz = observer
    tx, ty, tz = target
    dx, dy, dz = tx - ox, ty - oy, tz - oz
    denom = dx ** 2 + dy ** 2
    if denom == 0:
        return {"visible": True, "blocked_by": None}
    for obs in obstacles:
        bx, by, bh = obs.get("x", 0), obs.get("y", 0), obs.get("height", 0)
        t_param = ((bx - ox) * dx + (by - oy) * dy) / denom
        if t_param < 0 or t_param > 1:
            continue
        line_z = oz + dz * t_param
        if line_z < bh:
            return {"visible": False, "blocked_by": obs.get("id", "unknown"),
                    "block_height": bh, "line_height_at_block": round(line_z, 2)}
    return {"visible": True, "blocked_by": None}


def environmental_impact_screening(
    project_footprint: Dict[str, Any],
    sensitive_areas: Sequence[Dict[str, Any]], *, buffer_m: float = 500,
) -> Dict[str, Any]:
    """Screen a project footprint against sensitive environmental areas."""
    px, py = project_footprint.get("x", 0), project_footprint.get("y", 0)
    conflicts = []
    for sa in sensitive_areas:
        dist = math.hypot(px - sa.get("x", 0), py - sa.get("y", 0))
        if dist <= buffer_m + sa.get("radius", 0):
            conflicts.append({"area": sa.get("name", ""), "type": sa.get("type", ""),
                              "distance_m": round(dist, 1)})
    return {"project": project_footprint.get("name", ""), "conflicts": conflicts,
            "conflict_count": len(conflicts), "requires_eia": len(conflicts) > 0}


def environmental_justice_screening(
    block_groups: Sequence[Dict[str, Any]], *,
    minority_threshold: float = 0.5, poverty_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Flag census block groups as EJ communities."""
    results = []
    for bg in block_groups:
        m_pct, p_pct = bg.get("minority_pct", 0), bg.get("poverty_pct", 0)
        is_ej = m_pct >= minority_threshold or p_pct >= poverty_threshold
        factors = []
        if m_pct >= minority_threshold:
            factors.append("minority")
        if p_pct >= poverty_threshold:
            factors.append("poverty")
        results.append({**bg, "ej_community": is_ej, "ej_factors": factors})
    return results


# ===========================================================================
# CRIME / PUBLIC HEALTH  (1327 – 1335)
# ===========================================================================

def crime_pattern_prediction(
    incidents: Sequence[Dict[str, Any]], *,
    grid_size: int = 20, bandwidth: float = 200,
) -> Dict[str, Any]:
    """Kernel density prediction of crime hotspots."""
    return _kde_grid([(i.get("x", 0), i.get("y", 0)) for i in incidents], grid_size, bandwidth)


def repeat_offender_spatial(
    offenders: Sequence[Dict[str, Any]], incidents: Sequence[Dict[str, Any]], *,
    link_radius_m: float = 500,
) -> List[Dict[str, Any]]:
    """Link repeat offenders to spatial clusters of incidents."""
    results = []
    for off in offenders:
        ox, oy = off.get("x", 0), off.get("y", 0)
        linked = sum(1 for i in incidents
                     if math.hypot(ox - i.get("x", 0), oy - i.get("y", 0)) <= link_radius_m)
        results.append({**off, "linked_incidents": linked, "repeat": linked > 1})
    return results


def public_health_cluster_detection(
    cases: Sequence[Dict[str, Any]], population: Sequence[Dict[str, Any]], *,
    search_radius_m: float = 1000,
) -> List[Dict[str, Any]]:
    """SaTScan-style spatial scan statistic (simplified Poisson)."""
    total_cases = len(cases)
    total_pop = sum(p.get("population", 1) for p in population)
    expected_rate = total_cases / total_pop if total_pop else 0
    clusters = []
    for pz in population:
        zx, zy = pz.get("x", 0), pz.get("y", 0)
        observed = sum(1 for c in cases
                       if math.hypot(zx - c.get("x", 0), zy - c.get("y", 0)) <= search_radius_m)
        expected = expected_rate * pz.get("population", 1)
        if expected > 0 and observed > expected:
            rr = observed / expected
            llr = observed * math.log(rr) - (observed - expected)
            clusters.append({"zone": pz.get("id", ""), "observed": observed,
                             "expected": round(expected, 2), "relative_risk": round(rr, 3),
                             "llr": round(llr, 3)})
    clusters.sort(key=lambda c: c["llr"], reverse=True)
    return clusters


def epidemiological_mapping(
    cases: Sequence[Dict[str, Any]], population_zones: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute incidence rates per zone."""
    results = []
    for z in population_zones:
        zx, zy, r = z.get("x", 0), z.get("y", 0), z.get("radius", 1000)
        count = sum(1 for c in cases
                    if math.hypot(zx - c.get("x", 0), zy - c.get("y", 0)) <= r)
        pop = z.get("population", 1)
        results.append({**z, "case_count": count,
                        "incidence_per_100k": round(count / pop * 100_000, 2) if pop else 0})
    return results


def contact_tracing_spatial(
    index_case: Dict[str, Any], contacts: Sequence[Dict[str, Any]], *,
    proximity_m: float = 2, duration_min: float = 15,
) -> List[Dict[str, Any]]:
    """Identify close contacts based on spatial proximity and duration."""
    ix, iy = index_case.get("x", 0), index_case.get("y", 0)
    close = []
    for c in contacts:
        dist = math.hypot(ix - c.get("x", 0), iy - c.get("y", 0))
        if dist <= proximity_m and c.get("duration_min", 0) >= duration_min:
            close.append({**c, "distance_m": round(dist, 2), "risk": "high"})
        elif dist <= proximity_m * 3:
            close.append({**c, "distance_m": round(dist, 2), "risk": "medium"})
    return close


def environmental_exposure_assessment(
    population_points: Sequence[Dict[str, Any]],
    pollution_sources: Sequence[Dict[str, Any]], *, decay_rate: float = 0.001,
) -> List[Dict[str, Any]]:
    """Estimate population exposure to pollution sources."""
    results = []
    for p in population_points:
        px, py = p.get("x", 0), p.get("y", 0)
        exposure = sum(
            s.get("intensity", 1.0) * math.exp(
                -decay_rate * math.hypot(px - s.get("x", 0), py - s.get("y", 0)))
            for s in pollution_sources)
        results.append({**p, "exposure": round(exposure, 4)})
    return results


def air_quality_network_design(
    study_area_bounds: Tuple[float, float, float, float],
    existing_stations: Sequence[Tuple[float, float]], *,
    n_new: int = 5, min_spacing_m: float = 2000,
) -> List[Dict[str, float]]:
    """Propose new air-quality monitoring station locations."""
    import random
    xmin, ymin, xmax, ymax = study_area_bounds
    rng = random.Random(42)
    candidates: List[Dict[str, float]] = []
    for _ in range(n_new * 200):
        cx, cy = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        dists = [math.hypot(cx - ex, cy - ey) for ex, ey in existing_stations]
        dists += [math.hypot(cx - c["x"], cy - c["y"]) for c in candidates]
        if all(d >= min_spacing_m for d in dists):
            candidates.append({"x": round(cx, 2), "y": round(cy, 2)})
            if len(candidates) >= n_new:
                break
    return candidates


def water_quality_network_design(
    reaches: Sequence[Dict[str, Any]], *, n_stations: int = 5,
) -> List[Dict[str, Any]]:
    """Select water-quality monitoring stations on stream reaches."""
    scored = []
    for r in reaches:
        score = (r.get("upstream_area_km2", 0) * 0.4
                 + r.get("population_served", 0) * 0.3
                 + r.get("pollution_risk", 0) * 0.3)
        scored.append({**r, "priority_score": round(score, 3)})
    scored.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored[:n_stations]


def soil_contamination_interpolation(
    samples: Sequence[Tuple[float, float, float]],
    grid_bounds: Tuple[float, float, float, float], *,
    grid_size: int = 30, power: float = 2.0,
) -> Dict[str, Any]:
    """IDW interpolation of soil contamination from sample points."""
    xmin, ymin, xmax, ymax = grid_bounds
    dx = (xmax - xmin) / grid_size
    dy = (ymax - ymin) / grid_size
    grid = []
    for i in range(grid_size):
        row = []
        cy = ymin + (i + 0.5) * dy
        for j in range(grid_size):
            cx = xmin + (j + 0.5) * dx
            row.append(round(_inverse_distance_weight(samples, (cx, cy), power), 4))
        grid.append(row)
    return {"grid": grid, "extent": list(grid_bounds), "cell_size": [dx, dy]}


# ===========================================================================
# GROUNDWATER / HYDROLOGY  (1336 – 1338)
# ===========================================================================

def groundwater_flow_bridge(config: Dict[str, Any]) -> Dict[str, Any]:
    """Bridge to MODFLOW via subprocess."""
    import subprocess
    exe = config.get("modflow_exe", "mf6")
    try:
        r = subprocess.run([exe, "-s", config.get("sim_path", ".")],
                           capture_output=True, text=True, timeout=600)
        return {"returncode": r.returncode, "stdout": r.stdout[:4000]}
    except FileNotFoundError:
        return failure_payload(
            code="MODFLOW_EXECUTABLE_NOT_FOUND",
            category="dependency",
            remediation="Install MODFLOW or set config['modflow_exe'] to a valid executable path.",
            error=f"{exe} not found",
            returncode=-1,
        )
    except subprocess.TimeoutExpired:
        return failure_payload(
            code="MODFLOW_TIMEOUT",
            category="timeout",
            remediation="Increase timeout or reduce simulation complexity.",
            error="MODFLOW timed out",
            returncode=-2,
        )


def wellhead_protection_area(
    well: Dict[str, Any], *,
    pumping_rate_m3d: float = 500, porosity: float = 0.25,
    aquifer_thickness_m: float = 20,
    time_days: Sequence[float] = (365, 1825, 3650),
) -> List[Dict[str, Any]]:
    """Calculate wellhead protection zones using calculated fixed-radius."""
    zones = []
    for t in time_days:
        r = math.sqrt(pumping_rate_m3d * t / (math.pi * porosity * aquifer_thickness_m))
        zones.append({"time_days": t, "radius_m": round(r, 1),
                      "center_x": well.get("x", 0), "center_y": well.get("y", 0)})
    return zones


def wetland_delineation(
    parcels: Sequence[Dict[str, Any]], *,
    hydric_soil: bool = True, hydrophytic_vegetation: bool = True,
    wetland_hydrology: bool = True,
) -> List[Dict[str, Any]]:
    """Three-parameter wetland delineation screening."""
    results = []
    for p in parcels:
        has_s = p.get("hydric_soil", hydric_soil)
        has_v = p.get("hydrophytic_vegetation", hydrophytic_vegetation)
        has_h = p.get("wetland_hydrology", wetland_hydrology)
        results.append({**p, "is_wetland": has_s and has_v and has_h,
                        "criteria_met": sum([has_s, has_v, has_h])})
    return results


# ===========================================================================
# NATURAL HAZARDS  (1339 – 1351)
# ===========================================================================

def floodplain_mapping_bridge(config: Dict[str, Any]) -> Dict[str, Any]:
    """Bridge to HEC-RAS for floodplain mapping."""
    import subprocess
    exe = config.get("hecras_exe", "RAS.exe")
    try:
        r = subprocess.run([exe, config.get("project", "")],
                           capture_output=True, text=True, timeout=600)
        return {"returncode": r.returncode, "stdout": r.stdout[:4000]}
    except FileNotFoundError:
        return failure_payload(
            code="HECRAS_EXECUTABLE_NOT_FOUND",
            category="dependency",
            remediation="Install HEC-RAS or set config['hecras_exe'] to a valid executable path.",
            error=f"{exe} not found",
            returncode=-1,
        )
    except subprocess.TimeoutExpired:
        return failure_payload(
            code="HECRAS_TIMEOUT",
            category="timeout",
            remediation="Increase timeout or simplify the floodplain model before retrying.",
            error="HEC-RAS timed out",
            returncode=-2,
        )


def coastal_erosion_model(
    shoreline_points: Sequence[Dict[str, Any]], *,
    wave_height_m: float = 1.5, sea_level_rise_m_yr: float = 0.003, years: int = 50,
) -> List[Dict[str, Any]]:
    """Bruun rule coastal erosion projection."""
    results = []
    for p in shoreline_points:
        slope = p.get("beach_slope", 0.02)
        cd = p.get("closure_depth_m", 8)
        bw = cd / slope if slope else 100
        slr = sea_level_rise_m_yr * years
        retreat = slr * (bw + cd) / cd if cd else 0
        retreat += wave_height_m * 0.1 * years
        results.append({**p, "retreat_m": round(retreat, 1), "years": years})
    return results


def tsunami_inundation_zone(
    coastline_points: Sequence[Dict[str, Any]], *,
    wave_height_m: float = 5, roughness: float = 0.03,
) -> List[Dict[str, Any]]:
    """Estimate tsunami run-up and inundation distance."""
    results = []
    for p in coastline_points:
        slope = p.get("land_slope", 0.01)
        runup = wave_height_m * (1 + 2 * slope ** 0.5)
        inun = runup / slope if slope > 0 else runup * 100
        results.append({**p, "runup_m": round(runup, 2),
                        "inundation_distance_m": round(inun, 1)})
    return results


def earthquake_shaking_intensity(
    epicenter: Tuple[float, float], magnitude: float,
    sites: Sequence[Tuple[float, float]], *, depth_km: float = 10,
) -> List[Dict[str, Any]]:
    """Estimate Modified Mercalli Intensity from magnitude & distance."""
    results = []
    for sx, sy in sites:
        dist = _haversine(epicenter[0], epicenter[1], sx, sy) / 1000
        hypo = math.sqrt(dist ** 2 + depth_km ** 2)
        mmi = 2.085 + 1.428 * magnitude - 1.402 * math.log10(max(hypo, 1))
        results.append({"x": sx, "y": sy, "distance_km": round(dist, 1),
                        "mmi": round(max(1, min(12, mmi)), 1)})
    return results


def liquefaction_susceptibility(sites: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score liquefaction susceptibility from soil properties."""
    results = []
    for s in sites:
        gw = s.get("groundwater_depth_m", 5)
        spt = s.get("spt_n", 15)
        soil = s.get("soil_type", "sand")
        score = 0.0
        if soil in ("sand", "silty_sand", "loose_sand"):
            score += 0.4
        if gw < 3:
            score += 0.3
        elif gw < 6:
            score += 0.15
        if spt < 10:
            score += 0.3
        elif spt < 20:
            score += 0.15
        cat = "high" if score >= 0.6 else "moderate" if score >= 0.3 else "low"
        results.append({**s, "susceptibility_score": round(score, 2), "category": cat})
    return results


def hurricane_surge_zone(
    coastline_points: Sequence[Dict[str, Any]], *, category: int = 3,
) -> List[Dict[str, Any]]:
    """Hurricane storm-surge inundation by Saffir-Simpson category."""
    surge = {1: 1.5, 2: 2.5, 3: 4.0, 4: 5.5, 5: 7.0}.get(category, 3.0)
    results = []
    for p in coastline_points:
        elev = p.get("elevation_m", 0)
        results.append({**p, "surge_height_m": surge, "inundated": elev < surge,
                        "water_depth_m": round(max(0, surge - elev), 2)})
    return results


def tornado_track_analysis(tracks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise tornado track statistics."""
    lengths = [t.get("length_km", 0) for t in tracks]
    widths = [t.get("width_m", 0) for t in tracks]
    ratings = [t.get("ef_rating", 0) for t in tracks]
    return {
        "count": len(tracks),
        "mean_length_km": round(statistics.mean(lengths), 2) if lengths else 0,
        "max_length_km": max(lengths) if lengths else 0,
        "mean_width_m": round(statistics.mean(widths), 1) if widths else 0,
        "ef_distribution": {f"EF{i}": ratings.count(i) for i in range(6)},
    }


def drought_severity_map(stations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simplified Palmer Drought Severity Index proxy."""
    results = []
    for s in stations:
        d = s.get("precipitation_mm", 50) - s.get("potential_et_mm", 80)
        pdsi = d / 25.0
        cat = ("extreme_drought" if pdsi < -4 else "severe_drought" if pdsi < -3
               else "moderate_drought" if pdsi < -2 else "mild_drought" if pdsi < -1
               else "normal" if pdsi < 1 else "moist" if pdsi < 2 else "very_moist")
        results.append({**s, "pdsi_proxy": round(pdsi, 2), "category": cat})
    return results


def avalanche_danger_zone(
    slopes: Sequence[Dict[str, Any]], *,
    critical_angle_deg: float = 30, runout_ratio: float = 2.5,
) -> List[Dict[str, Any]]:
    """Classify avalanche start/runout zones."""
    results = []
    for s in slopes:
        angle = s.get("slope_deg", 0)
        elev_drop = s.get("elevation_drop_m", 0)
        is_start = critical_angle_deg <= angle <= 55
        runout = elev_drop * runout_ratio / math.tan(math.radians(max(angle, 1)))
        results.append({**s, "start_zone": is_start,
                        "runout_distance_m": round(runout, 1)})
    return results


def hazard_mitigation_analysis(
    hazards: Sequence[Dict[str, Any]], assets: Sequence[Dict[str, Any]], *,
    buffer_m: float = 1000,
) -> List[Dict[str, Any]]:
    """Screen assets within hazard zones for mitigation planning."""
    results = []
    for a in assets:
        ax, ay = a.get("x", 0), a.get("y", 0)
        threats = []
        for h in hazards:
            dist = math.hypot(ax - h.get("x", 0), ay - h.get("y", 0))
            if dist <= buffer_m:
                threats.append({"hazard": h.get("type", ""), "distance_m": round(dist, 1)})
        results.append({**a, "threats": threats,
                        "risk_score": round(sum(1 / max(t["distance_m"], 1) for t in threats), 4)})
    return results


def fema_flood_zone_analysis(
    parcels: Sequence[Dict[str, Any]], flood_zones: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Classify parcels by FEMA flood zone proximity."""
    results = []
    for p in parcels:
        px, py = p.get("x", 0), p.get("y", 0)
        zone, min_dist = "X", float("inf")
        for fz in flood_zones:
            d = math.hypot(px - fz.get("x", 0), py - fz.get("y", 0)) - fz.get("radius", 0)
            if d < min_dist:
                min_dist = d
                if d <= 0:
                    zone = fz.get("zone", "AE")
        results.append({**p, "flood_zone": zone,
                        "distance_to_zone_m": round(max(0, min_dist), 1)})
    return results


def insurance_risk_scoring(
    properties: Sequence[Dict[str, Any]], *,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Spatial insurance risk scoring based on hazard proximity."""
    w = weights or {"flood": 0.3, "fire": 0.25, "earthquake": 0.2, "wind": 0.15, "crime": 0.1}
    results = []
    for p in properties:
        score = sum(p.get(f"{k}_risk", 0) * v for k, v in w.items())
        tier = "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"
        results.append({**p, "risk_score": round(score, 4), "risk_tier": tier})
    return results


# ===========================================================================
# AGRICULTURE  (1352 – 1358)
# ===========================================================================

def agriculture_field_boundary(
    ndvi_grid: Sequence[Sequence[float]], *, threshold: float = 0.3,
) -> Dict[str, Any]:
    """Detect field boundaries from NDVI grid using edge detection."""
    if not ndvi_grid:
        return {"boundaries": [], "boundary_pixel_count": 0}
    rows, cols = len(ndvi_grid), len(ndvi_grid[0])
    edges = []
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gx = ndvi_grid[i + 1][j] - ndvi_grid[i - 1][j]
            gy = ndvi_grid[i][j + 1] - ndvi_grid[i][j - 1]
            mag = math.sqrt(gx ** 2 + gy ** 2)
            if mag > threshold:
                edges.append({"row": i, "col": j, "magnitude": round(mag, 4)})
    return {"boundaries": edges, "boundary_pixel_count": len(edges)}


def crop_type_classification(
    pixels: Sequence[Dict[str, Any]], *,
    profiles: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> List[Dict[str, Any]]:
    """Simple crop classification from multi-spectral values."""
    if profiles is None:
        profiles = {
            "corn": {"ndvi": (0.6, 0.9), "ndwi": (-0.2, 0.1)},
            "wheat": {"ndvi": (0.4, 0.7), "ndwi": (-0.3, 0.0)},
            "soybean": {"ndvi": (0.5, 0.8), "ndwi": (-0.1, 0.15)},
            "fallow": {"ndvi": (0.0, 0.3), "ndwi": (-0.5, -0.1)},
        }
    results = []
    for px in pixels:
        best_crop, best_dist = "unknown", float("inf")
        for crop, bands in profiles.items():
            dist = sum((px.get(b, 0) - (lo + hi) / 2) ** 2 for b, (lo, hi) in bands.items())
            if dist < best_dist:
                best_dist, best_crop = dist, crop
        results.append({**px, "crop_type": best_crop,
                        "confidence": round(1 / (1 + best_dist), 3)})
    return results


def yield_estimation(
    fields: Sequence[Dict[str, Any]], *, model: str = "ndvi_linear",
) -> List[Dict[str, Any]]:
    """Estimate crop yield from field attributes."""
    results = []
    for f in fields:
        ndvi = f.get("mean_ndvi", 0.6)
        area = f.get("area_ha", 1)
        yt = max(0, 12 * ndvi - 2) if model == "ndvi_linear" else max(0, 10 * ndvi)
        results.append({**f, "yield_t_ha": round(yt, 2), "total_yield_t": round(yt * area, 2)})
    return results


def variable_rate_application(
    zones: Sequence[Dict[str, Any]], *, target_nutrient_kg_ha: float = 150,
) -> List[Dict[str, Any]]:
    """Calculate variable-rate fertiliser application map."""
    results = []
    for z in zones:
        deficit = max(0, target_nutrient_kg_ha - z.get("soil_nitrogen_kg_ha", 50))
        results.append({**z, "application_rate_kg_ha": round(deficit, 1),
                        "deficit_pct": round(deficit / target_nutrient_kg_ha * 100, 1)})
    return results


def irrigation_scheduling(
    fields: Sequence[Dict[str, Any]], *,
    et_mm_day: float = 5.0, soil_capacity_mm: float = 100,
) -> List[Dict[str, Any]]:
    """Irrigation scheduling based on soil water balance."""
    results = []
    for f in fields:
        current = f.get("soil_moisture_mm", soil_capacity_mm * 0.6)
        depletion = soil_capacity_mm - current
        dts = current / et_mm_day if et_mm_day > 0 else 999
        irrigate = dts < 3
        results.append({**f, "depletion_mm": round(depletion, 1),
                        "days_to_stress": round(dts, 1), "irrigate_now": irrigate,
                        "irrigation_amount_mm": round(depletion, 1) if irrigate else 0})
    return results


def soil_sampling_design(
    field_bounds: Tuple[float, float, float, float], *,
    n_samples: int = 20, strategy: str = "grid",
) -> List[Dict[str, float]]:
    """Design soil sampling locations within a field."""
    import random
    xmin, ymin, xmax, ymax = field_bounds
    rng = random.Random(42)
    if strategy == "random":
        return [{"x": round(rng.uniform(xmin, xmax), 2),
                 "y": round(rng.uniform(ymin, ymax), 2)} for _ in range(n_samples)]
    side = int(math.ceil(math.sqrt(n_samples)))
    dx, dy = (xmax - xmin) / side, (ymax - ymin) / side
    pts: List[Dict[str, float]] = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= n_samples:
                break
            pts.append({"x": round(xmin + (j + 0.5) * dx, 2),
                        "y": round(ymin + (i + 0.5) * dy, 2)})
    return pts


def precision_agriculture_zones(
    field_data: Sequence[Dict[str, Any]], *, n_zones: int = 3,
) -> List[Dict[str, Any]]:
    """Delineate management zones using yield/soil data quantiles."""
    if not field_data:
        return []
    values = sorted(d.get("productivity_index", d.get("yield", 0)) for d in field_data)
    thresholds = [values[int(len(values) * (i + 1) / n_zones) - 1] for i in range(n_zones - 1)]
    results = []
    for d in field_data:
        v = d.get("productivity_index", d.get("yield", 0))
        zone = 1 + sum(1 for t in thresholds if v > t)
        results.append({**d, "management_zone": zone})
    return results


# ===========================================================================
# FORESTRY  (1359 – 1365)
# ===========================================================================

def forest_inventory(plots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise forest inventory plot data."""
    ba = [p.get("basal_area_m2_ha", 0) for p in plots]
    sph = [p.get("stems_per_ha", 0) for p in plots]
    return {"n_plots": len(plots),
            "mean_basal_area_m2_ha": round(statistics.mean(ba), 2) if ba else 0,
            "mean_stems_per_ha": round(statistics.mean(sph), 1) if sph else 0}


def timber_volume_estimation(
    trees: Sequence[Dict[str, Any]], *, form_factor: float = 0.45,
) -> List[Dict[str, Any]]:
    """Estimate timber volume using DBH and height."""
    results = []
    for t in trees:
        dbh_m = t.get("dbh_cm", 30) / 100
        vol = math.pi / 4 * dbh_m ** 2 * t.get("height_m", 20) * form_factor
        results.append({**t, "volume_m3": round(vol, 3)})
    return results


def deforestation_monitoring(
    before_ndvi: Sequence[Sequence[float]],
    after_ndvi: Sequence[Sequence[float]], *, threshold: float = -0.2,
) -> Dict[str, Any]:
    """Detect deforestation from NDVI change."""
    if not before_ndvi or not after_ndvi:
        return {"deforested_pixels": 0, "total_pixels": 0, "deforestation_pct": 0}
    rows = min(len(before_ndvi), len(after_ndvi))
    cols = min(len(before_ndvi[0]), len(after_ndvi[0]))
    deforested = sum(1 for i in range(rows) for j in range(cols)
                     if after_ndvi[i][j] - before_ndvi[i][j] < threshold)
    total = rows * cols
    return {"deforested_pixels": deforested, "total_pixels": total,
            "deforestation_pct": round(deforested / total * 100, 2) if total else 0}


def reforestation_site_selection(
    candidates: Sequence[Dict[str, Any]], *,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Score candidate sites for reforestation suitability."""
    w = weights or {"soil_quality": 0.3, "slope_suitability": 0.25,
                    "water_access": 0.25, "connectivity": 0.2}
    return _score_features(candidates, list(w.keys()), list(w.values()))


def urban_tree_canopy(blocks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calculate urban tree canopy coverage metrics."""
    results = []
    for b in blocks:
        total = b.get("area_m2", 1)
        canopy = b.get("canopy_m2", 0)
        results.append({**b, "canopy_pct": round(canopy / total * 100, 1) if total else 0,
                        "gap_m2": round(total - canopy, 1)})
    return results


def green_infrastructure_mapping(features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify and summarise green infrastructure features."""
    cats: Dict[str, int] = {}
    area = 0.0
    for f in features:
        cats[f.get("gi_type", "other")] = cats.get(f.get("gi_type", "other"), 0) + 1
        area += f.get("area_m2", 0)
    return {"feature_count": len(features), "categories": cats, "total_area_m2": round(area, 1)}


# ===========================================================================
# URBAN / ENERGY / UTILITIES  (1367 – 1381)
# ===========================================================================

def cool_roof_benefit(
    buildings: Sequence[Dict[str, Any]], *,
    albedo_increase: float = 0.4, cooling_factor: float = 0.3,
) -> List[Dict[str, Any]]:
    """Estimate cool-roof temperature reduction and energy savings."""
    results = []
    for b in buildings:
        roof = b.get("roof_area_m2", 100)
        temp_red = albedo_increase * cooling_factor * 10
        results.append({**b, "temp_reduction_c": round(temp_red, 1),
                        "energy_savings_kwh_yr": round(roof * albedo_increase * 50, 0)})
    return results


def energy_efficiency_scoring(buildings: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score buildings on energy efficiency."""
    results = []
    for b in buildings:
        eui = b.get("eui_kwh_m2", 200)
        score = max(0, min(100, 100 - (eui - 50) * 0.5))
        grade = ("A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40
                 else "D" if score >= 20 else "F")
        results.append({**b, "efficiency_score": round(score, 1), "grade": grade})
    return results


def renewable_energy_siting(
    candidates: Sequence[Dict[str, Any]], *,
    energy_type: str = "solar", weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Score candidate sites for renewable energy development."""
    if energy_type == "wind":
        w = weights or {"wind_speed": 0.4, "grid_distance": 0.2, "road_access": 0.2, "land_cost": 0.2}
    else:
        w = weights or {"solar_irradiance": 0.4, "grid_distance": 0.2, "slope": 0.2, "land_cost": 0.2}
    return _score_features(candidates, list(w.keys()), list(w.values()))


def transmission_line_routing(
    start: Tuple[float, float], end: Tuple[float, float],
    obstacles: Sequence[Dict[str, Any]], *, grid_size: int = 50,
) -> Dict[str, Any]:
    """Least-cost path for transmission line routing."""
    sx, sy = start
    ex, ey = end
    dx = (ex - sx) / grid_size
    dy = (ey - sy) / grid_size
    costs = [[1.0] * grid_size for _ in range(grid_size)]
    for obs in obstacles:
        ox, oy, r = obs.get("x", 0), obs.get("y", 0), obs.get("radius", 0)
        for i in range(grid_size):
            for j in range(grid_size):
                if math.hypot(sx + (j + 0.5) * dx - ox, sy + (i + 0.5) * dy - oy) <= r:
                    costs[i][j] = 999
    path = [(0, 0)]
    ci, cj = 0, 0
    gi, gj = grid_size - 1, grid_size - 1
    while (ci, cj) != (gi, gj):
        best, best_cost = None, float("inf")
        for di, dj in [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0)]:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                c = costs[ni][nj] + math.hypot(gi - ni, gj - nj)
                if c < best_cost:
                    best_cost, best = c, (ni, nj)
        if best is None:
            break
        ci, cj = best
        path.append(best)
    waypoints = [{"x": round(sx + (j + 0.5) * dx, 2),
                  "y": round(sy + (i + 0.5) * dy, 2)} for i, j in path]
    return {"waypoints": waypoints, "segments": len(path) - 1}


def pipeline_routing(
    start: Tuple[float, float], end: Tuple[float, float],
    constraints: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pipeline routing reusing transmission_line_routing logic."""
    return transmission_line_routing(start, end, constraints)


def right_of_way_analysis(
    centerline_points: Sequence[Tuple[float, float]], *, width_m: float = 30,
) -> List[Dict[str, Any]]:
    """Generate right-of-way buffer zones along a centerline."""
    return [{"x": x, "y": y, "buffer_m": width_m / 2, "segment": i}
            for i, (x, y) in enumerate(centerline_points)]


def easement_mapping(
    parcels: Sequence[Dict[str, Any]], easements: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Identify parcels affected by easements."""
    results = []
    for p in parcels:
        px, py = p.get("x", 0), p.get("y", 0)
        affected = [e.get("id", "") for e in easements
                    if math.hypot(px - e.get("x", 0), py - e.get("y", 0)) <= e.get("width_m", 10)]
        results.append({**p, "easements": affected, "has_easement": len(affected) > 0})
    return results


def utility_pole_inventory(poles: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise utility pole inventory statistics."""
    materials: Dict[str, int] = {}
    ages = []
    for p in poles:
        m = p.get("material", "wood")
        materials[m] = materials.get(m, 0) + 1
        ages.append(p.get("age_years", 0))
    return {"total_poles": len(poles), "by_material": materials,
            "mean_age_years": round(statistics.mean(ages), 1) if ages else 0}


def subsurface_utility_mapping(utilities: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify subsurface utilities by quality level (ASCE 38)."""
    levels = {"A": 0, "B": 0, "C": 0, "D": 0}
    for u in utilities:
        lvl = u.get("quality_level", "D")
        levels[lvl] = levels.get(lvl, 0) + 1
    return {"total": len(utilities), "by_quality_level": levels}


def joint_trench_design(
    utilities: Sequence[Dict[str, Any]], *, min_separation_m: float = 0.3,
) -> Dict[str, Any]:
    """Design joint trench cross-section with minimum separations."""
    sorted_u = sorted(utilities, key=lambda u: u.get("diameter_m", 0.1), reverse=True)
    positions = []
    x = 0.0
    for u in sorted_u:
        d = u.get("diameter_m", 0.1)
        positions.append({**u, "x_offset_m": round(x + d / 2, 3)})
        x += d + min_separation_m
    depth = max((u.get("depth_m", 1) for u in utilities), default=1) + 0.3
    return {"utilities": positions, "trench_width_m": round(x, 3),
            "trench_depth_m": round(depth, 2)}


def duct_bank_capacity(
    duct_bank: Dict[str, Any], cables: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check duct bank capacity for cable installation."""
    total = duct_bank.get("ducts", 4)
    duct_d = duct_bank.get("duct_diameter_mm", 100)
    used = sum(1 for c in cables if c.get("installed", False))
    fill = [round((c.get("diameter_mm", 30) / duct_d) ** 2 * 100, 1) for c in cables]
    return {"total_ducts": total, "used": used, "available": total - used,
            "fill_ratios_pct": fill}


def manhole_inspection_tracking(
    manholes: Sequence[Dict[str, Any]], *, inspection_interval_days: int = 365,
) -> List[Dict[str, Any]]:
    """Track manhole inspection schedules."""
    now = time.time()
    results = []
    for m in manholes:
        days = (now - m.get("last_inspection_epoch", 0)) / 86400
        overdue = days > inspection_interval_days
        results.append({**m, "days_since_inspection": round(days, 0), "overdue": overdue})
    return results


def valve_inventory_management(valves: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise valve inventory by type and condition."""
    by_type: Dict[str, int] = {}
    by_cond: Dict[str, int] = {}
    for v in valves:
        by_type[v.get("type", "gate")] = by_type.get(v.get("type", "gate"), 0) + 1
        by_cond[v.get("condition", "good")] = by_cond.get(v.get("condition", "good"), 0) + 1
    return {"total": len(valves), "by_type": by_type, "by_condition": by_cond}


def hydrant_inspection_tracking(
    hydrants: Sequence[Dict[str, Any]], *, inspection_interval_days: int = 180,
) -> List[Dict[str, Any]]:
    """Track fire hydrant inspection schedules."""
    now = time.time()
    results = []
    for h in hydrants:
        days = (now - h.get("last_inspection_epoch", 0)) / 86400
        gpm = h.get("flow_gpm", 0)
        color = ("red" if gpm < 500 else "orange" if gpm < 1000
                 else "green" if gpm < 1500 else "blue")
        results.append({**h, "days_since_inspection": round(days, 0),
                        "overdue": days > inspection_interval_days, "nfpa_color": color})
    return results


# ===========================================================================
# FIELD / MOBILE  (1384 – 1395)
# ===========================================================================

def mobile_work_order_schema() -> Dict[str, Any]:
    """Return a work-order JSON schema for mobile integration."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"}, "asset_id": {"type": "string"},
            "type": {"type": "string", "enum": ["inspection", "repair", "install", "remove"]},
            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
            "location": {"type": "object",
                         "properties": {"x": {"type": "number"}, "y": {"type": "number"}}},
            "status": {"type": "string", "enum": ["pending", "in_progress", "complete"]},
        },
        "required": ["id", "asset_id", "type", "location"],
    }


def field_inspection_form(
    asset_type: str, *, fields: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate a field inspection form definition."""
    default = [
        {"name": "inspector", "type": "text", "required": True},
        {"name": "date", "type": "date", "required": True},
        {"name": "condition", "type": "choice",
         "options": ["good", "fair", "poor", "critical"], "required": True},
        {"name": "photo", "type": "photo", "required": False},
        {"name": "location", "type": "gps", "required": True},
    ]
    return {"asset_type": asset_type, "fields": list(fields) if fields else default}


def photo_gps_attachment(
    photo_path: str, latitude: float, longitude: float, *,
    accuracy_m: float = 5.0, heading: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a GPS-tagged photo attachment record."""
    return {"photo": Path(photo_path).name, "latitude": latitude,
            "longitude": longitude, "accuracy_m": accuracy_m,
            "heading": heading, "timestamp": time.time()}


def real_time_vehicle_tracking(
    vehicles: Sequence[Dict[str, Any]], *, stale_threshold_s: float = 300,
) -> List[Dict[str, Any]]:
    """Evaluate vehicle tracking freshness and status."""
    now = time.time()
    return [{**v, "age_s": round(now - v.get("last_ping_epoch", 0), 0),
             "status": "stale" if now - v.get("last_ping_epoch", 0) > stale_threshold_s
             else "active"} for v in vehicles]


def avl_integration(positions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Process automatic vehicle location (AVL) data."""
    if not positions:
        return {"vehicles": 0, "active": 0}
    now = time.time()
    active = sum(1 for p in positions if now - p.get("timestamp", 0) < 300)
    speeds = [p.get("speed_kmh", 0) for p in positions]
    return {"vehicles": len(positions), "active": active,
            "mean_speed_kmh": round(statistics.mean(speeds), 1) if speeds else 0}


def _nearest_neighbour_route(
    items: Sequence[Dict[str, Any]], depot: Tuple[float, float],
) -> List[Dict[str, Any]]:
    """Greedy nearest-neighbour routing helper."""
    remaining = list(items)
    route = []
    cx, cy = depot
    while remaining:
        best_idx, best_dist = 0, float("inf")
        for i, r in enumerate(remaining):
            d = math.hypot(cx - r.get("x", 0), cy - r.get("y", 0))
            if d < best_dist:
                best_dist, best_idx = d, i
        seg = remaining.pop(best_idx)
        route.append({**seg, "order": len(route) + 1})
        cx = seg.get("x_end", seg.get("x", 0))
        cy = seg.get("y_end", seg.get("y", 0))
    return route


def snow_plough_route_optimisation(
    roads: Sequence[Dict[str, Any]], *, depot: Tuple[float, float] = (0, 0),
) -> List[Dict[str, Any]]:
    """Greedy nearest-neighbour route for snow ploughing."""
    return _nearest_neighbour_route(roads, depot)


def refuse_collection_route(
    stops: Sequence[Dict[str, Any]], *,
    depot: Tuple[float, float] = (0, 0), capacity_kg: float = 10000,
) -> List[List[Dict[str, Any]]]:
    """Route optimisation for refuse collection with capacity constraint."""
    remaining = list(stops)
    trips: List[List[Dict[str, Any]]] = []
    while remaining:
        trip: List[Dict[str, Any]] = []
        load, cx, cy = 0.0, depot[0], depot[1]
        while remaining:
            best_idx, best_dist = -1, float("inf")
            for i, s in enumerate(remaining):
                if load + s.get("weight_kg", 50) > capacity_kg:
                    continue
                d = math.hypot(cx - s.get("x", 0), cy - s.get("y", 0))
                if d < best_dist:
                    best_dist, best_idx = d, i
            if best_idx < 0:
                break
            s = remaining.pop(best_idx)
            trip.append(s)
            load += s.get("weight_kg", 50)
            cx, cy = s.get("x", 0), s.get("y", 0)
        trips.append(trip)
    return trips


def street_sweeping_route(
    segments: Sequence[Dict[str, Any]], *, depot: Tuple[float, float] = (0, 0),
) -> List[Dict[str, Any]]:
    """Street sweeping route using nearest-neighbour heuristic."""
    return _nearest_neighbour_route(segments, depot)


def meter_reading_route(
    meters: Sequence[Dict[str, Any]], *, depot: Tuple[float, float] = (0, 0),
) -> List[Dict[str, Any]]:
    """Meter reading route using nearest-neighbour heuristic."""
    remaining = list(meters)
    route = []
    cx, cy = depot
    while remaining:
        best_idx, best_dist = 0, float("inf")
        for i, m in enumerate(remaining):
            d = math.hypot(cx - m.get("x", 0), cy - m.get("y", 0))
            if d < best_dist:
                best_dist, best_idx = d, i
        m = remaining.pop(best_idx)
        route.append({**m, "sequence": len(route) + 1})
        cx, cy = m.get("x", 0), m.get("y", 0)
    return route


# ===========================================================================
# PLANNING & TRANSPORTATION  (1402 – 1424)
# ===========================================================================

def tax_parcel_valuation(parcels: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calculate assessed property values."""
    results = []
    for p in parcels:
        land = p.get("land_value_per_m2", 100) * p.get("area_m2", 500)
        assessed = (land + p.get("improvement_value", 0)) * p.get("assessment_ratio", 0.85)
        results.append({**p, "assessed_value": round(assessed, 2), "land_value": round(land, 2)})
    return results


def property_tax_equity(parcels: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyse property tax equity using coefficient of dispersion."""
    ratios = [p.get("assessed_value", 0) / p.get("market_value", 1)
              for p in parcels if p.get("market_value", 0) > 0]
    if not ratios:
        return {"cod": 0, "median_ratio": 0, "n": 0}
    med = statistics.median(ratios)
    cod = statistics.mean(abs(r - med) for r in ratios) / med * 100 if med else 0
    return {"cod": round(cod, 2), "median_ratio": round(med, 4), "n": len(ratios)}


def land_use_change_detection(
    before: Sequence[Dict[str, Any]], after: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Detect land-use changes between two time periods."""
    bm = {b.get("id"): b.get("land_use") for b in before}
    changes = [{"id": a.get("id"), "from": bm.get(a.get("id")), "to": a.get("land_use")}
               for a in after if bm.get(a.get("id")) and bm[a["id"]] != a.get("land_use")]
    return {"changes": changes, "change_count": len(changes),
            "unchanged": len(after) - len(changes)}


def urban_growth_boundary(
    developed: Sequence[Dict[str, Any]], *, density_threshold: float = 500,
) -> List[Dict[str, Any]]:
    """Identify parcels at the urban growth boundary."""
    results = []
    for d in developed:
        density = d.get("pop_density_per_km2", 0)
        results.append({**d, "at_ugb": abs(density - density_threshold) < density_threshold * 0.3,
                        "status": "urban" if density >= density_threshold else "rural"})
    return results


def complete_streets_analysis(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score street segments on complete-streets criteria."""
    results = []
    for s in segments:
        score = (20 * s.get("has_sidewalk", 0) + 20 * s.get("has_bike_lane", 0)
                 + 20 * s.get("has_transit", 0) + 15 * s.get("has_crosswalk", 0)
                 + (15 if s.get("speed_limit_kmh", 50) <= 40 else 0)
                 + 10 * s.get("has_street_trees", 0))
        grade = ("A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40
                 else "D" if score >= 20 else "F")
        results.append({**s, "complete_streets_score": score, "grade": grade})
    return results


def level_of_traffic_stress(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Classify road segments by Level of Traffic Stress (LTS 1-4)."""
    results = []
    for s in segments:
        speed = s.get("speed_limit_kmh", 50)
        lanes = s.get("lanes", 2)
        bl = s.get("has_bike_lane", False)
        if bl and speed <= 40 and lanes <= 2:
            lts = 1
        elif speed <= 40 or (bl and lanes <= 3):
            lts = 2
        elif speed <= 60:
            lts = 3
        else:
            lts = 4
        results.append({**s, "lts": lts})
    return results


def crash_heatmap(
    crashes: Sequence[Dict[str, Any]], *, grid_size: int = 30, bandwidth: float = 300,
) -> Dict[str, Any]:
    """Generate crash density heatmap."""
    return _kde_grid([(c.get("x", 0), c.get("y", 0)) for c in crashes], grid_size, bandwidth)


def speed_study_analysis(observations: Sequence[float]) -> Dict[str, Any]:
    """Analyse speed study data."""
    if not observations:
        return {"n": 0}
    s = sorted(observations)
    n = len(s)
    return {"n": n, "mean_kmh": round(statistics.mean(s), 1),
            "p85_kmh": round(s[int(n * 0.85)], 1),
            "std_dev": round(statistics.stdev(s), 1) if n > 1 else 0}


def turning_movement_count(counts: Dict[str, int]) -> Dict[str, Any]:
    """Summarise turning movement count data at an intersection."""
    total = sum(counts.values())
    return {"total_vehicles": total, "movements": counts,
            "percentages": {k: round(v / total * 100, 1) if total else 0
                            for k, v in counts.items()}}


def parking_inventory(lots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise parking inventory."""
    total = sum(l.get("spaces", 0) for l in lots)
    occupied = sum(l.get("occupied", 0) for l in lots)
    return {"lots": len(lots), "total_spaces": total, "occupied": occupied,
            "available": total - occupied,
            "occupancy_pct": round(occupied / total * 100, 1) if total else 0}


def curb_management(segments: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify curb usage along street segments."""
    uses: Dict[str, float] = {}
    for s in segments:
        u = s.get("curb_use", "parking")
        uses[u] = uses.get(u, 0) + s.get("length_m", 0)
    return {"total_length_m": round(sum(uses.values()), 1),
            "by_use": {k: round(v, 1) for k, v in uses.items()}}


def ada_ramp_inventory(ramps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise ADA curb ramp compliance."""
    compliant = sum(1 for r in ramps if r.get("compliant", False))
    return {"total": len(ramps), "compliant": compliant,
            "non_compliant": len(ramps) - compliant,
            "compliance_pct": round(compliant / len(ramps) * 100, 1) if ramps else 0}


def sidewalk_condition_assessment(
    segments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rate sidewalk condition on PCI-like scale."""
    return [{**s, "rating": "good" if s.get("condition_index", 70) >= 70
             else "fair" if s.get("condition_index", 70) >= 40 else "poor",
             "needs_repair": s.get("condition_index", 70) < 40} for s in segments]


def trail_network_analysis(trails: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise trail network statistics."""
    total = sum(t.get("length_km", 0) for t in trails)
    surfaces: Dict[str, float] = {}
    for t in trails:
        k = t.get("surface", "dirt")
        surfaces[k] = surfaces.get(k, 0) + t.get("length_km", 0)
    return {"total_km": round(total, 2), "trail_count": len(trails),
            "by_surface_km": {k: round(v, 2) for k, v in surfaces.items()}}


def park_access_analysis(
    population_points: Sequence[Dict[str, Any]],
    parks: Sequence[Dict[str, Any]], *, walk_distance_m: float = 800,
) -> Dict[str, Any]:
    """Calculate population within walking distance of parks."""
    served, total = 0, 0
    for p in population_points:
        pop = p.get("population", 1)
        total += pop
        px, py = p.get("x", 0), p.get("y", 0)
        if any(math.hypot(px - pk.get("x", 0), py - pk.get("y", 0)) <= walk_distance_m
               for pk in parks):
            served += pop
    return {"total_population": total, "served_population": served,
            "access_pct": round(served / total * 100, 1) if total else 0}


def recreational_facility_siting(
    candidates: Sequence[Dict[str, Any]],
    population_points: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Score candidate sites for recreational facility placement."""
    results = []
    for c in candidates:
        cx, cy = c.get("x", 0), c.get("y", 0)
        pop = sum(p.get("population", 1) for p in population_points
                  if math.hypot(cx - p.get("x", 0), cy - p.get("y", 0)) <= 2000)
        results.append({**c, "population_served": pop})
    results.sort(key=lambda x: x["population_served"], reverse=True)
    return results


def open_space_preservation_scoring(
    parcels: Sequence[Dict[str, Any]], *, weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Score parcels for open-space preservation priority."""
    w = weights or {"ecological_value": 0.3, "connectivity": 0.25,
                    "scenic_value": 0.2, "threat_of_development": 0.25}
    return _score_features(parcels, list(w.keys()), list(w.values()))


def scenic_viewpoint_analysis(
    viewpoints: Sequence[Dict[str, Any]], targets: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Score viewpoints by number of visible scenic targets."""
    results = []
    for vp in viewpoints:
        vx, vy, vz = vp.get("x", 0), vp.get("y", 0), vp.get("z", 0)
        visible = sum(1 for t in targets
                      if t.get("z", 0) >= (vz + t.get("z", 0)) / 2 - 10
                      or math.hypot(vx - t.get("x", 0), vy - t.get("y", 0)) < 1)
        results.append({**vp, "visible_targets": visible})
    return results


def dark_sky_analysis(
    light_sources: Sequence[Dict[str, Any]],
    observation_point: Tuple[float, float], *, radius_m: float = 5000,
) -> Dict[str, Any]:
    """Estimate light pollution at an observation point."""
    ox, oy = observation_point
    total_flux = 0.0
    count = 0
    for ls in light_sources:
        dist = math.hypot(ox - ls.get("x", 0), oy - ls.get("y", 0))
        if dist <= radius_m:
            count += 1
            total_flux += ls.get("lumens", 1000) / (dist ** 2 + 1)
    bortle = min(9, max(1, int(math.log10(total_flux + 1) * 2 + 1)))
    return {"total_flux": round(total_flux, 2), "sources_in_range": count,
            "bortle_class": bortle}


# ===========================================================================
# CULTURAL / HISTORIC  (1425 – 1428)
# ===========================================================================

def cultural_resource_inventory(resources: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise cultural resource inventory."""
    by_type: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    for r in resources:
        by_type[r.get("type", "other")] = by_type.get(r.get("type", "other"), 0) + 1
        by_status[r.get("nrhp_status", "not_evaluated")] = by_status.get(
            r.get("nrhp_status", "not_evaluated"), 0) + 1
    return {"total": len(resources), "by_type": by_type, "by_status": by_status}


def archaeological_site_sensitivity(
    parcels: Sequence[Dict[str, Any]], *, weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Model archaeological site sensitivity."""
    w = weights or {"proximity_to_water": 0.3, "slope": 0.2,
                    "soil_type": 0.2, "known_sites_nearby": 0.3}
    return _score_features(parcels, list(w.keys()), list(w.values()))


def historic_district_boundary(
    properties: Sequence[Dict[str, Any]], *, contributing_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate historic district boundary criteria."""
    contributing = sum(1 for p in properties if p.get("contributing", False))
    pct = contributing / len(properties) * 100 if properties else 0
    return {"total_properties": len(properties), "contributing": contributing,
            "contributing_pct": round(pct, 1),
            "meets_threshold": pct >= contributing_threshold * 100}


def cemetery_mapping(plots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise cemetery plot inventory."""
    occupied = sum(1 for p in plots if p.get("occupied", False))
    return {"total_plots": len(plots), "occupied": occupied,
            "available": len(plots) - occupied,
            "occupancy_pct": round(occupied / len(plots) * 100, 1) if plots else 0}


# ===========================================================================
# EMERGENCY MANAGEMENT  (1434 – 1446)
# ===========================================================================

def e911_msag_validation(
    addresses: Sequence[Dict[str, Any]], msag_ranges: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate addresses against Master Street Address Guide (MSAG)."""
    results = []
    for addr in addresses:
        street = addr.get("street", "").upper()
        number = addr.get("number", 0)
        matched = any(r.get("street", "").upper() == street
                      and r.get("low", 0) <= number <= r.get("high", 99999)
                      for r in msag_ranges)
        results.append({**addr, "msag_valid": matched})
    return results


def ng911_gis_data_model(layers: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Return NG-911 GIS data model schema (NENA i3)."""
    default = ["RoadCenterline", "AddressPoint", "ServiceBoundary_PSAP",
               "ServiceBoundary_Fire", "ServiceBoundary_Law", "ServiceBoundary_EMS"]
    schema = {}
    for layer in (layers or default):
        schema[layer] = {
            "required_fields": ["DiscrpAgID", "DateUpdate", "Effective", "Expire"],
            "geometry_type": ("Point" if "Point" in layer
                              else "Polygon" if "Boundary" in layer else "LineString"),
        }
    return schema


def search_and_rescue_coverage(
    search_areas: Sequence[Dict[str, Any]], teams: Sequence[Dict[str, Any]], *,
    search_speed_kmh: float = 2,
) -> Dict[str, Any]:
    """Estimate search-and-rescue area coverage."""
    total_area = sum(a.get("area_km2", 0) for a in search_areas)
    cap = sum(t.get("members", 4) * search_speed_kmh * t.get("hours", 8) for t in teams)
    return {"total_area_km2": round(total_area, 2),
            "search_capacity_km2": round(cap, 2),
            "coverage_pct": round(min(100, cap / total_area * 100), 1) if total_area else 100}


def incident_command_post_siting(
    incident: Dict[str, Any], candidates: Sequence[Dict[str, Any]], *,
    min_distance_m: float = 300, max_distance_m: float = 2000,
) -> List[Dict[str, Any]]:
    """Rank candidate ICP locations by suitability."""
    ix, iy = incident.get("x", 0), incident.get("y", 0)
    optimal = (min_distance_m + max_distance_m) / 2
    results = []
    for c in candidates:
        dist = math.hypot(ix - c.get("x", 0), iy - c.get("y", 0))
        if min_distance_m <= dist <= max_distance_m:
            ds = max(0, 1 - abs(dist - optimal) / optimal)
            amenity = 0.3 * c.get("has_parking", 0) + 0.2 * c.get("has_power", 0)
            results.append({**c, "distance_m": round(dist, 1),
                            "suitability_score": round(ds + amenity, 3)})
    results.sort(key=lambda x: x["suitability_score"], reverse=True)
    return results


def radiation_dispersion_model(
    source: Dict[str, Any], *,
    wind_speed_ms: float = 5, wind_direction_deg: float = 270,
    release_rate_bq_s: float = 1e6,
    distances_m: Sequence[float] = (500, 1000, 2000, 5000),
) -> List[Dict[str, Any]]:
    """Gaussian plume model for radiation dispersion."""
    results = []
    for d in distances_m:
        sy = 0.22 * d / (1 + 0.0001 * d) ** 0.5
        sz = 0.20 * d
        conc = (release_rate_bq_s / (2 * math.pi * sy * sz * wind_speed_ms)
                if sy and sz and wind_speed_ms else 0)
        results.append({"distance_m": d, "concentration_bq_m3": round(conc, 2)})
    return results


def pandemic_response_zones(
    zones: Sequence[Dict[str, Any]], *, case_threshold_per_100k: float = 100,
) -> List[Dict[str, Any]]:
    """Classify zones by pandemic risk level."""
    results = []
    for z in zones:
        rate = z.get("cases_per_100k", 0)
        level = ("critical" if rate >= case_threshold_per_100k * 3
                 else "high" if rate >= case_threshold_per_100k
                 else "moderate" if rate >= case_threshold_per_100k * 0.5 else "low")
        results.append({**z, "risk_level": level})
    return results


def vaccine_distribution_optimisation(
    sites: Sequence[Dict[str, Any]], supply: int,
) -> List[Dict[str, Any]]:
    """Allocate vaccine supply proportionally to population served."""
    total_pop = sum(s.get("population", 0) for s in sites)
    return [{**s, "allocation": int(supply * s.get("population", 0) / total_pop)
             if total_pop else 0} for s in sites]


def testing_site_accessibility(
    population_points: Sequence[Dict[str, Any]],
    testing_sites: Sequence[Dict[str, Any]], *, drive_distance_m: float = 16000,
) -> Dict[str, Any]:
    """Evaluate testing site accessibility for population."""
    total, served = 0, 0
    for p in population_points:
        pop = p.get("population", 1)
        total += pop
        px, py = p.get("x", 0), p.get("y", 0)
        if any(math.hypot(px - ts.get("x", 0), py - ts.get("y", 0)) <= drive_distance_m
               for ts in testing_sites):
            served += pop
    return {"total_population": total, "served": served,
            "access_pct": round(served / total * 100, 1) if total else 0}


# ===========================================================================
# LOGISTICS  (1447 – 1454)
# ===========================================================================

def supply_chain_spatial_analysis(
    facilities: Sequence[Dict[str, Any]], demand_points: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign demand points to nearest facility."""
    results = []
    for d in demand_points:
        dx, dy = d.get("x", 0), d.get("y", 0)
        best_fac, best_dist = "", float("inf")
        for f in facilities:
            dist = math.hypot(dx - f.get("x", 0), dy - f.get("y", 0))
            if dist < best_dist:
                best_dist, best_fac = dist, f.get("id", "")
        results.append({**d, "assigned_facility": best_fac,
                        "distance_m": round(best_dist, 1)})
    return results


def warehouse_location_optimisation(
    candidates: Sequence[Dict[str, Any]], demand_points: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank warehouse candidates by weighted distance to demand."""
    results = []
    for c in candidates:
        cx, cy = c.get("x", 0), c.get("y", 0)
        wd = sum(math.hypot(cx - d.get("x", 0), cy - d.get("y", 0)) * d.get("demand", 1)
                 for d in demand_points)
        results.append({**c, "weighted_distance": round(wd, 1)})
    results.sort(key=lambda x: x["weighted_distance"])
    return results


def last_mile_delivery_analysis(
    depot: Tuple[float, float], deliveries: Sequence[Dict[str, Any]], *,
    vehicle_capacity: int = 50,
) -> List[List[Dict[str, Any]]]:
    """Split deliveries into routes with capacity constraint."""
    return refuse_collection_route(deliveries, depot=depot,
                                   capacity_kg=vehicle_capacity * 20)


def port_hinterland_analysis(
    port: Dict[str, Any], regions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Analyse port hinterland by trade volume and distance."""
    px, py = port.get("x", 0), port.get("y", 0)
    results = []
    for r in regions:
        dist = math.hypot(px - r.get("x", 0), py - r.get("y", 0))
        gravity = r.get("trade_volume", 0) / (dist ** 2 + 1)
        results.append({**r, "distance_m": round(dist, 1), "gravity_score": round(gravity, 4)})
    results.sort(key=lambda x: x["gravity_score"], reverse=True)
    return results


def airspace_management_zones(
    airport: Dict[str, Any], *, zone_radii_nm: Sequence[float] = (5, 10, 30),
) -> List[Dict[str, Any]]:
    """Define airspace management zones around an airport."""
    names = ["Class D", "Class C", "Class B"]
    return [{"center_x": airport.get("x", 0), "center_y": airport.get("y", 0),
             "radius_m": round(r * 1852, 0),
             "class": names[i] if i < len(names) else f"Zone{i + 1}"}
            for i, r in enumerate(zone_radii_nm)]


def drone_flight_path(
    waypoints: Sequence[Tuple[float, float, float]], *, max_altitude_m: float = 120,
) -> Dict[str, Any]:
    """Plan a drone flight path with altitude constraints."""
    total = 0.0
    clamped = []
    for i, (x, y, z) in enumerate(waypoints):
        zc = min(z, max_altitude_m)
        clamped.append({"x": x, "y": y, "z": zc})
        if i > 0:
            px, py, pz = waypoints[i - 1]
            total += math.sqrt((x - px) ** 2 + (y - py) ** 2
                               + (zc - min(pz, max_altitude_m)) ** 2)
    return {"waypoints": clamped, "total_distance_m": round(total, 1)}


def uas_airspace_deconfliction(
    flights: Sequence[Dict[str, Any]], *, separation_m: float = 500,
) -> List[Dict[str, Any]]:
    """Check UAS flight plans for airspace conflicts."""
    conflicts = []
    for i, a in enumerate(flights):
        for j in range(i + 1, len(flights)):
            b = flights[j]
            dist = math.sqrt((a.get("x", 0) - b.get("x", 0)) ** 2
                             + (a.get("y", 0) - b.get("y", 0)) ** 2
                             + (a.get("z", 0) - b.get("z", 0)) ** 2)
            if dist < separation_m:
                conflicts.append({"flight_a": a.get("id", i), "flight_b": b.get("id", j),
                                  "separation_m": round(dist, 1)})
    return conflicts


# ===========================================================================
# MARITIME  (1455 – 1466)
# ===========================================================================

def ais_track_analysis(positions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyse AIS vessel track data."""
    if not positions:
        return {"positions": 0}
    speeds = [p.get("sog_knots", 0) for p in positions]
    total_nm = sum(
        _haversine(positions[i - 1].get("lat", 0), positions[i - 1].get("lon", 0),
                   positions[i].get("lat", 0), positions[i].get("lon", 0)) / 1852
        for i in range(1, len(positions)))
    return {"positions": len(positions), "total_distance_nm": round(total_nm, 2),
            "mean_speed_knots": round(statistics.mean(speeds), 1) if speeds else 0}


def vessel_traffic_density(
    tracks: Sequence[Dict[str, Any]], *, grid_size: int = 20,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """Compute vessel traffic density grid."""
    pts = [(t.get("lon", t.get("x", 0)), t.get("lat", t.get("y", 0))) for t in tracks]
    if not pts:
        return {"grid": [], "max_count": 0}
    if bounds is None:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        bounds = (min(xs), min(ys), max(xs), max(ys))
    xmin, ymin, xmax, ymax = bounds
    dx = (xmax - xmin) / grid_size if xmax > xmin else 1
    dy = (ymax - ymin) / grid_size if ymax > ymin else 1
    grid = [[0] * grid_size for _ in range(grid_size)]
    for px, py in pts:
        ci = min(grid_size - 1, max(0, int((py - ymin) / dy)))
        cj = min(grid_size - 1, max(0, int((px - xmin) / dx)))
        grid[ci][cj] += 1
    return {"grid": grid, "max_count": max(max(row) for row in grid),
            "extent": list(bounds)}


def port_facility_mapping(facilities: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise port facility inventory."""
    by_type: Dict[str, int] = {}
    cap = 0
    for f in facilities:
        t = f.get("type", "berth")
        by_type[t] = by_type.get(t, 0) + 1
        cap += f.get("capacity_teu", 0)
    return {"facilities": len(facilities), "by_type": by_type, "total_capacity_teu": cap}


def oil_spill_trajectory(
    release_point: Tuple[float, float], *,
    volume_m3: float = 100, wind_speed_ms: float = 5, wind_dir_deg: float = 180,
    current_speed_ms: float = 0.5, current_dir_deg: float = 90, hours: int = 24,
) -> List[Dict[str, Any]]:
    """Simple oil-spill trajectory model (wind + current drift)."""
    wx = 0.03 * wind_speed_ms * math.sin(math.radians(wind_dir_deg))
    wy = 0.03 * wind_speed_ms * math.cos(math.radians(wind_dir_deg))
    cx = current_speed_ms * math.sin(math.radians(current_dir_deg))
    cy = current_speed_ms * math.cos(math.radians(current_dir_deg))
    dmx, dmy = (wx + cx) * 3600, (wy + cy) * 3600
    trajectory = []
    x, y = release_point
    for h in range(hours + 1):
        spread = math.sqrt(volume_m3 * (h + 1)) * 10
        trajectory.append({"hour": h, "x": round(x, 6), "y": round(y, 6),
                           "spread_m": round(spread, 1)})
        x += dmx / 111_000
        y += dmy / 111_000
    return trajectory


def fisheries_management_zone(
    zones: Sequence[Dict[str, Any]], catches: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Summarise catch data by fisheries management zone."""
    zc: Dict[str, float] = {}
    for c in catches:
        z = c.get("zone", "")
        zc[z] = zc.get(z, 0) + c.get("catch_kg", 0)
    return [{**z, "total_catch_kg": round(zc.get(z.get("id", ""), 0), 1)} for z in zones]


def marine_protected_area_analysis(
    mpa: Dict[str, Any], activities: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Screen activities within a marine protected area."""
    mx, my, r = mpa.get("x", 0), mpa.get("y", 0), mpa.get("radius_m", 5000)
    prohibited = mpa.get("prohibited_activities", ["fishing", "mining"])
    violations = sum(1 for a in activities
                     if math.hypot(mx - a.get("x", 0), my - a.get("y", 0)) <= r
                     and a.get("type") in prohibited)
    return {"mpa": mpa.get("name", ""), "activities_screened": len(activities),
            "violations": violations}


def coral_reef_health(transects: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise coral reef health from transect surveys."""
    cover = [t.get("coral_cover_pct", 0) for t in transects]
    bleach = [t.get("bleaching_pct", 0) for t in transects]
    if not cover:
        return {"n_transects": 0, "health_rating": "unknown"}
    mc, mb = statistics.mean(cover), statistics.mean(bleach)
    rating = "good" if mc > 30 and mb < 10 else "fair" if mc > 15 else "poor"
    return {"n_transects": len(transects), "mean_coral_cover_pct": round(mc, 1),
            "mean_bleaching_pct": round(mb, 1), "health_rating": rating}


def bathymetric_surface(
    soundings: Sequence[Tuple[float, float, float]],
    grid_bounds: Tuple[float, float, float, float], *, grid_size: int = 25,
) -> Dict[str, Any]:
    """Interpolate bathymetric surface from soundings (IDW)."""
    return soil_contamination_interpolation(soundings, grid_bounds, grid_size=grid_size)


def tidal_datum_mapping(tide_gauges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute tidal datum values from gauge records."""
    results = []
    for g in tide_gauges:
        highs, lows = g.get("high_water", []), g.get("low_water", [])
        mhw = statistics.mean(highs) if highs else 0
        mlw = statistics.mean(lows) if lows else 0
        results.append({**g, "MHW": round(mhw, 3), "MLW": round(mlw, 3),
                        "tidal_range": round(mhw - mlw, 3)})
    return results


def coastal_setback_line(
    shoreline_points: Sequence[Dict[str, Any]], *,
    setback_m: float = 50, erosion_rate_m_yr: float = 0.5,
    planning_horizon_yr: int = 50,
) -> List[Dict[str, Any]]:
    """Calculate coastal setback line accounting for erosion."""
    total = setback_m + erosion_rate_m_yr * planning_horizon_yr
    return [{**p, "setback_m": round(total, 1)} for p in shoreline_points]


def dune_erosion_model(
    profiles: Sequence[Dict[str, Any]], *,
    storm_surge_m: float = 2.0, wave_height_m: float = 3.0,
) -> List[Dict[str, Any]]:
    """Simple dune erosion volume estimation."""
    results = []
    for p in profiles:
        dh = p.get("dune_crest_m", 5)
        dw = p.get("dune_width_m", 30)
        collision = storm_surge_m + wave_height_m * 0.5
        vol = dw * dh * 0.5 if collision >= dh else dw * collision * 0.3
        results.append({**p, "eroded_volume_m3_per_m": round(vol, 1),
                        "overwash_risk": collision >= dh})
    return results


def beach_nourishment_design(
    beach_length_m: float, *, target_width_m: float = 30,
    berm_height_m: float = 2.0, overfill_ratio: float = 1.3,
) -> Dict[str, Any]:
    """Calculate beach nourishment sand volume."""
    pv = target_width_m * berm_height_m / 2
    return {"profile_volume_m3_per_m": round(pv, 1),
            "total_volume_m3": round(pv * beach_length_m * overfill_ratio, 0),
            "beach_length_m": beach_length_m, "overfill_ratio": overfill_ratio}


# ===========================================================================
# MINING / GEOLOGY  (1467 – 1476)
# ===========================================================================

def mining_lease_boundary(leases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise mining lease boundaries."""
    by_status: Dict[str, int] = {}
    for l in leases:
        s = l.get("status", "active")
        by_status[s] = by_status.get(s, 0) + 1
    return {"leases": len(leases),
            "total_area_ha": round(sum(l.get("area_ha", 0) for l in leases), 1),
            "by_status": by_status}


def borehole_mapping(boreholes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise borehole locations and depths."""
    depths = [b.get("depth_m", 0) for b in boreholes]
    return {"count": len(boreholes),
            "mean_depth_m": round(statistics.mean(depths), 1) if depths else 0,
            "max_depth_m": max(depths) if depths else 0}


def geological_cross_section(
    boreholes: Sequence[Dict[str, Any]], *,
    section_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """Generate a geological cross-section from borehole data."""
    length = math.hypot(section_line[1][0] - section_line[0][0],
                        section_line[1][1] - section_line[0][1]) if section_line else 1000
    profiles = [{"borehole": b.get("id", ""), "x": b.get("x", 0), "y": b.get("y", 0),
                 "surface_elevation": b.get("elevation_m", 0),
                 "layers": b.get("layers", [])} for b in boreholes]
    return {"profiles": profiles, "section_length_m": round(length, 1)}


def ore_grade_interpolation(
    samples: Sequence[Tuple[float, float, float]],
    grid_bounds: Tuple[float, float, float, float], *, grid_size: int = 25,
) -> Dict[str, Any]:
    """IDW interpolation of ore grade from sample data."""
    return soil_contamination_interpolation(samples, grid_bounds, grid_size=grid_size)


def mine_plan_pit_shell(
    block_model: Sequence[Dict[str, Any]], *,
    cutoff_grade: float = 0.5, strip_ratio_max: float = 4.0,
) -> Dict[str, Any]:
    """Simple pit shell optimisation using cutoff grade."""
    ore = [b for b in block_model if b.get("grade", 0) >= cutoff_grade]
    waste = [b for b in block_model if b.get("grade", 0) < cutoff_grade]
    sr = len(waste) / len(ore) if ore else 999
    return {"ore_blocks": len(ore), "waste_blocks": len(waste),
            "strip_ratio": round(sr, 2), "economic": sr <= strip_ratio_max,
            "total_ore_tonnes": sum(b.get("tonnes", 0) for b in ore)}


def tailings_dam_monitoring(
    sensors: Sequence[Dict[str, Any]], *,
    pore_pressure_limit_kpa: float = 200, displacement_limit_mm: float = 50,
) -> List[Dict[str, Any]]:
    """Monitor tailings dam sensor readings."""
    results = []
    for s in sensors:
        pp, disp = s.get("pore_pressure_kpa", 0), s.get("displacement_mm", 0)
        alert = pp > pore_pressure_limit_kpa or disp > displacement_limit_mm
        reason = ("pore_pressure" if pp > pore_pressure_limit_kpa
                  else "displacement" if disp > displacement_limit_mm else None)
        results.append({**s, "alert": alert, "alert_reason": reason})
    return results


def subsidence_monitoring(
    points: Sequence[Dict[str, Any]], *, threshold_mm_yr: float = 20,
) -> List[Dict[str, Any]]:
    """Classify subsidence monitoring points."""
    return [{**p, "category": ("critical" if p.get("subsidence_mm_yr", 0) > threshold_mm_yr * 2
                               else "warning" if p.get("subsidence_mm_yr", 0) > threshold_mm_yr
                               else "stable")} for p in points]


def seismic_survey_planning(
    area_bounds: Tuple[float, float, float, float], *,
    line_spacing_m: float = 200, receiver_spacing_m: float = 25,
) -> Dict[str, Any]:
    """Plan seismic survey line layout."""
    xmin, ymin, xmax, ymax = area_bounds
    width, height = xmax - xmin, ymax - ymin
    n_lines = int(width / line_spacing_m) + 1
    rpl = int(height / receiver_spacing_m) + 1
    return {"n_lines": n_lines, "receivers_per_line": rpl,
            "total_receivers": n_lines * rpl,
            "total_line_km": round(n_lines * height / 1000, 2)}


def well_pad_placement(
    candidates: Sequence[Dict[str, Any]], constraints: Sequence[Dict[str, Any]], *,
    min_setback_m: float = 300,
) -> List[Dict[str, Any]]:
    """Evaluate well pad placement against constraints."""
    results = []
    for c in candidates:
        cx, cy = c.get("x", 0), c.get("y", 0)
        violations = [{"constraint": con.get("type", ""),
                       "distance_m": round(math.hypot(cx - con.get("x", 0),
                                                      cy - con.get("y", 0)), 1)}
                      for con in constraints
                      if math.hypot(cx - con.get("x", 0), cy - con.get("y", 0)) < min_setback_m]
        results.append({**c, "feasible": not violations, "violations": violations})
    return results


def fracking_radius_analysis(
    well: Dict[str, Any], *,
    stimulation_radius_m: float = 300, monitoring_radius_m: float = 1500,
) -> Dict[str, Any]:
    """Define fracking stimulation and monitoring zones."""
    return {"well_id": well.get("id", ""), "x": well.get("x", 0), "y": well.get("y", 0),
            "stimulation_radius_m": stimulation_radius_m,
            "monitoring_radius_m": monitoring_radius_m}


# ===========================================================================
# ENERGY  (1477 – 1483)
# ===========================================================================

def renewable_energy_capacity(
    sites: Sequence[Dict[str, Any]], *, energy_type: str = "solar",
) -> List[Dict[str, Any]]:
    """Estimate renewable energy generation capacity."""
    results = []
    for s in sites:
        area = s.get("area_m2", 10000)
        if energy_type == "solar":
            kwh = area * s.get("irradiance_kwh_m2_yr", 1500) * s.get("efficiency", 0.20)
        else:
            kwh = 0.5 * 1.225 * area * 0.01 * s.get("mean_wind_ms", 7) ** 3 * 0.35 * 8760
        results.append({**s, "annual_kwh": round(kwh, 0), "annual_mwh": round(kwh / 1000, 1)})
    return results


def solar_irradiance_map(
    points: Sequence[Dict[str, Any]], *, latitude_correction: bool = True,
) -> List[Dict[str, Any]]:
    """Estimate annual solar irradiance at points."""
    results = []
    for p in points:
        lat = p.get("latitude", 35)
        factor = max(0.3, math.cos(math.radians(lat))) if latitude_correction else 1.0
        slope_f = 1 + 0.01 * p.get("slope_deg", 0) * math.cos(
            math.radians(p.get("aspect_deg", 180) - 180))
        results.append({**p, "irradiance_kwh_m2_yr": round(2000 * factor * slope_f, 0)})
    return results


def wind_resource_map(
    points: Sequence[Dict[str, Any]], *,
    hub_height_m: float = 80, roughness_length_m: float = 0.03,
) -> List[Dict[str, Any]]:
    """Estimate wind speed at hub height using log wind profile."""
    results = []
    for p in points:
        ms = p.get("wind_speed_ms", 5)
        mh = p.get("measurement_height_m", 10)
        if roughness_length_m > 0 and mh > roughness_length_m:
            hs = ms * math.log(hub_height_m / roughness_length_m) / math.log(mh / roughness_length_m)
        else:
            hs = ms
        results.append({**p, "hub_speed_ms": round(hs, 2),
                        "power_density_w_m2": round(0.5 * 1.225 * hs ** 3, 1)})
    return results


def tidal_energy_potential(sites: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Estimate tidal energy potential at candidate sites."""
    results = []
    for s in sites:
        v = s.get("current_speed_ms", 2)
        a = s.get("rotor_area_m2", 100)
        pw = 0.5 * 1025 * a * v ** 3 * 0.35 / 1000
        results.append({**s, "rated_power_kw": round(pw, 1),
                        "annual_mwh": round(pw * 8760 * s.get("capacity_factor", 0.25) / 1000, 1)})
    return results


def geothermal_resource_map(points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Classify geothermal resource potential."""
    results = []
    for p in points:
        score = p.get("geothermal_gradient_c_km", 30) * 0.5 + p.get("heat_flow_mw_m2", 60) * 0.5
        results.append({**p, "geothermal_score": round(score, 1),
                        "category": "high" if score > 50 else "moderate" if score > 25 else "low"})
    return results


def carbon_capture_site_suitability(
    candidates: Sequence[Dict[str, Any]], *, weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Score sites for carbon capture and storage suitability."""
    w = weights or {"storage_capacity": 0.3, "seal_quality": 0.25,
                    "injectivity": 0.25, "proximity_to_source": 0.2}
    return _score_features(candidates, list(w.keys()), list(w.values()))


def ev_charging_station_siting(
    candidates: Sequence[Dict[str, Any]], demand_points: Sequence[Dict[str, Any]], *,
    max_distance_m: float = 5000,
) -> List[Dict[str, Any]]:
    """Score EV charging station candidates by demand coverage."""
    results = []
    for c in candidates:
        cx, cy = c.get("x", 0), c.get("y", 0)
        demand = sum(d.get("ev_registrations", 1) for d in demand_points
                     if math.hypot(cx - d.get("x", 0), cy - d.get("y", 0)) <= max_distance_m)
        results.append({**c, "demand_served": demand})
    results.sort(key=lambda x: x["demand_served"], reverse=True)
    return results


# ===========================================================================
# SMART CITY / TELECOM  (1484 – 1493)
# ===========================================================================

def smart_city_sensor_placement(
    candidates: Sequence[Dict[str, Any]], coverage_targets: Sequence[Dict[str, Any]], *,
    sensor_range_m: float = 200,
) -> List[Dict[str, Any]]:
    """Greedy set-cover sensor placement."""
    uncovered = list(range(len(coverage_targets)))
    placed = []
    remaining = list(candidates)
    while uncovered and remaining:
        best_idx, best_covered = 0, []
        for i, c in enumerate(remaining):
            cx, cy = c.get("x", 0), c.get("y", 0)
            covered = [u for u in uncovered
                       if math.hypot(cx - coverage_targets[u].get("x", 0),
                                     cy - coverage_targets[u].get("y", 0)) <= sensor_range_m]
            if len(covered) > len(best_covered):
                best_covered, best_idx = covered, i
        if not best_covered:
            break
        placed.append({**remaining.pop(best_idx), "covers": len(best_covered)})
        for u in best_covered:
            if u in uncovered:
                uncovered.remove(u)
    return placed


def iot_device_coverage(
    devices: Sequence[Dict[str, Any]], area_bounds: Tuple[float, float, float, float], *,
    range_m: float = 100, grid_size: int = 20,
) -> Dict[str, Any]:
    """Compute IoT device coverage percentage over an area."""
    xmin, ymin, xmax, ymax = area_bounds
    dx, dy = (xmax - xmin) / grid_size, (ymax - ymin) / grid_size
    covered = 0
    total = grid_size * grid_size
    for i in range(grid_size):
        cy = ymin + (i + 0.5) * dy
        for j in range(grid_size):
            cx = xmin + (j + 0.5) * dx
            if any(math.hypot(cx - d.get("x", 0), cy - d.get("y", 0)) <= range_m
                   for d in devices):
                covered += 1
    return {"coverage_pct": round(covered / total * 100, 1),
            "covered_cells": covered, "total_cells": total}


def five_g_small_cell_placement(
    candidates: Sequence[Dict[str, Any]], demand_points: Sequence[Dict[str, Any]], *,
    range_m: float = 300,
) -> List[Dict[str, Any]]:
    """5G small-cell placement using demand coverage scoring."""
    return ev_charging_station_siting(candidates, demand_points, max_distance_m=range_m)


def wifi_coverage_mapping(
    access_points: Sequence[Dict[str, Any]], *,
    grid_bounds: Tuple[float, float, float, float] = (0, 0, 100, 100),
    grid_size: int = 20,
) -> Dict[str, Any]:
    """Map Wi-Fi signal strength over an area."""
    xmin, ymin, xmax, ymax = grid_bounds
    dx, dy = (xmax - xmin) / grid_size, (ymax - ymin) / grid_size
    grid = []
    for i in range(grid_size):
        row = []
        cy = ymin + (i + 0.5) * dy
        for j in range(grid_size):
            cx = xmin + (j + 0.5) * dx
            best = -100.0
            for ap in access_points:
                dist = math.hypot(cx - ap.get("x", 0), cy - ap.get("y", 0))
                pl = (20 * math.log10(max(dist, 0.1)) + 20 * math.log10(2400) + 32.45 - 60
                      if dist > 0 else 0)
                rssi = ap.get("tx_power_dbm", 20) - pl
                if rssi > best:
                    best = rssi
            row.append(round(best, 1))
        grid.append(row)
    return {"grid": grid, "extent": list(grid_bounds)}


def satellite_ground_track(
    *, inclination_deg: float = 51.6, altitude_km: float = 408,
    period_min: float = 92.68, n_points: int = 100,
) -> List[Dict[str, float]]:
    """Simplified satellite ground-track projection."""
    points = []
    for i in range(n_points):
        t = i / n_points * period_min * 60
        omega = 2 * math.pi / (period_min * 60)
        angle = omega * t
        lat = math.degrees(math.asin(math.sin(math.radians(inclination_deg)) * math.sin(angle)))
        lon = math.degrees(angle - 7.292e-5 * t) % 360
        if lon > 180:
            lon -= 360
        points.append({"lat": round(lat, 4), "lon": round(lon, 4)})
    return points


def radar_coverage_footprint(
    radar: Dict[str, Any], *, max_range_km: float = 200,
    beam_width_deg: float = 360,
) -> Dict[str, Any]:
    """Calculate radar coverage footprint."""
    return {"center_x": radar.get("x", 0), "center_y": radar.get("y", 0),
            "max_range_km": max_range_km,
            "coverage_area_km2": round(math.pi * max_range_km ** 2 * beam_width_deg / 360, 1)}


def radio_propagation_model(
    frequency_mhz: float, tx_height_m: float, rx_height_m: float, distance_km: float, *,
    model: str = "okumura_hata",
) -> Dict[str, float]:
    """Compute radio path loss using Okumura-Hata model."""
    if model == "okumura_hata" and 150 <= frequency_mhz <= 1500:
        a_hm = ((1.1 * math.log10(frequency_mhz) - 0.7) * rx_height_m
                 - (1.56 * math.log10(frequency_mhz) - 0.8))
        L = (69.55 + 26.16 * math.log10(max(frequency_mhz, 1))
             - 13.82 * math.log10(max(tx_height_m, 1)) - a_hm
             + (44.9 - 6.55 * math.log10(max(tx_height_m, 1)))
             * math.log10(max(distance_km, 0.01)))
    else:
        L = (20 * math.log10(max(distance_km, 0.01))
             + 20 * math.log10(max(frequency_mhz, 1)) + 32.45)
    return {"path_loss_db": round(L, 2), "model": model}


def cell_tower_signal_strength(
    towers: Sequence[Dict[str, Any]], test_points: Sequence[Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """Estimate received signal strength from cell towers."""
    results = []
    for tx, ty in test_points:
        best_rssi, best_tower = -120.0, ""
        for t in towers:
            dist = math.hypot(tx - t.get("x", 0), ty - t.get("y", 0)) / 1000
            pl = (20 * math.log10(max(dist, 0.01))
                  + 20 * math.log10(t.get("frequency_mhz", 900)) + 32.45)
            rssi = t.get("tx_power_dbm", 43) + t.get("antenna_gain_dbi", 15) - pl
            if rssi > best_rssi:
                best_rssi, best_tower = rssi, t.get("id", "")
        results.append({"x": tx, "y": ty, "rssi_dbm": round(best_rssi, 1),
                        "serving_tower": best_tower})
    return results


def terrain_masking_analysis(
    observer: Tuple[float, float, float],
    targets: Sequence[Tuple[float, float, float]],
    terrain_profile: Sequence[Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """Check which targets are masked by terrain from observer."""
    ox, oy, oz = observer
    results = []
    for tx, ty, tz in targets:
        dist = math.hypot(tx - ox, ty - oy)
        masked = False
        if dist > 0:
            for pd, ph in terrain_profile:
                if 0 < pd < dist:
                    line_z = oz + (tz - oz) * pd / dist
                    if ph > line_z:
                        masked = True
                        break
        results.append({"x": tx, "y": ty, "visible": not masked})
    return results


def link_budget_calculation(
    tx_power_dbm: float, tx_gain_dbi: float, rx_gain_dbi: float,
    path_loss_db: float, *, cable_loss_db: float = 2,
    margin_db: float = 10, rx_sensitivity_dbm: float = -100,
) -> Dict[str, float]:
    """Calculate wireless link budget."""
    eirp = tx_power_dbm + tx_gain_dbi - cable_loss_db
    received = eirp - path_loss_db + rx_gain_dbi
    fade = received - rx_sensitivity_dbm
    return {"eirp_dbm": round(eirp, 2), "received_dbm": round(received, 2),
            "fade_margin_db": round(fade, 2), "link_viable": fade >= margin_db}


# ===========================================================================
# 3D / BIM / DIGITAL TWIN  (1494 – 1500)
# ===========================================================================

def indoor_spatial_analysis(floor_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Analyse indoor floor plan data."""
    rooms = floor_plan.get("rooms", [])
    by_type: Dict[str, float] = {}
    for r in rooms:
        t = r.get("type", "office")
        by_type[t] = by_type.get(t, 0) + r.get("area_m2", 0)
    return {"room_count": len(rooms),
            "total_area_m2": round(sum(r.get("area_m2", 0) for r in rooms), 1),
            "by_type_m2": {k: round(v, 1) for k, v in by_type.items()}}


def bim_to_gis_conversion(
    bim_elements: Sequence[Dict[str, Any]], *,
    crs_epsg: int = 4326, origin: Tuple[float, float, float] = (0, 0, 0),
) -> List[Dict[str, Any]]:
    """Convert BIM elements to GIS features with real-world coordinates."""
    ox, oy, oz = origin
    return [{"type": "Feature",
             "properties": {"ifc_type": el.get("ifc_type", ""), "name": el.get("name", ""),
                            "z": el.get("local_z", 0) + oz},
             "geometry": {"type": "Point",
                          "coordinates": [el.get("local_x", 0) + ox,
                                          el.get("local_y", 0) + oy]}}
            for el in bim_elements]


def citygml_model(
    buildings: Sequence[Dict[str, Any]], *, lod: int = 1,
) -> str:
    """Generate CityGML-like XML for 3D city model."""
    parts = ['<?xml version="1.0"?>',
             '<CityModel xmlns="http://www.opengis.net/citygml/2.0">']
    for b in buildings:
        bid, h = b.get("id", "bldg"), b.get("height_m", 10)
        x, y = b.get("x", 0), b.get("y", 0)
        parts.append(f'  <cityObjectMember><Building gml:id="{bid}">'
                     f'<measuredHeight uom="m">{h}</measuredHeight>'
                     f'<lod{lod}Solid><pos>{x} {y} 0 {x} {y} {h}</pos>'
                     f'</lod{lod}Solid></Building></cityObjectMember>')
    parts.append('</CityModel>')
    return "\n".join(parts)


def digital_twin_layer(
    assets: Sequence[Dict[str, Any]], sensor_readings: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge asset geometry with live sensor data for digital twin."""
    sm: Dict[str, list] = {}
    for s in sensor_readings:
        sm.setdefault(s.get("asset_id", ""), []).append(s)
    return [{**a, "live_data": sm.get(a.get("id", ""), [{}])[-1],
             "reading_count": len(sm.get(a.get("id", ""), []))} for a in assets]


def reality_capture_integration(
    point_cloud: Sequence[Dict[str, Any]], *, voxel_size_m: float = 0.5,
) -> Dict[str, Any]:
    """Downsample point cloud data via voxelisation."""
    voxels: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
    for p in point_cloud:
        key = (int(p.get("x", 0) / voxel_size_m),
               int(p.get("y", 0) / voxel_size_m),
               int(p.get("z", 0) / voxel_size_m))
        voxels.setdefault(key, []).append(p)
    centroids = [{"x": round(statistics.mean(pt.get("x", 0) for pt in pts), 3),
                  "y": round(statistics.mean(pt.get("y", 0) for pt in pts), 3),
                  "z": round(statistics.mean(pt.get("z", 0) for pt in pts), 3),
                  "point_count": len(pts)} for pts in voxels.values()]
    return {"voxel_count": len(centroids), "original_points": len(point_cloud),
            "centroids": centroids}


def ar_vr_georeferenced_scene(
    anchor: Dict[str, Any], objects: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a geo-referenced AR/VR scene descriptor."""
    ax, ay, az = anchor.get("x", 0), anchor.get("y", 0), anchor.get("z", 0)
    return {"anchor": anchor,
            "objects": [{"name": o.get("name", ""), "model": o.get("model", ""),
                         "offset_x": o.get("x", 0) - ax, "offset_y": o.get("y", 0) - ay,
                         "offset_z": o.get("z", 0) - az} for o in objects],
            "crs_epsg": anchor.get("crs_epsg", 4326)}


def game_engine_terrain_export(
    heightmap: Sequence[Sequence[float]], *,
    cell_size_m: float = 1.0, vertical_scale: float = 1.0, engine: str = "unreal",
) -> Dict[str, Any]:
    """Export terrain heightmap for game engines (Unreal/Unity)."""
    if not heightmap:
        return {"error": "empty heightmap"}
    rows, cols = len(heightmap), len(heightmap[0])
    flat = [round(v * vertical_scale, 3) for row in heightmap for v in row]
    return {"engine": engine, "rows": rows, "cols": cols,
            "cell_size_m": cell_size_m, "vertical_scale": vertical_scale,
            "min_height": min(flat), "max_height": max(flat),
            "heightmap_values": flat,
            "format": "R16" if engine == "unreal" else "RAW"}
