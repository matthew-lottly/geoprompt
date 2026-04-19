"""Geocoding, addressing, and mobility utilities for GeoPrompt.

Provides forward/reverse geocoding with pluggable providers, address parsing,
route helpers, and fleet optimization. HTTP providers require ``requests``
(optional dependency).
"""
from __future__ import annotations

import importlib
import math
import re
from typing import Any, Sequence


Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

def _get_requests() -> Any:
    try:
        return importlib.import_module("requests")
    except ImportError as exc:
        raise RuntimeError(
            "Install requests ('pip install requests') to use geocoding features."
        ) from exc


# ---------------------------------------------------------------------------
# A. Forward Geocoding
# ---------------------------------------------------------------------------


def forward_geocode(
    address: str,
    *,
    provider: str = "nominatim",
    api_key: str | None = None,
    limit: int = 1,
) -> list[dict[str, Any]]:
    """Forward geocode an address string to coordinates.

    Providers: ``"nominatim"`` (free, rate-limited), ``"arcgis"``.

    Returns list of dicts with ``"lat"``, ``"lon"``, ``"display_name"``,
    ``"score"``.
    """
    if provider == "nominatim":
        return _geocode_nominatim(address, limit)
    if provider == "arcgis":
        return _geocode_arcgis(address, limit, api_key)
    raise ValueError(f"unknown geocoding provider: {provider}")


def _geocode_nominatim(address: str, limit: int) -> list[dict[str, Any]]:
    requests = _get_requests()
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": address, "format": "json", "limit": limit},
        headers={"User-Agent": "GeoPrompt/1.0"},
        timeout=10,
    )
    resp.raise_for_status()
    return [
        {
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "display_name": r.get("display_name", ""),
            "score": float(r.get("importance", 0)),
        }
        for r in resp.json()
    ]


def _geocode_arcgis(address: str, limit: int, api_key: str | None) -> list[dict[str, Any]]:
    requests = _get_requests()
    params: dict[str, Any] = {
        "SingleLine": address,
        "f": "json",
        "maxLocations": limit,
    }
    if api_key:
        params["token"] = api_key
    resp = requests.get(
        "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates",
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "lat": c["location"]["y"],
            "lon": c["location"]["x"],
            "display_name": c.get("address", ""),
            "score": c.get("score", 0),
        }
        for c in data.get("candidates", [])
    ]


def batch_forward_geocode(
    addresses: Sequence[str],
    *,
    provider: str = "nominatim",
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Geocode multiple addresses. Returns one result per address."""
    results: list[dict[str, Any]] = []
    for addr in addresses:
        try:
            hits = forward_geocode(addr, provider=provider, api_key=api_key, limit=1)
            results.append(hits[0] if hits else {"lat": None, "lon": None, "display_name": addr, "score": 0})
        except Exception as e:
            results.append({"lat": None, "lon": None, "display_name": addr, "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# B. Reverse Geocoding
# ---------------------------------------------------------------------------


def reverse_geocode(
    lat: float,
    lon: float,
    *,
    provider: str = "nominatim",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Reverse geocode a coordinate to an address."""
    if provider == "nominatim":
        return _reverse_nominatim(lat, lon)
    if provider == "arcgis":
        return _reverse_arcgis(lat, lon, api_key)
    raise ValueError(f"unknown provider: {provider}")


def _reverse_nominatim(lat: float, lon: float) -> dict[str, Any]:
    requests = _get_requests()
    resp = requests.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={"lat": lat, "lon": lon, "format": "json"},
        headers={"User-Agent": "GeoPrompt/1.0"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "lat": lat,
        "lon": lon,
        "display_name": data.get("display_name", ""),
        "address_components": data.get("address", {}),
    }


def _reverse_arcgis(lat: float, lon: float, api_key: str | None) -> dict[str, Any]:
    requests = _get_requests()
    params: dict[str, Any] = {
        "location": f"{lon},{lat}",
        "f": "json",
    }
    if api_key:
        params["token"] = api_key
    resp = requests.get(
        "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/reverseGeocode",
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    addr = data.get("address", {})
    return {
        "lat": lat,
        "lon": lon,
        "display_name": addr.get("LongLabel", addr.get("Match_addr", "")),
        "address_components": addr,
    }


def batch_reverse_geocode(
    coordinates: Sequence[tuple[float, float]],
    *,
    provider: str = "nominatim",
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Reverse geocode multiple coordinates."""
    return [
        reverse_geocode(lat, lon, provider=provider, api_key=api_key)
        for lat, lon in coordinates
    ]


# ---------------------------------------------------------------------------
# C. Address Parsing & Standardization
# ---------------------------------------------------------------------------


_DIRECTIONALS = {"N": "North", "S": "South", "E": "East", "W": "West",
                 "NE": "Northeast", "NW": "Northwest", "SE": "Southeast", "SW": "Southwest"}
_SUFFIX_MAP = {
    "ST": "Street", "AVE": "Avenue", "BLVD": "Boulevard", "DR": "Drive",
    "LN": "Lane", "RD": "Road", "CT": "Court", "PL": "Place",
    "WAY": "Way", "CIR": "Circle", "PKWY": "Parkway", "HWY": "Highway",
    "TRL": "Trail", "TER": "Terrace", "LOOP": "Loop",
}


def parse_address(address: str) -> dict[str, str | None]:
    """Parse an address string into components.

    Returns dict with ``"number"``, ``"pre_direction"``, ``"street_name"``,
    ``"suffix"``, ``"post_direction"``, ``"unit"``, ``"city"``, ``"state"``,
    ``"zip_code"``.
    """
    addr = address.strip()
    result: dict[str, str | None] = {
        "number": None, "pre_direction": None, "street_name": None,
        "suffix": None, "post_direction": None, "unit": None,
        "city": None, "state": None, "zip_code": None,
    }

    # Extract zip
    zip_match = re.search(r"\b(\d{5}(?:-\d{4})?)\s*$", addr)
    if zip_match:
        result["zip_code"] = zip_match.group(1)
        addr = addr[:zip_match.start()].strip().rstrip(",").strip()

    # Split by comma for city/state
    parts = [p.strip() for p in addr.split(",")]
    street_part = parts[0]
    if len(parts) >= 3:
        result["city"] = parts[-2]
        state_part = parts[-1]
        state_match = re.match(r"([A-Za-z]{2})\b", state_part)
        if state_match:
            result["state"] = state_match.group(1).upper()
    elif len(parts) == 2:
        cs = parts[1].strip()
        cs_match = re.match(r"(.+?)\s+([A-Z]{2})\s*$", cs)
        if cs_match:
            result["city"] = cs_match.group(1)
            result["state"] = cs_match.group(2)
        else:
            result["city"] = cs

    # Parse street part
    tokens = street_part.split()
    if tokens and re.match(r"\d+", tokens[0]):
        result["number"] = tokens.pop(0)

    if tokens and tokens[0].upper() in _DIRECTIONALS:
        result["pre_direction"] = tokens.pop(0).upper()

    # Unit detection
    for i, t in enumerate(tokens):
        if t.upper() in ("APT", "STE", "SUITE", "UNIT", "#"):
            result["unit"] = " ".join(tokens[i:])
            tokens = tokens[:i]
            break

    if tokens and tokens[-1].upper() in _DIRECTIONALS:
        result["post_direction"] = tokens.pop().upper()

    if tokens and tokens[-1].upper().rstrip(".") in _SUFFIX_MAP:
        result["suffix"] = _SUFFIX_MAP.get(tokens.pop().upper().rstrip("."))

    result["street_name"] = " ".join(tokens) if tokens else None
    return result


def standardize_address(address: str) -> str:
    """Standardize an address string to a canonical form."""
    parsed = parse_address(address)
    parts: list[str] = []
    if parsed["number"]:
        parts.append(parsed["number"])
    if parsed["pre_direction"]:
        parts.append(parsed["pre_direction"])
    if parsed["street_name"]:
        parts.append(parsed["street_name"].title())
    if parsed["suffix"]:
        parts.append(parsed["suffix"])
    if parsed["post_direction"]:
        parts.append(parsed["post_direction"])
    if parsed["unit"]:
        parts.append(parsed["unit"])

    street = " ".join(parts)
    location_parts: list[str] = [street]
    if parsed["city"]:
        location_parts.append(parsed["city"].title())
    if parsed["state"]:
        location_parts.append(parsed["state"])
    result = ", ".join(location_parts)
    if parsed["zip_code"]:
        result += " " + parsed["zip_code"]
    return result


# ---------------------------------------------------------------------------
# D. Intersection & Milepost Lookup
# ---------------------------------------------------------------------------


def intersection_lookup(
    street_a: str,
    street_b: str,
    *,
    provider: str = "nominatim",
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Look up the coordinates of a street intersection."""
    query = f"{street_a} and {street_b}"
    results = forward_geocode(query, provider=provider, api_key=api_key, limit=1)
    return results[0] if results else None


def route_milepost(
    route_points: Sequence[tuple[float, float]],
    target_milepost: float,
) -> tuple[float, float] | None:
    """Interpolate a position along a route at the given milepost distance.

    Milepost is in the same units as coordinate distances.
    """
    if len(route_points) < 2:
        return None

    accumulated = 0.0
    for i in range(len(route_points) - 1):
        x0, y0 = route_points[i]
        x1, y1 = route_points[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if accumulated + seg_len >= target_milepost:
            remaining = target_milepost - accumulated
            frac = remaining / seg_len if seg_len > 0 else 0.0
            return (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0))
        accumulated += seg_len
    return route_points[-1]


# ---------------------------------------------------------------------------
# E. Travel Mode Presets
# ---------------------------------------------------------------------------


TRAVEL_MODE_PRESETS: dict[str, dict[str, Any]] = {
    "walking": {"speed_kmh": 5.0, "impedance": "distance", "restrictions": ["no_highway"]},
    "driving": {"speed_kmh": 60.0, "impedance": "time", "restrictions": []},
    "trucking": {"speed_kmh": 50.0, "impedance": "time", "restrictions": ["no_low_clearance"]},
    "emergency": {"speed_kmh": 80.0, "impedance": "time", "restrictions": [], "priority": True},
    "cycling": {"speed_kmh": 15.0, "impedance": "distance", "restrictions": ["no_highway"]},
}


def get_travel_mode(mode: str) -> dict[str, Any]:
    """Retrieve a travel mode preset."""
    if mode not in TRAVEL_MODE_PRESETS:
        raise ValueError(f"unknown travel mode: {mode}; choose from {list(TRAVEL_MODE_PRESETS)}")
    return dict(TRAVEL_MODE_PRESETS[mode])


# ---------------------------------------------------------------------------
# F. Time-Dependent Impedance
# ---------------------------------------------------------------------------


def time_dependent_impedance(
    base_cost: float,
    *,
    hour_of_day: int = 12,
    peak_hours: Sequence[int] = (7, 8, 9, 17, 18),
    peak_multiplier: float = 1.5,
    off_peak_multiplier: float = 0.8,
) -> float:
    """Adjust travel cost based on time of day.

    During *peak_hours* the base cost is multiplied by *peak_multiplier*;
    otherwise by *off_peak_multiplier*.
    """
    if hour_of_day in peak_hours:
        return base_cost * peak_multiplier
    return base_cost * off_peak_multiplier


# ---------------------------------------------------------------------------
# G. Route Directions Narrative
# ---------------------------------------------------------------------------


def route_directions_narrative(
    route_points: Sequence[tuple[float, float]],
    *,
    segment_names: Sequence[str] | None = None,
    unit: str = "km",
) -> list[dict[str, Any]]:
    """Generate turn-by-turn directions from a route geometry.

    Returns step dicts with ``"instruction"``, ``"distance"``, ``"bearing"``.
    """
    steps: list[dict[str, Any]] = []
    for i in range(len(route_points) - 1):
        x0, y0 = route_points[i]
        x1, y1 = route_points[i + 1]
        dist = math.hypot(x1 - x0, y1 - y0)
        bearing = math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360

        direction = _bearing_to_cardinal(bearing)
        name = segment_names[i] if segment_names and i < len(segment_names) else f"segment {i + 1}"

        if i == 0:
            instr = f"Start heading {direction} on {name}"
        else:
            prev_bearing = steps[-1]["bearing"]
            turn = (bearing - prev_bearing + 360) % 360
            if turn < 30 or turn > 330:
                turn_str = "Continue straight"
            elif turn < 150:
                turn_str = "Turn right"
            elif turn < 210:
                turn_str = "Make a U-turn"
            else:
                turn_str = "Turn left"
            instr = f"{turn_str} onto {name}"

        steps.append({
            "step": i + 1,
            "instruction": instr,
            "distance": round(dist, 4),
            "distance_unit": unit,
            "bearing": round(bearing, 1),
        })

    return steps


def _bearing_to_cardinal(bearing: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(bearing / 45) % 8
    return dirs[idx]


# ---------------------------------------------------------------------------
# H. Fleet Routing / Depot Optimization
# ---------------------------------------------------------------------------


def fleet_routing_optimize(
    depot: tuple[float, float],
    stops: Sequence[tuple[float, float]],
    *,
    n_vehicles: int = 1,
    max_stops_per_vehicle: int | None = None,
) -> list[dict[str, Any]]:
    """Simple nearest-neighbor fleet routing from a depot.

    Assigns stops to vehicles and computes routes.

    Returns per-vehicle dicts with ``"vehicle"``, ``"route"`` (ordered stops),
    ``"total_distance"``.
    """
    remaining = list(range(len(stops)))
    vehicles: list[dict[str, Any]] = []
    max_per = max_stops_per_vehicle or (len(stops) // n_vehicles + 1)

    for v in range(n_vehicles):
        route: list[int] = []
        total_dist = 0.0
        current = depot

        while remaining and len(route) < max_per:
            best_idx = -1
            best_dist = float("inf")
            for idx in remaining:
                d = math.hypot(stops[idx][0] - current[0], stops[idx][1] - current[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_idx < 0:
                break
            route.append(best_idx)
            total_dist += best_dist
            current = stops[best_idx]
            remaining.remove(best_idx)

        # Return to depot
        total_dist += math.hypot(current[0] - depot[0], current[1] - depot[1])
        vehicles.append({
            "vehicle": v,
            "route": route,
            "route_coords": [depot] + [stops[i] for i in route] + [depot],
            "total_distance": round(total_dist, 4),
            "stop_count": len(route),
        })

    return vehicles


def stop_sequence_optimize(
    stops: Sequence[tuple[float, float]],
    *,
    start: tuple[float, float] | None = None,
) -> list[int]:
    """Optimize stop sequence using nearest-neighbor heuristic.

    Returns ordered list of stop indices.
    """
    remaining = list(range(len(stops)))
    sequence: list[int] = []
    current = start or stops[0] if stops else (0.0, 0.0)

    while remaining:
        best = min(remaining, key=lambda i: math.hypot(stops[i][0] - current[0], stops[i][1] - current[1]))
        sequence.append(best)
        current = stops[best]
        remaining.remove(best)

    return sequence


__all__ = [
    "TRAVEL_MODE_PRESETS",
    "batch_forward_geocode",
    "batch_reverse_geocode",
    "fleet_routing_optimize",
    "forward_geocode",
    "get_travel_mode",
    "intersection_lookup",
    "parse_address",
    "reverse_geocode",
    "route_directions_narrative",
    "route_milepost",
    "standardize_address",
    "stop_sequence_optimize",
    "time_dependent_impedance",
]
