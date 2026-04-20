"""Environmental and hazard modelling utilities for GeoPrompt.

Pure-Python implementations covering roadmap items in A4/A9 related to
environmental science, hydrology, hazards, and natural-resource management.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 628. Gaussian plume air-quality dispersion
# ---------------------------------------------------------------------------

def gaussian_plume(
    source: Tuple[float, float],
    emission_rate: float,
    wind_speed: float,
    wind_direction: float,
    receptor_points: Sequence[Tuple[float, float]],
    *,
    stack_height: float = 0.0,
    stability_class: str = "D",
) -> List[float]:
    """Gaussian plume dispersion model for air-quality analysis.

    Returns concentration (µg/m³) at each receptor point.
    """
    _sigma_params = {
        "A": (0.22, 0.0001, 0.20, 0.0),
        "B": (0.16, 0.0001, 0.12, 0.0),
        "C": (0.11, 0.0001, 0.08, 0.0002),
        "D": (0.08, 0.0001, 0.06, 0.0015),
        "E": (0.06, 0.0001, 0.03, 0.0003),
        "F": (0.04, 0.0001, 0.016, 0.0003),
    }
    params = _sigma_params.get(stability_class.upper(), _sigma_params["D"])
    a_y, b_y, a_z, b_z = params
    wd_rad = math.radians(wind_direction)
    cos_wd = math.cos(wd_rad)
    sin_wd = math.sin(wd_rad)
    results: List[float] = []
    for rx, ry in receptor_points:
        dx = rx - source[0]
        dy = ry - source[1]
        x_along = dx * sin_wd + dy * cos_wd
        y_cross = dx * cos_wd - dy * sin_wd
        if x_along <= 0:
            results.append(0.0)
            continue
        sigma_y = a_y * x_along / math.sqrt(1 + b_y * x_along)
        sigma_z = a_z * x_along / math.sqrt(1 + b_z * x_along)
        if sigma_y < 1e-10 or sigma_z < 1e-10:
            results.append(0.0)
            continue
        exp_y = math.exp(-0.5 * (y_cross / sigma_y) ** 2)
        exp_z1 = math.exp(-0.5 * (stack_height / sigma_z) ** 2)
        exp_z2 = math.exp(-0.5 * (stack_height / sigma_z) ** 2)
        conc = (emission_rate / (2 * math.pi * wind_speed * sigma_y * sigma_z)) * exp_y * (exp_z1 + exp_z2)
        results.append(max(0.0, conc))
    return results


# ---------------------------------------------------------------------------
# 629. Flood inundation from DEM + water level
# ---------------------------------------------------------------------------

def flood_inundation(
    dem: Sequence[Sequence[float]],
    water_level: float,
    *,
    nodata: float = -9999.0,
) -> List[List[int]]:
    """Simple flood-fill inundation model from a DEM and water-level.

    Returns a grid of 1 (flooded) / 0 (dry).
    """
    rows = len(dem)
    cols = len(dem[0]) if rows else 0
    result = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            elev = dem[r][c]
            if elev != nodata and elev <= water_level:
                result[r][c] = 1
    return result


# ---------------------------------------------------------------------------
# 630. Sea-level rise inundation
# ---------------------------------------------------------------------------

def sea_level_rise_inundation(
    dem: Sequence[Sequence[float]],
    sea_level_rise: float,
    *,
    current_sea_level: float = 0.0,
    nodata: float = -9999.0,
) -> List[List[int]]:
    """Sea-level rise inundation grid (bathtub model)."""
    threshold = current_sea_level + sea_level_rise
    return flood_inundation(dem, threshold, nodata=nodata)


# ---------------------------------------------------------------------------
# 631. Rainfall-runoff SCS-CN
# ---------------------------------------------------------------------------

def scs_curve_number_runoff(
    rainfall: float,
    curve_number: float,
    *,
    initial_abstraction_ratio: float = 0.2,
) -> float:
    """SCS Curve Number method – compute direct runoff from rainfall.

    Returns runoff depth in same units as *rainfall*.
    """
    if curve_number <= 0 or curve_number >= 100:
        raise ValueError("curve_number must be between 0 and 100 exclusive")
    s = (1000.0 / curve_number) - 10.0
    ia = initial_abstraction_ratio * s
    if rainfall <= ia:
        return 0.0
    return ((rainfall - ia) ** 2) / (rainfall - ia + s)


# ---------------------------------------------------------------------------
# 632. Erosion risk (RUSLE)
# ---------------------------------------------------------------------------

def rusle_erosion(
    r_factor: float,
    k_factor: float,
    ls_factor: float,
    c_factor: float,
    p_factor: float,
) -> float:
    """Revised Universal Soil Loss Equation (RUSLE): A = R * K * LS * C * P."""
    return r_factor * k_factor * ls_factor * c_factor * p_factor


# ---------------------------------------------------------------------------
# 633. Landslide susceptibility (simple weight-of-evidence)
# ---------------------------------------------------------------------------

def landslide_susceptibility(
    factors: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> List[float]:
    """Weighted-overlay landslide susceptibility score per cell."""
    n = len(factors[0]) if factors else 0
    scores = [0.0] * n
    for fi, w in zip(factors, weights):
        for i in range(n):
            scores[i] += fi[i] * w
    return scores


# ---------------------------------------------------------------------------
# 634. Wildfire spread (simple CA model)
# ---------------------------------------------------------------------------

def wildfire_spread(
    fuel_grid: Sequence[Sequence[float]],
    ignition: Tuple[int, int],
    *,
    spread_threshold: float = 0.5,
    max_steps: int = 100,
) -> List[List[int]]:
    """Simplified cellular-automata wildfire spread model.

    Returns grid of burn-time (step number) or 0 if unburned.
    """
    rows = len(fuel_grid)
    cols = len(fuel_grid[0]) if rows else 0
    burned = [[0] * cols for _ in range(rows)]
    if not (0 <= ignition[0] < rows and 0 <= ignition[1] < cols):
        return burned
    burned[ignition[0]][ignition[1]] = 1
    front = [ignition]
    for step in range(2, max_steps + 2):
        new_front: List[Tuple[int, int]] = []
        for r, c in front:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and burned[nr][nc] == 0:
                    if fuel_grid[nr][nc] >= spread_threshold:
                        burned[nr][nc] = step
                        new_front.append((nr, nc))
        if not new_front:
            break
        front = new_front
    return burned


# ---------------------------------------------------------------------------
# 635. Habitat suitability model (HSI)
# ---------------------------------------------------------------------------

def habitat_suitability_index(
    factors: Dict[str, Sequence[float]],
    weights: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Compute weighted Habitat Suitability Index (HSI) per cell.

    Each factor should have values normalised 0–1. *weights* default to equal.
    """
    keys = list(factors.keys())
    if not keys:
        return []
    n = len(factors[keys[0]])
    if weights is None:
        weights = {k: 1.0 / len(keys) for k in keys}
    total_w = sum(weights.get(k, 0.0) for k in keys)
    scores = [0.0] * n
    for k in keys:
        w = weights.get(k, 0.0) / total_w if total_w else 0.0
        for i in range(n):
            scores[i] += factors[k][i] * w
    return scores


# ---------------------------------------------------------------------------
# 627. Noise propagation model (ISO 9613-2 simplified)
# ---------------------------------------------------------------------------

def noise_propagation(
    source: Tuple[float, float],
    source_power_db: float,
    receptor_points: Sequence[Tuple[float, float]],
    *,
    ground_absorption: float = 0.5,
    frequency_hz: float = 500.0,
) -> List[float]:
    """Simplified noise propagation (geometric spreading + ground attenuation).

    Returns estimated dB(A) at each receptor.
    """
    results: List[float] = []
    for rx, ry in receptor_points:
        d = math.sqrt((rx - source[0]) ** 2 + (ry - source[1]) ** 2)
        if d < 1.0:
            d = 1.0
        geometric_loss = 20 * math.log10(d) + 11
        alpha = 0.005 * frequency_hz / 1000.0
        atm_absorption = alpha * d / 1000.0
        ground_effect = ground_absorption * 4.8 - (2 + ground_absorption * 14.4) * (d / 1000.0)
        level = source_power_db - geometric_loss - atm_absorption - max(0, ground_effect)
        results.append(level)
    return results


# ---------------------------------------------------------------------------
# 625. Solar radiation surface (simple)
# ---------------------------------------------------------------------------

def solar_radiation_surface(
    slope_grid: Sequence[Sequence[float]],
    aspect_grid: Sequence[Sequence[float]],
    latitude: float,
    *,
    day_of_year: int = 172,
) -> List[List[float]]:
    """Estimate daily solar radiation (MJ/m²) for a terrain surface."""
    rows = len(slope_grid)
    cols = len(slope_grid[0]) if rows else 0
    solar_constant = 1367.0
    declination = 23.45 * math.sin(math.radians(360 / 365 * (284 + day_of_year)))
    lat_rad = math.radians(latitude)
    decl_rad = math.radians(declination)
    result = []
    for r in range(rows):
        row_vals: List[float] = []
        for c in range(cols):
            slope_rad = math.radians(slope_grid[r][c])
            aspect_rad = math.radians(aspect_grid[r][c])
            cos_inc = (math.sin(lat_rad) * math.cos(slope_rad) -
                       math.cos(lat_rad) * math.sin(slope_rad) * math.cos(aspect_rad))
            cos_inc = max(0.0, cos_inc)
            omega_s = math.acos(max(-1, min(1, -math.tan(lat_rad) * math.tan(decl_rad))))
            daily = (24 * 3600 * solar_constant / math.pi *
                     (1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)) *
                     cos_inc * omega_s) / 1e6
            row_vals.append(max(0.0, daily))
        result.append(row_vals)
    return result


# ---------------------------------------------------------------------------
# 626. Wind exposure surface
# ---------------------------------------------------------------------------

def wind_exposure_index(
    elevation_grid: Sequence[Sequence[float]],
    *,
    search_radius: int = 3,
) -> List[List[float]]:
    """Topographic wind exposure index (ratio of point elevation to neighbourhood mean)."""
    rows = len(elevation_grid)
    cols = len(elevation_grid[0]) if rows else 0
    result = []
    for r in range(rows):
        row_vals: List[float] = []
        for c in range(cols):
            vals: List[float] = []
            for dr in range(-search_radius, search_radius + 1):
                for dc in range(-search_radius, search_radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        vals.append(elevation_grid[nr][nc])
            mean_val = sum(vals) / len(vals) if vals else 0.0
            elev = elevation_grid[r][c]
            idx = elev / mean_val if mean_val > 0 else 1.0
            row_vals.append(idx)
        result.append(row_vals)
    return result


# ---------------------------------------------------------------------------
# 624. Sun shadow volume computation
# ---------------------------------------------------------------------------

def sun_shadow_map(
    elevation_grid: Sequence[Sequence[float]],
    sun_altitude: float,
    sun_azimuth: float,
    *,
    cell_size: float = 1.0,
) -> List[List[int]]:
    """Compute binary shadow map from DEM and sun position.

    Returns 1 = shadow, 0 = sunlit.
    """
    rows = len(elevation_grid)
    cols = len(elevation_grid[0]) if rows else 0
    shadow = [[0] * cols for _ in range(rows)]
    if sun_altitude <= 0:
        return [[1] * cols for _ in range(rows)]
    tan_alt = math.tan(math.radians(sun_altitude))
    az_rad = math.radians(sun_azimuth)
    dx = -math.sin(az_rad)
    dy = -math.cos(az_rad)
    for r in range(rows):
        for c in range(cols):
            elev = elevation_grid[r][c]
            step = 1
            in_shadow = False
            while True:
                sr = r + int(round(step * dy / cell_size))
                sc = c + int(round(step * dx / cell_size))
                if not (0 <= sr < rows and 0 <= sc < cols):
                    break
                blocking_elev = elevation_grid[sr][sc]
                dist = step * cell_size
                required_elev = elev + dist * tan_alt
                if blocking_elev >= required_elev:
                    in_shadow = True
                    break
                step += 1
            shadow[r][c] = 1 if in_shadow else 0
    return shadow


# ---------------------------------------------------------------------------
# 1258. Storm-water runoff calculation (SCS-CN) - for area
# ---------------------------------------------------------------------------

def stormwater_runoff_volume(
    rainfall_mm: float,
    area_m2: float,
    curve_number: float,
) -> float:
    """Estimate stormwater runoff volume (m³) for an area using SCS-CN."""
    rainfall_in = rainfall_mm / 25.4
    runoff_in = scs_curve_number_runoff(rainfall_in, curve_number)
    runoff_mm = runoff_in * 25.4
    return area_m2 * runoff_mm / 1000.0


# ---------------------------------------------------------------------------
# 1257. BMP sizing
# ---------------------------------------------------------------------------

def bmp_sizing(
    design_storm_mm: float,
    drainage_area_m2: float,
    runoff_coefficient: float = 0.9,
    *,
    bmp_type: str = "retention",
    target_capture_pct: float = 0.9,
) -> Dict[str, float | str]:
    """Size a best-management practice (BMP) for stormwater treatment."""
    runoff_volume = drainage_area_m2 * design_storm_mm / 1000.0 * runoff_coefficient
    capture_volume = runoff_volume * target_capture_pct
    if bmp_type == "retention":
        depth = 1.5
    elif bmp_type == "bioretention":
        depth = 0.9
    elif bmp_type == "swale":
        depth = 0.6
    else:
        depth = 1.0
    surface_area = capture_volume / depth
    return {
        "runoff_volume_m3": runoff_volume,
        "capture_volume_m3": capture_volume,
        "surface_area_m2": surface_area,
        "depth_m": depth,
        "bmp_type": bmp_type,
    }


# ---------------------------------------------------------------------------
# 580. Voltage drop calculation
# ---------------------------------------------------------------------------

def voltage_drop(
    current_amps: float,
    length_m: float,
    conductor_resistance_per_km: float,
    *,
    voltage_supply: float = 240.0,
    phases: int = 1,
) -> Dict[str, float]:
    """Calculate voltage drop for a conductor run."""
    r = conductor_resistance_per_km * length_m / 1000.0
    if phases == 3:
        vd = math.sqrt(3) * current_amps * r
    else:
        vd = 2 * current_amps * r
    pct = (vd / voltage_supply) * 100.0 if voltage_supply else 0.0
    return {"voltage_drop_v": vd, "voltage_drop_pct": pct, "compliant": pct <= 5.0}


# ---------------------------------------------------------------------------
# 575. Pipe sizing from flow/velocity
# ---------------------------------------------------------------------------

def pipe_sizing(
    flow_rate_m3s: float,
    target_velocity_ms: float = 1.5,
) -> Dict[str, float]:
    """Size a pipe from flow rate and target velocity."""
    area = flow_rate_m3s / target_velocity_ms
    diameter = math.sqrt(4 * area / math.pi)
    return {"diameter_m": diameter, "area_m2": area, "velocity_ms": target_velocity_ms}


# ---------------------------------------------------------------------------
# 570. Demand estimation from land use
# ---------------------------------------------------------------------------

def demand_from_land_use(
    land_use_areas: Dict[str, float],
    demand_rates: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Estimate water/utility demand from land-use areas (m²) and per-m² rates."""
    defaults: Dict[str, float] = {
        "residential": 0.0005,
        "commercial": 0.0003,
        "industrial": 0.0008,
        "agricultural": 0.001,
        "open_space": 0.0001,
    }
    rates = demand_rates or defaults
    demands: Dict[str, float] = {}
    total = 0.0
    for lu, area in land_use_areas.items():
        rate = rates.get(lu.lower(), 0.0002)
        d = area * rate
        demands[lu] = d
        total += d
    return {"demands_by_type": demands, "total_demand": total}


# ---------------------------------------------------------------------------
# 1441. Hazardous material plume model
# ---------------------------------------------------------------------------

def hazmat_plume(
    release_rate_kg_s: float,
    wind_speed_ms: float,
    wind_direction_deg: float,
    source: Tuple[float, float],
    distances: Sequence[float],
    *,
    stability_class: str = "D",
) -> List[Dict[str, float]]:
    """Simplified Gaussian hazmat plume centreline concentration at downwind distances."""
    _sigma_coeffs = {
        "A": (0.22, 0.20), "B": (0.16, 0.12), "C": (0.11, 0.08),
        "D": (0.08, 0.06), "E": (0.06, 0.03), "F": (0.04, 0.016),
    }
    ay, az = _sigma_coeffs.get(stability_class.upper(), (0.08, 0.06))
    results: List[Dict[str, float]] = []
    for d in distances:
        if d <= 0:
            results.append({"distance_m": d, "concentration_kg_m3": 0.0})
            continue
        sigma_y = ay * d
        sigma_z = az * d
        conc = release_rate_kg_s / (math.pi * wind_speed_ms * sigma_y * sigma_z)
        results.append({"distance_m": d, "concentration_kg_m3": max(0.0, conc)})
    return results


# ---------------------------------------------------------------------------
# 1442. Blast-zone buffer
# ---------------------------------------------------------------------------

def blast_zone_radii(
    tnt_equivalent_kg: float,
) -> Dict[str, float]:
    """Estimate blast-zone radii for a given TNT-equivalent charge weight.

    Returns radiating-glass, evacuation, and lethal radius in metres.
    """
    cube_root = tnt_equivalent_kg ** (1.0 / 3.0)
    return {
        "lethal_m": 3.5 * cube_root,
        "severe_injury_m": 6.0 * cube_root,
        "glass_breakage_m": 22.0 * cube_root,
        "evacuation_m": 45.0 * cube_root,
    }


# ---------------------------------------------------------------------------
# 1366. Heat island analysis
# ---------------------------------------------------------------------------

def heat_island_intensity(
    urban_temps: Sequence[float],
    rural_temps: Sequence[float],
) -> Dict[str, float]:
    """Calculate Urban Heat Island intensity metrics."""
    if not urban_temps or not rural_temps:
        return {"uhi_mean": 0.0, "uhi_max": 0.0}
    mean_u = sum(urban_temps) / len(urban_temps)
    mean_r = sum(rural_temps) / len(rural_temps)
    max_u = max(urban_temps)
    max_r = max(rural_temps)
    return {"uhi_mean": mean_u - mean_r, "uhi_max": max_u - max_r}


# ---------------------------------------------------------------------------
# 1361. Carbon stock estimation
# ---------------------------------------------------------------------------

def carbon_stock_estimate(
    biomass_tons_per_ha: float,
    area_ha: float,
    *,
    carbon_fraction: float = 0.47,
    below_ground_ratio: float = 0.26,
) -> Dict[str, float]:
    """Estimate carbon stock from above-ground biomass."""
    above_ground = biomass_tons_per_ha * area_ha * carbon_fraction
    below_ground = above_ground * below_ground_ratio
    total = above_ground + below_ground
    co2_equivalent = total * 3.67
    return {
        "above_ground_tc": above_ground,
        "below_ground_tc": below_ground,
        "total_carbon_tc": total,
        "co2_equivalent_t": co2_equivalent,
    }


# ---------------------------------------------------------------------------
# 1406. Sprawl index
# ---------------------------------------------------------------------------

def sprawl_index(
    population_density: float,
    employment_density: float,
    centrality: float,
    street_connectivity: float,
) -> float:
    """Compute composite urban-sprawl index (higher = more compact)."""
    return (
        0.25 * min(population_density / 5000.0, 1.0) +
        0.25 * min(employment_density / 3000.0, 1.0) +
        0.25 * min(centrality, 1.0) +
        0.25 * min(street_connectivity / 150.0, 1.0)
    ) * 100.0


# ---------------------------------------------------------------------------
# 1451. Airport noise contour (simplified)
# ---------------------------------------------------------------------------

def airport_noise_contour(
    runway_coords: Tuple[Tuple[float, float], Tuple[float, float]],
    operations_per_day: int,
    receptor_points: Sequence[Tuple[float, float]],
) -> List[float]:
    """Simplified airport noise contour (DNL/Ldn) at receptor points."""
    mid_x = (runway_coords[0][0] + runway_coords[1][0]) / 2
    mid_y = (runway_coords[0][1] + runway_coords[1][1]) / 2
    base_noise = 10 * math.log10(max(1, operations_per_day)) + 65
    results: List[float] = []
    for rx, ry in receptor_points:
        d = math.sqrt((rx - mid_x) ** 2 + (ry - mid_y) ** 2)
        if d < 100:
            d = 100
        attenuation = 20 * math.log10(d / 100)
        results.append(base_noise - attenuation)
    return results


# ---------------------------------------------------------------------------
# 488. Trend analysis (Mann-Kendall spatial)
# ---------------------------------------------------------------------------

def mann_kendall_trend(
    time_series: Sequence[float],
) -> Dict[str, Any]:
    """Mann-Kendall trend test for a time series."""
    n = len(time_series)
    if n < 4:
        raise ValueError("need at least 4 observations")
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = time_series[j] - time_series[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / math.sqrt(var_s) if var_s > 0 else 0.0
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s) if var_s > 0 else 0.0
    else:
        z = 0.0
    significant = abs(z) > 1.96
    if s > 0:
        trend = "increasing"
    elif s < 0:
        trend = "decreasing"
    else:
        trend = "no trend"
    slopes: List[float] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((time_series[j] - time_series[i]) / (j - i))
    slopes.sort()
    sen_slope = slopes[len(slopes) // 2] if slopes else 0.0
    return {"s": s, "z": z, "trend": trend, "significant": significant, "sen_slope": sen_slope}


# ---------------------------------------------------------------------------
# 489. Change-point detection
# ---------------------------------------------------------------------------

def change_point_detection(
    series: Sequence[float],
    *,
    method: str = "cusum",
) -> Dict[str, Any]:
    """Detect change points in a time series using CUSUM."""
    n = len(series)
    if n < 4:
        return {"change_points": [], "scores": []}
    mean_all = sum(series) / n
    cusum = [0.0]
    for v in series:
        cusum.append(cusum[-1] + (v - mean_all))
    max_diff = 0.0
    cp = 0
    for i in range(1, n):
        diff = abs(cusum[i] - cusum[0])
        if diff > max_diff:
            max_diff = diff
            cp = i
    threshold = 1.5 * (max(series) - min(series)) if max(series) != min(series) else 0
    change_points = [cp] if max_diff > threshold and 0 < cp < n - 1 else []
    return {"change_points": change_points, "scores": cusum[1:], "max_statistic": max_diff}


# ---------------------------------------------------------------------------
# 1274. Telecom signal attenuation model
# ---------------------------------------------------------------------------

def telecom_signal_attenuation(
    frequency_mhz: float,
    distance_km: float,
    *,
    tx_power_dbm: float = 43.0,
    tx_gain_dbi: float = 15.0,
    rx_gain_dbi: float = 0.0,
    environment: str = "urban",
) -> Dict[str, float]:
    """Simplified Hata model for telecom signal attenuation."""
    if distance_km <= 0:
        return {"path_loss_db": 0.0, "received_power_dbm": tx_power_dbm + tx_gain_dbi}
    f = frequency_mhz
    d = max(distance_km, 0.001)
    hb = 30.0
    hm = 1.5
    ahm = (1.1 * math.log10(f) - 0.7) * hm - (1.56 * math.log10(f) - 0.8)
    path_loss = (69.55 + 26.16 * math.log10(f) - 13.82 * math.log10(hb) -
                 ahm + (44.9 - 6.55 * math.log10(hb)) * math.log10(d))
    if environment == "suburban":
        path_loss -= 2 * (math.log10(f / 28)) ** 2 - 5.4
    elif environment == "rural":
        path_loss -= 4.78 * (math.log10(f)) ** 2 + 18.33 * math.log10(f) - 40.94
    received = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss
    return {"path_loss_db": path_loss, "received_power_dbm": received}


# ---------------------------------------------------------------------------
# 1275. Telecom coverage area estimation
# ---------------------------------------------------------------------------

def telecom_coverage_radius(
    frequency_mhz: float,
    tx_power_dbm: float = 43.0,
    sensitivity_dbm: float = -100.0,
    *,
    environment: str = "urban",
) -> float:
    """Estimate max coverage radius (km) for a base station."""
    low, high = 0.01, 100.0
    for _ in range(50):
        mid = (low + high) / 2
        result = telecom_signal_attenuation(frequency_mhz, mid,
                                             tx_power_dbm=tx_power_dbm,
                                             environment=environment)
        if result["received_power_dbm"] > sensitivity_dbm:
            low = mid
        else:
            high = mid
    return (low + high) / 2
