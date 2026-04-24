"""Coordinate system parsers for military, amateur-radio, and open grid formats.

Covers MGRS, USNG, GEOREF, GARS, Open Location Code (Plus-codes),
and Maidenhead locator grid systems.  All functions are pure-Python
with no external dependencies.
"""
from __future__ import annotations

import math
import re
import string
import warnings
from typing import Any

from ._capabilities import check_capability

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_LETTERS = string.ascii_uppercase


def _letter_index(char: str) -> int:
    idx = ord(char.upper()) - ord("A")
    if idx < 0 or idx > 25:
        raise ValueError(f"invalid letter: {char!r}")
    return idx


# ---------------------------------------------------------------------------
# MGRS  (items 274)
# ---------------------------------------------------------------------------

_MGRS_BANDS = "CDEFGHJKLMNPQRSTUVWX"
_UTM_COL_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # I/O omitted in some specs
_UTM_ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"       # 20-char cycle


def _utm_zone_from_lon_lat(lon: float, lat: float) -> tuple[int, str]:
    """Return (zone_number, band_letter) for WGS-84 lon/lat."""
    zone = int((lon + 180) / 6) + 1
    if zone > 60:
        zone = 60
    idx = int((lat + 80) / 8)
    idx = max(0, min(idx, len(_MGRS_BANDS) - 1))
    return zone, _MGRS_BANDS[idx]


def mgrs_to_lonlat(mgrs_string: str) -> tuple[float, float]:
    """Parse an MGRS coordinate string and return (longitude, latitude).

    Supports precision from 1 m to 10 km (1–5 digit easting/northing pairs).

    >>> mgrs_to_lonlat("18SUJ2337006519")
    (-77.038..., 38.897...)
    """
    text = mgrs_string.strip().replace(" ", "").upper()
    m = re.match(r"^(\d{1,2})([C-X])([A-Z]{2})(\d*)$", text)
    if not m:
        raise ValueError(f"invalid MGRS string: {mgrs_string!r}")

    zone_number = int(m.group(1))
    band_letter = m.group(2)
    col_letter, row_letter = m.group(3)
    digits = m.group(4)

    if len(digits) % 2 != 0:
        raise ValueError(f"MGRS numeric part must have even length, got {len(digits)}")

    precision = len(digits) // 2
    if precision == 0:
        easting_m = 0.0
        northing_m = 0.0
        cell_size = 100_000.0
    else:
        scale = 10 ** (5 - precision)
        easting_m = int(digits[:precision]) * scale
        northing_m = int(digits[precision:]) * scale
        cell_size = scale

    # 100 km column: depends on zone set
    set_index = (zone_number - 1) % 6
    col_origin = (set_index % 3) * 8  # simplified col origin
    col_idx = _UTM_COL_LETTERS.index(col_letter) - col_origin
    if col_idx < 0:
        col_idx += 24

    easting = (col_idx + 1) * 100_000 + easting_m

    # 100 km row
    row_cycle = 0 if (zone_number % 2) == 1 else 5
    row_idx = _UTM_ROW_LETTERS.index(row_letter)
    northing_100k_raw = (row_idx - row_cycle) % 20

    # Band latitude lower bound
    band_idx = _MGRS_BANDS.index(band_letter)
    lat_min = -80 + band_idx * 8

    # Approximate northing from band
    base_northing = lat_min * 111_320.0  # rough m per degree
    northing_base = (northing_100k_raw * 100_000)
    # Adjust to nearest match
    full_northing = northing_base + northing_m
    while full_northing < base_northing - 100_000:
        full_northing += 2_000_000
    while full_northing > base_northing + 900_000:
        full_northing -= 2_000_000

    northing = full_northing + cell_size / 2

    # Convert UTM to lon/lat (simplified inverse)
    lat, lon = _utm_to_lonlat(zone_number, easting + cell_size / 2, northing)
    return (lon, lat)


def _utm_to_lonlat(zone: int, easting: float, northing: float) -> tuple[float, float]:
    """Simplified UTM → WGS-84 conversion (accurate to ~1 m for most zones)."""
    a = 6_378_137.0
    f = 1 / 298.257223563
    e = math.sqrt(2 * f - f * f)
    e2 = e * e
    k0 = 0.9996
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))

    x = easting - 500_000.0
    y = northing

    M = y / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))

    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
             + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
             + (151 * e1 ** 3 / 96) * math.sin(6 * mu))

    sin_phi = math.sin(phi1)
    cos_phi = math.cos(phi1)
    tan_phi = math.tan(phi1)
    N1 = a / math.sqrt(1 - e2 * sin_phi ** 2)
    T1 = tan_phi ** 2
    C1 = (e2 / (1 - e2)) * cos_phi ** 2
    R1 = a * (1 - e2) / (1 - e2 * sin_phi ** 2) ** 1.5
    D = x / (N1 * k0)

    lat = phi1 - (N1 * tan_phi / R1) * (
        D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * (e2 / (1 - e2))) * D ** 4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * (e2 / (1 - e2)) - 3 * C1 ** 2) * D ** 6 / 720
    )

    lon = (D - (1 + 2 * T1 + C1) * D ** 3 / 6
           + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * (e2 / (1 - e2)) + 24 * T1 ** 2) * D ** 5 / 120) / cos_phi

    lat_deg = math.degrees(lat)
    lon0 = (zone - 1) * 6 - 180 + 3
    lon_deg = lon0 + math.degrees(lon)

    return (lat_deg, lon_deg)


def lonlat_to_mgrs(lon: float, lat: float, precision: int = 5) -> str:
    """Convert WGS-84 lon/lat to an MGRS string.

    *precision* controls the numeric digit pairs (5 = 1 m, 1 = 10 km).
    """
    if lat < -80 or lat > 84:
        raise ValueError("MGRS is valid for latitudes -80 to 84")
    if precision < 0 or precision > 5:
        raise ValueError("precision must be 0-5")

    zone, band = _utm_zone_from_lon_lat(lon, lat)
    easting, northing = _lonlat_to_utm(lon, lat, zone)

    # 100-km column / row letters
    set_index = (zone - 1) % 6
    col_origin = (set_index % 3) * 8
    col_idx = int(easting / 100_000) - 1
    col_letter = _UTM_COL_LETTERS[(col_origin + col_idx) % 24]

    row_cycle = 0 if (zone % 2) == 1 else 5
    row_idx = int(northing / 100_000) % 20
    row_letter = _UTM_ROW_LETTERS[(row_idx + row_cycle) % 20]

    e_remainder = easting % 100_000
    n_remainder = northing % 100_000

    if precision == 0:
        digits = ""
    else:
        scale = 10 ** (5 - precision)
        e_digits = str(int(e_remainder / scale)).zfill(precision)
        n_digits = str(int(n_remainder / scale)).zfill(precision)
        digits = e_digits + n_digits

    return f"{zone:02d}{band}{col_letter}{row_letter}{digits}"


def _lonlat_to_utm(lon: float, lat: float, zone: int) -> tuple[float, float]:
    """Simplified WGS-84 → UTM conversion."""
    a = 6_378_137.0
    f = 1 / 298.257223563
    e = math.sqrt(2 * f - f * f)
    e2 = e * e
    k0 = 0.9996

    lat_rad = math.radians(lat)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    lon_rad = math.radians(lon)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    tan_lat = math.tan(lat_rad)
    N = a / math.sqrt(1 - e2 * sin_lat ** 2)
    T = tan_lat ** 2
    C = (e2 / (1 - e2)) * cos_lat ** 2
    A_val = (lon_rad - lon0) * cos_lat

    M = a * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat_rad)
        + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat_rad)
        - (35 * e2 ** 3 / 3072) * math.sin(6 * lat_rad)
    )

    easting = k0 * N * (
        A_val + (1 - T + C) * A_val ** 3 / 6
        + (5 - 18 * T + T ** 2 + 72 * C - 58 * (e2 / (1 - e2))) * A_val ** 5 / 120
    ) + 500_000.0

    northing = k0 * (
        M + N * tan_lat * (
            A_val ** 2 / 2
            + (5 - T + 9 * C + 4 * C ** 2) * A_val ** 4 / 24
            + (61 - 58 * T + T ** 2 + 600 * C - 330 * (e2 / (1 - e2))) * A_val ** 6 / 720
        )
    )

    if lat < 0:
        northing += 10_000_000.0

    return (easting, northing)


# ---------------------------------------------------------------------------
# USNG  (item 275) — effectively the same grid as MGRS for CONUS
# ---------------------------------------------------------------------------

def usng_to_lonlat(usng_string: str) -> tuple[float, float]:
    """Parse a US National Grid (USNG) string to (longitude, latitude).

    USNG is identical to MGRS within the CONUS and its territories.
    """
    return mgrs_to_lonlat(usng_string)


def lonlat_to_usng(lon: float, lat: float, precision: int = 5) -> str:
    """Convert WGS-84 lon/lat to a USNG string."""
    return lonlat_to_mgrs(lon, lat, precision=precision)


# ---------------------------------------------------------------------------
# GEOREF  (item 276) — World Geographic Reference System
# ---------------------------------------------------------------------------

_GEOREF_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # I and O omitted (24 chars)


def georef_to_lonlat(georef_string: str) -> tuple[float, float]:
    """Parse a GEOREF string and return (longitude, latitude).

    >>> georef_to_lonlat("FJQE1234512345")
    """
    text = georef_string.strip().replace(" ", "").upper()
    if len(text) < 4:
        raise ValueError(f"GEOREF string too short: {georef_string!r}")

    # First two letters: 15° longitude quadrangle
    lon_quad = _GEOREF_LETTERS.index(text[0])
    lat_quad = _GEOREF_LETTERS.index(text[1])

    # Next two letters: 1° cell within quadrangle
    lon_cell = _GEOREF_LETTERS.index(text[2])
    lat_cell = _GEOREF_LETTERS.index(text[3])

    digits = text[4:]
    if len(digits) % 2 != 0:
        raise ValueError(f"GEOREF numeric part must have even length, got {len(digits)}")

    half = len(digits) // 2
    if half == 0:
        lon_frac = 0.0
        lat_frac = 0.0
    else:
        # Minutes or fractional minutes
        lon_minutes = int(digits[:half])
        lat_minutes = int(digits[half:])
        divisor = 10 ** (half - 2) if half > 2 else 1
        lon_frac = lon_minutes / (60.0 * max(divisor, 1)) if half > 2 else lon_minutes / 60.0
        lat_frac = lat_minutes / (60.0 * max(divisor, 1)) if half > 2 else lat_minutes / 60.0

    lon = -180 + lon_quad * 15 + lon_cell + lon_frac
    lat = -90 + lat_quad * 15 + lat_cell + lat_frac

    return (lon, lat)


def lonlat_to_georef(lon: float, lat: float, precision: int = 4) -> str:
    """Convert WGS-84 lon/lat to a GEOREF string.

    *precision* is the total digit count for each of lon/lat (2=minutes, 4=0.01 min).
    """
    lon_shifted = lon + 180
    lat_shifted = lat + 90

    lon_quad = int(lon_shifted / 15)
    lat_quad = int(lat_shifted / 15)
    lon_remainder = lon_shifted - lon_quad * 15
    lat_remainder = lat_shifted - lat_quad * 15

    lon_cell = int(lon_remainder)
    lat_cell = int(lat_remainder)

    lon_min = (lon_remainder - lon_cell) * 60
    lat_min = (lat_remainder - lat_cell) * 60

    half = max(precision // 2, 1)
    if half <= 2:
        lon_digits = str(int(lon_min)).zfill(2)
        lat_digits = str(int(lat_min)).zfill(2)
    else:
        scale = 10 ** (half - 2)
        lon_digits = str(int(lon_min * scale)).zfill(half)
        lat_digits = str(int(lat_min * scale)).zfill(half)

    return (
        _GEOREF_LETTERS[min(lon_quad, 23)]
        + _GEOREF_LETTERS[min(lat_quad, 23)]
        + _GEOREF_LETTERS[min(lon_cell, 23)]
        + _GEOREF_LETTERS[min(lat_cell, 23)]
        + lon_digits + lat_digits
    )


# ---------------------------------------------------------------------------
# GARS  (item 277) — Global Area Reference System
# ---------------------------------------------------------------------------

def gars_to_lonlat(gars_string: str) -> tuple[float, float]:
    """Parse a GARS cell reference and return the centre (longitude, latitude).

    GARS cells are 30' × 30' at the base level, sub-divided into quadrants and key-pads.

    >>> gars_to_lonlat("361HN")
    """
    text = gars_string.strip().replace(" ", "").upper()
    if len(text) < 5:
        raise ValueError(f"GARS string too short: {gars_string!r}")

    lon_band = int(text[:3])
    lat_letters = text[3:5]

    lat_idx = (_letter_index(lat_letters[0])) * 26 + _letter_index(lat_letters[1])

    lon_centre = -180 + (lon_band - 1) * 0.5 + 0.25
    lat_centre = -90 + lat_idx * 0.5 + 0.25

    # Optional quadrant (1-4) and keypad (1-9)
    if len(text) > 5:
        quadrant = int(text[5]) if len(text) > 5 else 0
        keypad = int(text[6]) if len(text) > 6 else 0

        if quadrant >= 1:
            q_lon = -0.125 if quadrant in (1, 3) else 0.125
            q_lat = 0.125 if quadrant in (1, 2) else -0.125
            lon_centre += q_lon
            lat_centre += q_lat

        if keypad >= 1:
            kp_col = (keypad - 1) % 3 - 1  # -1, 0, 1
            kp_row = 1 - (keypad - 1) // 3  # 1, 0, -1
            lon_centre += kp_col * (0.25 / 3)
            lat_centre += kp_row * (0.25 / 3)

    return (lon_centre, lat_centre)


def lonlat_to_gars(lon: float, lat: float) -> str:
    """Convert WGS-84 lon/lat to a base GARS cell (30' resolution)."""
    lon_band = int((lon + 180) / 0.5) + 1
    lat_idx = int((lat + 90) / 0.5)

    lat_first = _LETTERS[lat_idx // 26]
    lat_second = _LETTERS[lat_idx % 26]

    return f"{lon_band:03d}{lat_first}{lat_second}"


# ---------------------------------------------------------------------------
# Open Location Code / Plus-codes  (item 279)
# ---------------------------------------------------------------------------

_OLC_ALPHABET = "23456789CFGHJMPQRVWX"
_OLC_BASE = len(_OLC_ALPHABET)  # 20


def pluscode_to_lonlat(code: str) -> tuple[float, float]:
    """Decode an Open Location Code (Plus-code) to (longitude, latitude).

    Returns the centre of the code area.

    >>> pluscode_to_lonlat("87C4VXQH+68")
    """
    text = code.strip().replace(" ", "").upper()
    text = text.replace("+", "")
    # Remove padding zeros
    text = text.replace("0", "")

    lat = 0.0
    lon = 0.0
    lat_res = 180.0
    lon_res = 360.0

    for i, char in enumerate(text):
        if i >= 10 and i % 2 == 0:
            # Grid refinement pairs after initial 10 chars
            lat_res /= 5
            lon_res /= 4
        elif i < 10:
            if i % 2 == 0:
                lat_res /= _OLC_BASE
            else:
                lon_res /= _OLC_BASE

        val = _OLC_ALPHABET.index(char)

        if i < 10:
            if i % 2 == 0:
                lat += val * lat_res
            else:
                lon += val * lon_res
        else:
            if i % 2 == 0:
                lat += (val // 4) * lat_res
                lon += (val % 4) * lon_res
            else:
                lat += (val // 4) * lat_res
                lon += (val % 4) * lon_res

    lat -= 90
    lon -= 180

    # Return centre of final cell
    lat += lat_res / 2
    lon += lon_res / 2

    return (lon, lat)


def lonlat_to_pluscode(lon: float, lat: float, code_length: int = 10) -> str:
    """Encode WGS-84 lon/lat as an Open Location Code (Plus-code).

    Default *code_length* of 10 gives ~14m × 14m precision.
    """
    if code_length < 2 or code_length > 15:
        raise ValueError("code_length must be 2-15")

    lat_val = lat + 90.0
    lon_val = lon + 180.0

    # Clamp
    lat_val = max(0, min(lat_val, 180.0 - 1e-10))
    lon_val = max(0, min(lon_val, 360.0 - 1e-10))

    code_chars: list[str] = []
    lat_res = 180.0
    lon_res = 360.0

    # First 10 characters: 5 pairs
    for i in range(min(code_length, 10)):
        if i % 2 == 0:
            lat_res /= _OLC_BASE
            digit = int(lat_val / lat_res)
            digit = min(digit, _OLC_BASE - 1)
            lat_val -= digit * lat_res
            code_chars.append(_OLC_ALPHABET[digit])
        else:
            lon_res /= _OLC_BASE
            digit = int(lon_val / lon_res)
            digit = min(digit, _OLC_BASE - 1)
            lon_val -= digit * lon_res
            code_chars.append(_OLC_ALPHABET[digit])

    # Grid refinement after 10 chars
    for _i in range(10, code_length):
        lat_res /= 5
        lon_res /= 4
        lat_digit = min(int(lat_val / lat_res), 4)
        lon_digit = min(int(lon_val / lon_res), 3)
        lat_val -= lat_digit * lat_res
        lon_val -= lon_digit * lon_res
        code_chars.append(_OLC_ALPHABET[lat_digit * 4 + lon_digit])

    # Pad to 8 with zeros, insert + after position 8
    while len(code_chars) < 8:
        code_chars.append("0")

    result = "".join(code_chars[:8]) + "+" + "".join(code_chars[8:])
    return result


# ---------------------------------------------------------------------------
# Maidenhead locator  (item 280)
# ---------------------------------------------------------------------------

def maidenhead_to_lonlat(locator: str) -> tuple[float, float]:
    """Decode a Maidenhead grid locator to (longitude, latitude).

    Supports 2-, 4-, 6-, 8-, and 10-character locators.

    >>> maidenhead_to_lonlat("FN31pr")
    """
    text = locator.strip()
    if len(text) < 2:
        raise ValueError(f"Maidenhead locator too short: {locator!r}")

    lon = -180.0
    lat = -90.0

    # Field (first pair — letters A-R)
    lon += _letter_index(text[0]) * 20
    lat += _letter_index(text[1]) * 10

    lon_res = 20.0
    lat_res = 10.0

    if len(text) >= 4:
        # Square (digits 0-9)
        lon_res = 2.0
        lat_res = 1.0
        lon += int(text[2]) * lon_res
        lat += int(text[3]) * lat_res

    if len(text) >= 6:
        # Sub-square (letters a-x, case-insensitive)
        lon_res = 2.0 / 24
        lat_res = 1.0 / 24
        lon += _letter_index(text[4]) * lon_res
        lat += _letter_index(text[5]) * lat_res

    if len(text) >= 8:
        lon_res = (2.0 / 24) / 10
        lat_res = (1.0 / 24) / 10
        lon += int(text[6]) * lon_res
        lat += int(text[7]) * lat_res

    if len(text) >= 10:
        lon_res = (2.0 / 24) / 10 / 24
        lat_res = (1.0 / 24) / 10 / 24
        lon += _letter_index(text[8]) * lon_res
        lat += _letter_index(text[9]) * lat_res

    # Return centre of the cell
    lon += lon_res / 2
    lat += lat_res / 2

    return (lon, lat)


def lonlat_to_maidenhead(lon: float, lat: float, precision: int = 3) -> str:
    """Encode WGS-84 lon/lat as a Maidenhead grid locator.

    *precision* 1 = field (2 chars), 2 = square (4 chars), 3 = sub-square (6 chars),
    4 = extended (8 chars), 5 = super-extended (10 chars).
    """
    if precision < 1 or precision > 5:
        raise ValueError("precision must be 1-5")

    lon_val = lon + 180.0
    lat_val = lat + 90.0

    # Clamp
    lon_val = max(0.0, min(lon_val, 360.0 - 1e-10))
    lat_val = max(0.0, min(lat_val, 180.0 - 1e-10))

    parts: list[str] = []

    # Field
    lon_field = int(lon_val / 20)
    lat_field = int(lat_val / 10)
    parts.append(chr(ord("A") + lon_field))
    parts.append(chr(ord("A") + lat_field))
    lon_val -= lon_field * 20
    lat_val -= lat_field * 10

    if precision >= 2:
        lon_sq = int(lon_val / 2)
        lat_sq = int(lat_val / 1)
        parts.append(str(lon_sq))
        parts.append(str(lat_sq))
        lon_val -= lon_sq * 2
        lat_val -= lat_sq * 1

    if precision >= 3:
        lon_sub = int(lon_val / (2 / 24))
        lat_sub = int(lat_val / (1 / 24))
        lon_sub = min(lon_sub, 23)
        lat_sub = min(lat_sub, 23)
        parts.append(chr(ord("a") + lon_sub))
        parts.append(chr(ord("a") + lat_sub))
        lon_val -= lon_sub * (2 / 24)
        lat_val -= lat_sub * (1 / 24)

    if precision >= 4:
        lon_ext = int(lon_val / (2 / 24 / 10))
        lat_ext = int(lat_val / (1 / 24 / 10))
        lon_ext = min(lon_ext, 9)
        lat_ext = min(lat_ext, 9)
        parts.append(str(lon_ext))
        parts.append(str(lat_ext))
        lon_val -= lon_ext * (2 / 24 / 10)
        lat_val -= lat_ext * (1 / 24 / 10)

    if precision >= 5:
        lon_sup = int(lon_val / (2 / 24 / 10 / 24))
        lat_sup = int(lat_val / (1 / 24 / 10 / 24))
        lon_sup = min(lon_sup, 23)
        lat_sup = min(lat_sup, 23)
        parts.append(chr(ord("a") + lon_sup))
        parts.append(chr(ord("a") + lat_sup))

    return "".join(parts)


# ---------------------------------------------------------------------------
# Helmert 7-parameter datum transformation  (item 213)
# ---------------------------------------------------------------------------

def helmert_transform(
    x: float, y: float, z: float,
    *,
    dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
    rx: float = 0.0, ry: float = 0.0, rz: float = 0.0,
    ds: float = 0.0,
) -> tuple[float, float, float]:
    """Apply a 7-parameter Helmert transformation (position vector convention).

    Parameters *dx/dy/dz* are translation in metres, *rx/ry/rz* are rotations
    in arc-seconds, and *ds* is the scale difference in ppm.

    Returns transformed (X, Y, Z) geocentric coordinates.
    """
    # Convert rotations from arc-seconds to radians
    as2rad = math.pi / (180 * 3600)
    rx_r = rx * as2rad
    ry_r = ry * as2rad
    rz_r = rz * as2rad
    s = 1 + ds * 1e-6

    x_out = dx + s * (x - rz_r * y + ry_r * z)
    y_out = dy + s * (rz_r * x + y - rx_r * z)
    z_out = dz + s * (-ry_r * x + rx_r * y + z)

    return (x_out, y_out, z_out)


def geodetic_to_ecef(
    lon: float, lat: float, height: float = 0.0,
    *, a: float = 6_378_137.0, f: float = 1 / 298.257223563,
) -> tuple[float, float, float]:
    """Convert geodetic (lon, lat, height) to ECEF (X, Y, Z)."""
    e2 = 2 * f - f * f
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    sin_lat = math.sin(lat_r)
    cos_lat = math.cos(lat_r)
    N = a / math.sqrt(1 - e2 * sin_lat ** 2)
    x = (N + height) * cos_lat * math.cos(lon_r)
    y = (N + height) * cos_lat * math.sin(lon_r)
    z = (N * (1 - e2) + height) * sin_lat
    return (x, y, z)


def ecef_to_geodetic(
    x: float, y: float, z: float,
    *, a: float = 6_378_137.0, f: float = 1 / 298.257223563,
) -> tuple[float, float, float]:
    """Convert ECEF (X, Y, Z) to geodetic (lon, lat, height)."""
    e2 = 2 * f - f * f
    lon = math.degrees(math.atan2(y, x))
    p = math.sqrt(x ** 2 + y ** 2)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1 - e2 * sin_lat ** 2)
        lat_new = math.atan2(z + e2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sin_lat ** 2)
    height = p / math.cos(lat) - N if abs(math.cos(lat)) > 1e-10 else abs(z) - a * (1 - e2) / math.sqrt(1 - e2 * sin_lat ** 2)
    return (lon, math.degrees(lat), height)


# ---------------------------------------------------------------------------
# Validate CRS chain accuracy  (item 300)
# ---------------------------------------------------------------------------

def validate_crs_chain(
    source_crs: str,
    target_crs: str,
    test_points: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Validate the accuracy of a CRS transformation chain.

    Performs a forward-then-inverse roundtrip and reports residuals.
    """
    if test_points is None:
        test_points = [(0.0, 0.0), (90.0, 45.0), (-90.0, -45.0), (180.0, 0.0)]

    residuals: list[dict[str, float]] = []

    if not check_capability("pyproj"):
        warnings.warn(
            "validate_crs_chain requires pyproj for CRS roundtrip validation; install geoprompt[projection].",
            UserWarning,
            stacklevel=2,
        )
        return {
            "source_crs": source_crs,
            "target_crs": target_crs,
            "test_points": len(test_points),
            "residuals": residuals,
            "max_residual_m": float("nan"),
            "acceptable": False,
        }

    from .crs import _project_xy  # noqa: F811

    projection_failures = 0
    for lon, lat in test_points:
        try:
            fwd_x, fwd_y = _project_xy(lon, lat, source_crs, target_crs)
            inv_lon, inv_lat = _project_xy(fwd_x, fwd_y, target_crs, source_crs)
            residuals.append({
                "original_lon": lon,
                "original_lat": lat,
                "residual_lon": abs(inv_lon - lon),
                "residual_lat": abs(inv_lat - lat),
                "residual_m": math.sqrt((inv_lon - lon) ** 2 + (inv_lat - lat) ** 2) * 111_320,
            })
        except Exception as exc:
            # Intentional: individual projection failures should degrade to NaN rows
            # so the audit can report partial coverage instead of aborting outright.
            projection_failures += 1
            residuals.append({
                "original_lon": lon,
                "original_lat": lat,
                "residual_lon": float("nan"),
                "residual_lat": float("nan"),
                "residual_m": float("nan"),
            })
            warnings.warn(
                f"validate_crs_chain failed for point ({lon}, {lat}): {exc}",
                UserWarning,
                stacklevel=2,
            )

    max_residual = max((r["residual_m"] for r in residuals if not math.isnan(r["residual_m"])), default=float("nan"))
    return {
        "source_crs": source_crs,
        "target_crs": target_crs,
        "test_points": len(test_points),
        "residuals": residuals,
        "max_residual_m": max_residual,
        "acceptable": projection_failures == 0 and not math.isnan(max_residual) and max_residual < 1.0,
    }


# ---------------------------------------------------------------------------
# GeoHash encoding/decoding  (item 1792)
# ---------------------------------------------------------------------------

_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_encode(lon: float, lat: float, precision: int = 12) -> str:
    """Encode lon/lat to a GeoHash string of given precision."""
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    bits: list[int] = []
    is_lon = True

    while len(bits) < precision * 5:
        if is_lon:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon >= mid:
                bits.append(1)
                lon_range[0] = mid
            else:
                bits.append(0)
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                bits.append(1)
                lat_range[0] = mid
            else:
                bits.append(0)
                lat_range[1] = mid
        is_lon = not is_lon

    result: list[str] = []
    for i in range(0, len(bits), 5):
        chunk = bits[i:i + 5]
        val = 0
        for bit in chunk:
            val = (val << 1) | bit
        result.append(_GEOHASH_ALPHABET[val])

    return "".join(result)


def geohash_decode(geohash: str) -> tuple[float, float]:
    """Decode a GeoHash string to (longitude, latitude) centre point."""
    bits: list[int] = []
    for char in geohash.lower():
        idx = _GEOHASH_ALPHABET.index(char)
        for i in range(4, -1, -1):
            bits.append((idx >> i) & 1)

    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    is_lon = True

    for bit in bits:
        if is_lon:
            mid = (lon_range[0] + lon_range[1]) / 2
            if bit:
                lon_range[0] = mid
            else:
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if bit:
                lat_range[0] = mid
            else:
                lat_range[1] = mid
        is_lon = not is_lon

    lon = (lon_range[0] + lon_range[1]) / 2
    lat = (lat_range[0] + lat_range[1]) / 2
    return (lon, lat)


def geohash_neighbors(geohash: str) -> dict[str, str]:
    """Return the 8 neighboring GeoHash cells."""
    lon, lat = geohash_decode(geohash)
    precision = len(geohash)

    # Approximate cell size
    lat_bits = (precision * 5) // 2
    lon_bits = (precision * 5 + 1) // 2
    lat_err = 180.0 / (2 ** lat_bits)
    lon_err = 360.0 / (2 ** lon_bits)

    directions = {
        "n": (0, lat_err),
        "s": (0, -lat_err),
        "e": (lon_err, 0),
        "w": (-lon_err, 0),
        "ne": (lon_err, lat_err),
        "nw": (-lon_err, lat_err),
        "se": (lon_err, -lat_err),
        "sw": (-lon_err, -lat_err),
    }

    result: dict[str, str] = {}
    for direction, (dlon, dlat) in directions.items():
        result[direction] = geohash_encode(lon + dlon, lat + dlat, precision)
    return result


# ---------------------------------------------------------------------------
# Hilbert curve spatial index  (item 1797)
# ---------------------------------------------------------------------------

def hilbert_xy_to_d(n: int, x: int, y: int) -> int:
    """Convert (x, y) to Hilbert curve distance for an n×n grid (n must be power of 2)."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def hilbert_d_to_xy(n: int, d: int) -> tuple[int, int]:
    """Convert Hilbert curve distance to (x, y) for an n×n grid."""
    x = 0
    y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) != 0 else 0
        ry = 1 if ((d & 1) ^ rx) != 0 else 0
        # Rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return (x, y)


def hilbert_encode_lonlat(lon: float, lat: float, level: int = 16) -> int:
    """Encode lon/lat to a Hilbert curve index at given level."""
    n = 2 ** level
    x = int((lon + 180) / 360 * n)
    y = int((lat + 90) / 180 * n)
    x = max(0, min(x, n - 1))
    y = max(0, min(y, n - 1))
    return hilbert_xy_to_d(n, x, y)


def hilbert_decode_to_lonlat(d: int, level: int = 16) -> tuple[float, float]:
    """Decode a Hilbert curve index to (longitude, latitude)."""
    n = 2 ** level
    x, y = hilbert_d_to_xy(n, d)
    lon = x / n * 360 - 180 + (180 / n)
    lat = y / n * 180 - 90 + (90 / n)
    return (lon, lat)


# ---------------------------------------------------------------------------
# K-D tree spatial index  (item 1799)
# ---------------------------------------------------------------------------

class KDTree:
    """Simple pure-Python K-D tree for 2D point queries."""

    __slots__ = ("_points", "_indices", "_root")

    class _Node:
        __slots__ = ("point", "index", "left", "right", "axis")

        def __init__(self, point: tuple[float, float], index: int, axis: int) -> None:
            self.point = point
            self.index = index
            self.left: KDTree._Node | None = None
            self.right: KDTree._Node | None = None
            self.axis = axis

    def __init__(self, points: list[tuple[float, float]]) -> None:
        self._points = points
        self._indices = list(range(len(points)))
        self._root = self._build(list(range(len(points))), 0)

    def _build(self, indices: list[int], depth: int) -> _Node | None:
        if not indices:
            return None
        axis = depth % 2
        indices.sort(key=lambda i: self._points[i][axis])
        mid = len(indices) // 2
        node = self._Node(self._points[indices[mid]], indices[mid], axis)
        node.left = self._build(indices[:mid], depth + 1)
        node.right = self._build(indices[mid + 1:], depth + 1)
        return node

    def query_radius(self, center: tuple[float, float], radius: float) -> list[tuple[int, float]]:
        """Return all points within *radius* of *center* as (index, distance) pairs."""
        results: list[tuple[int, float]] = []
        self._search_radius(self._root, center, radius * radius, results)
        return [(idx, math.sqrt(d2)) for idx, d2 in results]

    def _search_radius(
        self, node: _Node | None, center: tuple[float, float], r2: float,
        results: list[tuple[int, float]],
    ) -> None:
        if node is None:
            return
        dx = node.point[0] - center[0]
        dy = node.point[1] - center[1]
        dist2 = dx * dx + dy * dy
        if dist2 <= r2:
            results.append((node.index, dist2))
        diff = center[node.axis] - node.point[node.axis]
        close = node.left if diff < 0 else node.right
        far = node.right if diff < 0 else node.left
        self._search_radius(close, center, r2, results)
        if diff * diff <= r2:
            self._search_radius(far, center, r2, results)

    def query_nearest(self, center: tuple[float, float], k: int = 1) -> list[tuple[int, float]]:
        """Return the *k* nearest points as (index, distance) pairs."""
        best: list[tuple[float, int]] = []  # (dist2, index)
        self._search_knn(self._root, center, k, best)
        best.sort()
        return [(idx, math.sqrt(d2)) for d2, idx in best]

    def _search_knn(
        self, node: _Node | None, center: tuple[float, float],
        k: int, best: list[tuple[float, int]],
    ) -> None:
        if node is None:
            return
        dx = node.point[0] - center[0]
        dy = node.point[1] - center[1]
        dist2 = dx * dx + dy * dy
        if len(best) < k:
            best.append((dist2, node.index))
            best.sort(reverse=True)
        elif dist2 < best[0][0]:
            best[0] = (dist2, node.index)
            best.sort(reverse=True)

        diff = center[node.axis] - node.point[node.axis]
        close = node.left if diff < 0 else node.right
        far = node.right if diff < 0 else node.left
        self._search_knn(close, center, k, best)
        worst_dist = best[0][0] if len(best) == k else float("inf")
        if diff * diff < worst_dist:
            self._search_knn(far, center, k, best)


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    # MGRS
    "mgrs_to_lonlat",
    "lonlat_to_mgrs",
    # USNG
    "usng_to_lonlat",
    "lonlat_to_usng",
    # GEOREF
    "georef_to_lonlat",
    "lonlat_to_georef",
    # GARS
    "gars_to_lonlat",
    "lonlat_to_gars",
    # Plus-codes
    "pluscode_to_lonlat",
    "lonlat_to_pluscode",
    # Maidenhead
    "maidenhead_to_lonlat",
    "lonlat_to_maidenhead",
    # Helmert
    "helmert_transform",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    # Validation
    "validate_crs_chain",
    # GeoHash
    "geohash_encode",
    "geohash_decode",
    "geohash_neighbors",
    # Hilbert
    "hilbert_xy_to_d",
    "hilbert_d_to_xy",
    "hilbert_encode_lonlat",
    "hilbert_decode_to_lonlat",
    # K-D tree
    "KDTree",
]
