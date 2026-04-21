"""Classification and colour ramp utilities for cartographic rendering.

Pure-Python implementations of map classification methods, colour ramps,
and report-generation helpers covering roadmap items from A6
(Cartography & Visualisation 854-866, 881-893).
"""
from __future__ import annotations

import colorsys
import math
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Classification methods (860-866)
# ---------------------------------------------------------------------------


def classify_natural_breaks(
    values: Sequence[float],
    n_classes: int = 5,
) -> list[float]:
    """Natural breaks (Jenks) classification.

    Returns a list of *n_classes - 1* break values.
    Uses Fisher's exact algorithm (1D dynamic programming).
    """
    n = len(values)
    if n < n_classes:
        raise ValueError("not enough values for requested number of classes")
    sv = sorted(values)
    if n_classes == 1:
        return []
    # Compute sum-of-squared-deviations matrix
    mat1 = [[0.0] * (n + 1) for _ in range(n + 1)]
    mat2 = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        s = 0.0
        ss = 0.0
        for m in range(1, i + 1):
            idx = i - m
            val = sv[idx]
            s += val
            ss += val * val
            ssd = ss - (s * s) / m
            if idx > 0:
                for j in range(2, n_classes + 1):
                    if mat2[i][j] == 0 or mat2[idx][j - 1] + ssd < mat2[i][j]:
                        mat1[i][j] = idx
                        mat2[i][j] = mat2[idx][j - 1] + ssd
            else:
                mat1[i][1] = 0
                mat2[i][1] = ssd

    breaks = []
    k = n
    for j in range(n_classes, 1, -1):
        brk_idx = int(mat1[k][j])
        breaks.append(sv[brk_idx])
        k = brk_idx
    breaks.sort()
    return breaks


def classify_equal_interval(
    values: Sequence[float],
    n_classes: int = 5,
) -> list[float]:
    """Equal interval classification."""
    lo, hi = min(values), max(values)
    step = (hi - lo) / n_classes
    return [lo + step * i for i in range(1, n_classes)]


def classify_quantile(
    values: Sequence[float],
    n_classes: int = 5,
) -> list[float]:
    """Quantile classification."""
    n = len(values)
    sv = sorted(values)
    return [sv[min(int(i * n / n_classes), n - 1)] for i in range(1, n_classes)]


def classify_standard_deviation(
    values: Sequence[float],
    *,
    interval: float = 1.0,
) -> list[float]:
    """Standard deviation classification.

    Returns breaks at mean ± k * interval * std for k = -2, -1, 0, 1, 2.
    """
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return [mean + k * interval * std for k in (-2, -1, 0, 1, 2)]


def classify_manual_breaks(
    values: Sequence[float],
    breaks: Sequence[float],
) -> list[int]:
    """Classify values using manually specified break values."""
    sb = sorted(breaks)
    labels = []
    for v in values:
        cls = 0
        for b in sb:
            if v > b:
                cls += 1
        labels.append(cls)
    return labels


def classify_geometric_interval(
    values: Sequence[float],
    n_classes: int = 5,
) -> list[float]:
    """Geometric interval classification."""
    lo = min(v for v in values if v > 0) if any(v > 0 for v in values) else 1.0
    hi = max(values)
    if lo >= hi:
        return [lo] * (n_classes - 1)
    ratio = (hi / lo) ** (1 / n_classes)
    return [lo * ratio ** i for i in range(1, n_classes)]


def classify_defined_interval(
    values: Sequence[float],
    interval: float,
) -> list[float]:
    """Defined interval classification with fixed step size."""
    lo = min(values)
    hi = max(values)
    breaks = []
    current = lo + interval
    while current < hi:
        breaks.append(current)
        current += interval
    return breaks


def classify_values(
    values: Sequence[float],
    breaks: Sequence[float],
) -> list[int]:
    """Assign class labels to values based on break points."""
    sb = sorted(breaks)
    labels = []
    for v in values:
        cls = 0
        for b in sb:
            if v > b:
                cls += 1
        labels.append(cls)
    return labels


# ---------------------------------------------------------------------------
# Colour ramps (854-859)
# ---------------------------------------------------------------------------

def _hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def sequential_color_ramp(
    n: int,
    *,
    start: tuple[int, int, int] = (255, 255, 229),
    end: tuple[int, int, int] = (0, 68, 27),
) -> list[str]:
    """Generate a sequential colour ramp of *n* hex colours."""
    if n <= 0:
        return []
    if n == 1:
        return [_hex(*end)]
    return [_hex(*_lerp_color(start, end, i / (n - 1))) for i in range(n)]


def diverging_color_ramp(
    n: int,
    *,
    low: tuple[int, int, int] = (69, 117, 180),
    mid: tuple[int, int, int] = (255, 255, 191),
    high: tuple[int, int, int] = (215, 48, 39),
) -> list[str]:
    """Generate a diverging colour ramp of *n* hex colours."""
    if n <= 0:
        return []
    if n == 1:
        return [_hex(*mid)]
    colors = []
    for i in range(n):
        t = i / (n - 1)
        if t <= 0.5:
            c = _lerp_color(low, mid, t * 2)
        else:
            c = _lerp_color(mid, high, (t - 0.5) * 2)
        colors.append(_hex(*c))
    return colors


def qualitative_color_palette(n: int) -> list[str]:
    """Generate a qualitative colour palette of *n* distinct colours."""
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        colors.append(_hex(int(r * 255), int(g * 255), int(b * 255)))
    return colors


def colorblind_safe_palette(n: int) -> list[str]:
    """Return a colour-blind-safe palette (up to 8 colours, cycling if more)."""
    CB_PALETTE = [
        "#332288", "#88CCEE", "#44AA99", "#117733",
        "#999933", "#DDCC77", "#CC6677", "#882255",
    ]
    return [CB_PALETTE[i % len(CB_PALETTE)] for i in range(n)]


def custom_color_ramp(
    stops: Sequence[tuple[float, tuple[int, int, int]]],
    n: int,
) -> list[str]:
    """Build a multi-stop colour ramp.

    *stops* is a sequence of (position, (r, g, b)) where position is 0-1.
    """
    if n <= 0 or not stops:
        return []
    sorted_stops = sorted(stops, key=lambda s: s[0])
    colors = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.5
        # Find bounding stops
        lo_stop = sorted_stops[0]
        hi_stop = sorted_stops[-1]
        for k in range(len(sorted_stops) - 1):
            if sorted_stops[k][0] <= t <= sorted_stops[k + 1][0]:
                lo_stop = sorted_stops[k]
                hi_stop = sorted_stops[k + 1]
                break
        span = hi_stop[0] - lo_stop[0]
        frac = (t - lo_stop[0]) / span if span > 0 else 0.5
        c = _lerp_color(lo_stop[1], hi_stop[1], frac)
        colors.append(_hex(*c))
    return colors


COLOR_RAMP_PRESETS: dict[str, list[tuple[int, int, int]]] = {
    "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    "plasma": [(13, 8, 135), (126, 3, 168), (204, 71, 120), (248, 149, 64), (240, 249, 33)],
    "inferno": [(0, 0, 4), (87, 16, 110), (188, 55, 84), (249, 142, 9), (252, 255, 164)],
    "cividis": [(0, 32, 77), (58, 76, 100), (119, 120, 113), (186, 168, 109), (253, 231, 37)],
    "blues": [(247, 251, 255), (107, 174, 214), (8, 48, 107)],
    "reds": [(255, 245, 240), (251, 106, 74), (103, 0, 13)],
    "greens": [(247, 252, 245), (116, 196, 118), (0, 68, 27)],
}


def get_color_ramp_preset(name: str, n: int) -> list[str]:
    """Return *n* colours from a named preset ramp."""
    stops_rgb = COLOR_RAMP_PRESETS.get(name)
    if stops_rgb is None:
        raise ValueError(f"unknown preset: {name}. Available: {list(COLOR_RAMP_PRESETS)}")
    stops = [(i / (len(stops_rgb) - 1), c) for i, c in enumerate(stops_rgb)]
    return custom_color_ramp(stops, n)


# ---------------------------------------------------------------------------
# Report generation helpers (881-886, 891-893)
# ---------------------------------------------------------------------------

def generate_report_markdown(
    title: str,
    sections: Sequence[dict[str, Any]],
    *,
    author: str | None = None,
) -> str:
    """Generate a Markdown report from sections.

    Each section: {'heading': str, 'body': str, 'table'?: list[dict]}.
    """
    lines = [f"# {title}", ""]
    if author:
        lines.append(f"*Author: {author}*\n")
    for sec in sections:
        lines.append(f"## {sec['heading']}")
        lines.append("")
        if "body" in sec:
            lines.append(sec["body"])
            lines.append("")
        if "table" in sec and sec["table"]:
            headers = list(sec["table"][0].keys())
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in sec["table"]:
                lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
            lines.append("")
    return "\n".join(lines)


def generate_report_html(
    title: str,
    sections: Sequence[dict[str, Any]],
    *,
    author: str | None = None,
    css: str | None = None,
) -> str:
    """Generate an HTML report from sections."""
    default_css = "body{font-family:sans-serif;margin:2em;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ddd;padding:8px;text-align:left;} th{background:#f4f4f4;}"
    style = css or default_css
    parts = [f"<!DOCTYPE html><html><head><title>{title}</title><style>{style}</style></head><body>"]
    parts.append(f"<h1>{title}</h1>")
    if author:
        parts.append(f"<p><em>Author: {author}</em></p>")
    for sec in sections:
        parts.append(f"<h2>{sec['heading']}</h2>")
        if "body" in sec:
            parts.append(f"<p>{sec['body']}</p>")
        if "table" in sec and sec["table"]:
            headers = list(sec["table"][0].keys())
            parts.append("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>")
            for row in sec["table"]:
                parts.append("<tr>" + "".join(f"<td>{row.get(h, '')}</td>" for h in headers) + "</tr>")
            parts.append("</tbody></table>")
    parts.append("</body></html>")
    return "\n".join(parts)


def report_template(
    template_str: str,
    context: dict[str, Any],
) -> str:
    """Simple template engine for reports.

    Replaces {{key}} placeholders with values from *context*.
    """
    result = template_str
    for key, value in context.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


def geojson_preview_text(
    features: Sequence[dict[str, Any]],
    *,
    max_features: int = 10,
) -> str:
    """Produce a compact text preview of GeoJSON features for terminal display."""
    lines = [f"GeoJSON Preview ({len(features)} features total)"]
    lines.append("-" * 50)
    for i, f in enumerate(features[:max_features]):
        geom = f.get("geometry", {})
        props = f.get("properties", {})
        gtype = geom.get("type", "?")
        coords = geom.get("coordinates", [])
        if gtype == "Point" and coords:
            loc = f"({coords[0]:.4f}, {coords[1]:.4f})"
        else:
            loc = f"{gtype}"
        prop_str = ", ".join(f"{k}={v}" for k, v in list(props.items())[:5])
        lines.append(f"  [{i}] {loc} | {prop_str}")
    if len(features) > max_features:
        lines.append(f"  ... and {len(features) - max_features} more")
    return "\n".join(lines)


def wkt_preview_text(
    wkt_strings: Sequence[str],
    *,
    max_items: int = 10,
    max_length: int = 80,
) -> str:
    """Produce a compact text preview of WKT geometries."""
    lines = [f"WKT Preview ({len(wkt_strings)} geometries)"]
    lines.append("-" * 50)
    for i, w in enumerate(wkt_strings[:max_items]):
        display = w[:max_length] + "..." if len(w) > max_length else w
        lines.append(f"  [{i}] {display}")
    if len(wkt_strings) > max_items:
        lines.append(f"  ... and {len(wkt_strings) - max_items} more")
    return "\n".join(lines)


def executive_summary(
    stats: dict[str, Any],
    *,
    title: str = "Executive Summary",
) -> str:
    """Auto-generate an executive summary paragraph from statistics dict."""
    lines = [f"**{title}**\n"]
    if "total_features" in stats:
        lines.append(f"Dataset contains {stats['total_features']} features.")
    if "geometry_types" in stats:
        lines.append(f"Geometry types: {', '.join(str(t) for t in stats['geometry_types'])}.")
    if "field_count" in stats:
        lines.append(f"Number of attribute fields: {stats['field_count']}.")
    if "bbox" in stats:
        lines.append(f"Bounding box: {stats['bbox']}.")
    if "crs" in stats:
        lines.append(f"Coordinate Reference System: {stats['crs']}.")
    for k, v in stats.items():
        if k not in {"total_features", "geometry_types", "field_count", "bbox", "crs"}:
            lines.append(f"{k.replace('_', ' ').title()}: {v}.")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# G24 additions — classification aliases
# ---------------------------------------------------------------------------

# Alias: classify_jenks is the Jenks/Natural Breaks classifier
classify_jenks = classify_natural_breaks  # type: ignore[name-defined]

# Alias: classify_std_dev matches common short-form naming
classify_std_dev = classify_standard_deviation  # type: ignore[name-defined]
