"""Cartography, mapping, and reporting utilities for GeoPrompt.

Provides layer styling, map layouts, atlas/series generation, chart helpers,
annotation, and export utilities. All heavy dependencies (matplotlib, folium,
plotly) are lazily imported and optional.
"""
from __future__ import annotations

import html as _html
import json as _json
import math as _math
from pathlib import Path
from typing import Any, Sequence


Record = dict[str, Any]


# ---------------------------------------------------------------------------
# A. Layer Style / Template System
# ---------------------------------------------------------------------------


class LayerStyle:
    """Reusable map layer style definition."""

    def __init__(
        self,
        *,
        fill_color: str = "#3388ff",
        stroke_color: str = "#333333",
        stroke_width: float = 1.0,
        opacity: float = 0.7,
        radius: float = 5.0,
        label_field: str | None = None,
        label_font_size: int = 10,
    ) -> None:
        self.fill_color = fill_color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.opacity = opacity
        self.radius = radius
        self.label_field = label_field
        self.label_font_size = label_font_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "fill_color": self.fill_color,
            "stroke_color": self.stroke_color,
            "stroke_width": self.stroke_width,
            "opacity": self.opacity,
            "radius": self.radius,
            "label_field": self.label_field,
            "label_font_size": self.label_font_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LayerStyle":
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})


def save_style_template(style: LayerStyle, path: str | Path) -> str:
    """Save a LayerStyle to a JSON template file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps(style.to_dict(), indent=2))
    return str(p)


def load_style_template(path: str | Path) -> LayerStyle:
    """Load a LayerStyle from a JSON template file."""
    return LayerStyle.from_dict(_json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# B. Renderer Factories
# ---------------------------------------------------------------------------


def graduated_symbol_renderer(
    records: Sequence[Record],
    value_field: str,
    *,
    n_classes: int = 5,
    min_size: float = 3.0,
    max_size: float = 20.0,
    color_ramp: Sequence[str] | None = None,
    geometry_field: str = "geometry",
) -> list[dict[str, Any]]:
    """Assign graduated symbol sizes/colors to features based on a numeric field.

    Returns feature dicts augmented with ``"_symbol_size"`` and ``"_symbol_color"``.
    """
    vals = [float(r.get(value_field, 0)) for r in records]
    lo, hi = (min(vals), max(vals)) if vals else (0, 1)
    rng = hi - lo if hi > lo else 1.0

    colors = list(color_ramp) if color_ramp else [
        "#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026",
    ]
    while len(colors) < n_classes:
        colors.append(colors[-1])

    result: list[dict[str, Any]] = []
    for rec in records:
        v = float(rec.get(value_field, 0))
        norm = (v - lo) / rng
        cls = min(int(norm * n_classes), n_classes - 1)
        size = min_size + norm * (max_size - min_size)
        result.append({**rec, "_symbol_size": size, "_symbol_color": colors[cls], "_class": cls})
    return result


def unique_value_renderer(
    records: Sequence[Record],
    category_field: str,
    *,
    color_map: dict[str, str] | None = None,
    default_color: str = "#cccccc",
    geometry_field: str = "geometry",
) -> list[dict[str, Any]]:
    """Assign colors to features based on categorical values."""
    if color_map is None:
        palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
                    "#ffff33", "#a65628", "#f781bf", "#999999"]
        uniq = sorted(set(str(r.get(category_field, "")) for r in records))
        color_map = {v: palette[i % len(palette)] for i, v in enumerate(uniq)}

    return [{**rec, "_symbol_color": color_map.get(str(rec.get(category_field, "")), default_color)}
            for rec in records]


def heatmap_renderer(
    points: Sequence[tuple[float, float]],
    *,
    weights: Sequence[float] | None = None,
    radius: float = 25.0,
    blur: float = 15.0,
    max_intensity: float | None = None,
) -> dict[str, Any]:
    """Prepare heatmap rendering parameters for a point layer.

    Returns a specification dict compatible with Leaflet/Folium heatmap plugins.
    """
    w = list(weights) if weights else [1.0] * len(points)
    data = [[p[0], p[1], w[i]] for i, p in enumerate(points)]
    return {
        "type": "heatmap",
        "data": data,
        "radius": radius,
        "blur": blur,
        "max_intensity": max_intensity or max(w) if w else 1.0,
    }


def rule_based_renderer(
    records: Sequence[Record],
    rules: Sequence[tuple[str, str, LayerStyle]],
    *,
    default_style: LayerStyle | None = None,
) -> list[dict[str, Any]]:
    """Apply rule-based styling to features.

    Each rule is ``(field_name, operator_value, style)`` where operator_value
    is like ``">100"``, ``"==A"``, ``"contains road"``.
    """
    ds = default_style or LayerStyle()

    result: list[dict[str, Any]] = []
    for rec in records:
        matched = False
        for field_name, condition, style in rules:
            val = rec.get(field_name)
            if _eval_rule(val, condition):
                result.append({**rec, **{f"_style_{k}": v for k, v in style.to_dict().items()}})
                matched = True
                break
        if not matched:
            result.append({**rec, **{f"_style_{k}": v for k, v in ds.to_dict().items()}})
    return result


def _eval_rule(val: Any, condition: str) -> bool:
    """Evaluate a simple rule condition against a value."""
    if val is None:
        return False
    s = str(val)
    if condition.startswith(">="):
        try:
            return float(val) >= float(condition[2:])
        except (ValueError, TypeError):
            return False
    if condition.startswith("<="):
        try:
            return float(val) <= float(condition[2:])
        except (ValueError, TypeError):
            return False
    if condition.startswith(">"):
        try:
            return float(val) > float(condition[1:])
        except (ValueError, TypeError):
            return False
    if condition.startswith("<"):
        try:
            return float(val) < float(condition[1:])
        except (ValueError, TypeError):
            return False
    if condition.startswith("=="):
        return s == condition[2:]
    if condition.startswith("!="):
        return s != condition[2:]
    if condition.startswith("contains "):
        return condition[9:] in s
    return s == condition


# ---------------------------------------------------------------------------
# C. Labeling Engine
# ---------------------------------------------------------------------------


def label_features(
    records: Sequence[Record],
    label_field: str,
    *,
    font_size: int = 10,
    font_color: str = "#000000",
    halo_color: str = "#ffffff",
    halo_width: float = 1.0,
    placement: str = "centroid",
    max_labels: int | None = None,
    priority_field: str | None = None,
    geometry_field: str = "geometry",
) -> list[dict[str, Any]]:
    """Generate label specifications for features with collision avoidance.

    Returns label dicts with ``"text"``, ``"x"``, ``"y"``, ``"style"`` keys.
    """
    from .geometry import geometry_centroid

    labeled: list[dict[str, Any]] = []
    placed_boxes: list[tuple[float, float, float, float]] = []

    items = list(records)
    if priority_field:
        items.sort(key=lambda r: float(r.get(priority_field, 0)), reverse=True)

    for rec in items:
        if max_labels and len(labeled) >= max_labels:
            break

        text = str(rec.get(label_field, ""))
        if not text:
            continue

        geom = rec.get(geometry_field)
        if not geom:
            continue

        c = geometry_centroid(geom)
        x, y = c["x"], c["y"]

        # Simple collision: check overlap with existing labels
        approx_w = len(text) * font_size * 0.6
        approx_h = font_size * 1.4
        box = (x - approx_w / 2, y - approx_h / 2, x + approx_w / 2, y + approx_h / 2)

        collision = any(
            not (box[2] < pb[0] or box[0] > pb[2] or box[3] < pb[1] or box[1] > pb[3])
            for pb in placed_boxes
        )
        if collision:
            continue

        placed_boxes.append(box)
        labeled.append({
            "text": text,
            "x": x,
            "y": y,
            "style": {
                "font_size": font_size,
                "font_color": font_color,
                "halo_color": halo_color,
                "halo_width": halo_width,
            },
        })

    return labeled


# ---------------------------------------------------------------------------
# D. Map Series / Atlas Generation
# ---------------------------------------------------------------------------


def map_series(
    features: Sequence[Record],
    group_field: str,
    *,
    title_template: str = "Map: {value}",
    style: LayerStyle | None = None,
    geometry_field: str = "geometry",
) -> list[dict[str, Any]]:
    """Generate a map series (atlas) — one page specification per group value.

    Returns a list of page dicts with ``"title"``, ``"features"``, ``"bounds"``.
    """
    from .geometry import geometry_bounds

    groups: dict[str, list[Record]] = {}
    for rec in features:
        key = str(rec.get(group_field, "default"))
        groups.setdefault(key, []).append(rec)

    pages: list[dict[str, Any]] = []
    for value, recs in sorted(groups.items()):
        all_bounds: list[tuple[float, float, float, float]] = []
        for r in recs:
            geom = r.get(geometry_field)
            if geom:
                min_x, min_y, max_x, max_y = geometry_bounds(geom)
                all_bounds.append((min_x, min_y, max_x, max_y))

        if all_bounds:
            extent = (
                min(b[0] for b in all_bounds),
                min(b[1] for b in all_bounds),
                max(b[2] for b in all_bounds),
                max(b[3] for b in all_bounds),
            )
        else:
            extent = (0, 0, 1, 1)

        pages.append({
            "title": title_template.format(value=value),
            "group_value": value,
            "features": recs,
            "bounds": extent,
            "feature_count": len(recs),
            "style": (style or LayerStyle()).to_dict(),
        })
    return pages


# ---------------------------------------------------------------------------
# E. Print Layout
# ---------------------------------------------------------------------------


def print_layout(
    *,
    title: str = "Map",
    subtitle: str = "",
    map_bounds: tuple[float, float, float, float] | None = None,
    legend_items: Sequence[dict[str, str]] | None = None,
    north_arrow: bool = True,
    scale_bar: bool = True,
    logo_path: str | None = None,
    page_size: str = "letter",
) -> dict[str, Any]:
    """Define a print-ready map layout specification.

    Returns a layout dict that downstream renderers can consume.
    """
    sizes = {"letter": (8.5, 11), "a4": (8.27, 11.69), "tabloid": (11, 17)}
    w, h = sizes.get(page_size.lower(), (8.5, 11))

    return {
        "title": title,
        "subtitle": subtitle,
        "page_width_in": w,
        "page_height_in": h,
        "map_bounds": map_bounds,
        "legend_items": list(legend_items or []),
        "north_arrow": north_arrow,
        "scale_bar": scale_bar,
        "logo_path": logo_path,
    }


# ---------------------------------------------------------------------------
# F. Interactive Dashboard
# ---------------------------------------------------------------------------


def interactive_dashboard(
    sections: Sequence[dict[str, Any]],
    *,
    title: str = "GeoPrompt Dashboard",
) -> str:
    """Generate an interactive HTML dashboard with linked sections.

    Each section dict has ``"type"`` (``"map"``, ``"chart"``, ``"table"``,
    ``"metric"``), ``"title"``, and ``"data"``.
    """
    parts: list[str] = [
        "<!DOCTYPE html><html><head>",
        f"<title>{_html.escape(title)}</title>",
        "<style>",
        "body{font-family:sans-serif;margin:20px}",
        ".section{border:1px solid #ccc;padding:15px;margin:10px 0;border-radius:4px}",
        ".metric{font-size:2em;font-weight:bold;color:#2c3e50}",
        "table{border-collapse:collapse;width:100%}",
        "th,td{border:1px solid #ddd;padding:8px;text-align:left}",
        "th{background:#f2f2f2}",
        "</style></head><body>",
        f"<h1>{_html.escape(title)}</h1>",
    ]

    for sec in sections:
        sec_type = sec.get("type", "text")
        sec_title = sec.get("title", "")
        data = sec.get("data")

        parts.append(f'<div class="section"><h2>{_html.escape(sec_title)}</h2>')

        if sec_type == "metric":
            parts.append(f'<div class="metric">{_html.escape(str(data))}</div>')
        elif sec_type == "table" and isinstance(data, list):
            parts.append(_html_table(data))
        elif sec_type == "chart" and isinstance(data, dict):
            parts.append(f"<pre>{_html.escape(_json.dumps(data, indent=2))}</pre>")
        else:
            parts.append(f"<p>{_html.escape(str(data))}</p>")

        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def _html_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "<p>No data</p>"
    cols = list(records[0].keys())
    rows = ["<table><thead><tr>" + "".join(f"<th>{_html.escape(str(c))}</th>" for c in cols) + "</tr></thead><tbody>"]
    for rec in records:
        rows.append("<tr>" + "".join(f"<td>{_html.escape(str(rec.get(c, '')))}</td>" for c in cols) + "</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# G. Story Map HTML
# ---------------------------------------------------------------------------


def story_map_html(
    slides: Sequence[dict[str, Any]],
    *,
    title: str = "Story Map",
) -> str:
    """Generate a scrolling story-map HTML page.

    Each slide dict has ``"title"``, ``"text"``, optional ``"image_url"``
    and ``"map_config"`` keys.
    """
    parts: list[str] = [
        "<!DOCTYPE html><html><head>",
        f"<title>{_html.escape(title)}</title>",
        "<style>",
        "body{font-family:Georgia,serif;margin:0;padding:0}",
        ".slide{min-height:100vh;padding:60px 40px;box-sizing:border-box;border-bottom:2px solid #eee}",
        ".slide:nth-child(even){background:#f9f9f9}",
        "h1{text-align:center;padding:40px}",
        "h2{color:#2c3e50}",
        "img{max-width:100%;border-radius:8px;margin:10px 0}",
        "</style></head><body>",
        f"<h1>{_html.escape(title)}</h1>",
    ]

    for slide in slides:
        parts.append('<div class="slide">')
        parts.append(f"<h2>{_html.escape(slide.get('title', ''))}</h2>")
        if slide.get("image_url"):
            parts.append(f'<img src="{_html.escape(slide["image_url"])}" alt="">')
        parts.append(f"<p>{_html.escape(slide.get('text', ''))}</p>")
        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# H. Bookmark / Extent Manager
# ---------------------------------------------------------------------------


class BookmarkManager:
    """Manage named spatial bookmarks for repeated map production."""

    def __init__(self) -> None:
        self._bookmarks: dict[str, tuple[float, float, float, float]] = {}

    def add(self, name: str, bounds: tuple[float, float, float, float]) -> None:
        self._bookmarks[name] = bounds

    def remove(self, name: str) -> None:
        self._bookmarks.pop(name, None)

    def get(self, name: str) -> tuple[float, float, float, float] | None:
        return self._bookmarks.get(name)

    def list_bookmarks(self) -> list[str]:
        return sorted(self._bookmarks.keys())

    def to_dict(self) -> dict[str, tuple[float, float, float, float]]:
        return dict(self._bookmarks)

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps({k: list(v) for k, v in self._bookmarks.items()}, indent=2))
        return str(p)

    @classmethod
    def load(cls, path: str | Path) -> "BookmarkManager":
        bm = cls()
        data = _json.loads(Path(path).read_text())
        for k, v in data.items():
            bm.add(k, tuple(v))  # type: ignore[arg-type]
        return bm


# ---------------------------------------------------------------------------
# I. Annotation / Callout Generation
# ---------------------------------------------------------------------------


def annotation_callouts(
    features: Sequence[Record],
    text_field: str,
    *,
    geometry_field: str = "geometry",
    offset_x: float = 15.0,
    offset_y: float = -10.0,
    style: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate annotation callouts for features.

    Returns dicts with ``"text"``, ``"anchor_x"``, ``"anchor_y"``,
    ``"label_x"``, ``"label_y"``, ``"style"``.
    """
    from .geometry import geometry_centroid

    default_style = {
        "font_size": 9,
        "font_color": "#333",
        "line_color": "#666",
        "background": "#fff",
        **(style or {}),
    }

    annotations: list[dict[str, Any]] = []
    for rec in features:
        geom = rec.get(geometry_field)
        if not geom:
            continue
        c = geometry_centroid(geom)
        annotations.append({
            "text": str(rec.get(text_field, "")),
            "anchor_x": c["x"],
            "anchor_y": c["y"],
            "label_x": c["x"] + offset_x,
            "label_y": c["y"] + offset_y,
            "style": default_style,
        })
    return annotations


# ---------------------------------------------------------------------------
# J. Chart Suite
# ---------------------------------------------------------------------------


def chart_histogram(
    values: Sequence[float],
    *,
    bins: int = 20,
    title: str = "Histogram",
    x_label: str = "Value",
    y_label: str = "Frequency",
) -> dict[str, Any]:
    """Build a histogram chart specification."""
    if not values:
        return {"type": "histogram", "title": title, "bins": [], "counts": []}
    lo, hi = min(values), max(values)
    if lo == hi:
        return {"type": "histogram", "title": title, "bins": [lo], "counts": [len(values)]}

    step = (hi - lo) / bins
    bin_edges = [lo + i * step for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1

    return {
        "type": "histogram",
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "bins": bin_edges,
        "counts": counts,
    }


def chart_scatter(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    title: str = "Scatter Plot",
    x_label: str = "X",
    y_label: str = "Y",
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a scatter plot chart specification."""
    return {
        "type": "scatter",
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "x": list(x_values),
        "y": list(y_values),
        "labels": list(labels) if labels else None,
    }


def chart_trend_line(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    title: str = "Trend",
    order: int = 1,
) -> dict[str, Any]:
    """Fit a trend line and return chart spec with coefficients."""
    n = len(x_values)
    if n < 2:
        return {"type": "trend", "title": title, "coefficients": [], "r_squared": 0.0}

    # Simple linear regression
    mx = sum(x_values) / n
    my = sum(y_values) / n
    ss_xx = sum((x - mx) ** 2 for x in x_values)
    ss_xy = sum((x - mx) * (y - my) for x, y in zip(x_values, y_values))
    if ss_xx == 0:
        return {"type": "trend", "title": title, "coefficients": [my, 0.0], "r_squared": 0.0}

    slope = ss_xy / ss_xx
    intercept = my - slope * mx
    predicted = [intercept + slope * x for x in x_values]
    ss_res = sum((y - p) ** 2 for y, p in zip(y_values, predicted))
    ss_tot = sum((y - my) ** 2 for y in y_values)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "type": "trend",
        "title": title,
        "coefficients": [intercept, slope],
        "r_squared": r_sq,
        "x": list(x_values),
        "y_actual": list(y_values),
        "y_predicted": predicted,
    }


def chart_distribution(
    values: Sequence[float],
    *,
    title: str = "Distribution",
) -> dict[str, Any]:
    """Compute distribution statistics and build a chart spec."""
    n = len(values)
    if n == 0:
        return {"type": "distribution", "title": title, "stats": {}}

    sorted_v = sorted(values)
    mean = sum(values) / n
    median = sorted_v[n // 2] if n % 2 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    q1 = sorted_v[n // 4]
    q3 = sorted_v[3 * n // 4]
    std = _math.sqrt(sum((v - mean) ** 2 for v in values) / n)

    return {
        "type": "distribution",
        "title": title,
        "stats": {
            "mean": mean,
            "median": median,
            "std": std,
            "min": sorted_v[0],
            "max": sorted_v[-1],
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "count": n,
        },
    }


# ---------------------------------------------------------------------------
# K. Export Helpers
# ---------------------------------------------------------------------------


def export_map_image(
    features: Sequence[Record],
    output_path: str | Path,
    *,
    format: str = "png",
    width: int = 800,
    height: int = 600,
    style: LayerStyle | None = None,
    title: str = "",
    geometry_field: str = "geometry",
) -> str:
    """Export features as a static map image (PNG, SVG, or PDF).

    Uses matplotlib if available, otherwise generates a minimal SVG.
    """
    from .geometry import geometry_centroid

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    st = style or LayerStyle()

    if format.lower() == "svg" or format.lower() == "html":
        return _export_svg_map(features, p, width, height, st, title, geometry_field)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100))
        for rec in features:
            geom = rec.get(geometry_field)
            if not geom:
                continue
            c = geometry_centroid(geom)
            ax.plot(c["x"], c["y"], "o", color=st.fill_color,
                    markersize=st.radius, alpha=st.opacity)
        if title:
            ax.set_title(title)
        ax.set_aspect("equal")
        fig.savefig(str(p), format=format, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(p)
    except ImportError:
        return _export_svg_map(features, p.with_suffix(".svg"), width, height, st, title, geometry_field)


def _export_svg_map(
    features: Sequence[Record],
    path: Path,
    w: int,
    h: int,
    style: LayerStyle,
    title: str,
    geometry_field: str,
) -> str:
    from .geometry import geometry_centroid, geometry_bounds

    all_bounds: list[tuple[float, float, float, float]] = []
    for rec in features:
        geom = rec.get(geometry_field)
        if geom:
            b = geometry_bounds(geom)
            all_bounds.append((b["min_x"], b["min_y"], b["max_x"], b["max_y"]))

    if all_bounds:
        ext = (min(b[0] for b in all_bounds), min(b[1] for b in all_bounds),
               max(b[2] for b in all_bounds), max(b[3] for b in all_bounds))
    else:
        ext = (0, 0, 1, 1)

    sx = w / max(ext[2] - ext[0], 1e-12)
    sy = h / max(ext[3] - ext[1], 1e-12)
    s = min(sx, sy)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    if title:
        parts.append(f'<text x="10" y="20" font-size="14">{_html.escape(title)}</text>')

    for rec in features:
        geom = rec.get(geometry_field)
        if not geom:
            continue
        c = geometry_centroid(geom)
        px = (c["x"] - ext[0]) * s
        py = h - (c["y"] - ext[1]) * s
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{style.radius}" '
            f'fill="{style.fill_color}" opacity="{style.opacity}"/>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return str(path)


# ---------------------------------------------------------------------------
# L. Comparison View
# ---------------------------------------------------------------------------


def comparison_view(
    scenarios: Sequence[dict[str, Any]],
    *,
    title: str = "Scenario Comparison",
    metric_fields: Sequence[str] | None = None,
) -> str:
    """Generate an HTML comparison view for multiple scenarios.

    Each scenario dict should have ``"name"`` and metric fields.
    """
    if not scenarios:
        return "<p>No scenarios</p>"

    if metric_fields is None:
        metric_fields = [k for k in scenarios[0] if isinstance(scenarios[0].get(k), (int, float))]

    rows: list[str] = []
    rows.append(f"<h2>{_html.escape(title)}</h2>")
    rows.append("<table><thead><tr><th>Metric</th>")
    for sc in scenarios:
        rows.append(f"<th>{_html.escape(str(sc.get('name', 'Scenario')))}</th>")
    rows.append("</tr></thead><tbody>")

    for field in metric_fields:
        rows.append(f"<tr><td>{_html.escape(field)}</td>")
        vals = [float(sc.get(field, 0)) for sc in scenarios]
        best = max(vals) if vals else 0
        for v in vals:
            cls = ' style="background:#d4edda;font-weight:bold"' if v == best and len(vals) > 1 else ""
            rows.append(f"<td{cls}>{v:.2f}</td>")
        rows.append("</tr>")

    rows.append("</tbody></table>")
    return "\n".join(rows)


__all__ = [
    "BookmarkManager",
    "LayerStyle",
    "annotation_callouts",
    "chart_distribution",
    "chart_histogram",
    "chart_scatter",
    "chart_trend_line",
    "comparison_view",
    "export_map_image",
    "graduated_symbol_renderer",
    "heatmap_renderer",
    "interactive_dashboard",
    "label_features",
    "load_style_template",
    "map_series",
    "print_layout",
    "rule_based_renderer",
    "save_style_template",
    "story_map_html",
    "unique_value_renderer",
]
