"""Visualization module for interactive maps and static plots.

Provides helpers for creating Folium maps, choropleths, and quick static
plots from GeoPromptFrame data.  All mapping dependencies (folium, branca,
matplotlib) are optional and lazily imported.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any, Sequence

from .geometry import Geometry, geometry_bounds, geometry_type
from .tools import compare_scenarios


Record = dict[str, Any]


def _load_folium() -> Any:
    try:
        return importlib.import_module("folium")
    except ImportError as exc:
        raise RuntimeError(
            "Install visualization support with 'pip install folium' before using map functions."
        ) from exc


def _load_plotly() -> Any:
    try:
        return importlib.import_module("plotly.graph_objects")
    except ImportError as exc:
        raise RuntimeError(
            "Install visualization support with 'pip install geoprompt[viz]' to use Plotly dashboards."
        ) from exc


def _geojson_coords(geometry: Geometry) -> Any:
    """Convert internal geometry dict to GeoJSON-compatible structure."""
    gtype = str(geometry.get("type", ""))
    coords = geometry.get("coordinates")
    if gtype == "Point":
        return {"type": "Point", "coordinates": list(coords)}
    if gtype == "LineString":
        return {"type": "LineString", "coordinates": [list(c) for c in coords]}
    if gtype == "Polygon":
        return {"type": "Polygon", "coordinates": [[list(c) for c in coords]]}
    if gtype == "MultiPoint":
        return {"type": "MultiPoint", "coordinates": [list(c) for c in coords]}
    if gtype == "MultiLineString":
        return {"type": "MultiLineString", "coordinates": [[list(c) for c in line] for line in coords]}
    if gtype == "MultiPolygon":
        return {"type": "MultiPolygon", "coordinates": [[[list(c) for c in poly]] for poly in coords]}
    return {"type": gtype, "coordinates": coords}


def _frame_bounds(rows: Sequence[Record], geometry_column: str) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) across all rows."""
    xs, ys = [], []
    for row in rows:
        min_x, min_y, max_x, max_y = geometry_bounds(row[geometry_column])
        xs.extend([min_x, max_x])
        ys.extend([min_y, max_y])
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def to_folium_map(
    frame: Any,
    *,
    color: str = "blue",
    fill_color: str | None = None,
    weight: float = 2,
    fill_opacity: float = 0.4,
    tooltip_columns: Sequence[str] | None = None,
    popup_columns: Sequence[str] | None = None,
    style_column: str | None = None,
    style_map: dict[Any, dict[str, Any]] | None = None,
    tiles: str = "OpenStreetMap",
    custom_tile_url: str | None = None,
    cluster_points: bool = False,
    fullscreen: bool = False,
    minimap: bool = False,
    measure_control: bool = False,
    zoom_start: int | None = None,
    width: str | int = "100%",
    height: str | int = 600,
    add_layer_control: bool = True,
) -> Any:
    """Create a Folium map from a GeoPromptFrame.

    Args:
        frame: A :class:`~geoprompt.frame.GeoPromptFrame`.
        color: Stroke color for features.
        fill_color: Fill color; defaults to ``color`` for polygons.
        weight: Stroke width.
        fill_opacity: Polygon fill opacity (0-1).
        tooltip_columns: Columns to show on hover.
        popup_columns: Columns to show on click.
        style_column: Column used to pick per-feature style from ``style_map``.
        style_map: Dict mapping style_column values to folium style kwargs.
        tiles: Tile layer name.
        custom_tile_url: Optional XYZ tile template URL.
        cluster_points: If ``True``, cluster point markers.
        fullscreen: If ``True``, add a fullscreen map control.
        minimap: If ``True``, add a mini-map overview control.
        measure_control: If ``True``, add a line/area measurement control.
        zoom_start: Initial zoom level; auto-fit if ``None``.
        width: Map width (CSS value or pixels).
        height: Map height (CSS value or pixels).

    Returns:
        A ``folium.Map`` object.
    """
    folium = _load_folium()
    rows = list(frame)
    geometry_column = frame.geometry_column

    if not rows:
        return folium.Map(location=[0, 0], zoom_start=2, tiles=tiles, width=width, height=height)

    min_x, min_y, max_x, max_y = _frame_bounds(rows, geometry_column)
    center_lat = (min_y + max_y) / 2
    center_lon = (min_x + max_x) / 2

    base_tiles = None if custom_tile_url else tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        tiles=base_tiles,
        width=width,
        height=height,
    )

    if custom_tile_url:
        folium.TileLayer(tiles=custom_tile_url, attr="Custom tiles", name="basemap").add_to(m)

    if fullscreen or minimap or measure_control:
        try:
            plugins = importlib.import_module("folium.plugins")
            if fullscreen:
                plugins.Fullscreen().add_to(m)
            if minimap:
                plugins.MiniMap(toggle_display=True).add_to(m)
            if measure_control:
                plugins.MeasureControl().add_to(m)
        except ImportError:
            pass

    if zoom_start is not None:
        m.zoom_start = zoom_start
    else:
        m.fit_bounds([[min_y, min_x], [max_y, max_x]])

    feature_group = folium.FeatureGroup(name="features")
    point_group: Any = feature_group
    if cluster_points:
        try:
            plugins = importlib.import_module("folium.plugins")
            point_group = plugins.MarkerCluster(name="points")
        except ImportError:
            point_group = feature_group

    for row in rows:
        geom = row[geometry_column]
        gtype = geometry_type(geom)
        feat_color = color
        feat_fill = fill_color or color
        feat_weight = weight
        feat_fill_opacity = fill_opacity

        if style_column and style_map and row.get(style_column) in style_map:
            style = style_map[row[style_column]]
            feat_color = style.get("color", feat_color)
            feat_fill = style.get("fill_color", feat_fill)
            feat_weight = style.get("weight", feat_weight)
            feat_fill_opacity = style.get("fill_opacity", feat_fill_opacity)

        tooltip_text = None
        if tooltip_columns:
            parts = [f"<b>{c}</b>: {row.get(c, '')}" for c in tooltip_columns if c != geometry_column]
            tooltip_text = "<br>".join(parts) if parts else None

        popup_text = None
        if popup_columns:
            parts = [f"<b>{c}</b>: {row.get(c, '')}" for c in popup_columns if c != geometry_column]
            popup_text = "<br>".join(parts) if parts else None

        tooltip = folium.Tooltip(tooltip_text) if tooltip_text else None
        popup = folium.Popup(popup_text, max_width=300) if popup_text else None

        if gtype == "Point":
            coords = geom["coordinates"]
            folium.CircleMarker(
                location=[coords[1], coords[0]],
                radius=6,
                color=feat_color,
                fill=True,
                fill_color=feat_fill,
                fill_opacity=feat_fill_opacity,
                weight=feat_weight,
                tooltip=tooltip,
                popup=popup,
            ).add_to(point_group)
        else:
            geojson = _geojson_coords(geom)
            folium.GeoJson(
                {"type": "Feature", "geometry": geojson, "properties": {}},
                style_function=lambda _f, _c=feat_color, _fc=feat_fill, _w=feat_weight, _fo=feat_fill_opacity: {
                    "color": _c,
                    "fillColor": _fc,
                    "weight": _w,
                    "fillOpacity": _fo,
                },
                tooltip=tooltip,
                popup=popup,
            ).add_to(feature_group)

    if point_group is not feature_group:
        point_group.add_to(m)
    feature_group.add_to(m)
    if add_layer_control:
        folium.LayerControl().add_to(m)

    return m


def to_choropleth(
    frame: Any,
    value_column: str,
    *,
    id_column: str = "site_id",
    fill_color: str = "YlOrRd",
    fill_opacity: float = 0.7,
    line_opacity: float = 0.3,
    legend_name: str | None = None,
    tiles: str = "OpenStreetMap",
    width: str | int = "100%",
    height: str | int = 600,
) -> Any:
    """Create a choropleth map colored by a numeric column.

    Args:
        frame: A :class:`~geoprompt.frame.GeoPromptFrame`.
        value_column: Numeric column to use for coloring.
        id_column: Column to use as feature identifiers.
        fill_color: Color scale name (e.g. ``"YlOrRd"``, ``"BuGn"``).
        fill_opacity: Polygon fill opacity.
        line_opacity: Border line opacity.
        legend_name: Legend title; defaults to ``value_column``.
        tiles: Tile layer name.
        width: Map width.
        height: Map height.

    Returns:
        A ``folium.Map`` object with choropleth layer.
    """
    folium = _load_folium()
    rows = list(frame)
    geometry_column = frame.geometry_column

    features = []
    for row in rows:
        geojson_geom = _geojson_coords(row[geometry_column])
        features.append({
            "type": "Feature",
            "id": str(row.get(id_column, "")),
            "geometry": geojson_geom,
            "properties": {k: v for k, v in row.items() if k != geometry_column},
        })

    geojson_collection = {"type": "FeatureCollection", "features": features}

    min_x, min_y, max_x, max_y = _frame_bounds(rows, geometry_column)
    center = [(min_y + max_y) / 2, (min_x + max_x) / 2]

    m = folium.Map(location=center, tiles=tiles, width=width, height=height)
    m.fit_bounds([[min_y, min_x], [max_y, max_x]])

    data = {str(row.get(id_column, "")): float(row.get(value_column, 0)) for row in rows}

    try:
        import pandas as pd
        series = pd.Series(data, name=value_column)
    except ImportError:
        series = data

    folium.Choropleth(
        geo_data=geojson_collection,
        data=series if not isinstance(series, dict) else None,
        key_on="feature.id",
        fill_color=fill_color,
        fill_opacity=fill_opacity,
        line_opacity=line_opacity,
        legend_name=legend_name or value_column,
    ).add_to(m)

    return m


def save_map(folium_map: Any, output_path: str | Path) -> str:
    """Save a Folium map to an HTML file.

    Args:
        folium_map: A ``folium.Map`` object.
        output_path: File path.

    Returns:
        The resolved output path string.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    folium_map.save(str(path))
    return str(path)


def plot_scenario_dashboard(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    *,
    higher_is_better: Sequence[str] | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> Any:
    """Create a simple side-by-side scenario dashboard figure."""
    plt = importlib.import_module("matplotlib.pyplot")
    comparison = compare_scenarios(
        baseline_metrics,
        candidate_metrics,
        higher_is_better=higher_is_better,
    )
    metrics = list(comparison.keys())
    baseline = [float(comparison[m]["baseline"]) for m in metrics]
    candidate = [float(comparison[m]["candidate"]) for m in metrics]
    deltas = [float(comparison[m]["delta_percent"]) for m in metrics]
    colors = ["#2b8a3e" if comparison[m]["direction"] == "improved" else "#c92a2a" for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(metrics) * 1.2)))
    ax_left, ax_right = axes

    positions = list(range(len(metrics)))
    width = 0.38
    ax_left.barh([p - width / 2 for p in positions], baseline, height=width, label="baseline", color="#94a3b8")
    ax_left.barh([p + width / 2 for p in positions], candidate, height=width, label="candidate", color="#2563eb")
    ax_left.set_yticks(positions)
    ax_left.set_yticklabels(metrics)
    ax_left.set_title("Metric values")
    ax_left.legend()
    ax_left.grid(True, axis="x", alpha=0.25)

    ax_right.barh(positions, deltas, color=colors)
    ax_right.set_yticks(positions)
    ax_right.set_yticklabels(metrics)
    ax_right.axvline(0, color="#555", linewidth=1)
    ax_right.set_title("Percent change")
    ax_right.grid(True, axis="x", alpha=0.25)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), bbox_inches="tight", dpi=150)

    return fig


def plotly_scenario_dashboard(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    *,
    higher_is_better: Sequence[str] | None = None,
    title: str | None = None,
) -> Any:
    """Create an interactive Plotly scenario dashboard."""
    go = _load_plotly()
    comparison = compare_scenarios(
        baseline_metrics,
        candidate_metrics,
        higher_is_better=higher_is_better,
    )
    metrics = list(comparison.keys())
    baseline = [float(comparison[m]["baseline"]) for m in metrics]
    candidate = [float(comparison[m]["candidate"]) for m in metrics]
    deltas = [float(comparison[m]["delta_percent"]) for m in metrics]
    colors = ["#2b8a3e" if comparison[m]["direction"] == "improved" else "#c92a2a" for m in metrics]

    fig = go.Figure()
    fig.add_bar(name="baseline", y=metrics, x=baseline, orientation="h", marker_color="#94a3b8")
    fig.add_bar(name="candidate", y=metrics, x=candidate, orientation="h", marker_color="#2563eb")
    fig.add_scatter(
        name="delta %",
        y=metrics,
        x=deltas,
        mode="markers+text",
        marker=dict(color=colors, size=11),
        text=[f"{d:.1f}%" for d in deltas],
        textposition="middle right",
        xaxis="x2",
    )
    fig.update_layout(
        title=title or "Scenario Dashboard",
        barmode="group",
        template="plotly_white",
        xaxis=dict(title="metric value"),
        xaxis2=dict(title="percent change", overlaying="x", side="top", showgrid=False, zeroline=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=80, r=40, t=80, b=40),
    )
    return fig


def save_plotly_html(figure: Any, output_path: str | Path) -> str:
    """Save a Plotly figure to an HTML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs="cdn")
    return str(path)


# --- Resilience styling presets ---

RESILIENCE_STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "critical": {"color": "#c92a2a", "fill_color": "#ff6b6b", "weight": 3, "fill_opacity": 0.6},
    "high": {"color": "#e8590c", "fill_color": "#ff922b", "weight": 2.5, "fill_opacity": 0.5},
    "medium": {"color": "#e67700", "fill_color": "#fcc419", "weight": 2, "fill_opacity": 0.4},
    "low": {"color": "#2b8a3e", "fill_color": "#51cf66", "weight": 1.5, "fill_opacity": 0.3},
    "minimal": {"color": "#1971c2", "fill_color": "#4dabf7", "weight": 1, "fill_opacity": 0.2},
}


def resilience_style_map(tier_column: str = "risk_tier") -> tuple[str, dict[str, dict[str, Any]]]:
    """Return a ``(style_column, style_map)`` tuple for use with :func:`to_folium_map`.

    The returned mapping colours features by resilience tier using the
    built-in :data:`RESILIENCE_STYLE_PRESETS`.

    Args:
        tier_column: Column name containing tier labels.

    Returns:
        Tuple of ``(tier_column, style_map)`` ready for ``to_folium_map``.
    """
    return tier_column, dict(RESILIENCE_STYLE_PRESETS)


def to_outage_overlay_map(
    frame: Any,
    *,
    outage_column: str = "is_outage",
    restored_column: str | None = None,
    tooltip_columns: Sequence[str] | None = None,
    tiles: str = "OpenStreetMap",
    width: str | int = "100%",
    height: str | int = 600,
) -> Any:
    """Create a Folium map with outage / restored overlay colouring.

    Features where *outage_column* is truthy are shown in red.
    Features where *restored_column* is truthy are shown in green.
    All other features appear grey.

    Args:
        frame: A :class:`~geoprompt.frame.GeoPromptFrame`.
        outage_column: Boolean column indicating an outage.
        restored_column: Optional boolean column for restored features.
        tooltip_columns: Columns to show on hover.
        tiles: Tile layer name.
        width: Map width.
        height: Map height.

    Returns:
        A ``folium.Map`` with outage/restored colour coding.
    """
    folium = _load_folium()
    rows = list(frame)
    geometry_column = frame.geometry_column

    if not rows:
        return folium.Map(location=[0, 0], zoom_start=2, tiles=tiles, width=width, height=height)

    min_x, min_y, max_x, max_y = _frame_bounds(rows, geometry_column)
    center = [(min_y + max_y) / 2, (min_x + max_x) / 2]
    m = folium.Map(location=center, tiles=tiles, width=width, height=height)
    m.fit_bounds([[min_y, min_x], [max_y, max_x]])

    outage_group = folium.FeatureGroup(name="Outage")
    restored_group = folium.FeatureGroup(name="Restored")
    normal_group = folium.FeatureGroup(name="Normal")

    for row in rows:
        geom = row[geometry_column]
        gtype = geometry_type(geom)

        is_outage = bool(row.get(outage_column))
        is_restored = bool(row.get(restored_column)) if restored_column else False

        if is_outage and not is_restored:
            feat_color, feat_fill, opacity, target = "#c92a2a", "#ff6b6b", 0.6, outage_group
        elif is_restored:
            feat_color, feat_fill, opacity, target = "#2b8a3e", "#51cf66", 0.5, restored_group
        else:
            feat_color, feat_fill, opacity, target = "#868e96", "#adb5bd", 0.3, normal_group

        tooltip = None
        if tooltip_columns:
            parts = [f"<b>{c}</b>: {row.get(c, '')}" for c in tooltip_columns if c != geometry_column]
            tooltip = folium.Tooltip("<br>".join(parts)) if parts else None

        if gtype == "Point":
            coords = geom["coordinates"]
            folium.CircleMarker(
                location=[coords[1], coords[0]],
                radius=6,
                color=feat_color,
                fill=True,
                fill_color=feat_fill,
                fill_opacity=opacity,
                tooltip=tooltip,
            ).add_to(target)
        else:
            geojson = _geojson_coords(geom)
            folium.GeoJson(
                {"type": "Feature", "geometry": geojson, "properties": {}},
                style_function=lambda _f, _c=feat_color, _fc=feat_fill, _o=opacity: {
                    "color": _c,
                    "fillColor": _fc,
                    "weight": 2,
                    "fillOpacity": _o,
                },
                tooltip=tooltip,
            ).add_to(target)

    outage_group.add_to(m)
    restored_group.add_to(m)
    normal_group.add_to(m)
    folium.LayerControl().add_to(m)
    return m


def plot_restoration_timeline(
    events: Sequence[dict[str, Any]],
    *,
    time_column: str = "restored_at",
    label_column: str = "node_id",
    title: str | None = None,
    output_path: str | Path | None = None,
) -> Any:
    """Plot a horizontal timeline of restoration events.

    Each event dict should have at least *time_column* (numeric hours or
    a sortable value) and *label_column*.

    Args:
        events: Sequence of event dicts.
        time_column: Column with the time value.
        label_column: Column with the event label.
        title: Optional title.
        output_path: Save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    plt = importlib.import_module("matplotlib.pyplot")
    sorted_events = sorted(events, key=lambda e: float(e.get(time_column, 0)))
    labels = [str(e.get(label_column, "")) for e in sorted_events]
    times = [float(e.get(time_column, 0)) for e in sorted_events]

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.5)))
    colors = ["#2b8a3e" if i < len(times) // 2 else "#e67700" for i in range(len(times))]
    ax.barh(range(len(labels)), times, color=colors, edgecolor="#333")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time")
    ax.set_title(title or "Restoration Timeline")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), bbox_inches="tight", dpi=150)

    return fig


def quickplot(
    frame: Any,
    *,
    color: str = "steelblue",
    edge_color: str = "black",
    figsize: tuple[int, int] = (10, 8),
    title: str | None = None,
    label_column: str | None = None,
    output_path: str | Path | None = None,
) -> Any:
    """Create a quick static matplotlib plot of frame geometries.

    Args:
        frame: A :class:`~geoprompt.frame.GeoPromptFrame`.
        color: Fill color for geometries.
        edge_color: Edge color for geometries.
        figsize: Figure size as ``(width, height)`` inches.
        title: Plot title.
        label_column: Column name for point labels.
        output_path: If given, save the figure to this path.

    Returns:
        A ``matplotlib.figure.Figure`` object.
    """
    plt = importlib.import_module("matplotlib.pyplot")
    fig, ax = plt.subplots(figsize=figsize)

    rows = list(frame)
    geometry_column = frame.geometry_column

    for row in rows:
        geom = row[geometry_column]
        gtype = geometry_type(geom)
        coords = geom.get("coordinates")

        if gtype == "Point":
            ax.plot(coords[0], coords[1], "o", color=color, markersize=6, markeredgecolor=edge_color)
            if label_column and label_column in row:
                ax.annotate(str(row[label_column]), (coords[0], coords[1]), fontsize=8, ha="left", va="bottom")
        elif gtype == "MultiPoint":
            for pt in coords:
                ax.plot(pt[0], pt[1], "o", color=color, markersize=6, markeredgecolor=edge_color)
        elif gtype == "LineString":
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, "-", color=color, linewidth=1.5)
        elif gtype == "MultiLineString":
            for line in coords:
                xs = [c[0] for c in line]
                ys = [c[1] for c in line]
                ax.plot(xs, ys, "-", color=color, linewidth=1.5)
        elif gtype == "Polygon":
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.fill(xs, ys, color=color, alpha=0.4, edgecolor=edge_color, linewidth=1)
        elif gtype == "MultiPolygon":
            for poly in coords:
                xs = [c[0] for c in poly]
                ys = [c[1] for c in poly]
                ax.fill(xs, ys, color=color, alpha=0.4, edgecolor=edge_color, linewidth=1)

    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), bbox_inches="tight", dpi=150)

    return fig


# ---------------------------------------------------------------------------
#  Basemap and symbol preset families
# ---------------------------------------------------------------------------

BASEMAP_PRESETS: dict[str, str] = {
    "light": "CartoDB positron",
    "dark": "CartoDB dark_matter",
    "satellite": "Esri.WorldImagery",
    "terrain": "Stamen Terrain",
    "street": "OpenStreetMap",
    "report": "CartoDB positron",
}

SYMBOL_PRESETS: dict[str, dict[str, str]] = {
    "risk": {
        "high": "#d7191c",
        "medium": "#fdae61",
        "low": "#1a9641",
        "unknown": "#999999",
    },
    "outage": {
        "active": "#e31a1c",
        "restored": "#33a02c",
        "planned": "#ff7f00",
        "unknown": "#999999",
    },
    "severity": {
        "critical": "#d7191c",
        "major": "#fdae61",
        "minor": "#ffffbf",
        "none": "#1a9641",
    },
    "asset_class": {
        "electric": "#e41a1c",
        "water": "#377eb8",
        "gas": "#ff7f00",
        "telecom": "#984ea3",
        "stormwater": "#4daf4a",
    },
    "recommendation": {
        "do_now": "#d7191c",
        "plan_next": "#fdae61",
        "monitor": "#1a9641",
        "defer": "#999999",
    },
}

MAP_STYLE_PACKS: dict[str, dict[str, str]] = {
    "utilities": {
        "accent": "#1565c0",
        "accent_soft": "#e8f1ff",
        "background": "#f7fbff",
        "text": "#17324d",
    },
    "resilience": {
        "accent": "#b02a37",
        "accent_soft": "#fdecef",
        "background": "#fff8f8",
        "text": "#4a1f24",
    },
    "planning": {
        "accent": "#6f42c1",
        "accent_soft": "#f3ecff",
        "background": "#fbf9ff",
        "text": "#33204c",
    },
    "environmental": {
        "accent": "#2b8a3e",
        "accent_soft": "#e9f7ed",
        "background": "#f8fff9",
        "text": "#1f3a28",
    },
}


def portfolio_scorecard(
    records: Sequence[Record],
    *,
    title: str = "Portfolio Scorecard",
    metrics: Sequence[str] | None = None,
) -> str:
    """Render a simple HTML portfolio scorecard from a set of records.

    Each record should have a ``"name"`` key and one or more numeric metric
    columns.

    Args:
        records: Sequence of dicts representing scored items.
        title: Scorecard title.
        metrics: Metric columns to display (default: all numeric columns).

    Returns:
        An HTML string suitable for embedding or saving.
    """
    if not records:
        return f"<h2>{title}</h2><p>No data.</p>"

    if metrics is None:
        metrics = [k for k, v in records[0].items() if isinstance(v, (int, float)) and k != "name"]

    header = "".join(f"<th>{m}</th>" for m in metrics)
    rows_html: list[str] = []
    for rec in records:
        cells = "".join(f"<td>{rec.get(m, '')}</td>" for m in metrics)
        rows_html.append(f"<tr><td><strong>{rec.get('name', '')}</strong></td>{cells}</tr>")

    return (
        f"<h2>{title}</h2>"
        f"<table border='1' cellpadding='4' cellspacing='0'>"
        f"<tr><th>Item</th>{header}</tr>"
        + "".join(rows_html)
        + "</table>"
    )


def recommendation_card(
    item_name: str,
    recommendation: str,
    *,
    explanation: str = "",
    score: float | None = None,
    color: str | None = None,
) -> str:
    """Render an HTML recommendation card for a single item.

    Args:
        item_name: Name of the asset or project.
        recommendation: Short recommendation label (e.g. ``"Do Now"``).
        explanation: Longer explanation text.
        score: Optional numeric score to display.
        color: Optional background color for the recommendation badge.

    Returns:
        An HTML string.
    """
    badge_color = color or SYMBOL_PRESETS["recommendation"].get(
        recommendation.lower().replace(" ", "_"), "#999999"
    )
    score_html = f"<p>Score: <strong>{score:.2f}</strong></p>" if score is not None else ""
    return (
        f"<div style='border:1px solid #ccc; padding:12px; margin:8px; border-radius:6px;'>"
        f"<h3>{item_name}</h3>"
        f"<span style='background:{badge_color}; color:white; padding:4px 10px; border-radius:4px;'>"
        f"{recommendation}</span>"
        f"{score_html}"
        f"<p>{explanation}</p>"
        f"</div>"
    )


def before_after_comparison(
    before_records: Sequence[Record],
    after_records: Sequence[Record],
    *,
    title: str = "Before / After Comparison",
    key_column: str = "name",
    metric_columns: Sequence[str] | None = None,
) -> str:
    """Render an HTML before/after comparison table.

    Args:
        before_records: Records representing the baseline scenario.
        after_records: Records representing the proposed scenario.
        title: Table title.
        key_column: Column used to match rows across scenarios.
        metric_columns: Numeric columns to compare (auto-detected if omitted).

    Returns:
        An HTML string.
    """
    if not before_records:
        return f"<h2>{title}</h2><p>No baseline data.</p>"

    if metric_columns is None:
        metric_columns = [k for k, v in before_records[0].items() if isinstance(v, (int, float)) and k != key_column]

    after_map = {r.get(key_column): r for r in after_records}

    header = "".join(f"<th>{m} (before)</th><th>{m} (after)</th><th>Δ</th>" for m in metric_columns)
    rows_html: list[str] = []
    for rec in before_records:
        name = rec.get(key_column, "")
        after_rec = after_map.get(name, {})
        cells = ""
        for m in metric_columns:
            bv = rec.get(m)
            av = after_rec.get(m)
            delta = ""
            if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                delta = f"{av - bv:+.2f}"
            cells += f"<td>{bv}</td><td>{av if av is not None else ''}</td><td>{delta}</td>"
        rows_html.append(f"<tr><td><strong>{name}</strong></td>{cells}</tr>")

    return (
        f"<h2>{title}</h2>"
        f"<table border='1' cellpadding='4' cellspacing='0'>"
        f"<tr><th>Item</th>{header}</tr>"
        + "".join(rows_html)
        + "</table>"
    )


def build_executive_briefing_pack(
    sections: Sequence[dict[str, Any]],
    *,
    title: str = "Executive Briefing",
    organization: str = "GeoPrompt",
    theme: str = "utilities",
    output_path: str | Path | None = None,
) -> str:
    """Build a polished, brandable HTML briefing pack for executive delivery."""
    pack = MAP_STYLE_PACKS.get(theme, MAP_STYLE_PACKS["utilities"])
    cards: list[str] = []
    for section in sections:
        section_title = str(section.get("title", "Section"))
        section_type = str(section.get("type", "note"))
        content = section.get("content", section.get("data", ""))
        if section_type == "metric":
            body = f"<div class='metric' aria-label='{section_title} metric'>{content}</div>"
        elif isinstance(content, list) and content and isinstance(content[0], dict):
            body = _html_table(content)
        elif isinstance(content, (list, tuple)):
            body = "<ul>" + "".join(f"<li>{item}</li>" for item in content) + "</ul>"
        else:
            body = f"<p>{content}</p>"
        cards.append(
            f"<section class='card' aria-labelledby='section-{len(cards)}'>"
            f"<h2 id='section-{len(cards)}'>{section_title}</h2>{body}</section>"
        )

    html = (
        "<!DOCTYPE html>"
        "<html lang='en'><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<style>"
        f"body{{font-family:Segoe UI,Arial,sans-serif;background:{pack['background']};color:{pack['text']};margin:0;padding:0;}}"
        f"header{{background:{pack['accent']};color:white;padding:24px 28px;}}"
        ".sub{opacity:.9;font-size:0.95rem;margin-top:6px;}"
        ".wrap{padding:18px 24px 30px 24px;}"
        f".card{{background:white;border-left:6px solid {pack['accent']};padding:16px 18px;margin:14px 0;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);}}"
        f".metric{{font-size:2rem;font-weight:700;color:{pack['accent']};}}"
        f".badge{{display:inline-block;background:{pack['accent_soft']};color:{pack['text']};padding:4px 10px;border-radius:999px;font-weight:600;}}"
        "table{border-collapse:collapse;width:100%;}th,td{border:1px solid #d0d7de;padding:8px;text-align:left;}"
        "th{background:#f6f8fa;}footer{padding:12px 24px 24px 24px;font-size:.9rem;color:#555;}"
        "</style></head><body>"
        f"<header><h1>{title}</h1><div class='sub'>{organization} executive briefing · theme: <span class='badge'>{theme}</span></div></header>"
        f"<main class='wrap'>{''.join(cards)}</main>"
        "<footer>Generated from GeoPrompt reporting and scenario outputs.</footer>"
        "</body></html>"
    )
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    return html


def audit_html_accessibility(html: str) -> dict[str, Any]:
    """Run a lightweight accessibility audit for generated HTML outputs."""
    lowered = html.lower()
    issues: list[str] = []
    if "<title>" not in lowered:
        issues.append("missing_title")
    if "<main" not in lowered:
        issues.append("missing_main_landmark")
    if "<h1" not in lowered:
        issues.append("missing_h1")

    html_tag = re.search(r"<html\b[^>]*>", html, flags=re.IGNORECASE)
    if html_tag and "lang=" not in html_tag.group(0).lower():
        issues.append("missing_lang")

    for tag in re.findall(r"<img\b[^>]*>", html, flags=re.IGNORECASE):
        if "alt=" not in tag.lower():
            issues.append("image_missing_alt")
    return {"passed": not issues, "issues": issues, "issue_count": len(issues)}


__all__ = [
    "BASEMAP_PRESETS",
    "MAP_STYLE_PACKS",
    "RESILIENCE_STYLE_PRESETS",
    "SYMBOL_PRESETS",
    "audit_html_accessibility",
    "before_after_comparison",
    "build_executive_briefing_pack",
    "plot_restoration_timeline",
    "plot_scenario_dashboard",
    "plotly_scenario_dashboard",
    "portfolio_scorecard",
    "quickplot",
    "recommendation_card",
    "resilience_style_map",
    "save_map",
    "save_plotly_html",
    "to_choropleth",
    "to_folium_map",
    "to_outage_overlay_map",
]
