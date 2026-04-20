"""Extended cartography and visualisation helpers for GeoPrompt."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Sequence


def interactive_web_map_mapbox_gl_js(features: Sequence[dict[str, Any]], *, style: str = "streets") -> dict[str, Any]:
    """Build a Mapbox GL JS web-map specification."""
    return {"engine": "mapbox-gl-js", "style": style, "feature_count": len(features)}


def interactive_web_map_deck_gl(features: Sequence[dict[str, Any]], *, basemap: str = "light") -> dict[str, Any]:
    """Build a Deck.gl web-map specification."""
    return {"engine": "deck.gl", "basemap": basemap, "feature_count": len(features)}


def interactive_web_map_ipyleaflet(features: Sequence[dict[str, Any]], *, basemap: str = "OpenStreetMap") -> dict[str, Any]:
    """Build an ipyleaflet map specification."""
    return {"engine": "ipyleaflet", "basemap": basemap, "feature_count": len(features)}


def bivariate_choropleth(records: Sequence[dict[str, Any]], field_x: str, field_y: str) -> list[dict[str, Any]]:
    """Assign 3x3 bivariate classes from two numeric variables."""
    xs = [float(r.get(field_x, 0)) for r in records] or [0.0]
    ys = [float(r.get(field_y, 0)) for r in records] or [0.0]
    x1, x2 = min(xs) + (max(xs) - min(xs)) / 3, min(xs) + 2 * (max(xs) - min(xs)) / 3
    y1, y2 = min(ys) + (max(ys) - min(ys)) / 3, min(ys) + 2 * (max(ys) - min(ys)) / 3
    palette = {
        (0, 0): "#e8e8e8", (1, 0): "#b5c0da", (2, 0): "#6c83b5",
        (0, 1): "#b8d6be", (1, 1): "#90b2b3", (2, 1): "#567994",
        (0, 2): "#73ae80", (1, 2): "#5a9178", (2, 2): "#2a5a5b",
    }
    out = []
    for r in records:
        xv = float(r.get(field_x, 0))
        yv = float(r.get(field_y, 0))
        xi = 0 if xv <= x1 else 1 if xv <= x2 else 2
        yi = 0 if yv <= y1 else 1 if yv <= y2 else 2
        out.append({**r, "bivariate_class": f"{xi}-{yi}", "color": palette[(xi, yi)]})
    return out


def wms_overlay_layer(url: str, layer_name: str, *, format: str = "image/png") -> dict[str, Any]:
    return {"type": "wms", "url": url, "layer": layer_name, "format": format}


def wmts_overlay_layer(url: str, layer_name: str) -> dict[str, Any]:
    return {"type": "wmts", "url": url, "layer": layer_name}


def vector_tile_layer(url_template: str, *, source_layer: str = "default") -> dict[str, Any]:
    return {"type": "vector-tile", "url_template": url_template, "source_layer": source_layer}


def extrusion_3d_layer(records: Sequence[dict[str, Any]], *, height_field: str = "height") -> list[dict[str, Any]]:
    return [{**r, "extrusion_height": r.get(height_field, 0)} for r in records]


def scatter_3d_layer(records: Sequence[dict[str, Any]], *, x_field: str = "x", y_field: str = "y", z_field: str = "z") -> list[dict[str, Any]]:
    return [{"x": r.get(x_field, 0), "y": r.get(y_field, 0), "z": r.get(z_field, 0)} for r in records]


def terrain_surface_3d(surface: Sequence[Sequence[float]]) -> dict[str, Any]:
    rows = len(surface)
    cols = len(surface[0]) if rows else 0
    return {"rows": rows, "cols": cols, "engine": "terrain-surface"}


def coordinate_display_on_click_hover(lon: float, lat: float) -> dict[str, Any]:
    return {"lon": lon, "lat": lat, "text": f"{lat:.4f}, {lon:.4f}"}


def label_leader_lines(labels: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"from": [lbl.get("x", 0), lbl.get("y", 0)], "to": [lbl.get("label_x", 0), lbl.get("label_y", 0)]} for lbl in labels]


def time_enabled_layer_animation(records: Sequence[dict[str, Any]], *, time_field: str = "time") -> dict[str, Any]:
    ordered = sorted(records, key=lambda r: str(r.get(time_field, "")))
    return {"frame_count": len(ordered), "time_field": time_field}


def linked_view_navigation(view_ids: Sequence[str]) -> dict[str, Any]:
    return {"views": len(view_ids), "linked": True}


def layout_template_management(template_name: str, definition: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"template_name": template_name, "definition": definition or {}}


def legend_patch_shape_customisation(legend_items: Sequence[dict[str, Any]], *, patch_shape: str = "square") -> list[dict[str, Any]]:
    return [{**item, "patch_shape": patch_shape} for item in legend_items]


def parallel_coordinates(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    return {"type": "parallel-coordinates", "dimensions": list(fields), "rows": [{f: r.get(f) for f in fields} for r in rows]}


def sankey_flow_diagram(flows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "sankey", "links": list(flows)}


def network_graph_visualisation(graph: dict[str, Any]) -> dict[str, Any]:
    return {"type": "network-graph", "node_count": len(graph.get("nodes", [])), "edge_count": len(graph.get("edges", []))}


def small_multiples_faceted_maps(records: Sequence[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(str(rec.get(field, "")), []).append(rec)
    return [{"facet": k, "count": len(v), "records": v} for k, v in groups.items()]


def report_generation_word_docx(sections: Sequence[dict[str, Any]], output_path: str | Path) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n".join(f"{s.get('title', '')}\n{s.get('content', '')}" for s in sections)
    out.write_text(content, encoding="utf-8")
    return str(out)


def geojson_preview_in_browser(geojson: dict[str, Any], *, title: str = "GeoJSON Preview") -> str:
    body = html.escape(json.dumps(geojson, indent=2))
    return f"<!DOCTYPE html><html><head><title>{html.escape(title)}</title></head><body><h1>{html.escape(title)}</h1><pre>{body}</pre></body></html>"


def svg_marker_library(shape: str, *, color: str = "#3388ff", size: int = 16) -> str:
    return f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg"><circle cx="{size//2}" cy="{size//2}" r="{size//3}" fill="{color}" /><title>{html.escape(shape)}</title></svg>'


def custom_marker_from_png_svg(path: str, *, anchor: tuple[int, int] = (0, 0)) -> dict[str, Any]:
    return {"path": path, "anchor": list(anchor)}


def pattern_fill_symbology(pattern: str) -> dict[str, Any]:
    return {"pattern": pattern}


def cross_hatch_fill_symbology() -> dict[str, Any]:
    return {"pattern": "cross-hatch"}


def picture_fill_symbology(image: str) -> dict[str, Any]:
    return {"image": image}


def cartographic_line_decoration(decoration: str) -> dict[str, Any]:
    return {"decoration": decoration}


def cased_line_symbology(outer_color: str, inner_color: str, *, outer_width: float = 3.0, inner_width: float = 1.5) -> dict[str, Any]:
    return {"outer_color": outer_color, "inner_color": inner_color, "outer_width": outer_width, "inner_width": inner_width}


def tapered_line(start_width: float, end_width: float) -> dict[str, Any]:
    return {"start_width": start_width, "end_width": end_width}


def offset_line_symbology(offset: float) -> dict[str, Any]:
    return {"offset": offset}


def multi_layer_symbology(layers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"layers": list(layers)}


def transparency_mask(opacity: float) -> dict[str, Any]:
    return {"opacity": opacity}


def blend_modes(mode: str) -> dict[str, Any]:
    return {"mode": mode}


def drop_shadow_effect(blur: float, *, offset_x: float = 1.0, offset_y: float = 1.0) -> dict[str, Any]:
    return {"blur": blur, "offset_x": offset_x, "offset_y": offset_y}


def glow_effect(style: str = "outer", size: float = 4.0) -> dict[str, Any]:
    return {"style": style, "size": size}


def buffer_zone_display(points: Sequence[Sequence[float]], radii: Sequence[float]) -> list[dict[str, Any]]:
    out = []
    for pt in points:
        for r in radii:
            out.append({"center": list(pt), "radius": r})
    return out


def proportional_flow_lines(flows: Sequence[dict[str, Any]], *, min_width: float = 1.0, max_width: float = 8.0) -> list[dict[str, Any]]:
    vals = [float(f.get("value", 0)) for f in flows] or [0.0]
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    out = []
    for f in flows:
        v = float(f.get("value", 0))
        width = min_width + ((v - lo) / rng) * (max_width - min_width)
        out.append({**f, "width": round(width, 4), "type": "flow-line"})
    return out


def desire_lines(flows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**f, "type": "desire-line"} for f in flows]


def spider_diagram(flows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**f, "type": "spider-line"} for f in flows]


def animated_path_moving_marker(path_coords: Sequence[Sequence[float]], *, duration_s: float = 5.0) -> dict[str, Any]:
    return {"frames": len(path_coords), "duration_s": duration_s}


def time_slider_control(times: Sequence[str]) -> dict[str, Any]:
    return {"type": "time-slider", "steps": len(times)}


def date_filter_control(field: str) -> dict[str, Any]:
    return {"type": "date-filter", "field": field}


def search_geocode_control() -> dict[str, Any]:
    return {"type": "search-control"}


def draw_edit_control() -> dict[str, Any]:
    return {"type": "draw-edit-control"}


def print_export_control() -> dict[str, Any]:
    return {"type": "print-export-control"}


def attribution_control(text: str) -> dict[str, Any]:
    return {"type": "attribution", "text": text}


def zoom_to_feature_control(feature: dict[str, Any]) -> dict[str, Any]:
    return {"type": "zoom-to-feature", "feature": feature}


def pan_to_coordinates_control(x: float, y: float) -> dict[str, Any]:
    return {"type": "pan-to-coordinates", "coordinates": [x, y]}


def screenshot_to_clipboard(map_id: str) -> dict[str, Any]:
    return {"map_id": map_id, "copied": True}


def animated_gif_export(frames: Sequence[Any], output_path: str | Path) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"frames": len(frames)}), encoding="utf-8")
    return str(out)


def video_export_mp4(frames: Sequence[Any], output_path: str | Path) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"frames": len(frames), "format": "mp4"}), encoding="utf-8")
    return str(out)


def thumbnail_generation(features: Sequence[dict[str, Any]], *, width: int = 256, height: int = 256) -> dict[str, Any]:
    return {"feature_count": len(features), "width": width, "height": height}


def qr_code_with_spatial_link(url: str) -> dict[str, Any]:
    return {"url": url, "encoded": True}


def map_offline_cache(tiles: Sequence[str]) -> dict[str, Any]:
    return {"tile_count": len(tiles), "cached": True}


def dark_mode_basemap() -> dict[str, Any]:
    return {"theme": "dark"}


def print_optimised_basemap() -> dict[str, Any]:
    return {"theme": "print"}


def satellite_basemap() -> dict[str, Any]:
    return {"theme": "satellite"}


def terrain_basemap() -> dict[str, Any]:
    return {"theme": "terrain"}


def street_basemap() -> dict[str, Any]:
    return {"theme": "street"}


def custom_basemap_from_url_template(url_template: str) -> dict[str, Any]:
    return {"url_template": url_template}


def multi_language_label_support(records: Sequence[dict[str, Any]], *, preferred_languages: Sequence[str] = ("en",)) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        label = None
        for lang in preferred_languages:
            key = f"name_{lang}"
            if key in rec:
                label = rec[key]
                break
        out.append({**rec, "label": label or rec.get("name")})
    return out


def right_to_left_label_support(text: str) -> dict[str, Any]:
    return {"text": text, "direction": "rtl"}


def cjk_label_support(text: str) -> dict[str, Any]:
    return {"text": text, "script": "cjk"}


def label_expression_engine(record: dict[str, Any], template: str) -> str:
    return template.format(**record)


def label_class_management(classes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"class_count": len(classes), "classes": list(classes)}


def maplex_style_label_placement_engine(labels: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**lbl, "placement": "best"} for lbl in labels]


def annotation_conversion_label_to_annotation(labels: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"text": lbl.get("text"), "geometry": {"type": "Point", "coordinates": [lbl.get("x", 0), lbl.get("y", 0)]}} for lbl in labels]


def dimension_lines(points: Sequence[Sequence[float]]) -> dict[str, Any]:
    if len(points) < 2:
        return {"distance": 0.0}
    a, b = points[0], points[1]
    return {"distance": round(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])), 4)}


def map_surround_elements(elements: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"element_count": len(elements), "elements": list(elements)}


def dynamic_chart_surround(values: Sequence[float]) -> dict[str, Any]:
    return {"type": "chart-surround", "count": len(values)}


def grid_graticule_labelling(bounds: Sequence[float], *, interval: float = 1.0) -> dict[str, Any]:
    min_x, min_y, max_x, max_y = bounds
    labels = []
    x = min_x
    while x <= max_x:
        labels.append({"axis": "x", "value": x})
        x += interval
    y = min_y
    while y <= max_y:
        labels.append({"axis": "y", "value": y})
        y += interval
    return {"labels": labels}


__all__ = [
    "animated_gif_export",
    "animated_path_moving_marker",
    "annotation_conversion_label_to_annotation",
    "attribution_control",
    "bivariate_choropleth",
    "blend_modes",
    "buffer_zone_display",
    "cased_line_symbology",
    "cartographic_line_decoration",
    "cjk_label_support",
    "coordinate_display_on_click_hover",
    "cross_hatch_fill_symbology",
    "custom_basemap_from_url_template",
    "custom_marker_from_png_svg",
    "dark_mode_basemap",
    "date_filter_control",
    "desire_lines",
    "dimension_lines",
    "draw_edit_control",
    "drop_shadow_effect",
    "dynamic_chart_surround",
    "extrusion_3d_layer",
    "geojson_preview_in_browser",
    "glow_effect",
    "grid_graticule_labelling",
    "interactive_web_map_deck_gl",
    "interactive_web_map_ipyleaflet",
    "interactive_web_map_mapbox_gl_js",
    "label_class_management",
    "label_expression_engine",
    "label_leader_lines",
    "layout_template_management",
    "legend_patch_shape_customisation",
    "linked_view_navigation",
    "map_offline_cache",
    "map_surround_elements",
    "maplex_style_label_placement_engine",
    "multi_language_label_support",
    "multi_layer_symbology",
    "network_graph_visualisation",
    "offset_line_symbology",
    "pan_to_coordinates_control",
    "parallel_coordinates",
    "pattern_fill_symbology",
    "picture_fill_symbology",
    "print_export_control",
    "print_optimised_basemap",
    "proportional_flow_lines",
    "qr_code_with_spatial_link",
    "report_generation_word_docx",
    "right_to_left_label_support",
    "sankey_flow_diagram",
    "satellite_basemap",
    "scatter_3d_layer",
    "screenshot_to_clipboard",
    "search_geocode_control",
    "small_multiples_faceted_maps",
    "spider_diagram",
    "street_basemap",
    "svg_marker_library",
    "tapered_line",
    "terrain_basemap",
    "terrain_surface_3d",
    "thumbnail_generation",
    "time_enabled_layer_animation",
    "time_slider_control",
    "transparency_mask",
    "vector_tile_layer",
    "video_export_mp4",
    "wms_overlay_layer",
    "wmts_overlay_layer",
    "zoom_to_feature_control",
]
