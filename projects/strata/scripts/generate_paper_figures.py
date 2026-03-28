#!/usr/bin/env python3
"""Generate paper figures from benchmark outputs.

Produces grouped bar charts for marginal coverage, mean width, and ECE from
`outputs/real_method_comparison.csv` and `outputs/real_data_benchmark.csv`.
Also produces georeferenced site maps using real bus coordinates on
OpenStreetMap basemaps via contextily.
"""
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly palette (Okabe-Ito)
LAYER_COLORS = {
    "power": "#E69F00",   # orange
    "water": "#56B4E9",   # sky blue
    "telecom": "#009E73",  # bluish green
}
LAYER_MARKERS = {
    "power": "o",
    "water": "s",
    "telecom": "^",
}


def read_csv_safe(path):
    p = Path(path)
    if not p.exists():
        print(f"Missing: {p}")
        return None
    return pd.read_csv(p)


def plot_grouped_metrics(df, dataset_col, method_col, metrics, out_prefix):
    grouped = df.groupby([dataset_col, method_col]).mean().reset_index()
    datasets = grouped[dataset_col].unique()
    methods = grouped[method_col].unique()

    # Colorblind-friendly method palette (Okabe-Ito extended)
    method_colors = ["#0072B2", "#D55E00", "#CC79A7", "#F0E442",
                     "#56B4E9", "#009E73", "#E69F00", "#999999", "#000000"]

    metric_labels = {
        'marginal_cov': 'Marginal Coverage',
        'mean_width': 'Mean Interval Width',
        'ece': 'Expected Calibration Error',
    }

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        width = 0.8 / len(methods)
        x = range(len(datasets))
        for i, m in enumerate(sorted(methods)):
            vals = []
            for d in datasets:
                row = grouped[(grouped[dataset_col] == d) & (grouped[method_col] == m)]
                if not row.empty:
                    vals.append(float(row.iloc[0][metric]))
                else:
                    vals.append(0.0)
            color = method_colors[i % len(method_colors)]
            ax.bar([xi + i*width for xi in x], vals, width=width, label=m.upper(),
                   color=color, edgecolor='white', linewidth=0.5)
        ax.set_xticks([xi + width*(len(methods)-1)/2 for xi in x])
        ax.set_xticklabels(datasets, fontsize=11)
        ylabel = metric_labels.get(metric, metric.replace('_', ' ').title())
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=13, fontweight='bold')
        ax.legend(framealpha=0.9, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        out = OUT / f"{out_prefix}_{metric}.png"
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Wrote {out}")
        plt.close(fig)


def draw_georeferenced_map(graph, name):
    """Draw a georeferenced site map with real lat/lon coordinates on a basemap."""
    try:
        import geopandas as gpd
        import contextily as ctx
        from shapely.geometry import Point, LineString
    except ImportError:
        print(f"geopandas/contextily not available; falling back to schematic for {name}")
        draw_schematic_fallback(graph, name)
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each layer with distinct color and marker
    handles = []
    for ntype in ["power", "water", "telecom"]:
        positions = graph.node_positions.get(ntype)
        if positions is None or len(positions) == 0:
            continue

        # positions are stored as [lon, lat]
        lons = positions[:, 0]
        lats = positions[:, 1]
        points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
        gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=3857)  # Web Mercator for basemap

        color = LAYER_COLORS[ntype]
        marker = LAYER_MARKERS[ntype]
        sizes = {"power": 40, "water": 25, "telecom": 30}

        gdf.plot(
            ax=ax,
            color=color,
            marker=marker,
            markersize=sizes[ntype],
            alpha=0.8,
            zorder=3,
        )
        handles.append(
            mlines.Line2D([], [], color=color, marker=marker, linestyle='None',
                          markersize=8, label=f"{ntype.capitalize()} ({len(positions)})")
        )

    # Draw intra-layer edges
    for edge_key, edge_idx in graph.edge_index.items():
        if edge_idx.shape[1] == 0:
            continue
        src_type = edge_key[0]
        positions_src = graph.node_positions.get(src_type)
        if len(edge_key) == 3:
            dst_type = edge_key[2]
        else:
            dst_type = src_type
        positions_dst = graph.node_positions.get(dst_type)
        if positions_src is None or positions_dst is None:
            continue

        # Only draw a subset of edges to keep the plot readable
        n_edges = edge_idx.shape[1]
        max_draw = min(n_edges, 500)
        sample = np.random.default_rng(42).choice(n_edges, max_draw, replace=False) if n_edges > max_draw else range(n_edges)

        edge_color = LAYER_COLORS.get(src_type, "#999999")
        if src_type != dst_type:
            edge_color = "#BBBBBB"

        for idx in sample:
            si, di = int(edge_idx[0, idx]), int(edge_idx[1, idx])
            if si >= len(positions_src) or di >= len(positions_dst):
                continue
            x0, y0 = _to_mercator(positions_src[si, 0], positions_src[si, 1])
            x1, y1 = _to_mercator(positions_dst[di, 0], positions_dst[di, 1])
            ax.plot([x0, x1], [y0, y1], color=edge_color, alpha=0.15, linewidth=0.5, zorder=1)

    # Add basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=9)
    except Exception:
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=9)
        except Exception:
            print(f"  Could not fetch basemap tiles for {name}; plotting without basemap.")

    ax.legend(handles=handles, loc='upper right', framealpha=0.9)
    ax.set_title(f"{name} — Heterogeneous Infrastructure Graph", fontsize=12)
    ax.set_axis_off()
    fig.tight_layout()
    out = OUT / f"{name.replace(' ', '_').lower()}_site_map.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out}")


def _to_mercator(lon, lat):
    """Convert lon/lat to Web Mercator (EPSG:3857)."""
    import math
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return x, y


def draw_schematic_fallback(graph_or_nx, name):
    """Fallback schematic using spring layout when basemap is unavailable."""
    G = graph_or_nx
    if not isinstance(G, nx.Graph):
        try:
            G = nx.from_numpy_array(G)
        except Exception:
            try:
                n = int(getattr(G, 'number_of_nodes', lambda: len(G))())
            except Exception:
                n = 50
            G = nx.erdos_renyi_graph(n, 0.05)
    pos = nx.spring_layout(G, seed=2)
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=30, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    ax.set_title(name)
    ax.set_axis_off()
    out = OUT / f"{name.replace(' ', '_').lower()}_site_map.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def try_draw_real_maps():
    """Try to draw georeferenced site maps from real data loaders."""
    sys.path.insert(0, str(ROOT))
    try:
        from src.hetero_conformal.real_data import load_activsg200, load_ieee118
        print("Found real-data loaders; generating georeferenced maps...")
        try:
            G1 = load_activsg200()
            draw_georeferenced_map(G1, "ACTIVSg200")
        except Exception:
            print("Could not draw ACTIVSg200 georeferenced map:")
            traceback.print_exc()
        try:
            G2 = load_ieee118()
            draw_georeferenced_map(G2, "IEEE118")
        except Exception:
            print("Could not draw IEEE118 georeferenced map:")
            traceback.print_exc()
    except Exception:
        print("Real-data loaders not available; drawing placeholder schematics.")
        draw_schematic_fallback(nx.erdos_renyi_graph(200, 0.02, seed=1), "ACTIVSg200_placeholder")
        draw_schematic_fallback(nx.erdos_renyi_graph(118, 0.03, seed=2), "IEEE118_placeholder")


def main():
    df_cmp = read_csv_safe(ROOT / 'outputs' / 'real_method_comparison.csv')
    df_bench = read_csv_safe(ROOT / 'outputs' / 'real_data_benchmark.csv')
    metrics = ['marginal_cov', 'mean_width', 'ece']
    if df_cmp is not None:
        plot_grouped_metrics(df_cmp, 'dataset', 'method', metrics, 'method_comparison')
    if df_bench is not None:
        try:
            plot_grouped_metrics(df_bench, 'dataset', 'method', metrics, 'real_benchmark')
        except KeyError:
            print("Skipping real_benchmark plots: expected column 'method' not found.")

    try_draw_real_maps()


if __name__ == '__main__':
    main()
