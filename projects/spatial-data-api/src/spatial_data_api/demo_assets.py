from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from spatial_data_api.core.config import DEFAULT_DATA_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "charts"


@dataclass(frozen=True)
class MonitoringSite:
    feature_id: str
    name: str
    category: str
    region: str
    status: str
    coordinates: tuple[float, float]


def load_monitoring_sites(input_path: Path = DEFAULT_DATA_PATH) -> list[MonitoringSite]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return [
        MonitoringSite(
            feature_id=feature["properties"]["featureId"],
            name=feature["properties"]["name"],
            category=feature["properties"]["category"],
            region=feature["properties"]["region"],
            status=feature["properties"]["status"],
            coordinates=(
                float(feature["geometry"]["coordinates"][0]),
                float(feature["geometry"]["coordinates"][1]),
            ),
        )
        for feature in payload["features"]
    ]


def export_monitoring_status_map(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    input_path: Path = DEFAULT_DATA_PATH,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "monitoring-status-footprint.png"
    sites = load_monitoring_sites(input_path)
    status_colors = {
        "alert": "#d94841",
        "normal": "#4c956c",
        "offline": "#52616b",
    }

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.set_facecolor("#f4f1ea")
    figure.patch.set_facecolor("#f4f1ea")

    for status in sorted({site.status for site in sites}):
        matching_sites = [site for site in sites if site.status == status]
        axis.scatter(
            [site.coordinates[0] for site in matching_sites],
            [site.coordinates[1] for site in matching_sites],
            s=180,
            c=status_colors.get(status, "#457b9d"),
            edgecolors="#1f2933",
            linewidths=0.8,
            label=status.title(),
        )

    for site in sites:
        axis.annotate(
            site.name,
            xy=site.coordinates,
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#1f2933",
        )

    axis.set_title("Monitoring Status Footprint", fontsize=16, color="#1f2933")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(color="#d9d3c7", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False, loc="lower left")
    figure.tight_layout()
    figure.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return chart_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a monitoring status map from the sample feature collection.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated image output.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_DATA_PATH, help="GeoJSON feature collection used for the generated map.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_monitoring_status_map(output_dir=args.output_dir, input_path=args.input_path)
    print(f"Wrote monitoring status map to {output_path}")


__all__ = ["MonitoringSite", "export_monitoring_status_map", "load_monitoring_sites", "main"]


if __name__ == "__main__":
    main()