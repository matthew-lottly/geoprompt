from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "service_layers.geojson"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_SEED_SQL_PATH = DEFAULT_OUTPUT_DIR / "sample_seed.sql"
DEFAULT_CHART_DIR_NAME = "charts"


@dataclass(frozen=True)
class LayerFeature:
    feature_id: str
    layer_name: str
    category: str
    region: str
    status: str
    owner: str
    coordinates: tuple[float, float]


def _escape_sql_text(value: str) -> str:
    return value.replace("'", "''")


def load_layer_features(input_path: Path = DEFAULT_INPUT_PATH) -> list[LayerFeature]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    features: list[LayerFeature] = []
    for entry in payload["features"]:
        properties = entry["properties"]
        longitude, latitude = entry["geometry"]["coordinates"]
        features.append(
            LayerFeature(
                feature_id=properties["featureId"],
                layer_name=properties["layerName"],
                category=properties["category"],
                region=properties["region"],
                status=properties["status"],
                owner=properties["owner"],
                coordinates=(float(longitude), float(latitude)),
            )
        )
    return features


def _build_bounds(features: list[LayerFeature]) -> dict[str, float]:
    longitudes = [feature.coordinates[0] for feature in features]
    latitudes = [feature.coordinates[1] for feature in features]
    return {
        "minLongitude": round(min(longitudes), 4),
        "minLatitude": round(min(latitudes), 4),
        "maxLongitude": round(max(longitudes), 4),
        "maxLatitude": round(max(latitudes), 4),
    }


def _build_layer_summary(features: list[LayerFeature]) -> dict[str, Any]:
    by_layer = Counter(feature.layer_name for feature in features)
    by_region = Counter(feature.region for feature in features)
    by_status = Counter(feature.status for feature in features)
    return {
        "featureCount": len(features),
        "layers": dict(sorted(by_layer.items())),
        "regions": dict(sorted(by_region.items())),
        "statuses": dict(sorted(by_status.items())),
        "bounds": _build_bounds(features),
    }


def _build_collections(features: list[LayerFeature]) -> list[dict[str, Any]]:
    groups: dict[str, list[LayerFeature]] = defaultdict(list)
    for feature in features:
        groups[feature.layer_name].append(feature)

    collections = []
    for layer_name, layer_features in sorted(groups.items()):
        categories = sorted({feature.category for feature in layer_features})
        regions = sorted({feature.region for feature in layer_features})
        collections.append(
            {
                "id": layer_name,
                "title": layer_name.replace("_", " ").title(),
                "featureCount": len(layer_features),
                "categories": categories,
                "regions": regions,
                "queryPatterns": [
                    "bbox filter",
                    "status filter",
                    "region filter",
                    "updated-since filter",
                ],
            }
        )
    return collections


def _build_endpoints(collections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    endpoints = [
        {
            "path": "/collections",
            "purpose": "List published spatial collections with summary metadata.",
        },
        {
            "path": "/collections/{collectionId}",
            "purpose": "Return metadata and query capabilities for a single collection.",
        },
        {
            "path": "/collections/{collectionId}/items",
            "purpose": "Serve filtered features with bbox, region, and status query parameters.",
        },
    ]
    for collection in collections:
        endpoints.append(
            {
                "path": f"/collections/{collection['id']}/items?bbox={{minLon,minLat,maxLon,maxLat}}",
                "purpose": f"Fetch filtered items for the {collection['title']} collection.",
            }
        )
    return endpoints


def _plot_service_footprint(features: list[LayerFeature], chart_path: Path) -> Path:
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    color_by_layer = {
        "maintenance_zones": "#b56576",
        "monitoring_sites": "#457b9d",
    }

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.set_facecolor("#f6f4ef")
    figure.patch.set_facecolor("#f6f4ef")

    for layer_name in sorted({feature.layer_name for feature in features}):
        matching_features = [feature for feature in features if feature.layer_name == layer_name]
        axis.scatter(
            [feature.coordinates[0] for feature in matching_features],
            [feature.coordinates[1] for feature in matching_features],
            s=180,
            c=color_by_layer.get(layer_name, "#4f5d75"),
            edgecolors="#1f2933",
            linewidths=0.8,
            label=layer_name.replace("_", " ").title(),
        )

    for feature in features:
        axis.annotate(
            f"{feature.layer_name.replace('_', ' ').title()}\n{feature.region}",
            xy=feature.coordinates,
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#1f2933",
        )

    axis.set_title("Published Service Footprint", fontsize=16, color="#1f2933")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(color="#d8d2c6", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False, loc="lower left")
    figure.tight_layout()
    figure.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return chart_path


def build_service_blueprint(project_root: Path = PROJECT_ROOT, service_name: str = "PostGIS Service Blueprint") -> dict[str, Any]:
    features = load_layer_features(project_root / "data" / "service_layers.geojson")
    collections = _build_collections(features)
    return {
        "serviceName": service_name,
        "summary": _build_layer_summary(features),
        "collections": collections,
        "endpoints": _build_endpoints(collections),
        "publicationPlan": {
            "database": "PostgreSQL + PostGIS",
            "deliveryOptions": ["FastAPI", "PostgREST", "OGC API Features gateway"],
            "indexes": ["GIST(geom)", "btree(status)", "btree(region)"],
        },
        "notes": [
            "The checked-in data is sample content for a public-safe spatial service blueprint.",
            "SQL assets in the repo show how source tables, views, and publication-ready collections can be organized.",
            "The exported blueprint JSON is intended as a handoff artifact for API or SQL-first service implementation.",
            "A generated seed SQL file can load the same sample collections into a local PostGIS container.",
        ],
    }


def export_service_blueprint(output_dir: Path = DEFAULT_OUTPUT_DIR, service_name: str = "PostGIS Service Blueprint") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = load_layer_features(PROJECT_ROOT / "data" / "service_layers.geojson")
    blueprint = build_service_blueprint(PROJECT_ROOT, service_name=service_name)
    chart_path = _plot_service_footprint(features, output_dir / DEFAULT_CHART_DIR_NAME / "published-service-footprint.png")
    blueprint["artifacts"] = {
        "charts": [
            {
                "name": "published_service_footprint",
                "chart": str(chart_path.relative_to(output_dir)).replace("\\", "/"),
            }
        ]
    }
    output_path = output_dir / "postgis_service_blueprint.json"
    output_path.write_text(json.dumps(blueprint, indent=2), encoding="utf-8")
    return output_path


def export_seed_sql(output_path: Path = DEFAULT_SEED_SQL_PATH, project_root: Path = PROJECT_ROOT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = load_layer_features(project_root / "data" / "service_layers.geojson")

    statements = [
        "TRUNCATE TABLE monitoring_features;",
    ]

    for feature in features:
        statements.append(
            "INSERT INTO monitoring_features (feature_id, layer_name, category, region, status, owner, geom) VALUES "
            f"('{_escape_sql_text(feature.feature_id)}', "
            f"'{_escape_sql_text(feature.layer_name)}', "
            f"'{_escape_sql_text(feature.category)}', "
            f"'{_escape_sql_text(feature.region)}', "
            f"'{_escape_sql_text(feature.status)}', "
            f"'{_escape_sql_text(feature.owner)}', "
            f"ST_SetSRID(ST_MakePoint({feature.coordinates[0]}, {feature.coordinates[1]}), 4326));"
        )

    output_path.write_text("\n".join(statements) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sample PostGIS service blueprint artifact.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated JSON output.")
    parser.add_argument("--service-name", default="PostGIS Service Blueprint", help="Display name embedded in the output artifact.")
    parser.add_argument(
        "--export-seed-sql",
        action="store_true",
        help="Also generate a PostGIS seed SQL file from the checked-in sample layer.",
    )
    parser.add_argument(
        "--seed-sql-path",
        type=Path,
        default=DEFAULT_SEED_SQL_PATH,
        help="Output path for the generated PostGIS seed SQL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_service_blueprint(args.output_dir, service_name=args.service_name)
    print(f"Wrote service blueprint to {output_path}")
    if args.export_seed_sql:
        seed_sql_path = export_seed_sql(args.seed_sql_path)
        print(f"Wrote PostGIS seed SQL to {seed_sql_path}")


if __name__ == "__main__":
    main()
