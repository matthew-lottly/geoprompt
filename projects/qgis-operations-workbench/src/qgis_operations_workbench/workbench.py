from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTS_PATH = PROJECT_ROOT / "data" / "station_review_points.geojson"
DEFAULT_ROUTES_PATH = PROJECT_ROOT / "data" / "inspection_routes.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class StationFeature:
    feature_id: str
    name: str
    category: str
    region: str
    priority: str
    status: str
    owner: str
    coordinates: tuple[float, float]


@dataclass(frozen=True)
class InspectionRoute:
    route_id: str
    region: str
    map_scale: int
    layout_name: str
    station_ids: tuple[str, ...]


def load_station_features(points_path: Path = DEFAULT_POINTS_PATH) -> list[StationFeature]:
    payload = json.loads(points_path.read_text(encoding="utf-8"))
    features: list[StationFeature] = []
    for entry in payload["features"]:
        properties = entry["properties"]
        coordinates = entry["geometry"]["coordinates"]
        features.append(
            StationFeature(
                feature_id=properties["featureId"],
                name=properties["name"],
                category=properties["category"],
                region=properties["region"],
                priority=properties["priority"],
                status=properties["status"],
                owner=properties["owner"],
                coordinates=(float(coordinates[0]), float(coordinates[1])),
            )
        )
    return features


def load_inspection_routes(routes_path: Path = DEFAULT_ROUTES_PATH) -> list[InspectionRoute]:
    routes: list[InspectionRoute] = []
    with routes_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            station_ids = tuple(part.strip() for part in row["stationIds"].split("|") if part.strip())
            routes.append(
                InspectionRoute(
                    route_id=row["routeId"],
                    region=row["region"],
                    map_scale=int(row["mapScale"]),
                    layout_name=row["layoutName"],
                    station_ids=station_ids,
                )
            )
    return routes


def _build_layer_summary(features: list[StationFeature]) -> dict[str, Any]:
    category_counts = Counter(feature.category for feature in features)
    status_counts = Counter(feature.status for feature in features)
    region_counts = Counter(feature.region for feature in features)
    return {
        "station_count": len(features),
        "categories": dict(sorted(category_counts.items())),
        "statuses": dict(sorted(status_counts.items())),
        "regions": dict(sorted(region_counts.items())),
    }


def _build_map_themes(features: list[StationFeature]) -> list[dict[str, Any]]:
    category_groups = defaultdict(list)
    for feature in features:
        category_groups[feature.category].append(feature.feature_id)

    themes = [
        {
            "name": "Alert Review",
            "description": "Highlights high-priority or alert stations for the morning analyst pass.",
            "featureIds": [
                feature.feature_id for feature in features if feature.status == "alert" or feature.priority == "high"
            ],
        },
        {
            "name": "Field Readiness",
            "description": "Shows stations due for inspection in route-oriented groups.",
            "featureIds": [feature.feature_id for feature in features if feature.status != "offline"],
        },
    ]

    for category, category_feature_ids in sorted(category_groups.items()):
        themes.append(
            {
                "name": f"{category.replace('_', ' ').title()} Review",
                "description": f"Focus theme for {category.replace('_', ' ')} operations.",
                "featureIds": category_feature_ids,
            }
        )

    return themes


def _build_bookmarks(features: list[StationFeature]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for feature in features:
        grouped[feature.region].append(feature.coordinates)

    bookmarks = []
    for region, coordinates in sorted(grouped.items()):
        longitudes = [coordinate[0] for coordinate in coordinates]
        latitudes = [coordinate[1] for coordinate in coordinates]
        bookmarks.append(
            {
                "name": f"{region} Operations",
                "center": {
                    "longitude": round(sum(longitudes) / len(longitudes), 4),
                    "latitude": round(sum(latitudes) / len(latitudes), 4),
                },
                "extentPadding": 0.35,
                "featureCount": len(coordinates),
            }
        )
    return bookmarks


def _build_layout_jobs(features: list[StationFeature], routes: list[InspectionRoute]) -> list[dict[str, Any]]:
    features_by_id = {feature.feature_id: feature for feature in features}
    layout_jobs = []
    for route in routes:
        stations = [features_by_id[station_id].name for station_id in route.station_ids if station_id in features_by_id]
        layout_jobs.append(
            {
                "layoutName": route.layout_name,
                "region": route.region,
                "mapScale": route.map_scale,
                "routeId": route.route_id,
                "stationNames": stations,
            }
        )
    return layout_jobs


def _build_review_tasks(features: list[StationFeature]) -> list[dict[str, Any]]:
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    sorted_features = sorted(
        features,
        key=lambda feature: (priority_rank.get(feature.priority, 99), feature.status != "alert", feature.name),
    )
    tasks = []
    for feature in sorted_features:
        if feature.priority == "low" and feature.status == "normal":
            continue
        tasks.append(
            {
                "featureId": feature.feature_id,
                "stationName": feature.name,
                "region": feature.region,
                "category": feature.category,
                "priority": feature.priority,
                "status": feature.status,
                "recommendedAction": _recommended_action(feature),
            }
        )
    return tasks


def _recommended_action(feature: StationFeature) -> str:
    if feature.status == "alert":
        return f"Validate the latest {feature.category.replace('_', ' ')} readings and export a reviewer map for {feature.region}."
    if feature.status == "offline":
        return "Confirm device connectivity and flag the station for the next field route."
    return "Include in the next standard inspection packet and verify recent field notes."


def build_workbench_pack(project_root: Path = PROJECT_ROOT, project_name: str = "QGIS Operations Workbench") -> dict[str, Any]:
    points_path = project_root / "data" / "station_review_points.geojson"
    routes_path = project_root / "data" / "inspection_routes.csv"
    features = load_station_features(points_path)
    routes = load_inspection_routes(routes_path)
    return {
        "projectName": project_name,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": _build_layer_summary(features),
        "layers": [
            {
                "name": "station_review_points",
                "source": str(points_path.relative_to(project_root)).replace("\\", "/"),
                "geometryType": "Point",
            }
        ],
        "mapThemes": _build_map_themes(features),
        "bookmarks": _build_bookmarks(features),
        "layoutJobs": _build_layout_jobs(features, routes),
        "reviewTasks": _build_review_tasks(features),
        "notes": [
            "Designed as a public-safe QGIS-oriented operations pack.",
            "The checked-in data is sample monitoring content for repeatable portfolio review.",
            "Project XML, layout templates, and PyQGIS automation can build on this pack in later iterations.",
        ],
    }


def export_workbench_pack(output_dir: Path = DEFAULT_OUTPUT_DIR, project_name: str = "QGIS Operations Workbench") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pack = build_workbench_pack(PROJECT_ROOT, project_name=project_name)
    output_path = output_dir / "qgis_workbench_pack.json"
    output_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sample QGIS operations workbench pack.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated JSON output.")
    parser.add_argument("--project-name", default="QGIS Operations Workbench", help="Display name embedded in the output pack.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_workbench_pack(args.output_dir, project_name=args.project_name)
    print(f"Wrote workbench pack to {output_path}")


if __name__ == "__main__":
    main()
