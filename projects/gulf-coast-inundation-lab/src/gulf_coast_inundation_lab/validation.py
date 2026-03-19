from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

from gulf_coast_inundation_lab.workflow_base import ReportWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "gauge_validation_sample.csv"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "gauge_validation_manifest.geojson"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_REGISTRY_NAME = "run_registry.json"
DEFAULT_CHART_DIR_NAME = "charts"


@dataclass(slots=True)
class ValidationRecord:
    gauge_id: str
    gauge_name: str
    date: str
    stage_ft: float
    percent_inundated: float


def _load_validation_records(input_path: Path) -> list[ValidationRecord]:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = [
            ValidationRecord(
                gauge_id=str(row["gaugeId"]).strip(),
                gauge_name=str(row["gaugeName"]).strip(),
                date=str(row["date"]).strip(),
                stage_ft=float(row["stageFt"]),
                percent_inundated=float(row["percentInundated"]),
            )
            for row in reader
        ]
    if not records:
        raise ValueError("Validation input must contain at least one record.")
    return records


def _coefficient_of_determination(stage_values: list[float], inundation_values: list[float]) -> tuple[float, float, float]:
    if len(stage_values) != len(inundation_values):
        raise ValueError("Stage and inundation series must have matching lengths.")
    if len(stage_values) < 2:
        raise ValueError("At least two observations are required to compute R^2.")

    mean_stage = mean(stage_values)
    mean_inundation = mean(inundation_values)
    variance_stage = sum((value - mean_stage) ** 2 for value in stage_values)
    variance_inundation = sum((value - mean_inundation) ** 2 for value in inundation_values)
    if variance_stage == 0 or variance_inundation == 0:
        return 0.0, 0.0, round(mean_inundation, 4)

    covariance = sum(
        (stage - mean_stage) * (inundation - mean_inundation)
        for stage, inundation in zip(stage_values, inundation_values, strict=True)
    )
    slope = covariance / variance_stage
    intercept = mean_inundation - slope * mean_stage
    predictions = [(slope * stage) + intercept for stage in stage_values]
    residual_sum = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(inundation_values, predictions, strict=True)
    )
    r_squared = max(0.0, min(1.0, 1.0 - (residual_sum / variance_inundation)))
    return round(r_squared, 4), round(slope, 4), round(intercept, 4)


def _validation_category(r_squared: float) -> str:
    if r_squared >= 0.8:
        return "very_strong"
    if r_squared >= 0.6:
        return "adequate"
    if r_squared >= 0.25:
        return "weak_to_moderate"
    return "poor"


def _plot_gauge_validation_footprint(chart_path: Path, manifest_path: Path = DEFAULT_MANIFEST_PATH) -> Path:
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    huc_colors = {
        "03": "#4c78a8",
        "08": "#f58518",
        "12": "#54a24b",
    }

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.set_facecolor("#f3f0e8")
    figure.patch.set_facecolor("#f3f0e8")

    for huc2 in sorted({feature["properties"]["huc2"] for feature in payload["features"]}):
        matching_features = [feature for feature in payload["features"] if feature["properties"]["huc2"] == huc2]
        axis.scatter(
            [float(feature["geometry"]["coordinates"][0]) for feature in matching_features],
            [float(feature["geometry"]["coordinates"][1]) for feature in matching_features],
            s=150,
            c=huc_colors.get(huc2, "#6d597a"),
            edgecolors="#1f2933",
            linewidths=0.8,
            label=f"HUC2 {huc2}",
        )

    for feature in payload["features"]:
        axis.annotate(
            feature["properties"]["gaugeId"],
            xy=(float(feature["geometry"]["coordinates"][0]), float(feature["geometry"]["coordinates"][1])),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#1f2933",
        )

    axis.set_title("Gauge Validation Footprint", fontsize=16, color="#1f2933")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(color="#d8d2c6", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False, loc="lower left")
    figure.tight_layout()
    figure.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return chart_path


def _plot_gauge_validation_ranking(gauges: list[dict[str, Any]], chart_path: Path) -> Path:
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    category_colors = {
        "very_strong": "#2a9d8f",
        "adequate": "#e9c46a",
        "weak_to_moderate": "#f4a261",
        "poor": "#e76f51",
    }

    labels = [str(gauge["gaugeId"]) for gauge in gauges]
    scores = [float(gauge["rSquared"]) for gauge in gauges]
    colors = [category_colors.get(str(gauge["category"]), "#6d597a") for gauge in gauges]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.set_facecolor("#f7f4ef")
    figure.patch.set_facecolor("#f7f4ef")
    bars = axis.bar(labels, scores, color=colors, edgecolor="#1f2933", linewidth=0.8)

    for bar, score in zip(bars, scores, strict=True):
        axis.text(
            bar.get_x() + (bar.get_width() / 2),
            score + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1f2933",
        )

    axis.set_ylim(0, 1.05)
    axis.set_title("Gauge Validation Ranking", fontsize=16, color="#1f2933")
    axis.set_xlabel("Gauge ID")
    axis.set_ylabel("R squared")
    axis.grid(axis="y", color="#d8d2c6", linewidth=0.7, alpha=0.7)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return chart_path


@dataclass(slots=True)
class GulfCoastValidationWorkflow(ReportWorkflow):
    input_path: Path = DEFAULT_INPUT_PATH
    report_name: str = "Gulf Coast Inundation Lab Validation Summary"
    run_label: str = "gauge-validation-review"
    registry_name: str = DEFAULT_REGISTRY_NAME

    @property
    def output_filename(self) -> str:
        return "gulf_coast_validation_summary.json"

    def build_report(self) -> dict[str, Any]:
        records = _load_validation_records(self.input_path)
        grouped: dict[str, list[ValidationRecord]] = {}
        for record in records:
            grouped.setdefault(record.gauge_id, []).append(record)

        gauges: list[dict[str, Any]] = []
        for gauge_id, gauge_records in grouped.items():
            stage_values = [record.stage_ft for record in gauge_records]
            inundation_values = [record.percent_inundated for record in gauge_records]
            r_squared, slope, intercept = _coefficient_of_determination(stage_values, inundation_values)
            gauges.append(
                {
                    "gaugeId": gauge_id,
                    "gaugeName": gauge_records[0].gauge_name,
                    "observationCount": len(gauge_records),
                    "stageRangeFt": round(max(stage_values) - min(stage_values), 3),
                    "inundationRangePct": round(max(inundation_values) - min(inundation_values), 3),
                    "rSquared": r_squared,
                    "slope": slope,
                    "intercept": intercept,
                    "category": _validation_category(r_squared),
                }
            )

        gauges.sort(key=lambda gauge: (-float(gauge["rSquared"]), gauge["gaugeId"]))
        adequate_gauges = [gauge for gauge in gauges if float(gauge["rSquared"]) >= 0.6]
        very_strong_gauges = [gauge for gauge in gauges if float(gauge["rSquared"]) >= 0.8]
        mean_r_squared = round(mean(float(gauge["rSquared"]) for gauge in gauges), 4)

        return {
            "reportName": self.report_name,
            "experiment": {
                "runLabel": self.run_label,
                "generatedAt": datetime.now(UTC).isoformat(),
                "inputFile": self.input_path.name,
                "registryFile": self.registry_name,
                "adequateThreshold": 0.6,
            },
            "summary": {
                "gaugeCount": len(gauges),
                "observationCount": len(records),
                "adequateGaugeCount": len(adequate_gauges),
                "veryStrongGaugeCount": len(very_strong_gauges),
                "meanRSquared": mean_r_squared,
                "strongestGaugeId": gauges[0]["gaugeId"],
                "weakestGaugeId": gauges[-1]["gaugeId"],
            },
            "gauges": gauges,
            "notes": [
                "This report summarizes merged gauge-stage and percent-inundated observations after Earth Engine export.",
                "An R^2 value above 0.6 is treated as adequate validation in line with the study framing.",
                "Low-support or weakly correlated gauges should be reviewed alongside observation counts and local scene coverage.",
            ],
        }

    def build_registry_entry(self, report: dict[str, Any], output_path: Path) -> dict[str, Any]:
        return {
            "runLabel": report["experiment"]["runLabel"],
            "generatedAt": report["experiment"]["generatedAt"],
            "reportName": report["reportName"],
            "reportFile": output_path.name,
            "gaugeCount": report["summary"]["gaugeCount"],
            "adequateGaugeCount": report["summary"]["adequateGaugeCount"],
            "meanRSquared": report["summary"]["meanRSquared"],
        }

    def export_report(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = self.build_report()
        ranking_chart_path = _plot_gauge_validation_ranking(
            report["gauges"],
            output_dir / DEFAULT_CHART_DIR_NAME / "gauge-validation-ranking.png",
        )
        footprint_chart_path = _plot_gauge_validation_footprint(
            output_dir / DEFAULT_CHART_DIR_NAME / "gauge-validation-footprint.png",
        )
        report["artifacts"] = {
            "charts": [
                {
                    "name": "gauge_validation_ranking",
                    "chart": str(ranking_chart_path.relative_to(output_dir)).replace("\\", "/"),
                },
                {
                    "name": "gauge_validation_footprint",
                    "chart": str(footprint_chart_path.relative_to(output_dir)).replace("\\", "/"),
                }
            ]
        }
        output_path = output_dir / self.output_filename
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._update_run_registry(output_dir, self.build_registry_entry(report, output_path))
        return output_path


def build_validation_report(input_path: Path = DEFAULT_INPUT_PATH) -> dict[str, Any]:
    workflow = GulfCoastValidationWorkflow(input_path=input_path)
    return workflow.build_report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Gulf Coast gauge-validation summary from merged AOI export data.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="Merged CSV containing gauge stage and percent inundated values.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated JSON output.")
    parser.add_argument("--report-name", default="Gulf Coast Inundation Lab Validation Summary", help="Display name embedded in the output report.")
    parser.add_argument("--run-label", default="gauge-validation-review", help="Label stored with the report output.")
    parser.add_argument("--registry-name", default=DEFAULT_REGISTRY_NAME, help="Name of the JSON file used to store appended run metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow = GulfCoastValidationWorkflow(
        input_path=args.input_path,
        report_name=args.report_name,
        run_label=args.run_label,
        registry_name=args.registry_name,
    )
    output_path = workflow.export_report(args.output_dir)
    print(f"Wrote Gulf Coast validation summary to {output_path}")


__all__ = [
    "GulfCoastValidationWorkflow",
    "ValidationRecord",
    "build_validation_report",
    "main",
]


if __name__ == "__main__":
    main()