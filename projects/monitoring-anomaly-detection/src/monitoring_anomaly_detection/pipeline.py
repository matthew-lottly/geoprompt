from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import matplotlib.pyplot as plt

from monitoring_anomaly_detection.workflow_base import ReportWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "station_observations.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHART_DIR_NAME = "charts"
DEFAULT_REGISTRY_NAME = "run_registry.json"
DEFAULT_WARMUP_WINDOW = 3
DETECTOR_THRESHOLDS = {
    "global_zscore": 1.35,
    "rolling_zscore": 1.2,
    "mad_score": 2.2,
    "delta_zscore": 1.25,
}


@dataclass(slots=True)
class ObservationRecord:
    station_id: str
    metric: str
    timestamp: str
    value: float
    is_known_event: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "stationId": self.station_id,
            "metric": self.metric,
            "timestamp": self.timestamp,
            "value": self.value,
            "isKnownEvent": self.is_known_event,
        }


@dataclass(slots=True)
class DetectorEvaluation:
    detector: str
    precision: float
    recall: float
    f1_score: float
    alert_count: int
    true_positives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "precision": self.precision,
            "recall": self.recall,
            "f1Score": self.f1_score,
            "alertCount": self.alert_count,
            "truePositives": self.true_positives,
            "falsePositives": self.false_positives,
            "falseNegatives": self.false_negatives,
        }


@dataclass(slots=True)
class ScoredEvent:
    station_id: str
    metric: str
    timestamp: str
    value: float
    is_known_event: bool
    scores: dict[str, float]
    flags: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stationId": self.station_id,
            "metric": self.metric,
            "timestamp": self.timestamp,
            "value": self.value,
            "isKnownEvent": self.is_known_event,
            "scores": self.scores,
            "flags": self.flags,
        }


@dataclass(slots=True)
class SelectedAlert:
    station_id: str
    metric: str
    timestamp: str
    value: float
    is_known_event: bool
    baseline_mean: float
    selected_score: float
    selected_detector: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stationId": self.station_id,
            "metric": self.metric,
            "timestamp": self.timestamp,
            "value": self.value,
            "isKnownEvent": self.is_known_event,
            "baselineMean": self.baseline_mean,
            "selectedScore": self.selected_score,
            "selectedDetector": self.selected_detector,
        }


def load_observations(path: Path) -> list[dict[str, Any]]:
    return [observation.to_dict() for observation in _load_observation_records(path)]


def _load_observation_records(path: Path) -> list[ObservationRecord]:
    with path.open(encoding="utf-8", newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    observations = [
        ObservationRecord(
            station_id=row["station_id"],
            metric=row["metric"],
            timestamp=row["timestamp"],
            value=float(row["value"]),
            is_known_event=row.get("is_known_event", "false").lower() == "true",
        )
        for row in rows
    ]
    observations.sort(key=lambda observation: (observation.station_id, observation.timestamp))
    return observations


def _safe_stddev(values: list[float]) -> float:
    return pstdev(values) or 1.0


def _mad_score(values: list[float], current_value: float) -> float:
    center = median(values)
    absolute_deviations = [abs(value - center) for value in values]
    mad = median(absolute_deviations)
    if mad == 0:
        return 0.0
    return 0.6745 * abs(current_value - center) / mad


def _delta_zscore(values: list[float], current_value: float) -> float:
    if len(values) < 2:
        return 0.0
    diffs = [later - earlier for earlier, later in zip(values[:-1], values[1:], strict=True)]
    current_delta = current_value - values[-1]
    return abs(current_delta - mean(diffs)) / _safe_stddev(diffs)


def _detector_scores(history: list[float], current_value: float, warmup_window: int) -> dict[str, float]:
    rolling_history = history[-warmup_window:]
    return {
        "global_zscore": abs(current_value - mean(history)) / _safe_stddev(history),
        "rolling_zscore": abs(current_value - mean(rolling_history)) / _safe_stddev(rolling_history),
        "mad_score": _mad_score(history, current_value),
        "delta_zscore": _delta_zscore(history, current_value),
    }


def _f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 2)


def _evaluate_detector(events: list[ScoredEvent], detector_name: str) -> DetectorEvaluation:
    true_positives = sum(event.is_known_event and event.flags[detector_name] for event in events)
    false_positives = sum((not event.is_known_event) and event.flags[detector_name] for event in events)
    false_negatives = sum(event.is_known_event and (not event.flags[detector_name]) for event in events)
    precision = round(true_positives / (true_positives + false_positives), 2) if true_positives + false_positives else 0.0
    recall = round(true_positives / (true_positives + false_negatives), 2) if true_positives + false_negatives else 0.0
    return DetectorEvaluation(
        detector=detector_name,
        precision=precision,
        recall=recall,
        f1_score=_f1_score(precision, recall),
        alert_count=sum(event.flags[detector_name] for event in events),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def _plot_selected_detector_ranking(output_dir: Path, selected_detector: str, ranked_events: list[dict[str, Any]]) -> str:
    chart_dir = output_dir / DEFAULT_CHART_DIR_NAME
    chart_dir.mkdir(parents=True, exist_ok=True)

    chart_path = chart_dir / "selected-detector-ranking.png"
    top_events = ranked_events[:6]
    labels = [f"{event['stationId']}\n{event['timestamp'][5:10]}" for event in top_events]
    scores = [event["scores"][selected_detector] for event in top_events]
    colors = ["#c76a2d" if event["isKnownEvent"] else "#3d6f8e" for event in top_events]

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(range(len(top_events)), scores, color=colors)
    axis.set_xticks(range(len(top_events)))
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel(f"{selected_detector} score")
    axis.set_title("Selected detector ranking review")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    return chart_path.relative_to(output_dir).as_posix()


@dataclass(slots=True)
class AnomalyDetectionWorkflow(ReportWorkflow):
    data_path: Path = DEFAULT_DATA_PATH
    report_name: str = "Monitoring Anomaly Detection"
    run_label: str = "detector-comparison-pass"
    warmup_window: int = DEFAULT_WARMUP_WINDOW
    registry_name: str = DEFAULT_REGISTRY_NAME

    @property
    def output_filename(self) -> str:
        return "anomaly_report.json"

    def load_observations(self) -> list[dict[str, Any]]:
        return load_observations(self.data_path)

    def load_observation_records(self) -> list[ObservationRecord]:
        return _load_observation_records(self.data_path)

    def _build_station_histories(
        self, observations: list[ObservationRecord]
    ) -> tuple[dict[str, list[ObservationRecord]], dict[str, list[float]]]:
        station_histories: dict[str, list[ObservationRecord]] = defaultdict(list)
        grouped_values: dict[str, list[float]] = defaultdict(list)
        for observation in observations:
            station_histories[observation.station_id].append(observation)
            grouped_values[observation.station_id].append(observation.value)
        return station_histories, grouped_values

    def _build_baselines(self, grouped_values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        return {
            station_id: {
                "mean": round(mean(values), 2),
                "stddev": round(_safe_stddev(values), 2),
            }
            for station_id, values in grouped_values.items()
        }

    def _build_scored_events(self, station_histories: dict[str, list[ObservationRecord]]) -> list[ScoredEvent]:
        scored_events: list[ScoredEvent] = []
        for station_id, station_observations in station_histories.items():
            for index, observation in enumerate(station_observations):
                if index < self.warmup_window:
                    continue
                history_values = [item.value for item in station_observations[:index]]
                scores = {
                    detector: round(score, 2)
                    for detector, score in _detector_scores(history_values, observation.value, self.warmup_window).items()
                }
                flags = {
                    detector: score >= DETECTOR_THRESHOLDS[detector]
                    for detector, score in scores.items()
                }
                scored_events.append(
                    ScoredEvent(
                        station_id=station_id,
                        metric=observation.metric,
                        timestamp=observation.timestamp,
                        value=observation.value,
                        is_known_event=observation.is_known_event,
                        scores=scores,
                        flags=flags,
                    )
                )
        return scored_events

    def _build_detector_leaderboard(self, scored_events: list[ScoredEvent]) -> list[DetectorEvaluation]:
        leaderboard = [_evaluate_detector(scored_events, detector) for detector in DETECTOR_THRESHOLDS]
        leaderboard.sort(key=lambda detector: (-detector.f1_score, -detector.precision, detector.detector))
        return leaderboard

    def _build_selected_alerts(
        self,
        scored_events: list[ScoredEvent],
        selected_detector: str,
        baselines: dict[str, dict[str, float]],
    ) -> tuple[list[SelectedAlert], dict[str, int]]:
        selected_alerts: list[SelectedAlert] = []
        station_alert_counts: dict[str, int] = defaultdict(int)
        for event in scored_events:
            if not event.flags[selected_detector]:
                continue
            baseline = baselines[event.station_id]
            selected_alerts.append(
                SelectedAlert(
                    station_id=event.station_id,
                    metric=event.metric,
                    timestamp=event.timestamp,
                    value=event.value,
                    is_known_event=event.is_known_event,
                    baseline_mean=baseline["mean"],
                    selected_score=event.scores[selected_detector],
                    selected_detector=selected_detector,
                )
            )
            station_alert_counts[event.station_id] += 1
        return selected_alerts, dict(sorted(station_alert_counts.items()))

    def build_report(self) -> dict[str, Any]:
        observations = self.load_observation_records()
        station_histories, grouped_values = self._build_station_histories(observations)
        baselines = self._build_baselines(grouped_values)
        scored_events = self._build_scored_events(station_histories)
        detector_leaderboard = self._build_detector_leaderboard(scored_events)
        selected_detector = detector_leaderboard[0].detector
        selected_alerts, station_alert_counts = self._build_selected_alerts(scored_events, selected_detector, baselines)
        detector_wins: Counter[str] = Counter()
        detector_wins[selected_detector] += 1
        scored_events.sort(key=lambda event: event.scores[selected_detector], reverse=True)
        selected_alerts.sort(key=lambda alert: alert.selected_score, reverse=True)

        return {
            "reportName": self.report_name,
            "experiment": {
                "runLabel": self.run_label,
                "generatedAt": datetime.now(UTC).isoformat(),
                "registryFile": self.registry_name,
                "warmupWindow": self.warmup_window,
                "detectorCount": len(DETECTOR_THRESHOLDS),
                "thresholds": DETECTOR_THRESHOLDS,
            },
            "summary": {
                "observationCount": len(observations),
                "stationCount": len(grouped_values),
                "scoredEventCount": len(scored_events),
                "knownEventCount": sum(observation.is_known_event for observation in observations),
                "selectedAlertCount": len(selected_alerts),
                "selectedDetector": selected_detector,
                "selectedDetectorF1": detector_leaderboard[0].f1_score,
                "detectorWins": dict(detector_wins),
            },
            "stationBaselines": baselines,
            "detectorLeaderboard": [detector.to_dict() for detector in detector_leaderboard],
            "rankedEvents": [event.to_dict() for event in scored_events],
            "selectedAlerts": [alert.to_dict() for alert in selected_alerts],
            "stationAlertCounts": station_alert_counts,
            "notes": [
                "Designed as a public-safe anomaly-detection workflow with detector comparison and experiment-style metadata.",
                "The selected detector is chosen by labeled-event F1 rather than a single hard-coded scoring rule.",
                "The same structure can later support richer event labels, rolling retraining, and external experiment tracking.",
            ],
        }

    def build_registry_entry(self, report: dict[str, Any], output_path: Path) -> dict[str, Any]:
        return {
            "runLabel": report["experiment"]["runLabel"],
            "generatedAt": report["experiment"]["generatedAt"],
            "reportName": report["reportName"],
            "reportFile": output_path.name,
            "stationCount": report["summary"]["stationCount"],
            "selectedDetector": report["summary"]["selectedDetector"],
            "selectedDetectorF1": report["summary"]["selectedDetectorF1"],
            "selectedAlertCount": report["summary"]["selectedAlertCount"],
        }

    def export_report(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = self.build_report()
        report["artifacts"] = {
            "charts": [
                {
                    "name": "selected_detector_ranking",
                    "chart": _plot_selected_detector_ranking(output_dir, report["summary"]["selectedDetector"], report["rankedEvents"]),
                }
            ]
        }
        output_path = output_dir / self.output_filename
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._update_run_registry(output_dir, self.build_registry_entry(report, output_path))
        return output_path


def build_anomaly_report(
    data_path: Path = DEFAULT_DATA_PATH,
    report_name: str = "Monitoring Anomaly Detection",
    run_label: str = "detector-comparison-pass",
    warmup_window: int = DEFAULT_WARMUP_WINDOW,
    registry_name: str = DEFAULT_REGISTRY_NAME,
) -> dict[str, Any]:
    workflow = AnomalyDetectionWorkflow(
        data_path=data_path,
        report_name=report_name,
        run_label=run_label,
        warmup_window=warmup_window,
        registry_name=registry_name,
    )
    return workflow.build_report()


def export_anomaly_report(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_name: str = "Monitoring Anomaly Detection",
    run_label: str = "detector-comparison-pass",
    warmup_window: int = DEFAULT_WARMUP_WINDOW,
    registry_name: str = DEFAULT_REGISTRY_NAME,
) -> Path:
    workflow = AnomalyDetectionWorkflow(
        report_name=report_name,
        run_label=run_label,
        warmup_window=warmup_window,
        registry_name=registry_name,
    )
    return workflow.export_report(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sample anomaly-detection report.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated JSON output.")
    parser.add_argument("--report-name", default="Monitoring Anomaly Detection", help="Display name embedded in the output report.")
    parser.add_argument("--run-label", default="detector-comparison-pass", help="Label stored with the experiment-style report output.")
    parser.add_argument("--registry-name", default=DEFAULT_REGISTRY_NAME, help="Name of the JSON file used to store appended run metadata.")
    parser.add_argument("--warmup-window", type=int, default=DEFAULT_WARMUP_WINDOW, help="Number of historical observations required before scoring a station event.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_anomaly_report(
        output_dir=args.output_dir,
        report_name=args.report_name,
        run_label=args.run_label,
        warmup_window=args.warmup_window,
        registry_name=args.registry_name,
    )
    print(f"Wrote anomaly report to {output_path}")


if __name__ == "__main__":
    main()