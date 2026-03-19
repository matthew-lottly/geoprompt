from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pywt

from arroyo_flood_forecasting_lab.charts import export_chart_pack, export_comparison_chart
from arroyo_flood_forecasting_lab.summary_page import render_comparison_summary, render_review_summary
from arroyo_flood_forecasting_lab.workflow_base import ReportWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "arroyo_stage_series.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_REGISTRY_NAME = "run_registry.json"
DEFAULT_FORECAST_HORIZON = 12
DEFAULT_MAX_ORDER = 6
DEFAULT_SIMULATION_COUNT = 250
DEFAULT_WAVELET = "db4"
DEFAULT_LEVEL = 2
DEFAULT_CHART_DIRNAME = "charts"
DEFAULT_SECONDARY_DATA_PATH = PROJECT_ROOT / "data" / "oso_creek_stage_series.json"
DEFAULT_COMPARISON_REPORT_NAME = "multi_site_comparison.json"


@dataclass(slots=True)
class SeriesData:
    series_name: str
    timestamps: list[str]
    stage_values: np.ndarray
    data_source: str
    source_url: str
    site_name: str
    site_code: str
    parameter_name: str
    latitude: float
    longitude: float
    review_threshold_ft: float


@dataclass(slots=True)
class ArModelFit:
    order: int
    intercept: float
    coefficients: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    residual_std: float
    fit_percent: float


@dataclass(slots=True)
class CandidateOrderResult:
    order: int
    pmse: float
    fit_percent: float
    residual_std: float
    forecast: list[float]
    coefficients: list[float]
    intercept: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "pmse": self.pmse,
            "fitPercent": self.fit_percent,
            "residualStd": self.residual_std,
            "forecast": self.forecast,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
        }


@dataclass(slots=True)
class CandidateOrderReview:
    candidates: list[CandidateOrderResult]

    @property
    def best(self) -> CandidateOrderResult:
        return self.candidates[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bestOrder": self.best.order,
            "bestPmse": self.best.pmse,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(slots=True)
class SiteComparisonEntry:
    site_name: str
    site_code: str
    series_name: str
    observation_count: int
    hour_to_hour_volatility: float
    stage_range: float
    raw_pmse: float
    denoised_pmse: float
    wavelet_pmse_benefit: float
    winning_signal: str
    raw_best_order: int
    denoised_best_order: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "siteName": self.site_name,
            "siteCode": self.site_code,
            "seriesName": self.series_name,
            "observationCount": self.observation_count,
            "hourToHourVolatility": self.hour_to_hour_volatility,
            "stageRange": self.stage_range,
            "rawPmse": self.raw_pmse,
            "denoisedPmse": self.denoised_pmse,
            "waveletPmseBenefit": self.wavelet_pmse_benefit,
            "winningSignal": self.winning_signal,
            "rawBestOrder": self.raw_best_order,
            "denoisedBestOrder": self.denoised_best_order,
        }


def load_series(path: Path) -> SeriesData:
    payload = json.loads(path.read_text(encoding="utf-8"))
    start = datetime.fromisoformat(str(payload["startTimestamp"]).replace("Z", "+00:00"))
    frequency_hours = int(payload.get("frequencyHours", 1))
    stage_values = np.array([float(value) for value in payload["stageFt"]], dtype=float)
    timestamps = [
        (start + timedelta(hours=index * frequency_hours)).astimezone(UTC).isoformat().replace("+00:00", "Z")
        for index in range(len(stage_values))
    ]
    return SeriesData(
        series_name=str(payload["seriesName"]),
        timestamps=timestamps,
        stage_values=stage_values,
        data_source=str(payload.get("dataSource", "Unknown source")),
        source_url=str(payload.get("sourceUrl", "")),
        site_name=str(payload.get("siteName", payload["seriesName"])),
        site_code=str(payload.get("siteCode", "unknown")),
        parameter_name=str(payload.get("parameterName", "Gage height, ft")),
        latitude=float(payload.get("latitude", 0.0)),
        longitude=float(payload.get("longitude", 0.0)),
        review_threshold_ft=float(payload.get("reviewThresholdFt", 0.0)),
    )


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _pmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return round(float(np.mean((actual - predicted) ** 2)), 4)


def _fit_percent(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if denominator == 0:
        return 100.0
    explained = 1.0 - float(np.sum((actual - predicted) ** 2)) / denominator
    return round(explained * 100.0, 2)


def _autocorrelation(series: np.ndarray, max_lag: int) -> list[float]:
    centered = series - np.mean(series)
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return [0.0 for _ in range(max_lag)]
    correlations: list[float] = []
    for lag in range(1, max_lag + 1):
        numerator = float(np.dot(centered[:-lag], centered[lag:]))
        correlations.append(round(numerator / denominator, 4))
    return correlations


def _partial_autocorrelation(series: np.ndarray, max_lag: int) -> list[float]:
    acf = [1.0, *_autocorrelation(series, max_lag)]
    phi = np.zeros((max_lag + 1, max_lag + 1), dtype=float)
    variance = np.zeros(max_lag + 1, dtype=float)
    variance[0] = 1.0
    pacf: list[float] = []
    for lag in range(1, max_lag + 1):
        adjustment = sum(phi[lag - 1, index] * acf[lag - index] for index in range(1, lag))
        phi[lag, lag] = (acf[lag] - adjustment) / variance[lag - 1]
        for index in range(1, lag):
            phi[lag, index] = phi[lag - 1, index] - phi[lag, lag] * phi[lag - 1, lag - index]
        variance[lag] = variance[lag - 1] * (1.0 - phi[lag, lag] ** 2)
        pacf.append(round(float(phi[lag, lag]), 4))
    return pacf


def _dominant_lags(values: list[float], max_items: int = 3) -> list[dict[str, float]]:
    ranked = sorted(enumerate(values, start=1), key=lambda item: abs(item[1]), reverse=True)
    return [{"lag": lag, "value": round(value, 4)} for lag, value in ranked[:max_items]]


def _denoise_series(series: np.ndarray, wavelet_name: str, requested_level: int) -> tuple[np.ndarray, int, float]:
    coeffs = pywt.wavedec(series, wavelet_name, mode="symmetric", level=max(1, requested_level))  # pyright: ignore[reportAttributeAccessIssue]
    level = len(coeffs) - 1
    sigma = float(np.median(np.abs(coeffs[-1])) / 0.6745) if coeffs[-1].size else 0.0
    threshold = sigma * sqrt(2.0 * np.log(series.size)) if sigma else 0.0
    denoised_coeffs = [coeffs[0], *[pywt.threshold(detail, threshold, mode="soft") for detail in coeffs[1:]]]
    denoised = pywt.waverec(denoised_coeffs, wavelet_name, mode="symmetric")[: series.size]  # pyright: ignore[reportAttributeAccessIssue]
    return denoised.astype(float), level, round(threshold, 4)


def _fit_ar_model(series: np.ndarray, order: int) -> ArModelFit:
    if series.size <= order:
        raise ValueError("Series length must be greater than the autoregressive order.")
    targets = series[order:]
    lag_columns = [series[order - lag : series.size - lag] for lag in range(1, order + 1)]
    design_matrix = np.column_stack([np.ones(targets.size), *lag_columns])
    solution, _, _, _ = np.linalg.lstsq(design_matrix, targets, rcond=None)
    fitted_values = design_matrix @ solution
    residuals = targets - fitted_values
    return ArModelFit(
        order=order,
        intercept=float(solution[0]),
        coefficients=solution[1:].astype(float),
        fitted_values=fitted_values.astype(float),
        residuals=residuals.astype(float),
        residual_std=float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0,
        fit_percent=_fit_percent(targets, fitted_values),
    )


def _forecast_with_ar(model: ArModelFit, history: np.ndarray, steps: int, noise: np.ndarray | None = None) -> np.ndarray:
    window = [float(value) for value in history]
    forecasts: list[float] = []
    for index in range(steps):
        lags = np.array(window[-model.order :][::-1], dtype=float)
        next_value = model.intercept + float(np.dot(model.coefficients, lags))
        if noise is not None:
            next_value += float(noise[index])
        forecasts.append(next_value)
        window.append(next_value)
    return np.array(forecasts, dtype=float)


def _evaluate_candidate_orders(series: np.ndarray, forecast_horizon: int, max_order: int) -> CandidateOrderReview:
    train = series[:-forecast_horizon]
    holdout = series[-forecast_horizon:]
    candidates: list[CandidateOrderResult] = []
    for order in range(1, max_order + 1):
        model = _fit_ar_model(train, order)
        forecast = _forecast_with_ar(model, train, forecast_horizon)
        candidates.append(
            CandidateOrderResult(
                order=order,
                pmse=_pmse(holdout, forecast),
                fit_percent=model.fit_percent,
                residual_std=round(model.residual_std, 4),
                forecast=[round(float(value), 3) for value in forecast],
                coefficients=[round(float(value), 4) for value in model.coefficients],
                intercept=round(model.intercept, 4),
            )
        )
    candidates.sort(key=lambda item: (item.pmse, -item.fit_percent, item.order))
    return CandidateOrderReview(candidates=candidates)


def _monte_carlo_summary(
    model: ArModelFit,
    history: np.ndarray,
    forecast_horizon: int,
    simulation_count: int,
    review_threshold_ft: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    shocks = rng.normal(0.0, model.residual_std, size=(simulation_count, forecast_horizon))
    paths = np.array([
        _forecast_with_ar(model, history, forecast_horizon, noise=shock)
        for shock in shocks
    ])
    median = np.median(paths, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    exceedance = np.mean(paths >= review_threshold_ft, axis=0)
    return {
        "simulationCount": simulation_count,
        "medianForecast": [round(float(value), 3) for value in median],
        "p10Forecast": [round(float(value), 3) for value in p10],
        "p90Forecast": [round(float(value), 3) for value in p90],
        "reviewThresholdExceedanceProbability": [round(float(value), 3) for value in exceedance],
        "maximumSimulatedStage": round(float(np.max(paths)), 3),
    }


def _hour_to_hour_volatility(series: np.ndarray) -> float:
    if series.size < 2:
        return 0.0
    return round(float(np.std(np.diff(series), ddof=1)), 4)


def build_site_comparison(
    data_paths: list[Path],
    forecast_horizon: int,
    max_order: int,
    wavelet_name: str,
    wavelet_level: int,
) -> dict[str, Any]:
    sites: list[SiteComparisonEntry] = []
    for data_path in data_paths:
        if not data_path.exists():
            continue
        series_data = load_series(data_path)
        denoised_values, _, _ = _denoise_series(series_data.stage_values, wavelet_name, wavelet_level)
        raw_review = _evaluate_candidate_orders(series_data.stage_values, forecast_horizon, max_order)
        denoised_review = _evaluate_candidate_orders(denoised_values, forecast_horizon, max_order)
        benefit = round(raw_review.best.pmse - denoised_review.best.pmse, 4)
        sites.append(
            SiteComparisonEntry(
                site_name=series_data.site_name,
                site_code=series_data.site_code,
                series_name=series_data.series_name,
                observation_count=int(series_data.stage_values.size),
                hour_to_hour_volatility=_hour_to_hour_volatility(series_data.stage_values),
                stage_range=round(float(np.max(series_data.stage_values) - np.min(series_data.stage_values)), 4),
                raw_pmse=raw_review.best.pmse,
                denoised_pmse=denoised_review.best.pmse,
                wavelet_pmse_benefit=benefit,
                winning_signal="denoised" if benefit >= 0 else "raw",
                raw_best_order=raw_review.best.order,
                denoised_best_order=denoised_review.best.order,
            )
        )

    sites.sort(key=lambda site: site.site_code)
    if not sites:
        return {"summary": {"siteCount": 0, "interpretation": "No comparison sites were available."}, "sites": []}

    noisiest_site = max(sites, key=lambda site: site.hour_to_hour_volatility)
    largest_benefit_site = max(sites, key=lambda site: site.wavelet_pmse_benefit)
    helps_more_on_noisy_series = noisiest_site.site_code == largest_benefit_site.site_code
    interpretation = (
        "The noisiest site also shows the largest wavelet benefit in this sample."
        if helps_more_on_noisy_series
        else "The noisiest site does not show the largest wavelet benefit in this sample."
    )
    return {
        "summary": {
            "siteCount": len(sites),
            "noisiestSiteCode": noisiest_site.site_code,
            "largestDenoisingBenefitSiteCode": largest_benefit_site.site_code,
            "denoisingHelpsMoreOnNoisySeries": helps_more_on_noisy_series,
            "interpretation": interpretation,
        },
        "sites": [site.to_dict() for site in sites],
    }


@dataclass(slots=True)
class ArroyoFloodForecastLab(ReportWorkflow):
    data_path: Path = DEFAULT_DATA_PATH
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON
    max_order: int = DEFAULT_MAX_ORDER
    simulation_count: int = DEFAULT_SIMULATION_COUNT
    wavelet_name: str = DEFAULT_WAVELET
    wavelet_level: int = DEFAULT_LEVEL
    report_name: str = "Arroyo Flood Forecasting Lab"
    run_label: str = "wavelet-ar-review"
    registry_name: str = DEFAULT_REGISTRY_NAME

    @property
    def output_filename(self) -> str:
        return "arroyo_flood_forecast_report.json"

    def load_series(self) -> SeriesData:
        return load_series(self.data_path)

    def _build_candidate_reviews(self, series_data: SeriesData, denoised_values: np.ndarray) -> tuple[CandidateOrderReview, CandidateOrderReview]:
        return (
            _evaluate_candidate_orders(series_data.stage_values, self.forecast_horizon, self.max_order),
            _evaluate_candidate_orders(denoised_values, self.forecast_horizon, self.max_order),
        )

    def _select_branch(
        self,
        series_data: SeriesData,
        denoised_values: np.ndarray,
        raw_review: CandidateOrderReview,
        denoised_review: CandidateOrderReview,
    ) -> tuple[str, np.ndarray, int, ArModelFit]:
        selected_series = "denoised" if denoised_review.best.pmse <= raw_review.best.pmse else "raw"
        selected_values = denoised_values if selected_series == "denoised" else series_data.stage_values
        selected_order = denoised_review.best.order if selected_series == "denoised" else raw_review.best.order
        selected_train = selected_values[:-self.forecast_horizon]
        return selected_series, selected_values, selected_order, _fit_ar_model(selected_train, selected_order)

    def _build_summary(
        self,
        series_data: SeriesData,
        selected_series: str,
        selected_order: int,
        raw_review: CandidateOrderReview,
        denoised_review: CandidateOrderReview,
    ) -> dict[str, Any]:
        threshold_exceedance_count = int(np.sum(series_data.stage_values >= series_data.review_threshold_ft))
        pmse_improvement = round(raw_review.best.pmse - denoised_review.best.pmse, 4)
        pmse_improvement_percent = round((pmse_improvement / raw_review.best.pmse) * 100.0, 2) if raw_review.best.pmse else 0.0
        return {
            "seriesName": series_data.series_name,
            "sourceSiteName": series_data.site_name,
            "sourceSiteCode": series_data.site_code,
            "dataSource": series_data.data_source,
            "observationCount": int(series_data.stage_values.size),
            "calibrationCount": int(series_data.stage_values.size - self.forecast_horizon),
            "holdoutCount": self.forecast_horizon,
            "selectedSeries": selected_series,
            "selectedOrder": selected_order,
            "rawPmse": raw_review.best.pmse,
            "denoisedPmse": denoised_review.best.pmse,
            "pmseImprovement": pmse_improvement,
            "pmseImprovementPercent": pmse_improvement_percent,
            "peakStageFt": round(float(np.max(series_data.stage_values)), 3),
            "meanStageFt": round(_mean(series_data.stage_values), 3),
            "reviewThresholdFt": round(series_data.review_threshold_ft, 3),
            "thresholdExceedanceCount": threshold_exceedance_count,
        }

    def _build_lag_diagnostics(self, series_data: SeriesData, denoised_values: np.ndarray) -> dict[str, Any]:
        raw_acf = _autocorrelation(series_data.stage_values, self.max_order)
        raw_pacf = _partial_autocorrelation(series_data.stage_values, self.max_order)
        denoised_acf = _autocorrelation(denoised_values, self.max_order)
        denoised_pacf = _partial_autocorrelation(denoised_values, self.max_order)
        return {
            "raw": {
                "acf": raw_acf,
                "pacf": raw_pacf,
                "dominantAcfLags": _dominant_lags(raw_acf),
                "dominantPacfLags": _dominant_lags(raw_pacf),
            },
            "denoised": {
                "acf": denoised_acf,
                "pacf": denoised_pacf,
                "dominantAcfLags": _dominant_lags(denoised_acf),
                "dominantPacfLags": _dominant_lags(denoised_pacf),
            },
        }

    def _build_source_context(self, series_data: SeriesData) -> dict[str, Any]:
        return {
            "siteName": series_data.site_name,
            "siteCode": series_data.site_code,
            "dataSource": series_data.data_source,
            "sourceUrl": series_data.source_url,
            "latitude": round(series_data.latitude, 6),
            "longitude": round(series_data.longitude, 6),
            "publicModelingNote": "This public South Texas gauge is used as a reproducible analog for an Arroyo-style forecasting workflow.",
        }

    def build_report(self) -> dict[str, Any]:
        series_data = self.load_series()
        if series_data.stage_values.size <= self.forecast_horizon + self.max_order:
            raise ValueError("Series is too short for the requested forecast horizon and AR order review.")

        denoised_values, level_used, threshold = _denoise_series(
            series_data.stage_values,
            self.wavelet_name,
            self.wavelet_level,
        )
        raw_review, denoised_review = self._build_candidate_reviews(series_data, denoised_values)
        selected_series, selected_values, selected_order, selected_model = self._select_branch(
            series_data,
            denoised_values,
            raw_review,
            denoised_review,
        )
        monte_carlo = _monte_carlo_summary(
            selected_model,
            selected_values[:-self.forecast_horizon],
            self.forecast_horizon,
            self.simulation_count,
            series_data.review_threshold_ft,
        )
        holdout_timestamps = series_data.timestamps[-self.forecast_horizon :]

        return {
            "reportName": self.report_name,
            "experiment": {
                "runLabel": self.run_label,
                "generatedAt": datetime.now(UTC).isoformat(),
                "registryFile": self.registry_name,
                "forecastHorizon": self.forecast_horizon,
                "candidateOrderCount": self.max_order,
                "simulationCount": self.simulation_count,
                "waveletName": self.wavelet_name,
                "waveletLevel": level_used,
                "threshold": threshold,
                "parameterName": series_data.parameter_name,
            },
            "summary": self._build_summary(series_data, selected_series, selected_order, raw_review, denoised_review),
            "sourceContext": self._build_source_context(series_data),
            "hydrographProfile": {
                "startTimestamp": series_data.timestamps[0],
                "endTimestamp": series_data.timestamps[-1],
                "holdoutWindow": holdout_timestamps,
                "rawStageTail": [round(float(value), 3) for value in series_data.stage_values[-self.forecast_horizon :]],
                "denoisedStageTail": [round(float(value), 3) for value in denoised_values[-self.forecast_horizon :]],
            },
            "lagDiagnostics": self._build_lag_diagnostics(series_data, denoised_values),
            "candidateModels": {
                "raw": raw_review.to_dict(),
                "denoised": denoised_review.to_dict(),
            },
            "monteCarlo": monte_carlo,
            "notes": [
                "This lab recreates the structure of a flood-forecasting case study with a real public USGS stage series from South Texas.",
                "The denoised branch preserves the original method story by comparing wavelet-preprocessed input against the raw signal.",
                "The review threshold is a reproducible percentile-based benchmark for scenario discussion, not an asserted official flood stage for Arroyo Colorado.",
                "Monte Carlo output is driven by residual uncertainty from the selected autoregressive fit rather than a single deterministic trace.",
            ],
        }

    def build_registry_entry(self, report: dict[str, Any], output_path: Path) -> dict[str, Any]:
        return {
            "runLabel": report["experiment"]["runLabel"],
            "generatedAt": report["experiment"]["generatedAt"],
            "reportName": report["reportName"],
            "reportFile": output_path.name,
            "selectedSeries": report["summary"]["selectedSeries"],
            "selectedOrder": report["summary"]["selectedOrder"],
            "rawPmse": report["summary"]["rawPmse"],
            "denoisedPmse": report["summary"]["denoisedPmse"],
            "chartCount": len(report.get("artifacts", {}).get("chartFiles", [])),
        }

    def export_report(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = self.build_report()
        series_data = self.load_series()
        denoised_values, _, _ = _denoise_series(
            series_data.stage_values,
            self.wavelet_name,
            self.wavelet_level,
        )
        chart_files = export_chart_pack(
            output_dir=output_dir,
            chart_dirname=DEFAULT_CHART_DIRNAME,
            report=report,
            timestamps=series_data.timestamps,
            raw_series=series_data.stage_values,
            denoised_series=denoised_values,
            review_threshold_ft=series_data.review_threshold_ft,
        )
        comparison_report = build_site_comparison(
            data_paths=[self.data_path, DEFAULT_SECONDARY_DATA_PATH],
            forecast_horizon=self.forecast_horizon,
            max_order=self.max_order,
            wavelet_name=self.wavelet_name,
            wavelet_level=self.wavelet_level,
        )
        comparison_report_path = output_dir / DEFAULT_COMPARISON_REPORT_NAME
        comparison_report_path.write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")
        comparison_chart_file = export_comparison_chart(output_dir, DEFAULT_CHART_DIRNAME, comparison_report)
        report["artifacts"] = {
            "reportFile": self.output_filename,
            "chartDirectory": DEFAULT_CHART_DIRNAME,
            "chartFiles": chart_files,
            "comparisonReportFile": comparison_report_path.name,
            "comparisonChartFile": comparison_chart_file,
        }
        summary_page_file = render_review_summary(output_dir, report, comparison_report)
        report["artifacts"]["summaryPage"] = summary_page_file
        comparison_summary_page = render_comparison_summary(output_dir, report, comparison_report)
        report["artifacts"]["comparisonSummaryPage"] = comparison_summary_page
        output_path = output_dir / self.output_filename
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._update_run_registry(output_dir, self.build_registry_entry(report, output_path))
        return output_path


def build_flood_report(
    data_path: Path = DEFAULT_DATA_PATH,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    max_order: int = DEFAULT_MAX_ORDER,
    simulation_count: int = DEFAULT_SIMULATION_COUNT,
    wavelet_name: str = DEFAULT_WAVELET,
    wavelet_level: int = DEFAULT_LEVEL,
    report_name: str = "Arroyo Flood Forecasting Lab",
    run_label: str = "wavelet-ar-review",
    registry_name: str = DEFAULT_REGISTRY_NAME,
) -> dict[str, Any]:
    lab = ArroyoFloodForecastLab(
        data_path=data_path,
        forecast_horizon=forecast_horizon,
        max_order=max_order,
        simulation_count=simulation_count,
        wavelet_name=wavelet_name,
        wavelet_level=wavelet_level,
        report_name=report_name,
        run_label=run_label,
        registry_name=registry_name,
    )
    return lab.build_report()


def export_flood_report(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    max_order: int = DEFAULT_MAX_ORDER,
    simulation_count: int = DEFAULT_SIMULATION_COUNT,
    wavelet_name: str = DEFAULT_WAVELET,
    wavelet_level: int = DEFAULT_LEVEL,
    report_name: str = "Arroyo Flood Forecasting Lab",
    run_label: str = "wavelet-ar-review",
    registry_name: str = DEFAULT_REGISTRY_NAME,
) -> Path:
    lab = ArroyoFloodForecastLab(
        forecast_horizon=forecast_horizon,
        max_order=max_order,
        simulation_count=simulation_count,
        wavelet_name=wavelet_name,
        wavelet_level=wavelet_level,
        report_name=report_name,
        run_label=run_label,
        registry_name=registry_name,
    )
    return lab.export_report(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a wavelet-assisted flood forecasting report.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated JSON output.")
    parser.add_argument("--forecast-horizon", type=int, default=DEFAULT_FORECAST_HORIZON, help="Trailing hours reserved for holdout scoring.")
    parser.add_argument("--max-order", type=int, default=DEFAULT_MAX_ORDER, help="Largest AR order reviewed for each signal.")
    parser.add_argument("--simulation-count", type=int, default=DEFAULT_SIMULATION_COUNT, help="Monte Carlo scenario count for the selected model.")
    parser.add_argument("--wavelet-name", default=DEFAULT_WAVELET, help="Wavelet family used for denoising.")
    parser.add_argument("--wavelet-level", type=int, default=DEFAULT_LEVEL, help="Requested decomposition level for denoising.")
    parser.add_argument("--report-name", default="Arroyo Flood Forecasting Lab", help="Display name embedded in the output report.")
    parser.add_argument("--run-label", default="wavelet-ar-review", help="Label stored with the experiment-style report output.")
    parser.add_argument("--registry-name", default=DEFAULT_REGISTRY_NAME, help="Name of the JSON file used to store appended run metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_flood_report(
        output_dir=args.output_dir,
        forecast_horizon=args.forecast_horizon,
        max_order=args.max_order,
        simulation_count=args.simulation_count,
        wavelet_name=args.wavelet_name,
        wavelet_level=args.wavelet_level,
        report_name=args.report_name,
        run_label=args.run_label,
        registry_name=args.registry_name,
    )
    print(f"Wrote arroyo flood forecast report to {output_path}")


if __name__ == "__main__":
    main()