from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


CHART_DPI = 300
PALETTE = {
    "ink": "#1F2A44",
    "blue": "#4C78A8",
    "teal": "#4E9F8A",
    "ochre": "#C58B39",
    "rose": "#B75D69",
    "grid": "#D9DEE7",
    "band": "#AFC5E4",
}
METHOD_LABELS = {
    "RegressionAdjustmentEstimator": "Regression adjustment",
    "PropensityMatcher": "Propensity matching",
    "IPWEstimator": "IPW",
    "DoublyRobustEstimator": "Doubly robust",
    "DifferenceInDifferences": "Difference-in-differences",
}


def results_to_frame(results: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        diagnostics = result["diagnostics"]
        balance_before = diagnostics["balance_before"]
        balance_after = diagnostics["balance_after"]
        rows.append(
            {
                "method": result["method"],
                "estimand": result["estimand"],
                "effect": result["effect"],
                "se": result.get("se"),
                "p_value": result.get("p_value"),
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "overlap_ok": diagnostics["overlap_ok"],
                "mean_abs_balance_before": sum(abs(value) for value in balance_before.values()) / len(balance_before),
                "mean_abs_balance_after": sum(abs(value) for value in balance_after.values()) / len(balance_after),
                "ess_treated": diagnostics.get("ess_treated"),
                "ess_control": diagnostics.get("ess_control"),
            }
        )
    return pd.DataFrame(rows)


def subgroup_to_frame(subgroups: list[dict]) -> pd.DataFrame:
    if not subgroups:
        return pd.DataFrame(columns=["subgroup", "rows", "treated_count", "control_count", "effect", "ci_low", "ci_high"])
    return pd.DataFrame(subgroups)


def sensitivity_to_frame(summary: dict) -> pd.DataFrame:
    scenarios = summary.get("scenarios", [])
    if not scenarios:
        return pd.DataFrame(columns=["bias", "adjusted_effect", "adjusted_ci_low", "adjusted_ci_high"])
    return pd.DataFrame(scenarios)


def export_dataset_artifacts(dataset_key: str, payload: dict, output_dir: Path) -> None:
    charts_dir = output_dir / "charts"
    tables_dir = output_dir / "tables"
    charts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    results_frame = results_to_frame(payload["results"])
    sensitivity_frame = sensitivity_to_frame(payload["primary_sensitivity"])
    subgroup_frame = subgroup_to_frame(payload["subgroups"])

    results_csv = tables_dir / f"{dataset_key}_estimator_summary.csv"
    subgroup_csv = tables_dir / f"{dataset_key}_subgroup_summary.csv"
    sensitivity_csv = tables_dir / f"{dataset_key}_sensitivity_summary.csv"
    results_md = tables_dir / f"{dataset_key}_estimator_summary.md"
    results_tex = tables_dir / f"{dataset_key}_estimator_summary.tex"

    results_frame.to_csv(results_csv, index=False)
    subgroup_frame.to_csv(subgroup_csv, index=False)
    sensitivity_frame.to_csv(sensitivity_csv, index=False)
    results_md.write_text(_frame_to_markdown(results_frame), encoding="utf-8")
    results_tex.write_text(_frame_to_latex(results_frame, caption=f"{_display_name(dataset_key)} estimator summary.", label=f"tab:{dataset_key}-estimators"), encoding="utf-8")

    _plot_estimator_comparison(
        results_frame,
        title=f"{_display_name(dataset_key)} estimator comparison",
        output_path=charts_dir / f"{dataset_key}_estimator_comparison.png",
    )
    _plot_balance_summary(
        results_frame,
        title=f"{_display_name(dataset_key)} balance summary",
        output_path=charts_dir / f"{dataset_key}_balance_summary.png",
    )
    _plot_sensitivity_summary(
        sensitivity_frame,
        title=f"{_display_name(dataset_key)} sensitivity curve",
        output_path=charts_dir / f"{dataset_key}_sensitivity_curve.png",
    )
    if not subgroup_frame.empty:
        _plot_subgroup_effects(
            subgroup_frame,
            title=f"{_display_name(dataset_key)} subgroup effects",
            output_path=charts_dir / f"{dataset_key}_subgroup_effects.png",
        )

    if payload["results"]:
        primary = payload["results"][-1]
        _plot_love(
            primary["diagnostics"]["balance_before"],
            primary["diagnostics"]["balance_after"],
            title=f"{_display_name(dataset_key)} covariate balance (Love plot)",
            output_path=charts_dir / f"{dataset_key}_love_plot.png",
        )


def export_benchmark_artifacts(report_payload: dict, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    benchmark_frame = benchmark_to_frame(report_payload)
    benchmark_csv = tables_dir / "cross_dataset_benchmark_summary.csv"
    benchmark_md = tables_dir / "cross_dataset_benchmark_summary.md"
    benchmark_tex = tables_dir / "cross_dataset_benchmark_summary.tex"
    benchmark_frame.to_csv(benchmark_csv, index=False)
    benchmark_md.write_text(_frame_to_markdown(benchmark_frame), encoding="utf-8")
    benchmark_tex.write_text(_frame_to_latex(benchmark_frame, caption="Cross-dataset benchmark summary.", label="tab:cross-dataset-benchmarks"), encoding="utf-8")


def benchmark_to_frame(report_payload: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_key, dataset in report_payload.items():
        if not isinstance(dataset, dict) or "results" not in dataset:
            continue
        results_frame = results_to_frame(dataset["results"]).copy()
        results_frame["dataset"] = dataset_key
        for _, row in results_frame.iterrows():
            ci_width = float(row["ci_high"] - row["ci_low"])
            rows.append(
                {
                    "dataset": dataset_key,
                    "method": row["method"],
                    "effect": float(row["effect"]),
                    "ci_width": ci_width,
                    "overlap_ok": bool(row["overlap_ok"]),
                    "mean_abs_balance_before": float(row["mean_abs_balance_before"]),
                    "mean_abs_balance_after": float(row["mean_abs_balance_after"]),
                    "balance_improvement": float(row["mean_abs_balance_before"] - row["mean_abs_balance_after"]),
                }
            )
    return pd.DataFrame(rows)


def _plot_estimator_comparison(results_frame: pd.DataFrame, title: str, output_path: Path) -> None:
    frame = results_frame.copy().sort_values("effect")
    effect = frame["effect"].astype(float).to_numpy()
    ci_low = frame["ci_low"].astype(float).to_numpy()
    ci_high = frame["ci_high"].astype(float).to_numpy()
    lower = (effect - ci_low).clip(min=0.0)
    upper = (ci_high - effect).clip(min=0.0)
    y_positions = np.arange(len(frame))
    labels = [_clean_method_name(method) for method in frame["method"]]
    fig, ax = plt.subplots(figsize=(8.6, max(4.6, 0.72 * len(frame) + 1.2)))
    _apply_publication_style(ax)
    ax.hlines(y_positions, ci_low, ci_high, color=PALETTE["ink"], linewidth=2.0, zorder=2)
    ax.scatter(effect, y_positions, s=60, color=PALETTE["blue"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(0.0, color=PALETTE["rose"], linestyle="--", linewidth=1.4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Estimated effect with 95% confidence interval")
    ax.set_title(title, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_balance_summary(results_frame: pd.DataFrame, title: str, output_path: Path) -> None:
    frame = results_frame.copy()
    frame["method_label"] = frame["method"].map(_clean_method_name)
    frame = frame.sort_values("mean_abs_balance_before", ascending=True)
    y_positions = np.arange(len(frame))
    before = frame["mean_abs_balance_before"].astype(float).to_numpy()
    after = frame["mean_abs_balance_after"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(8.6, max(4.6, 0.72 * len(frame) + 1.2)))
    _apply_publication_style(ax)
    ax.hlines(y_positions, after, before, color=PALETTE["grid"], linewidth=2.2, zorder=1)
    ax.scatter(before, y_positions, s=52, color=PALETTE["ochre"], label="Before adjustment", zorder=3)
    ax.scatter(after, y_positions, s=52, color=PALETTE["teal"], label="After adjustment", zorder=3)
    ax.axvline(0.1, color=PALETTE["rose"], linestyle=":", linewidth=1.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(frame["method_label"])
    ax.set_xlabel("Mean absolute standardized mean difference")
    ax.set_title(title, loc="left", pad=10)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_sensitivity_summary(sensitivity_frame: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    _apply_publication_style(ax)
    ax.plot(
        sensitivity_frame["bias"],
        sensitivity_frame["adjusted_effect"],
        marker="o",
        markersize=5.5,
        color=PALETTE["ink"],
        linewidth=2.2,
    )
    ax.fill_between(
        sensitivity_frame["bias"],
        sensitivity_frame["adjusted_ci_low"],
        sensitivity_frame["adjusted_ci_high"],
        color=PALETTE["band"],
        alpha=0.45,
    )
    ax.axhline(0.0, color=PALETTE["rose"], linestyle="--", linewidth=1.4)
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel("Additive hidden-bias shift")
    ax.set_ylabel("Adjusted effect")
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_subgroup_effects(subgroup_frame: pd.DataFrame, title: str, output_path: Path) -> None:
    frame = subgroup_frame.copy().sort_values("effect")
    effect = frame["effect"].astype(float).to_numpy()
    ci_low = frame["ci_low"].astype(float).to_numpy()
    ci_high = frame["ci_high"].astype(float).to_numpy()
    lower = (effect - ci_low).clip(min=0.0)
    upper = (ci_high - effect).clip(min=0.0)
    y_positions = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.72 * len(frame) + 1.2)))
    _apply_publication_style(ax)
    ax.hlines(y_positions, ci_low, ci_high, color=PALETTE["ink"], linewidth=2.0, zorder=2)
    ax.scatter(effect, y_positions, s=60, color=PALETTE["teal"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(0.0, color=PALETTE["rose"], linestyle="--", linewidth=1.4)
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel("Estimated effect with 95% confidence interval")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(frame["subgroup"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)


def export_propensity_overlap(
    propensity: "np.ndarray",
    treatment: "np.ndarray",
    title: str,
    output_path: Path,
) -> None:
    """Export a propensity-score overlap histogram by treatment group."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    _apply_publication_style(ax)
    bins = np.linspace(float(propensity.min()), float(propensity.max()), 28)
    ax.hist(
        propensity[treatment == 1],
        bins=bins,
        alpha=0.55,
        label="Treated",
        color=PALETTE["blue"],
        density=True,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        propensity[treatment == 0],
        bins=bins,
        alpha=0.55,
        label="Control",
        color=PALETTE["ochre"],
        density=True,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Propensity score")
    ax.set_ylabel("Density")
    ax.set_title(title, loc="left", pad=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)


def export_placebo_artifacts(placebo_results: list[dict], output_dir: Path) -> pd.DataFrame:
    """Export placebo/falsification test results to CSV."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(placebo_results)
    frame.to_csv(tables_dir / "placebo_test.csv", index=False)
    return frame


def export_rosenbaum_artifacts(rosenbaum_results: list[dict], output_dir: Path) -> pd.DataFrame:
    """Export Rosenbaum sensitivity bounds to CSV."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rosenbaum_results)
    frame.to_csv(tables_dir / "rosenbaum_bounds.csv", index=False)
    return frame


def _plot_love(
    balance_before: dict[str, float],
    balance_after: dict[str, float],
    title: str,
    output_path: Path,
) -> None:
    """Love plot: covariate-level |SMD| before and after adjustment."""
    covariates = sorted(balance_before.keys(), key=lambda covariate: abs(balance_before[covariate]), reverse=True)
    y_positions = list(range(len(covariates)))
    abs_before = [abs(balance_before[c]) for c in covariates]
    abs_after = [abs(balance_after[c]) for c in covariates]

    fig, ax = plt.subplots(figsize=(8.4, max(4.8, 0.52 * len(covariates) + 1.1)))
    _apply_publication_style(ax)
    for y_position, before_value, after_value in zip(y_positions, abs_before, abs_after):
        ax.hlines(y_position, min(before_value, after_value), max(before_value, after_value), color=PALETTE["grid"], linewidth=2.0, zorder=1)
    ax.scatter(abs_before, y_positions, marker="o", s=52, color=PALETTE["ochre"], label="Before adjustment", zorder=3)
    ax.scatter(abs_after, y_positions, marker="o", s=52, color=PALETTE["teal"], label="After adjustment", zorder=3)
    ax.axvline(0.1, color=PALETTE["rose"], linestyle="--", linewidth=1.2, label="|SMD| = 0.1")
    ax.axvline(0.25, color=PALETTE["rose"], linestyle=":", linewidth=1.0, alpha=0.7, label="|SMD| = 0.25")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(covariates)
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_title(title, loc="left", pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)


def _display_name(dataset_key: str) -> str:
    return dataset_key.replace("_", " ").title()


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    formatted = _format_table_frame(frame)
    columns = list(formatted.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, divider]
    for _, row in formatted.iterrows():
        values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows) + "\n"


def _frame_to_latex(frame: pd.DataFrame, caption: str, label: str) -> str:
    formatted = _format_table_frame(frame)
    return formatted.to_latex(index=False, escape=False, caption=caption, label=label)


def _format_table_frame(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in formatted.columns:
        if column == "method":
            formatted[column] = formatted[column].map(_clean_method_name)
            continue
        if pd.api.types.is_bool_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "Yes" if bool(value) else "No")
            continue
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: _format_numeric_value(column, value))
    return formatted


def _format_numeric_value(column: str, value: object) -> str:
    if pd.isna(value):
        return ""
    numeric_value = float(value)
    lower_column = column.lower()
    if any(token in lower_column for token in ["count", "rows", "ess"]):
        return f"{numeric_value:,.0f}" if numeric_value >= 100 else f"{numeric_value:.1f}".rstrip("0").rstrip(".")
    if "p_value" in lower_column:
        return "<0.001" if numeric_value < 0.001 else f"{numeric_value:.3f}"
    if "effect" in lower_column or "ci_" in lower_column:
        return f"{numeric_value:,.3f}" if abs(numeric_value) >= 100 else f"{numeric_value:.3f}"
    if "balance" in lower_column or "width" in lower_column or "bias" in lower_column or lower_column == "se":
        return f"{numeric_value:.3f}"
    return f"{numeric_value:.3f}"


def _clean_method_name(method: object) -> str:
    method_str = str(method)
    return METHOD_LABELS.get(method_str, method_str.replace("Estimator", "").replace("_", " "))


def _apply_publication_style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["ink"])
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax.yaxis.label.set_color(PALETTE["ink"])
    ax.title.set_color(PALETTE["ink"])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
