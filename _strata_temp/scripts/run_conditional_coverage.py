#!/usr/bin/env python3
"""Conditional coverage analysis: compare Mondrian vs CHMP width redistribution.

Produces CSV and console output showing:
- Coverage by width decile for Mondrian vs CHMP
- Coverage by node degree for Mondrian vs CHMP
- Width standard deviation comparison (redistribution evidence)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hetero_conformal.experiment import ExperimentConfig, run_experiment
from src.hetero_conformal.conformal import (
    HeteroConformalCalibrator,
    PropagationAwareCalibrator,
)
from src.hetero_conformal.diagnostics import (
    conditional_coverage_by_degree,
    conditional_coverage_by_width_decile,
)

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    seeds = [42, 123, 456]
    alpha = 0.10

    all_rows = []

    for seed in seeds:
        config = ExperimentConfig(
            n_power=200, n_water=150, n_telecom=100,
            feature_dim=8, hidden_dim=32, num_layers=2,
            epochs=100, lr=0.01, alpha=alpha, seed=seed,
        )

        # Run mondrian experiment
        config_mond = ExperimentConfig(**{**config.__dict__, "use_propagation_aware": False})
        exp_mond = run_experiment(config_mond, verbose=False)

        # Run CHMP experiment
        config_chmp = ExperimentConfig(**{**config.__dict__, "use_propagation_aware": True})
        exp_chmp = run_experiment(config_chmp, verbose=False)

        for method_name, exp_result in [("mondrian", exp_mond), ("chmp", exp_chmp)]:
            result = exp_result.conformal_result
            if result is None:
                print(f"  No conformal result for {method_name}, seed={seed}")
                continue

            # Build test labels
            test_labels = {}
            for nt in result.lower:
                test_labels[nt] = np.zeros(len(result.lower[nt]))

            # Width decile analysis (using labels=test node labels directly)
            # We need to get the actual graph to extract test labels
            # Re-run to get graph — generate same graph
            from src.hetero_conformal.graph import generate_synthetic_infrastructure
            np.random.seed(seed)
            graph = generate_synthetic_infrastructure(
                n_power=config.n_power, n_water=config.n_water, n_telecom=config.n_telecom,
                feature_dim=config.feature_dim, seed=seed,
            )

            test_masks_dict = {nt: graph.node_masks[nt]["test"] for nt in graph.node_masks}
            test_labels = {nt: graph.node_labels[nt][test_masks_dict[nt]] for nt in graph.node_labels}
            num_nodes = {nt: f.shape[0] for nt, f in graph.node_features.items()}

            decile_info = conditional_coverage_by_width_decile(
                result, test_labels, n_deciles=5,
            )

            degree_info = conditional_coverage_by_degree(
                result, graph.node_labels, test_masks_dict,
                graph.edge_index, num_nodes, n_bins=5,
            )

            # Width statistics
            all_widths = []
            for nt in result.lower:
                w = result.upper[nt] - result.lower[nt]
                all_widths.append(w)
            widths = np.concatenate(all_widths)

            row = {
                "seed": seed,
                "method": method_name,
                "mean_width": float(np.mean(widths)),
                "std_width": float(np.std(widths)),
                "cv_width": float(np.std(widths) / max(np.mean(widths), 1e-9)),
                "marginal_cov": exp_result.marginal_cov,
            }

            for i, cov in enumerate(decile_info["coverage"]):
                row[f"decile_{i+1}_cov"] = float(cov) if not np.isnan(cov) else ""

            all_rows.append(row)

            print(f"\nSeed={seed}, Method={method_name}")
            print(f"  Marginal cov: {exp_result.marginal_cov:.4f}")
            print(f"  Mean width: {row['mean_width']:.4f}, Std: {row['std_width']:.4f}, "
                  f"CV: {row['cv_width']:.4f}")
            print(f"  Decile coverage: {[f'{c:.3f}' for c in decile_info['coverage']]}")

            for nt, dinfo in degree_info.items():
                print(f"  Degree coverage ({nt}): bins={dinfo['degree_bins'].tolist()}, "
                      f"cov={[f'{c:.3f}' for c in dinfo['coverage']]}")

    # Write CSV
    csv_path = OUT / "conditional_coverage_analysis.csv"
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
