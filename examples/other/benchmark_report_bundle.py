from __future__ import annotations

from pathlib import Path

from geoprompt import benchmark_summary_table, build_comparison_report, export_comparison_bundle


"""Generate benchmark and correctness proof artifacts for release notes.

Requires the comparison extras to be installed:
    pip install geoprompt[compare]
"""


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_comparison_report(output_dir=output_dir)
    summary = benchmark_summary_table(report)
    written = export_comparison_bundle(report, output_dir)

    print("Top comparison rows:")
    for row in summary.head(5):
        print(row)

    print("\nWritten files:")
    for name, path in written.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
