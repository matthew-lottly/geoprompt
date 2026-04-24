from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .benchmark_registry import benchmark_assignment_details, benchmark_assignment_map
except ImportError:
    from benchmark_registry import benchmark_assignment_details, benchmark_assignment_map

DEFAULT_PARITY_TESTS: tuple[str, ...] = (
    "tests/parity/test_symbol_manifest.py",
    "tests/parity/test_registry_quality.py",
    "tests/parity/test_equation_parity.py",
    "tests/parity/test_vector_parity_geopandas.py",
    "tests/parity/test_crs_parity_pyproj.py",
    "tests/parity/test_raster_parity_rasterio.py",
    "tests/parity/test_network_parity_networkx.py",
    "tests/parity/test_reporting_output_parity.py",
)


def _parse_pytest_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for key in counts:
        match = re.search(rf"(\d+)\s+{key}", output)
        if match:
            counts[key] = int(match.group(1))
    return counts


def _run_pytest_file(test_file: str) -> dict[str, Any]:
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-q"],
        capture_output=True,
        text=True,
    )
    duration_seconds = time.perf_counter() - start
    combined = f"{result.stdout}\n{result.stderr}"
    counts = _parse_pytest_counts(combined)
    executed_total = counts["passed"] + counts["failed"] + counts["errors"]
    total = executed_total + counts["skipped"]
    if executed_total == 0:
        pass_rate = 1.0 if counts["skipped"] > 0 else 0.0
    else:
        pass_rate = counts["passed"] / executed_total
    execution_status = "passed" if counts["failed"] == 0 and counts["errors"] == 0 else "failed"

    return {
        "test_file": test_file,
        "duration_seconds": round(duration_seconds, 3),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "errors": counts["errors"],
        "exit_code": result.returncode,
        "pass_rate": round(pass_rate, 6),
        "execution_status": execution_status,
        "skipped_only": counts["skipped"] > 0 and executed_total == 0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_bar_chart_svg(path: Path, title: str, rows: list[tuple[str, float]], color: str) -> None:
    width = 980
    height = 520
    pad_left = 220
    pad_top = 60
    bar_h = 28
    gap = 14
    max_value = max((value for _, value in rows), default=1.0)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; fill: #1f2937; } .title { font-size: 20px; font-weight: 700; }</style>',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="24" y="34" class="title">{title}</text>',
    ]

    for idx, (label, value) in enumerate(rows):
        y = pad_top + idx * (bar_h + gap)
        bar_width = int((value / max(max_value, 1e-9)) * (width - pad_left - 70))
        svg_lines.append(f'<text x="18" y="{y + 19}">{label}</text>')
        svg_lines.append(
            f'<rect x="{pad_left}" y="{y}" width="{bar_width}" height="{bar_h}" rx="4" fill="{color}" opacity="0.85"/>'
        )
        svg_lines.append(
            f'<text x="{pad_left + bar_width + 8}" y="{y + 19}">{value:.3f}</text>'
        )

    svg_lines.append("</svg>")
    path.write_text("\n".join(svg_lines), encoding="utf-8")


def _write_markdown_summary(
    path: Path,
    class_counts: Counter[str],
    symbol_count: int,
    perf_rows: list[dict[str, Any]],
) -> None:
    class_lines = [
        "| benchmark_class | count | ratio |",
        "|---|---:|---:|",
    ]
    for klass, count in class_counts.most_common():
        class_lines.append(f"| {klass} | {count} | {count / max(symbol_count, 1):.2%} |")

    perf_lines = [
        "| test_file | duration_seconds | passed | failed | skipped | errors | pass_rate_executed | status |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in perf_rows:
        perf_lines.append(
            "| {test_file} | {duration_seconds:.3f} | {passed} | {failed} | {skipped} | {errors} | {pass_rate:.2%} | {execution_status} |".format(
                **row
            )
        )

    md = "\n".join(
        [
            "# Parity Audit",
            "",
            f"Total public symbols audited: {symbol_count}",
            "",
            "## Benchmark Class Distribution",
            *class_lines,
            "",
            "## Parity Test Performance",
            *perf_lines,
            "",
            "Artifacts:",
            "- parity-class-counts.csv",
            "- parity-test-performance.csv",
            "- parity-class-counts.svg",
            "- parity-test-performance.svg",
        ]
    )
    path.write_text(md, encoding="utf-8")


def build_audit_artifacts(
    private_dir: Path = Path("private") / "parity-audit",
    run_tests: bool = True,
) -> dict[str, Any]:
    assignments = benchmark_assignment_map()
    details = benchmark_assignment_details()
    class_counts: Counter[str] = Counter(assignments.values())
    symbol_count = len(assignments)

    private_dir.mkdir(parents=True, exist_ok=True)

    class_rows = [
        {
            "benchmark_class": klass,
            "count": count,
            "ratio": round(count / max(symbol_count, 1), 6),
        }
        for klass, count in class_counts.most_common()
    ]
    _write_csv(
        private_dir / "parity-class-counts.csv",
        class_rows,
        ["benchmark_class", "count", "ratio"],
    )

    _write_csv(
        private_dir / "parity-assignment-details.csv",
        details,
        ["symbol", "benchmark_class", "module"],
    )

    perf_rows: list[dict[str, Any]] = []
    if run_tests:
        for test_file in DEFAULT_PARITY_TESTS:
            perf_rows.append(_run_pytest_file(test_file))

    _write_csv(
        private_dir / "parity-test-performance.csv",
        perf_rows,
        [
            "test_file",
            "duration_seconds",
            "passed",
            "failed",
            "skipped",
            "errors",
            "exit_code",
            "pass_rate",
            "execution_status",
            "skipped_only",
        ],
    )

    _write_bar_chart_svg(
        private_dir / "parity-class-counts.svg",
        "Benchmark Class Counts",
        [(row["benchmark_class"], float(row["count"])) for row in class_rows],
        color="#0f766e",
    )
    _write_bar_chart_svg(
        private_dir / "parity-test-performance.svg",
        "Parity Test Duration (seconds)",
        [(row["test_file"], float(row["duration_seconds"])) for row in perf_rows],
        color="#1d4ed8",
    )

    _write_markdown_summary(
        private_dir / "parity-audit-summary.md",
        class_counts,
        symbol_count,
        perf_rows,
    )

    json_payload = {
        "symbol_count": symbol_count,
        "class_counts": dict(class_counts),
        "performance": perf_rows,
    }
    (private_dir / "parity-audit-summary.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return json_payload


if __name__ == "__main__":
    payload = build_audit_artifacts(run_tests=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
