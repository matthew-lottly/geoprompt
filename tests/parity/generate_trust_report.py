from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from benchmark_registry import benchmark_assignment_details, benchmark_assignment_map


if __name__ == "__main__":
    assignments = benchmark_assignment_map()
    assignment_details = benchmark_assignment_details()
    class_counts = Counter(assignments.values())
    symbol_count = len(assignments)
    payload = {
        "symbol_count": symbol_count,
        "class_counts": dict(class_counts),
        "class_ratios": {
            klass: round(count / max(symbol_count, 1), 6)
            for klass, count in class_counts.items()
        },
        "symbols": assignments,
        "details": assignment_details,
    }
    out = Path("outputs") / "parity-trust-report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out}")
