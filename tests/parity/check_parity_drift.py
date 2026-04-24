from __future__ import annotations

import json
from pathlib import Path

from benchmark_registry import benchmark_assignment_map, public_symbols


BASELINE_PATH = Path("data") / "parity_baseline.json"


if __name__ == "__main__":
    current = benchmark_assignment_map()
    symbols = set(public_symbols())
    if not BASELINE_PATH.exists():
        baseline_payload = {
            "symbol_count": len(current),
            "symbols": current,
        }
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Initialized baseline at {BASELINE_PATH}")
        raise SystemExit(0)

    baseline_payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_symbols = baseline_payload.get("symbols", {})

    missing = sorted(symbol for symbol in symbols if symbol not in baseline_symbols)
    stale = sorted(symbol for symbol in baseline_symbols if symbol not in symbols)
    changed = sorted(
        symbol
        for symbol, klass in baseline_symbols.items()
        if symbol in current and current[symbol] != klass
    )

    if missing or stale or changed:
        print("Parity drift detected")
        if missing:
            print(f"Missing explicit mappings for public symbols: {missing}")
        if stale:
            print(f"Stale explicit mappings not in public API: {stale}")
        if changed:
            print(f"Changed benchmark class assignments: {changed}")
        raise SystemExit(1)

    print("No parity drift detected")
