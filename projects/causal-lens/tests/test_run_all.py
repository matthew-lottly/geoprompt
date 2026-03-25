from __future__ import annotations

import importlib.util
from pathlib import Path


RUN_ALL_PATH = Path(__file__).resolve().parents[1] / "replications" / "run_all.py"
SPEC = importlib.util.spec_from_file_location("causal_lens_run_all", RUN_ALL_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_all = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_all)


def test_run_all_executes_replications_then_reporting(monkeypatch) -> None:
    calls: list[tuple[str, list[str] | None]] = []

    monkeypatch.setattr(run_all, "_run_script", lambda name, extra_args=None: calls.append((name, extra_args)))
    monkeypatch.setattr(run_all, "_run_reporting_bundle", lambda: calls.append(("reporting", None)))
    monkeypatch.setattr(run_all.argparse.ArgumentParser, "parse_args", lambda self: run_all.argparse.Namespace(skip_simulation=False, full=True))

    run_all.main()

    assert calls == [
        ("replicate_lalonde.py", None),
        ("replicate_nhefs.py", None),
        ("replicate_cross_design.py", None),
        ("replicate_simulation.py", ["--full"]),
        ("reporting", None),
    ]


def test_run_all_can_skip_simulation_and_still_refresh_reporting(monkeypatch) -> None:
    calls: list[tuple[str, list[str] | None]] = []

    monkeypatch.setattr(run_all, "_run_script", lambda name, extra_args=None: calls.append((name, extra_args)))
    monkeypatch.setattr(run_all, "_run_reporting_bundle", lambda: calls.append(("reporting", None)))
    monkeypatch.setattr(run_all.argparse.ArgumentParser, "parse_args", lambda self: run_all.argparse.Namespace(skip_simulation=True, full=False))

    run_all.main()

    assert calls == [
        ("replicate_lalonde.py", None),
        ("replicate_nhefs.py", None),
        ("replicate_cross_design.py", None),
        ("reporting", None),
    ]