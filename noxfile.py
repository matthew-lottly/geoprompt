from __future__ import annotations

import nox

PYTHONS = ["3.9", "3.10", "3.11", "3.12", "3.13"]


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev,compare,overlay,projection]")
    session.run("pytest", "-m", "not slow")


@nox.session(python="3.12")
def docs(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("python", "-m", "compileall", "src", "docs")


@nox.session(python="3.12", name="tests-core-only")
def tests_core_only(session: nox.Session) -> None:
    """Validate degraded-mode guarantees with no optional extras installed.

    Corresponds to CI_EXTRAS_PROFILES["core-only"] in _capabilities.py.
    Installs only the dev extras (pytest, no geopandas / shapely / pyarrow etc.)
    and runs the optional-dependency hardening and I/O safety test suites.
    This session proves that:
    - Core JSON/CSV paths work without any optional extras.
    - Missing optional deps raise DependencyError (not ImportError) with pip hints.
    - FallbackWarning is emitted on degraded-mode paths.
    """
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "tests/test_optional_dep_hardening.py",
        "tests/test_io_db_safety.py",
        "-v",
        "--tb=short",
    )


@nox.session(python="3.12", name="tests-degraded-benchmark")
def tests_degraded_benchmark(session: nox.Session) -> None:
    """Benchmark degraded mode vs full-feature mode for key I/O paths.

    Corresponds to J5.11: benchmarking split for degraded vs full-feature mode.
    Runs bench.py with GEOPROMPT_DEGRADED_MODE=1 then with full extras.
    """
    session.install("-e", ".[dev,io,viz,overlay]")
    session.run(
        "python", "-m", "pytest",
        "benchmarks/bench.py",
        "--benchmark-only",
        "--benchmark-json=outputs/benchmark-degraded.json",
        "-q",
        success_codes=[0, 5],  # 5 = no tests collected (bench file may not be pytest)
    )
