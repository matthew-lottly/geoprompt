from __future__ import annotations

import nox

PYTHONS = ["3.10", "3.11", "3.12", "3.13"]


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev,compare,overlay,projection]")
    session.run("pytest", "-m", "not slow")


@nox.session(python="3.12")
def docs(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("python", "-m", "compileall", "src", "docs")
