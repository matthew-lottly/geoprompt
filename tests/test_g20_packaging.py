from __future__ import annotations

import builtins
from pathlib import Path

import geoprompt as gp
from geoprompt import demo as demo_module


ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_g20_version_metadata_matches_public_api() -> None:
    pyproject = _read_text("pyproject.toml")
    assert 'version = "0.1.8"' in pyproject
    assert gp.__version__ == "0.1.8"


def test_g20_core_install_is_light_and_profiles_exist() -> None:
    pyproject = _read_text("pyproject.toml")
    assert '"matplotlib>=3.9,<4.0",' not in pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "analyst = [" in pyproject
    assert "developer = [" in pyproject
    assert "full = [" in pyproject
    assert "all = [" in pyproject


def test_g20_build_matrix_covers_python_39() -> None:
    ci = _read_text(".github/workflows/geoprompt-ci.yml")
    wheel = _read_text(".github/workflows/wheel-build.yml")
    noxfile = _read_text("noxfile.py")
    toxfile = _read_text("tox.ini")

    assert '"3.9"' in ci
    assert "cp39-*" in wheel
    assert '"3.9"' in noxfile
    assert "py39" in toxfile


def test_g20_demo_only_requires_matplotlib_when_chart_export_runs(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == "matplotlib.pyplot":
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    try:
        demo_module.export_pressure_plot([], ROOT / "outputs" / "_should_not_exist.png")
    except RuntimeError as exc:
        assert "matplotlib is required" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("expected export_pressure_plot to raise when matplotlib is unavailable")