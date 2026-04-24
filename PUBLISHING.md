# Publishing Guide

## Recommended Standalone Repository Name

- geoprompt

## Recommended Description

- Pure-Python spatial analysis toolkit with GeoJSON-native geometries and GeoPandas-style frame workflows.

## Suggested Topics

- python
- gis
- geospatial
- spatial-analysis
- package
- developer-tools

## Split Steps

1. Create a new empty repository named `geoprompt`.
2. Copy this project folder into the new repository root.
3. Preserve `data/`, `src/`, `tests/`, `docs/`, and `pyproject.toml`.
4. Keep the committed pressure plot in `assets/neighborhood-pressure-review-live.png` and regenerate it with `geoprompt-demo` if the package behavior changes.
5. Preserve the GitHub Actions workflows in `.github/workflows/geoprompt-ci.yml` and `.github/workflows/publish-pypi.yml` so validation and publishing follow the package.
6. Reference [docs/architecture.md](docs/architecture.md) and [docs/demo-storyboard.md](docs/demo-storyboard.md) from the README when polishing the public pitch.
7. Verify the README presentation strategy still works for both GitHub and PyPI before release; if GitHub-only Mermaid content is used, provide a PyPI-safe fallback or accept reduced PyPI rendering.

## Release Checklist

1. Run `pytest`.
2. Run `geoprompt-demo`.
3. Run `geoprompt-compare`.
4. Confirm the module and function list in `README.md` still accurately reflects `src/geoprompt/`.
5. Confirm the test count quoted in `README.md` matches the current `pytest` suite result.
6. Review `CHANGELOG.md` and verify the release section covers all new tools and behavior changes.
7. Run optional integration and performance gates locally before release: `GEOPROMPT_RUN_GEO_IO=1 python -m pytest tests/test_geoprompt.py::test_geospatial_integration_parquet_round_trip` and `GEOPROMPT_RUN_BENCHMARKS=1 python -m pytest tests/test_benchmark_regression.py`.
8. Confirm the `optional-gated` job in `.github/workflows/geoprompt-ci.yml` is passing in GitHub Actions.
9. Generate an SBOM and verify the release evidence bundle is current.
10. Run a secrets scan and config review for service-facing artifacts.
11. Confirm provenance notes, figure manifests, and benchmark outputs are refreshed for the release.
12. Run `python -m build`.
13. Run `python -m twine check dist/*`.
14. Review `outputs/geoprompt_comparison_report.json` and confirm all summary flags are `true`.
15. Confirm the PyPI Trusted Publisher is linked to the correct repository and workflow; expected pairs are monorepo `matthew-lottly/Matt-Powell` with `.github/workflows/publish-pypi.yml`, and standalone repo `matthew-lottly/geoprompt` with `.github/workflows/publish-pypi.yml`.
16. If the PyPI publisher entry expects an environment, set it to `pypi` so the OIDC claim matches the workflow.
17. Confirm `pyproject.toml` has the intended release version and that `README.md` still renders acceptably on the intended publish targets.
18. **Degraded-mode gate**: run `geoprompt capability-report` and confirm output is coherent (no unexpected hard-fail caps). Confirm `tests/test_optional_dep_hardening.py` and `tests/test_io_db_safety.py` are fully green. Verify no new `except ImportError: return None/[]` patterns were introduced since the last release (run `git diff HEAD~1 -- src/ | grep -E "except ImportError.*return"`).
19. Push a version tag such as `geoprompt-v0.1.7` to trigger the monorepo publish workflow, or run the workflow manually from GitHub Actions.

## PyPI Commands

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

## GitHub Actions Publish Flow

The standalone repository includes `.github/workflows/publish-pypi.yml`.

Use this flow when PyPI Trusted Publishing is connected to GitHub:

1. Ensure the Trusted Publisher entry in PyPI points at the repository that is actually running the workflow.
2. Set the workflow file to `.github/workflows/publish-pypi.yml`.
3. Set the environment name to `pypi` when using the current workflow.
4. Push a tag such as `geoprompt-v0.1.7`.
5. Watch the `Publish To PyPI` workflow in GitHub Actions.
Tag pushes can also be triggered manually from GitHub once the workflow and Trusted Publisher are aligned. Confirm the workflow run finishes green before announcing or documenting the release.
<!-- End of publishing guide -->
