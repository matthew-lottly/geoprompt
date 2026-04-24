# Contributing to GeoPrompt

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/<owner>/geoprompt.git
cd geoprompt
python -m venv .venv
.venv/Scripts/activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -q
```

## Code Style

This project uses **Ruff** for linting and formatting:

```bash
ruff check src/
ruff format src/
```

Pre-commit hooks are configured — install them with:

```bash
pre-commit install
```

## Pull Request Process

1. Fork the repo and create a feature branch from `main`.
2. Add or update tests for any new functionality.
3. Ensure `pytest` and `ruff check` pass with zero errors.
4. Open a pull request with a clear description of the change.

## Trust Regression Checklist for New APIs

For every new public API surface:

1. Add tier metadata (stable/beta/experimental/simulation-only) and ensure docs reflect the same tier.
2. Use capability checks for optional dependencies; do not expose raw `ImportError`.
3. Document degraded behavior and remediation guidance in docstrings and user docs.
4. Preserve error-contract shape (`code`, `category`, `remediation`, `error`) for user-facing failures.
5. Add or update tests for warnings, fallback policy behavior, and capability mismatch handling.

## Reporting Issues

Open an issue on GitHub with:

- A clear title and description
- Steps to reproduce (if applicable)
- Expected vs. actual behavior
