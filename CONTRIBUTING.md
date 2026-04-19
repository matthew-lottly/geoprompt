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

## Reporting Issues

Open an issue on GitHub with:
- A clear title and description
- Steps to reproduce (if applicable)
- Expected vs. actual behavior
