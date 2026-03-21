# Contributing to GeoPrompt

## Development Setup

```bash
# Clone and create a virtual environment
git clone https://github.com/matthew-lottly/Matt-Powell.git
cd projects/geoprompt
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Install with all extras
pip install -e ".[dev,compare,overlay,projection]"
```

## Running Tests

```bash
# Full suite
python -m pytest tests/ --tb=short -q -W ignore

# Specific test file
python -m pytest tests/test_geoprompt.py -q

# Reference parity tests (requires compare extras)
python -m pytest tests/test_reference_parity.py -q
```

## Running Benchmarks

```bash
python benchmarks/bench.py
```

## Project Structure

```
src/geoprompt/
├── __init__.py          # Public API exports
├── frame.py             # GeoPromptFrame class (130+ spatial tools)
├── geometry.py          # Pure-Python geometry functions
├── equations.py         # Spatial equations (gravity, accessibility, etc.)
├── overlay.py           # Buffer, dissolve, Shapely interop
├── spatial_index.py     # R-tree spatial index
├── io.py                # GeoJSON, CSV, flat-record I/O
└── py.typed             # PEP 561 type marker

tests/
├── test_geoprompt.py          # Core tool tests (338)
├── test_cross_validation.py   # Kriging cross-validation (35)
├── test_new_tools.py          # Tools 86-100 (35)
├── test_new_tools_p3.py       # Tools 101-130 (38)
├── test_reference_parity.py   # Parity vs PySAL/Shapely/etc (44)
└── test_spatial_weights.py    # SpatialWeights class (11)
```

## Adding a New Tool

1. Add the method to `GeoPromptFrame` in `frame.py`.
2. Follow the naming convention: `tool_name(self, ..., suffix: str = "shortname") -> GeoPromptFrame`.
3. Add at least one test in the appropriate test file.
4. Run the full suite to check for regressions.
5. Update `CHANGELOG.md` with the new tool.

## Conventions

- Tools that add columns use a configurable `suffix` parameter.
- Tools return new `GeoPromptFrame` instances (immutable pattern).
- Tools that return aggregate results (not per-row) return `dict` or `list`.
- Helper functions are module-level and prefixed with `_`.
- Keep zero required dependencies beyond `matplotlib`.

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Make your changes with tests.
3. Run the full test suite and benchmarks.
4. Update the changelog.
5. Submit a PR against `main`.
