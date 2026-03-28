# STRATA — Reproducibility Makefile
# Run `make all` to reproduce all results from scratch.

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
PAPER_DIR = paper/submission_jmlr

.PHONY: all test benchmark ablations fairness figures conditional pdf clean

all: test benchmark ablations fairness conditional figures

# ── Tests ──────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --tb=short

# ── Benchmarks ─────────────────────────────────────────────────
benchmark:
	$(PYTHON) scripts/run_benchmark.py
	$(PYTHON) scripts/run_real_data_benchmark.py
	$(PYTHON) scripts/run_real_method_comparison.py

# ── Ablations ──────────────────────────────────────────────────
ablations:
	$(PYTHON) scripts/run_ablations.py
	$(PYTHON) scripts/run_lambda_only.py

# ── Analysis ───────────────────────────────────────────────────
fairness:
	$(PYTHON) scripts/fairness_audit.py

conditional:
	$(PYTHON) scripts/run_conditional_coverage.py

# ── Figures ────────────────────────────────────────────────────
figures:
	$(PYTHON) scripts/generate_paper_figures.py
	$(PYTHON) scripts/generate_tables.py

# ── PDF Build (requires pdflatex + bibtex) ─────────────────────
pdf:
	cd $(PAPER_DIR) && pdflatex manuscript_draft.tex
	cd $(PAPER_DIR) && bibtex manuscript_draft
	cd $(PAPER_DIR) && pdflatex manuscript_draft.tex
	cd $(PAPER_DIR) && pdflatex manuscript_draft.tex

# ── Clean ──────────────────────────────────────────────────────
clean:
	find outputs/ -name "*.csv" -delete 2>/dev/null || true
	find outputs/ -name "*.png" -delete 2>/dev/null || true
	find outputs/ -name "*.json" -delete 2>/dev/null || true
