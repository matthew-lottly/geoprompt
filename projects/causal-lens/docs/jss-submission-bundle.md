# JSS Submission Bundle

This note records the exact bundle to prepare for the CausalLens Journal of Statistical Software submission.

## Core Files

- PDF manuscript generated from `.private-drafts/jss/template_extract/causal-lens.tex`
- LaTeX source: `.private-drafts/jss/template_extract/causal-lens.tex`
- BibTeX source: `.private-drafts/jss/template_extract/causal-lens.bib`
- Public source repository: the `projects/causal-lens` checkout
- License: `LICENSE`
- Citation metadata: `CITATION.cff`

## Reproduction Path

Run from the project root:

```bash
pip install -e .
python replications/run_all.py --skip-simulation
python replications/run_all.py --full
```

Expected outputs:

- replication CSVs under `replications/outputs/`
- manuscript-ready figures and tables under `outputs/paper/`
- supporting comparison and stability tables under `outputs/tables/`

## Reviewer-Facing Evidence Files

- `replications/outputs/lalonde_replication.csv`
- `replications/outputs/nhefs_replication.csv`
- `replications/outputs/cross_design_diagnostics.csv`
- `outputs/tables/external_comparison.csv`
- `outputs/tables/stability_summary.csv`
- `outputs/paper/tables/`
- `outputs/paper/figures/`

## Final Pre-Submission Check

1. Run the one-command replication workflow on the exact submission revision.
2. Confirm the manuscript benchmark numbers match the generated files in `outputs/paper/tables/`.
3. Confirm `outputs/tables/external_comparison.csv` still shows zero absolute differences.
4. Rebuild the manuscript PDF from the same revision.
5. Submit the PDF, LaTeX/BibTeX sources, and source-code/reproduction materials together.