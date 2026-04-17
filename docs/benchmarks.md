# Benchmarks and Proof

GeoPrompt should only claim competitive advantages when they are backed by repeatable evidence. The comparison lane generates three artifacts:

- JSON report for machine-readable snapshots
- Markdown summary for release notes and pull requests
- HTML summary for stakeholder review

## Generate a comparison bundle

```python
from pathlib import Path
from geoprompt import build_comparison_report, benchmark_summary_table, export_comparison_bundle

report = build_comparison_report(output_dir=Path("outputs"))
summary = benchmark_summary_table(report)
written = export_comparison_bundle(report, Path("outputs"))

print(summary.head(5))
print(written)
```

## Command-line workflow

```bash
python -m geoprompt.compare --output-dir outputs
```

This writes:

- outputs/geoprompt_comparison_report.json
- outputs/geoprompt_comparison_summary.md
- outputs/geoprompt_comparison_summary.html

## What to look for

1. Correctness checks should all pass before performance claims are highlighted.
2. Indexed workflows should show the clearest gains on repeated join and nearest operations.
3. Network workflows should emphasize decision-readiness, not only raw runtime.

## Suggested release habit

For each release:

1. run the comparison bundle
2. attach the markdown or HTML summary to the release notes
3. compare the new snapshot against the previous release
4. only publish better-than claims when the evidence is visible
