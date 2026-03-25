# JSS Submission Readiness for CausalLens

This note maps the current CausalLens repository and manuscript state to the Journal of Statistical Software requirements verified from the journal website on 2026-03-25.

## Verified Journal Facts

- JSS is free to submit, free to publish, and free to read.
- New submissions must provide a PDF manuscript written in the JSS LaTeX style.
- References must be managed through BibTeX rather than hard-coded inline.
- Source code must be available to reviewers.
- Replication materials must be included, preferably with a standalone script that reproduces the paper outputs.
- The software must use GPL-2, GPL-3, or a GPL-compatible license.
- The paper should discuss both the implemented methods and the software.
- The paper should compare the software with related implementations.
- The paper should include a non-trivial empirical illustration or case study and make the reported results reproducible.

## What CausalLens Already Meets

- Public source code repository exists.
- License is MIT, which is GPL-compatible for JSS purposes.
- Package metadata, versioning, and installation instructions already exist.
- Tests cover estimators, diagnostics, benchmarks, panel and IV methods, regression discontinuity, bunching, reporting, and reference-parity checks.
- Public benchmark datasets and manuscript evidence outputs already exist.
- Multiple replication scripts already exist under `replications/`.
- A single standalone replication runner now exists at `replications/run_all.py` to simplify JSS-style reproduction.
- A JSS-format LaTeX manuscript draft now exists in `.private-drafts/jss/template_extract/causal-lens.tex` with references in `.private-drafts/jss/template_extract/causal-lens.bib`.

## What Is Still Required Before Literal Submission

### 1. Final manuscript polish and PDF build

Status: not fully complete.

The project now has a JSS-shaped LaTeX manuscript draft, but it still needs a final author-information pass, reference cleanup, and a final PDF compiled from the exact submission version.

Minimum remaining actions:

1. finalize affiliation and contact details
2. finalize the title, abstract, and keywords for the exact submission framing
3. compile the manuscript PDF from the final LaTeX and BibTeX sources
4. do a final citation cleanup for capitalization, URLs, and DOI completeness

### 2. Final replication bundle check

Status: close, but should be packaged once before submission.

JSS prefers a replication path that is easy for reviewers to execute. The repository is close here, especially now that a single runner script exists, but the final submission should still verify that the manuscript tables and any final figures are regenerated exactly from the provided scripts.

Minimum remaining actions:

1. run the standalone replication runner on the final tagged submission state
2. verify that the manuscript's reported benchmark values match the exported files exactly
3. keep any final manuscript figures or tables in the replication outputs rather than as manual artifacts

### 3. Final software-positioning pass

Status: materially improved, but should be checked once more before submission.

JSS expects software papers to explain how the package relates to existing tools. The current manuscript draft now does this, but the final version should make sure the comparison remains precise and non-overclaiming.

Minimum remaining actions:

1. keep comparison claims limited to verified functionality and workflow differences
2. avoid implying parity with specialized RD, bunching, graph-based, or ML-first packages where parity does not exist
3. ensure the case-study and benchmark discussion remains software-validation oriented rather than novelty-claim oriented

## Practical Conclusion

CausalLens is much closer to a real JSS submission path than it was to a zero-cost JORS path.

The main remaining work is no longer journal-fit triage. It is submission packaging: final author metadata, final LaTeX/BibTeX cleanup, one exact PDF build, and one final replication pass against the submission version.

## Recommended Final Sequence

1. finalize the JSS manuscript text and BibTeX entries
2. run the standalone replication workflow on the release candidate
3. reconcile the manuscript numbers against the exported benchmark tables
4. compile the final PDF manuscript in JSS style
5. submit the PDF, source code, and replication materials together

## Sources Checked

- JSS mission and scope page: https://www.jstatsoft.org/about/mission
- JSS submission page: https://www.jstatsoft.org/about/submissions
- JSS style guide: https://www.jstatsoft.org/pages/view/style