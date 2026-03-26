JSS Submission Package for CausalLens

This folder contains the files needed for a Journal of Statistical Software submission.

Recommended uploads
1. causal-lens.pdf
   The manuscript PDF in JSS style.

2. causal-lens.tex, causal-lens.bib, jss.cls, jss.bst, jsslogo.jpg
   The LaTeX manuscript source and required JSS style files.

3. causal-lens-software-source.zip
   The Python package source for reviewers to inspect and install.
   Contents: pyproject.toml, README.md, LICENSE, CITATION.cff, src/, tests/.

4. causal-lens-replication-materials.zip
   The replication scripts and input data used to regenerate manuscript results.
   Contents: replications/, data/.

5. replication-output.log
   Saved output from running the replication workflow.

Notes
- The software is installable via the packaged pyproject.toml.
- The main reviewer workflow is: pip install -e . then python replications/run_all.py
- The manuscript PDF keeps the official JSS section-heading style from jss.cls.
- Publication metadata such as volume, issue, and DOI are controlled by JSS after acceptance.