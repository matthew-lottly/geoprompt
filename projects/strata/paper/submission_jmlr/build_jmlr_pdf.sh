#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f jmlr2e.sty ]; then
  cp jmlr-style-file/jmlr2e.sty .
fi

pdflatex -interaction=nonstopmode manuscript_draft.tex
bibtex manuscript_draft
pdflatex -interaction=nonstopmode manuscript_draft.tex
pdflatex -interaction=nonstopmode manuscript_draft.tex