#!/bin/bash
# Lab 2 pipeline: execute the analysis notebook (regenerates ../figs/ and
# ../tmp/ artifacts), then compile the report. Run from anywhere; always
# operates on the directories containing this script.
set -e
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 215a

# The notebook writes every figure and intermediate array into these, and
# neither directory ships with the submission.
mkdir -p ../figs ../tmp

# 1. Analysis notebook (imports clean.py; writes ../figs/ + ../tmp/).
#    Stored outputs in the committed .ipynb are refreshed in place.
#    Per-cell timeout disabled: t-SNE / k-means sweeps take minutes.
jupyter nbconvert \
    --to notebook \
    --execute lab2.ipynb \
    --inplace \
    --ExecutePreprocessor.timeout=-1

# 2. Report (pdflatex twice for references; MacTeX expected on PATH).
# Build log kept out of report/ so the submitted folder stays clean.
cd ../report
pdflatex -interaction=nonstopmode lab2.tex > ../tmp/build.log 2>&1
pdflatex -interaction=nonstopmode lab2.tex >> ../tmp/build.log 2>&1

conda deactivate
