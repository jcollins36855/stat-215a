#!/bin/bash

set -e

conda run -n stat215a python code/clean.py

conda run -n stat215a pandoc report/lab1.md \
    -o report/lab1.pdf \
    --resource-path=report \
    --pdf-engine=tectonic