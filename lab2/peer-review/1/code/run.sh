#!/bin/bash
# add stuff below here...

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stat215a

cd "$(dirname "$0")"

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  demo.ipynb