#!/bin/bash
#SBATCH -c 8
#SBATCH --job-name=kmeans_stability
#SBATCH --output=kmeans_stability.out

export OMP_NUM_THREADS=1
mkdir -p results
python kmeans_stability.py
