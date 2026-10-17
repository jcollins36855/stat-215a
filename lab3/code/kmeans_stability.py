"""Algorithm 1 (Ben-Hur et al.): k-means stability on lingBinary.

For every k from 2 to K_MAX and every repeat i from 1 to B: draw two new
subsamples of fraction M, run k-means on each, and compute the Jaccard
similarity on the people the two subsamples share.

Every (k, i) pass is independent, so joblib runs them in parallel.
Results are saved to results/S.csv (one row per repeat i, one column per k).

Usage (from the lab3/code folder): python kmeans_stability.py
On the SCF: sbatch run.sh
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from joblib import Parallel, delayed, parallel_config
from sklearn.cluster import KMeans

from similarity import similarity

# this file lives in lab3/code, so the lab3 folder is one level up
LAB_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = LAB_DIR / "data" / "lingBinary.Rdata"
RESULTS_PATH = LAB_DIR / "results" / "S.csv"

M = 0.8          # subsample fraction
B = 100          # repeats per k
K_MAX = 10       # largest number of clusters
BASE_SEED = 215  # main random seed
N_INIT = 1       # k-means restarts per fit


def load_binary_ling(path):
    """Load lingBinary.Rdata and keep only the 0/1 answer columns."""
    rdata = pyreadr.read_r(str(path))
    # grab the data frame called "lingBinary"
    frame = rdata["lingBinary"]
    # keep only columns that are all 0s and 1s (drops ID, city, state, zip, lat, long)
    binary_cols = [col for col in frame.columns
                   if frame[col].notna().all() and frame[col].isin([0, 1]).all()]
    return frame[binary_cols].to_numpy(dtype=np.float64)


def similarity_onepair(X, k, m, seed, n_init=1):
    """One pass of Algorithm 1 to get the similarity of two subsample clusterings."""
    rng = np.random.default_rng(seed)
    n_people = X.shape[0]
    n_sub = int(round(m * n_people))

    # subgroup 1 and subgroup 2: two random sets of people
    subgroup1 = np.sort(rng.choice(n_people, size=n_sub, replace=False))
    subgroup2 = np.sort(rng.choice(n_people, size=n_sub, replace=False))

    # k-means on each subgroup separately
    kmeans_labels1 = KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                            random_state=seed).fit_predict(X[subgroup1])
    kmeans_labels2 = KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                            random_state=seed + 1).fit_predict(X[subgroup2])

    # Find the people in both subgroups.
    # pos1 = where each shared person sits in subgroup1, pos2 = where they sit in subgroup2,
    # so kmeans_labels1[pos1] and kmeans_labels2[pos2] line up person by person.
    # The _ is the list of shared people's IDs, which we don't need.
    # assume_unique=True: no one appears twice in a subgroup, so this runs faster.
    _, pos1, pos2 = np.intersect1d(subgroup1, subgroup2,
                                   assume_unique=True, return_indices=True)
    return similarity(kmeans_labels1[pos1], kmeans_labels2[pos2])


def main():
    X = load_binary_ling(DATA_PATH)

    # On the SCF use the cores sbatch gave us; otherwise use every core
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    # one job for every (k, i) pair
    jobs = [(k, i) for k in range(2, K_MAX + 1) for i in range(1, B + 1)]

    start = time.perf_counter()
    # tell joblib: each worker gets 1 thread for its k-means,
    # since joblib itself is already using all the cores
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(similarity_onepair)(X, k, M, BASE_SEED + 2 * (k * B + i - 1), N_INIT)
            for k, i in jobs
        )
    print(f"{len(jobs)} passes on {n_jobs} cores in {time.perf_counter() - start:.1f} s")

    # one row per repeat i, one column per k
    S = pd.DataFrame(np.array(results).reshape(K_MAX - 1, B).T,
                     index=pd.RangeIndex(1, B + 1, name="i"),
                     columns=range(2, K_MAX + 1))
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    S.to_csv(RESULTS_PATH)
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
