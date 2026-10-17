"""Jaccard similarity between two clusterings of the same people.

Instead of building the q x q "same cluster?" matrix, we build a small
k1 x k2 table of how many people fall in each (cluster in run 1,
cluster in run 2) combination. A group of n people has n * (n - 1) / 2
pairs, so every pair count we need comes from this table.

Runtime is O(q + k1 * k2) and memory is O(k1 * k2).
"""
import numpy as np
from numba import njit


@njit
def _jaccard(mem1, mem2):
    """Numba-compiled Jaccard score between two membership vectors."""
    # Number of clusters in each run (labels go 0, 1, ..., so max + 1)
    # These are the number of rows and columns in the table
    k1 = mem1.max() + 1
    k2 = mem2.max() + 1

    # Initialize table with one row per run 1 cluster and one column per run 2 cluster
    # counts[cluster1, cluster2] = number of people in cluster1 in run 1 and cluster2 in run 2
    counts = np.zeros((k1, k2), dtype=np.int64)

    # Loop through each shared person one time
    # Each person adds 1 to the cell for their run 1 and run 2 clusters
    for i in range(mem1.shape[0]):
        counts[mem1[i], mem2[i]] += 1

    # Initialize pair counts
    together_both = 0    # pairs in the same cluster in both runs
    together_first = 0   # pairs in the same cluster in run 1
    together_second = 0  # pairs in the same cluster in run 2

    # Go through the table one row (run 1 cluster) at a time
    for cluster1 in range(k1):
        # Running total of people in run 1's cluster1
        row_total = 0
        for cluster2 in range(k2):
            # Number of people in cluster1 in run 1 AND cluster2 in run 2
            people_in_cell = counts[cluster1, cluster2]
            # A group of n people has n * (n - 1) // 2 pairs, all together in both runs
            together_both += people_in_cell * (people_in_cell - 1) // 2
            row_total += people_in_cell
        # row_total = size of run 1's cluster1, so add its pairs
        together_first += row_total * (row_total - 1) // 2

    # Go through the table one column (run 2 cluster) at a time
    for cluster2 in range(k2):
        # Running total of people in run 2's cluster2
        col_total = 0
        for cluster1 in range(k1):
            col_total += counts[cluster1, cluster2]
        # col_total = size of run 2's cluster2, so add its pairs
        together_second += col_total * (col_total - 1) // 2

    # Pairs together in at least one run
    # (subtract together_both because those pairs were counted in both totals)
    at_least_one = together_first + together_second - together_both

    # If no pair is ever together (every cluster has 1 person), call it a perfect match
    if at_least_one == 0:
        return 1.0

    # Jaccard formula = together in both / together in at least one
    return together_both / at_least_one


def similarity(mem1, mem2):
    """Jaccard similarity between two membership vectors.

    Args:
        mem1: cluster label of each shared person in run 1 (membership1).
        mem2: cluster label of the same people in run 2 (membership2).

    Returns:
        float: pairs together in both runs / pairs together in at least one.
    """
    mem1 = np.asarray(mem1, dtype=np.int64)
    mem2 = np.asarray(mem2, dtype=np.int64)
    if mem1.shape != mem2.shape:
        raise ValueError("Membership vectors must have the same length.")
    return _jaccard(mem1, mem2)
