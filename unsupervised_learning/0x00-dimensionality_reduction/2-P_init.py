#!/usr/bin/env python3
"""Initialize SNE"""

import numpy as np


def P_init(X, perplexity):
    """
    Initialize variables to calculate affinities t-SNE
    """

    n, d = X.shape
    # e distance for all pairs of points in input matrix X
    # ||Xi - Xj|| ** 2
    sum_X = np.sum(np.square(X), axis=1)
    # exp(-||Yi - Yj|| ** 2)
    D = (np.add(np.add(-2 * np.matmul(X, X.T), sum_X).T, sum_X))
    # fill diagonal of 0
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H
