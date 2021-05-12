#!/usr/bin/env python3
"""hat updates the weights of a neural network with Dropout
regularization using gradient descent:"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """hat updates the weights of a neural network with Dropout
    regularization using gradient descent:"""
    x = Y.shape[1]
    for i in range(L - 1, -1, -1):
        prev = str(i)
        layer = str(i + 1)
        if i == L - 1:
            dz = cache['A' + str(L)] - Y
        else:
            dz = da * (1 - cache['A' + layer] ** 2)
            dz *= cache['D' + layer]
            dz /= keep_prob
        dw = np.matmul(dz, cache['A' + prev].T) / x
        db = np.sum(dz, axis=1, keepdims=True) / x
        da = np.matmul(weights['W' + layer].T, dz)
        weights['W' + layer] = weights['W' + layer] - alpha * dw
        weights['b' + layer] = weights['b' + layer] - alpha * db
