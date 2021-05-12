#!/usr/bin/env python3
"""that conducts forward propagation using Dropout: """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ that conducts forward propagation using Dropout: """
    cache = {}
    cache['A0'] = X
    for i in range(L):
        layer = str(i + 1)
        prev = str(i)
        w = 'W' + layer
        b = 'b' + layer
        a = 'A' + layer
        ap = 'A' + prev
        drop = 'D' + layer
        z = np.matmul(weights[w], cache[ap]) + weights[b]
        if i == L - 1:
            cache[a] = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
        else:
            cache[a] = np.tanh(z)
            sh = cache[a].shape
            cache[drop] = (np.random.rand(sh[0], sh[1]) < keep_prob) * 1
            cache[a] = np.multiply(cache[a], cache[drop])
            cache[a] /= keep_prob
    return cache
