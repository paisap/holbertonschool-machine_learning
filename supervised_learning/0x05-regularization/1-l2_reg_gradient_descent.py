#!/usr/bin/env python3
""" that updates the weights and biases of a neural network
using gradient descent with L2 regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ that updates the weights and biases of a neural network
    using gradient descent with L2 regularization """
    x = Y.shape[1]
    for i in range(L - 1, -1, -1):
        prev = str(i)
        aux = str(i + 1)
        if i == L - 1:
            dz = cache['A' + str(L)] - Y
        else:
            dz = da * (1 - cache['A' + aux] ** 2)
        auxP = cache['A' + prev]
        dw = (np.matmul(dz, auxP.T) + lambtha * weights['W' + aux]) / x
        db = np.sum(dz, axis=1, keepdims=True) / x
        da = np.matmul(weights['W' + aux].T, dz)
        weights['W' + aux] = weights['W' + aux] - alpha * dw
        weights['b' + aux] = weights['b' + aux] - alpha * db
