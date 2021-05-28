#!/usr/bin/env python3
""" L2 Regularization Cost """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ that calculates the cost of a neural
    network with L2 regularization: """
    sumWeights = 0
    for i in range(1, L + 1):
        sumWeights += np.linalg.norm(weights['W' + str(i)])
    return cost + sumWeights * lambtha / (2 * m)
