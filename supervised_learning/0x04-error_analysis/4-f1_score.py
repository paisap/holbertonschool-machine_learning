#!/usr/bin/env python3
""" Create specificity """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ Create  specificity """
    FN = sensitivity(confusion)
    FP = precision(confusion)
    return (2 * (FN * FP)) / (FN + FP)
