#!/usr/bin/env python3
""" Create specificity """
import numpy as np


def specificity(confusion):
    """ Create  specificity """
    FP = np.sum(confusion, axis=0) - np.diag(confusion)
    FN = np.sum(confusion, axis=1) - np.diag(confusion)
    TN = np.sum(confusion) - (FP + FN + np.diag(confusion))
    speci = TN / (TN + FP)
    return speci
