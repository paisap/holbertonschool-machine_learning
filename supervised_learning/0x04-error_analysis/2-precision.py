#!/usr/bin/env python3
""" Create precision """
import numpy as np


def precision(confusion):
    """ Create  precision """
    sensitivy = np.diag(confusion) / np.sum(confusion, axis=0)
    return sensitivy
