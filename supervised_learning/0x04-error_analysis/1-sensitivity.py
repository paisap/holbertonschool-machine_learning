#!/usr/bin/env python3
""" Create sensitivity """
import numpy as np


def sensitivity(confusion):
    """ Create  sensitivity """
    sensitivy = np.diag(confusion) / np.sum(confusion, axis=0)
    return sensitivy
