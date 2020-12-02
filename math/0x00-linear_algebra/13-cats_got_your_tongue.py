#!/usr/bin/env python3
""" el numpy """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ concat with numpy """
    return np.concatenate((mat1, mat2), axis)
