#!/usr/bin/env python3
""" that slices a matrix along specific axes """
import numpy as np


def np_slice(matrix, axes={}):
    """ that slices a matrix along specific axes """
    empty_slicer = slice(None, None, None)
    slice_list = [empty_slicer] * len(matrix.shape)

    for key, value in sorted(axes.items()):
        slice_list[key] = slice(*value)
    matrix = matrix[tuple(slice_list)]
    return matrix
