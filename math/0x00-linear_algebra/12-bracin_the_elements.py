#!/usr/bin/env python3
""" function sum ,diference product """
import numpy as np


def np_elementwise(mat1, mat2):
    """ return a tuple """
    t = (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
    return t
