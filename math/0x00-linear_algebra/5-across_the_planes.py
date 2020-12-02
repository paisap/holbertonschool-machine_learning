#!/usr/bin/env python3
""" Implemetns add_matrices2D """


def add_matrices2D(mat1, mat2):
    """ add two matrices 2D """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[i + j for i, j in zip(y, x)] for y, x in zip(mat1, mat2)]
