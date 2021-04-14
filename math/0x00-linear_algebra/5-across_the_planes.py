#!/usr/bin/env python3
""" that adds two matrices element-wise: """


def add_matrices2D(mat1, mat2):
    """   that adds two matrices element-wise: """
    l1 = len(mat1)
    l2 = len(mat2)
    if (l1 != l2) or (len(mat1[0]) != len(mat2[0])):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(l1)] for i in range(l1)]
