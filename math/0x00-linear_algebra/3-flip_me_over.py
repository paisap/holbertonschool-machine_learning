#!/usr/bin/env python3
"""that returns the transpose of a 2D matrix """


def matrix_transpose(matrix):
    """ that returns the transpose of a 2D matrix """
    l1 = len(matrix)
    l2 = len(matrix[0])
    new_matrix = [[matrix[j][i] for j in range(l1)] for i in range(l2))]
    return new_matrix
