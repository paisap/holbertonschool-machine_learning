#!/usr/bin/env python3
"""that returns the transpose of a 2D matrix """


def matrix_transpose(matrix):
    """ that returns the transpose of a 2D matrix """
    new_matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return new_matrix
