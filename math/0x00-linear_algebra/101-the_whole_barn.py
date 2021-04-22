#!/usr/bin/env python3
""" el numpy """


def matrix_shape(matrix):
    """ know shape of matrix """
    if type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])


def add_two_matrices(mat1, mat2):
    """recursivly adds two matrices"""
    sum = []
    for i in range(len(mat1)):
        if type(mat1[i]) == list:
            sum.append(add_two_matrices(mat1[i], mat2[i]))
        else:
            sum.append(mat1[i] + mat2[i])
    return sum


def add_matrices(mat1, mat2):
    """ add two matices """

    size1 = matrix_shape(mat1)
    size2 = matrix_shape(mat2)
    if size1 != size2:
        return None
    return add_two_matrices(mat1, mat2)
