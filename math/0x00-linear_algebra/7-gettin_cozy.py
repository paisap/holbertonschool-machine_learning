#!/usr/bin/env python3
""" function """


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatanet """
    new_mat = []
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        for row in mat1:
            new_mat.append(list(row))
        for row in mat2:
            new_mat.append(list(row))
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        for row, element in zip(mat1, mat2):
            new_mat.append(list(row) + element)
    return new_mat
