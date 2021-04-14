#!/usr/bin/env python3
""" that concatenates two matrices along a specific axis """


def cat_matrices2D(mat1, mat2, axis=0):
    """   that concatenates two matrices along a specific axis """
    if axis == 0:
        return mat1 + mat2
    else:
        new_mat = []
        for i in range(len(mat1)):
            aux = mat1[i].copy()
            for j in range(axis):
                aux.append(mat2[i][j])
            new_mat.append(aux)
    return new_mat
