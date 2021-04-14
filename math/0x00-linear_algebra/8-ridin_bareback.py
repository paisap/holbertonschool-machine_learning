#!/usr/bin/env python3
""" that performs matrix multiplication: """


def mat_mul(mat1, mat2):
    """ that performs matrix multiplication: """
    if len(mat1[0]) != len(mat2):
        return None
    new_mat = []
    for i in range(len(mat1)):
        aux = []
        for j in range(len(mat2[0])):
            y = 0
            for x in range(len(mat1[0])):
                y += mat1[i][x] * mat2[x][j]
            aux.append(y)
        new_mat.append(aux)
    return new_mat
