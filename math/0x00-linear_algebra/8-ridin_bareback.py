#!/usr/bin/env python3


def mat_mul(mat1, mat2):
    """ mul two mat """
    if len(mat1[0]) != len(mat2):
        return None
    ward = 0
    mat = []
    while ward < len(mat1):
        aux = []
        for i in range(len(mat2[0])):
            multiply = 0
            for j in range(len(mat2)):
                multiply += mat1[ward][j] * mat2[j][i]
            aux.append(multiply)
        mat.append(aux)
        ward += 1
    return mat
