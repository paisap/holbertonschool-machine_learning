#!/usr/bin/env python3

def cat_matrices2D(mat1, mat2, axis=0):
    mat = []
    size = len(mat1) + len(mat2)
    for i in range(size):
        aux = mat1[i] + mat2[i]
        mat.append(aux)
    return mat
