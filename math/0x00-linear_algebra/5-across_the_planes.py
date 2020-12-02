#!/usr/bin/env python3

def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[i + j for i, j in zip(y, x)] for y, x in zip(mat1, mat2)]
