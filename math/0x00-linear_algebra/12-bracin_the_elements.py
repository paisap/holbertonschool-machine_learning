#!/usr/bin/env python3
""" that performs element-wise
    addition, subtraction, multiplication, and division: """


def np_elementwise(mat1, mat2):
    """ that performs element-wise
        addition, subtraction, multiplication, and division: """
    tupla = (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
    return tupla
