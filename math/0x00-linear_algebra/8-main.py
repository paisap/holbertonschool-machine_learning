#!/usr/bin/env python3

mat_mul = __import__('8-ridin_bareback').mat_mul

mat1 = [[3, 1],
        [0, -2]]
mat2 = [[1, -2, 5],
        [0, 4, 3]]
print(mat_mul(mat1, mat2))
