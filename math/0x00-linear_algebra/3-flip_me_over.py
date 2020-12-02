#!/usr/bin/env python3


def matrix_transpose(matrix):
    new_matrix = []
    contador = 0
    aux = []
    i = 0
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            aux.append(matrix[j][i])
            if (j + 1) == len(matrix):
                new_matrix.append(aux)
                aux = []
    return new_matrix
