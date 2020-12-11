#!/usr/bin/env python3
""" make a derivate """


def poly_derivative(poly):
    """ that calculates the derivative of a polynomial"""
    if type(poly) != list or len(poly) == 0:
        return None
    for i in poly:
        if type(i) != int and type(i) != float:
            return None
    if len(poly) == 1:
        return [0]
    aux = []
    for i in range(1, len(poly)):
        if poly[i] != 0:
            aux.append(poly[i] * i)
        else:
            aux.append(0)
    return aux
