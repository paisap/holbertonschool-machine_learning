#!/usr/bin/env python3
""" make a derivate """


def poly_derivative(poly):
    """ that calculates the derivative of a polynomial"""
    if isinstance(poly, list) == False:
        return None
    aux = []
    for i in range(1, len(poly)):
        if poly[i] != 0:
            aux.append(poly[i] * i)
        else:
            aux.append(0)
    return aux
