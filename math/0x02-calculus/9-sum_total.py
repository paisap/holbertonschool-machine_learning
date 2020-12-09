#!/usr/bin/env python3
""" make a sum """


def summ(n, i):
    """ te function recursive """
    if i == n:
        return i**2
    else:
        return i**2 + summ(n, (i + 1))


def summation_i_squared(n):
    """ recursivamente"""
    if type(n) != int or n <= 0:
        return None
    return summ(n, 1)
