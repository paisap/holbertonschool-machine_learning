#!/usr/bin/env python3
""" make a sum """


def summation_i_squared(n):
    """ recursivamente"""
    if type(n) != int or n <= 0:
        return None
    return int((n*(n+1)*((2*n)+1))/6)
