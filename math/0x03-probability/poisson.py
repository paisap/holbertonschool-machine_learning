#!/usr/bin/env python3
""" poisson distribution class"""


class Poisson:
    """ class"""

    def __init__(self, data=None, lambtha=1.):
        """ constructor """
        if data is not None:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """ calculate the pmf"""
        e = 2.7182818285
        factorial = 1
        k = int(k)

        if k < 0:
            return 0

        for i in range(1, k + 1):
            factorial = factorial * i

        return ((e**-self.lambtha)*(self.lambtha**k))/factorial

    def cdf(self, k):
        """ calculate the pmf """
        resultado = 0
        k = int(k)

        if k < 0:
            return 0

        for s in range(k + 1):
            resultado += self.pmf(s)
        return resultado
