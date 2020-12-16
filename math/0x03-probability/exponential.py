#!/usr/bin/env python3
""" poisson distribution class"""


class Exponential:
    """ class"""

    def __init__(self, data=None, lambtha=1.):
        """ class constructor """
        if data is not None:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            x = float(sum(data) / len(data))
            self.lambtha = float(1 / x)
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """ calculate the pmf"""
        e = 2.7182818285
        factorial = 1
        x = int(x)

        if x < 0:
            return 0

        for i in range(1, x + 1):
            factorial = factorial * i

        return ((e**-self.lambtha)*(self.lambtha**x))/factorial

    def cdf(self, x):
        """ calculate the pmf """
        resultado = 0
        x = int(x)

        if x < 0:
            return 0

        for s in range(x + 1):
            resultado += self.pdf(s)
        return resultado
