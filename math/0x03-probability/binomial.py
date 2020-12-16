#!/usr/bin/env python3
""" represents a  binomial class"""


class Binomial:
    """ class"""

    def __init__(self, data=None, n=1, p=0.5):
        """ constructor """
        if data is not None:
            if type(data) != list:
                raise TypeError("data must be a list")

            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            medio = float(sum(data) / len(data))
            lista = [(x - medio) ** 2 for x in data]
            variance = sum(lista) / len(data)
            p = 1 - variance / medio
            if ((medio / p) - (medio // p)) >= 0.5:
                self.n = 1 + int(medio / p)
            else:
                self.n = int(medio / p)
            self.p = float(medio / self.n)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
