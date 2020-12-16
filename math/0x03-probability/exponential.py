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
