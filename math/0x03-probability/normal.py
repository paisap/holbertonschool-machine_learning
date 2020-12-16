#!/usr/bin/env python3
""" represents a normal distribution class"""


class Normal:
    """ class"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """ constructor """
        if data is not None:
            if type(data) != list:
                raise TypeError("data must be a list")

            if len(data) <= 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))
            my_list = [(x - self.mean) ** 2 for x in data]
            self.stddev = float((sum(my_list) / (len(data))) ** 0.5)

        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """ return x_score """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ return z value """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """ calculate the pdf """
        e = 2.7182818285
        π = 3.1415926536

        formula_pdf = 1 / (self.stddev * (2 * π) ** 0.5) * \
            e ** (- (x - self.mean)**2 / (2 * self.stddev**2))

        return formula_pdf
