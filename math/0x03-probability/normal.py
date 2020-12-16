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
