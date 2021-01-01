#!/usr/bin/env python3
""" class neuron euron performing binary classification """
import numpy as np


class Neuron:
    """ class """

    def __init__(self, nx):
        """ contructor class """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("x must be a positive integer")
        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0
