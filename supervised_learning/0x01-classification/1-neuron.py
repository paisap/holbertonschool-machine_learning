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
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ getter function """
        return self.__W

    @property
    def b(self):
        """ getter function """
        return self.__b

    @property
    def A(self):
        """ getter function """
        return self.__A
