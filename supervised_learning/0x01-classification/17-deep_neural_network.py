#!/usr/bin/env python3
"""DeepNeuralNetwork class."""
import numpy as np


class DeepNeuralNetwork:
    """class that defines a deep neural network """

    def __init__(self, nx, layers):
        """Class constructor."""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__cache = {}
        self.__weights = {}
        self.__L = len(layers)

        for i in range(len(layers)):
            val = layers[i]

            if type(val) != int or val < 1:
                raise TypeError("layers must be a list of positive integers")

            nameW = 'W' + str(i + 1)
            nameb = 'b' + str(i + 1)
            neurons = layers[i]

            if i != 0:
                weights = layers[i - 1]
            else:
                weights = nx

            self.__weights[nameW] = np.random.randn(neurons,
                                                    weights)*np.sqrt(2/weights)
            self.__weights[nameb] = np.zeros((neurons, 1))

    @property
    def L(self):
        """ Getter method for attribute L """
        return self.__L

    @property
    def cache(self):
        """ Getter method for attribute cache """
        return self.__cache

    @property
    def weights(self):
        """ Getter method for attrubute weights """
        return self.__weights
