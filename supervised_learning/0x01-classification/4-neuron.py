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

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(z)
        return self.__A

    def sigmoid(self, Z):
        """ sigmoid function """
        return 1 / (1 + np.exp(-Z))

    def cost(self, Y, A):
        """ calculate the cost of neuron """
        cost = - np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost /= Y.size
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions and cost"""
        A = self.forward_prop(X)
        costo = self.cost(Y, A)
        predic = A
        predic[A < 0.5] = 0
        predic[A >= 0.5] = 1
        return [predic, costo]
