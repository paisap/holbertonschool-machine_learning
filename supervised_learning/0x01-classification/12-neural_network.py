#!/usr/bin/env python3
""" class neuron euron performing binary classification """
import numpy as np


class NeuralNetwork:
    """ that defines a neural network with one hidden layer
    performing binary classification """

    def __init__(self, nx, nodes):
        """ class constructor """

        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method for the private attribute W1."""
        return self.__W1

    @property
    def b1(self):
        """Getter method for the private attribute b1."""
        return self.__b1

    @property
    def A1(self):
        """Getter method for the private attribute A1."""
        return self.__A1

    @property
    def W2(self):
        """Getter method for the private attribute W2."""
        return self.__W2

    @property
    def b2(self):
        """Getter method for the private attribute b2."""
        return self.__b2

    @property
    def A2(self):
        """Getter method for the private attribute A2."""
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron """
        z = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(z)

        z1 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(z1)

        return self.__A1, self.__A2

    def sigmoid(self, Z):
        """ sigmoid function """
        return 1 / (1 + np.exp(-Z))

    def cost(self, Y, A):
        """ cost function """
        cost = - np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost /= Y.size
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions and cost"""
        A = self.forward_prop(X)
        costo = self.cost(Y, self.__A2)
        predic = self.__A2.copy()
        predic[self.__A2 < 0.5] = 0
        predic[self.__A2 >= 0.5] = 1
        predic = predic.astype('int')
        return [predic, costo]
