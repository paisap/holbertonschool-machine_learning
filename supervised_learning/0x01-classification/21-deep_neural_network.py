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

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        n_layers = self.__L
        cache = self.__cache
        weights = self.__weights

        cache['A0'] = X

        for i in range(n_layers):
            inputs = cache['A' + str(i)]
            weight = weights['W' + str(i + 1)]
            bias = weights['b' + str(i + 1)]

            cache['A' + str(i + 1)] = self.sigmoid(weight @ inputs + bias)

        return [cache['A' + str(i + 1)], cache]

    def sigmoid(self, Z):
        """ sigmoid function """
        return 1 / (1 + np.exp(-Z))

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression."""
        cost = - np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost /= Y.size

        return cost

    def evaluate(self, X, Y):
        """ evaluate de neural network """
        A, H = self.forward_prop(X)
        cost = self.cost(Y, A)

        predict = A.copy()
        predict[predict < 0.5] = 0
        predict[predict >= 0.5] = 1

        predict = predict.astype('int')

        return [predict, cost]

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network """
        n_layers = self.__L

        for i in range(n_layers, 0, -1):
            Il = cache['A' + str(i - 1)]  # Input for layer
            Wl = self.__weights['W' + str(i)]
            bl = self.__weights['b' + str(i)]
            Al = cache['A' + str(i)]

            if i != n_layers:
                dZl = (Wnl.T @ dZnl) * (Al * (1 - Al))
            else:
                Aout = cache['A' + str(n_layers)]
                dZl = Aout - Y

            Wnl = Wl.copy()

            Wl -= alpha * (dZl @ Il.T) / Y.size
            bl -= alpha * np.sum(dZl, axis=1, keepdims=True) / Y.size

            dZnl = dZl
