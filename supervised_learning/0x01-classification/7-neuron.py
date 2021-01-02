#!/usr/bin/env python3
""" class neuron euron performing binary classification """
import numpy as np
import matplotlib.pyplot as plt


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
        predic = A.copy()
        predic[A < 0.5] = 0
        predic[A >= 0.5] = 1
        predic = predic.astype('int')
        return [predic, costo]

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ make the gradient descent of problem """
        self.__W -= alpha * np.sum((A - Y) * X, axis=1) / Y.size
        self.__b -= alpha * np.sum((A - Y)) / Y.size

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ trains the neuron """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        A = self.forward_prop(X)
        iterat = []
        costs = []
        iterat.append(0)
        costs.append(self.cost(Y, A))
        if verbose:
            print("Cost after", 0, "iterations:", costs[-1])

        for i in range(1, iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if (verbose or graph) and (i % step == 0 or i == iterations):
                iterat.append(i)
                costs.append(self.cost(Y, A))
                if verbose:
                    print("Cost after", i, "iterations:", costs[-1])

        if graph:
            plt.plot(iterat, costs, '-b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
