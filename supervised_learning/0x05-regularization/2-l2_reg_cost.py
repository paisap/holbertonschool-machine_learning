#!/usr/bin/env python3
""" hat calculates the cost of a neural network with L2 regularization """
import numpy as np
import tensorflow as tf


def l2_reg_cost(cost):
    """ hat calculates the cost of a neural network with L2 regularization: """
    return cost + tf.losses.get_regularization_losses()
