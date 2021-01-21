#!/usr/bin/env python3
""" that creates a layer of a neural network using dropout:"""
import numpy as np


def dropout_create_layer(prev, n, activation, keep_prob):
    """ that creates a layer of a neural network using dropout: """
    aux = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=aux)
    drop = tf.layers.Dropout(keep_prob)

    return drop(layer(prev))
