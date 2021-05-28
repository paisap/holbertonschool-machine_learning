#!/usr/bin/env python3
"""creates a tensorflow layer that includes L2 regularization: """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ creates a tensorflow layer that includes L2 regularization: """
    He = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=He, kernel_regularizer=reg)
    return layer(prev)
