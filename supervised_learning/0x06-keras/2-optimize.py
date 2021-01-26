#!/usr/bin/env python3
""" that sets up Adam optimization for a
keras model with categorical crossentropy loss and accuracy metrics:"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ that sets up Adam optimization for a
    keras model with categorical crossentropy loss and accuracy metrics:"""
    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy', metrics=['accuracy'])
    return None
