#!/usr/bin/env python3
""" This module contains the function dense_block. """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected Convolutional
     Networks """
    He = K.initializers.he_normal()
    for i in range(layers):
        layer = K.layers.BatchNormalization()(X)
        layer = K.layers.Activation('relu')(layer)
        layer = K.layers.Conv2D(filters=4*growth_rate, kernel_size=1,
                                padding='same', kernel_initializer=He)(layer)
        layer = K.layers.BatchNormalization()(layer)
        layer = K.layers.Activation('relu')(layer)
        layer = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                padding='same', kernel_initializer=He)(layer)
        X = K.layers.concatenate([X, layer])
        nb_filters += growth_rate
    return X, nb_filters
