#!/usr/bin/env python3
""" This module contains the function transition_layer. """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected Convolutional
     Networks.
    """
    He = K.initializers.he_normal()
    layer = K.layers.BatchNormalization()(X)
    layer = K.layers.Activation('relu')(layer)
    nb_filters = int(nb_filters * compression)
    layer = K.layers.Conv2D(filters=nb_filters, kernel_size=1, padding='same',
                            kernel_initializer=He)(layer)
    layer = K.layers.AveragePooling2D(pool_size=2, padding='same')(layer)
    return layer, nb_filters
