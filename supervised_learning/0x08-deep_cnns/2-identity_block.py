#!/usr/bin/env python3
""" hat builds an inception block """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ that builds an inception block """
    He = K.initializers.he_normal()
    F11, F3, F12 = filters
    F11_layer = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                                padding='same', kernel_initializer=He)(A_prev)
    F11_layer = K.layers.BatchNormalization(axis=3)(F11_layer)
    F11_layer = K.layers.Activation('relu')(F11_layer)
    F3_layer = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                               kernel_initializer=He)(F11_layer)
    F3_layer = K.layers.BatchNormalization(axis=3)(F3_layer)
    F3_layer = K.layers.Activation('relu')(F3_layer)
    F12_layer = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                                padding='same',
                                kernel_initializer=He)(F3_layer)
    F12_layer = K.layers.BatchNormalization(axis=3)(F12_layer)
    iden = K.layers.Add()([F12_layer, A_prev])
    iden = K.layers.Activation('relu')(iden)
    return iden
