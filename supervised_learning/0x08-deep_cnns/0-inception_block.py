#!/usr/bin/env python3
""" hat builds an inception block """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ that builds an inception block """
    He = K.initializers.he_normal()
    F1, F3R, F3, F5R, F5, FPP = filters
    F1_layer = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding='same',
                               activation='relu',
                               kernel_initializer=He)(A_prev)
    F3R_layer = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1),
                                padding='same', activation='relu',
                                kernel_initializer=He)(A_prev)
    F3_layer = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                               activation='relu',
                               kernel_initializer=He)(F3R_layer)
    F5R_layer = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                                padding='same', activation='relu',
                                kernel_initializer=He)(A_prev)
    F5_layer = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same',
                               activation='relu',
                               kernel_initializer=He)(F5R_layer)
    pool = K.layers.MaxPool2D(pool_size=(3, 3), padding='same',
                              strides=(1, 1))(A_prev)
    FPP_layer = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                                padding='same', activation='relu',
                                kernel_initializer=He)(pool)
    stack = K.layers.concatenate([F1_layer, F3_layer, F5_layer, FPP_layer])
    return stack
