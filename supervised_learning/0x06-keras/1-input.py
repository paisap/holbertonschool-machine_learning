#!/usr/bin/env python3
""" input keras """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ that builds a neural network with the Keras library: """
    _inputs = K.Input(shape=(nx,))
    regs = K.regularizers.L1L2(l2=lambtha)
    dense = K.layers.Dense(units=layers[0], activation=activations[0],
                           kernel_regularizer=regs, input_shape=(nx,))(_inputs)
    for i in range(1, len(layers)):
        dense = K.layers.Dropout(1 - keep_prob)(dense)
        dense = K.layers.Dense(units=layers[i], activation=activations[i],
                               kernel_regularizer=regs)(dense)
    model = K.Model(inputs=_inputs, outputs=dense)
    return model
