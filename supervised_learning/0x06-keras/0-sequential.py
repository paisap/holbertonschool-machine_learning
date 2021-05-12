#!/usr/bin/env python3
""" Sequential keras """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ that builds a neural network with the Keras library: """
    NN = K.Sequential()
    regs = K.regularizers.L1L2(l2=lambtha)
    NN.add(K.layers.Dense(units=layers[0], activation=activations[0],
                          kernel_regularizer=regs, input_shape=(nx,)))
    for i in range(1, len(layers)):
        NN.add(K.layers.Dropout(1 - keep_prob))
        NN.add(K.layers.Dense(units=layers[i], activation=activations[i],
                              kernel_regularizer=regs))
    return NN
