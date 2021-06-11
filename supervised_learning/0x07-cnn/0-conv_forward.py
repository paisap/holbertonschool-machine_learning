#!/usr/bin/env python3
""" that performs forward propagation over a
convolutional layer of a neural network: """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ that performs forward propagation
    over a convolutional layer of a neural network: """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev1, c_new = W.shape
    sh, sw = stride
    ph = pw = 0
    if padding == 'same':
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2)
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2)
    padded = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), mode='constant')
    ansh = int((h_prev + 2 * ph - kh) / sh + 1)
    answ = int((w_prev + 2 * pw - kw) / sw + 1)
    new = np.zeros((m, ansh, answ, c_new))
    for i in range(ansh):
        for j in range(answ):
            for x in range(c_new):
                k = i * sh
                z = j * sw
                new[:, i, j, x] = (padded[:, k: k + kh, z: z + kw, :] *
                                   W[:, :, :, x]).sum(axis=(1, 2, 3))
    new = activation(new + b)
    return new
