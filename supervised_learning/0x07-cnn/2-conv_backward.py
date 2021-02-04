#!/usr/bin/env python3
""" Convolutional Back Prop """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ that performs back propagation over a convolutional
    layer of a neural network: """
    m, h_new, w_new, _ = dZ.shape
    ma, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    ph = pw = 0
    if padding == 'same':
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2) + 1
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2) + 1
    padded = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), mode='constant')
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    dA = np.zeros(padded.shape)
    dW = np.zeros(W.shape)
    for i in range(m):
        for j in range(h_new):
            for x in range(w_new):
                z = j * sh
                w = x * sw
                for l in range(c_new):
                    aux1 = dZ[i, j, x, l]
                    aux2 = padded[i, z: z + kh, w: w + kw, :]
                    dA[i, z: z + kh, w: w + kw, :] += aux1 * W[:, :, :, l]
                    dW[:, :, :, l] += aux1 * aux2
    if padding == 'same':
        dA = dA[:, ph:-ph, pw:-pw, :]
    return dA, dW, db
