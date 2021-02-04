#!/usr/bin/env python3
""" Pooling Back Prop """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ that performs back propagation over
    a pooling layer of a neural network """
    m, h_new, w_new, c_new = dA.shape
    ma, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = pw = 0
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        for j in range(h_new):
            for x in range(w_new):
                w = j * sh
                z = x * sw
                for l in range(c_new):
                    aux1 = A_prev[i, w: w + kh, z: z + kw, l]
                    aux2 = dA[i, j, x, l]
                    if mode == 'max':
                        general = np.zeros(kernel_shape)
                        maxV = np.amax(aux1)
                        np.place(general, aux1 == maxV, 1)
                        dA_prev[i, w: w + kh, z: z + kw, l] += general * aux2
                    else:
                        avg = aux2 / (kh * kw)
                        general = np.ones(kernel_shape)
                        dA_prev[i, w: w + kh, z: z + kw, l] += general * avg
    return dA_prev
