#!/usr/bin/env python3
""" tthat performs forward propagation over
a pooling layer of a neural network: """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ tthat performs forward propagation over
    a pooling layer of a neural network: """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    if mode == 'max':
        mode_np = np.max
    else:
        mode_np = np.average
    anhkh = int((h_prev - kh) / sh + 1)
    anwkw = int((w_prev - kw) / sw + 1)
    ans = np.zeros((m, anhkh, anwkw, c_prev))
    for i in range(anhkh):
        for j in range(anwkw):
            x = i * sh
            y = j * sw
            ans[:, i, j, :] = mode_np(A_prev[:, x: x + kh, y: y + kw, :],
                                      axis=(1, 2))
    return ans
