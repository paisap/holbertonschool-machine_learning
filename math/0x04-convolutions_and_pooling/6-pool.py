#!/usr/bin/env python3
""" that performs a valid convolution on grayscale images: """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ that performs a valid convolution on grayscale images """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    if mode == 'max':
        mode_np = np.max
    else:
        mode_np = np.average
    anhkh = int((h - kh) / sh + 1)
    anwkw = int((w - kw) / sw + 1)
    ans = np.zeros((m, anhkh, anwkw, c))
    for i in range(anhkh):
        for j in range(anwkw):
            x = i * sh
            y = j * sw
            ans[:, i, j, :] = mode_np(images[:, x: x + kh, y: y + kw, :],
                                      axis=(1, 2))
    return ans
