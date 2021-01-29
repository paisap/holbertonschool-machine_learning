#!/usr/bin/env python3
""" that performs a valid convolution on grayscale images: """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ that performs a valid convolution on grayscale images """
    m, h, w = images.shape
    kh, kw = kernel.shape
    x, y = padding
    anhkh = h + 2 * x - kh + 1
    anwkw = w + 2 * y - kw + 1
    padded = np.pad(images, ((0,), (x,), (y,)), 'constant')
    ans = np.zeros((m, anhkh, anwkw))
    for i in range(anhkh):
        for j in range(anwkw):
            ans[:, i, j] = (padded[:, i: i + kh, j: j + kw]
                            * kernel).sum(axis=(1, 2))
    return ans
