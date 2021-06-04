#!/usr/bin/env python3
""" that performs a valid convolution on grayscale images: """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ that performs a valid convolution on grayscale images """
    m, h, w = images.shape
    kh, kw = kernel.shape
    anhkh = max((kh - 1) // 2, kh // 2)
    anwkw = max((kw - 1) // 2, kw // 2)
    padded = np.pad(images, ((0,), (anhkh,), (anwkw,)), 'constant')
    ans = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            ans[:, i, j] = (padded[:, i: i + kh, j: j + kw]
                            * kernel).sum(axis=(1, 2))
    return ans
