#!/usr/bin/env python3
""" that performs a valid convolution on grayscale images: """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ that performs a valid convolution on grayscale images """
    m, h, w = images.shape
    kh, kw = kernel.shape
    anhkh = max((kh - 1) // 2, kh // 2)
    anwkw = max((kw - 1) // 2, kw // 2)
    padded = np.pad(images, ((0, 0), (anhkh, anhkh), (anwkw, anwkw)))
    ans = np.zeros((m, kh, kw))
    for i in range(kh):
        for j in range(kw):
            ans[:, i, j] = (padded[:, i: i + kh, j: j + kw]
                            * kernel).sum(axis=(1, 2))
    return ans
