#!/usr/bin/env python3
""" that performs a valid convolution on grayscale images: """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ that performs a valid convolution on grayscale images """
    m, h, w = images.shape
    kh, kw = kernel.shape
    anhkh = h - kh + 1
    anwkw = w - kw + 1
    ans = np.zeros((m, anhkh, anwkw))
    for i in range(anhkh):
        for j in range(anwkw):
            ans[:, i, j] = (images[:, i: i + kh, j: j + kw]
                            * kernel).sum(axis=(1, 2))
    return ans
