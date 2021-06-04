#!/usr/bin/env python3
""" that performs a valid convolution on grayscale images: """
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """ that performs a valid convolution on grayscale images """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = int(((h - 1) * sh - h + kh) / 2) + 1
        pw = int(((w - 1) * sw - w + kw) / 2) + 1
    else:
        ph = pw = 0
    anhkh = int((h + 2 * ph - kh) / sh + 1)
    anwkw = int((w + 2 * pw - kw) / sw + 1)
    padded = np.pad(images, ((0,), (ph,), (pw,), (0,)), 'constant')
    ans = np.zeros((m, anhkh, anwkw))
    for i in range(anhkh):
        for j in range(anwkw):
            x = i * sh
            y = j * sw
            ans[:, i, j] = (padded[:, x: x + kh, y: y + kw]
                            * kernel).sum(axis=(1, 2, 3))
    return ans
