#!/usr/bin/env python3
"""Shannon entropy and P affinities"""

import numpy as np


def HP(Di, beta):
    """
    Function that calculates shannon entropy
    """
    P = np.exp(-Di * beta)
    sumP = np.sum(P)
    Pi = P / sumP
    Hi = -np.sum(Pi * np.log2(Pi))
    return (Hi, Pi)
