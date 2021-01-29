#!/usr/bin/env python3
""" that performs a valid convolution on grayscale images: """
import numpy as np

ans = np.random.randint(1, 8, size=(5, 5, 5))
print(ans)
print("posiciones")
print(ans[:, 0:3, 0:3])