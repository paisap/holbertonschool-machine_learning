#!/usr/bin/env python3
""" Create Confusion """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ Create  Confusion """
    confusion = np.dot(labels.T, logits)
    return confusion
