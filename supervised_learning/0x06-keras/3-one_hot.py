#!/usr/bin/env python3
""" that converts a label vector into a one-hot matrix: """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ that converts a label vector into a one-hot matrix::"""
    x = K.utils.to_categorical(labels, classes)
    return x
