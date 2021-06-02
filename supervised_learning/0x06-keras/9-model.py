#!/usr/bin/env python3
""" This module contains the functions save_model and load_model. """
import tensorflow.keras as K


def save_model(network, filename):
    """ saves a model """
    K.models.save_model(network, filename)
    return None


def load_model(filename):
    """ loads the model """
    return K.models.load_model(filename)
