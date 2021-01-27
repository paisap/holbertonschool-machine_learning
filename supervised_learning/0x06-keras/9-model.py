#!/usr/bin/env python3
""" This module contains the functions save_model and load_model. """
import tensorflow.keras as K


def saver_model(network, filename):
    """ saves a model """
    network.save(filename)
    return None


def load_model(filename):
    """ loads the model """
    return K.models.load_model(filename)
