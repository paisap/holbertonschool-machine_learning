#!/usr/bin/env python3
""" This module contains the functions Save and Load Configuration. """
import tensorflow.keras as K


def save_config(network, filename):
    """  saves a model’s configuration in JSON format: """
    model_json = network.to_json()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(model_json)
    return None


def load_config(filename):
    """  loads a model with a specific configuration: """
    with open(filename, "r", encoding="utf-8") as f:
        config = f.read()
        loaded = K.models.model_from_json(config)
    return loaded
