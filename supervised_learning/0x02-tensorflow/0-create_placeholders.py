#!/usr/bin/env python3

import tensorflow as tf

def create_placeholders(nx, classes):
    """ create two place"""
    x = tf.placeholder(dtype=float32, shape=(None, nx), name="x")
    y = tf.placeholder(dtype=float32, shape=(None, classes), name="y")
    return x, y
