#!/usr/bin/env python3
""" that trains a model using mini-batch gradient descent: """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """ that trains a model using mini-batch gradient descent: """
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   mode='min',
                                                   patience=patience))
    return network.fit(data,
                       labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
