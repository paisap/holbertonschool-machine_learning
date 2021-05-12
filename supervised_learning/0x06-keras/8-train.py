#!/usr/bin/env python3
""" that trains a model using mini-batch gradient descent: """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):

    """ that trains a model using mini-batch gradient descent: """

    def lrd(epochs):
        """  Function for learning rate decay. """
        return alpha / (1 + decay_rate * epochs)

    callbacks = []
    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath,
                                                     save_best_only=True,
                                                     monitor='val_loss',
                                                     mode='min'))
    if validation_data:

        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                       mode='min',
                                                       patience=patience))
        if learning_rate_decay:
            callbacks.append(K.callbacks.LearningRateScheduler(lrd, verbose=1))

    return network.fit(data,
                       labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
