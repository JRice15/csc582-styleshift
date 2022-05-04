import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def lstm1(input_shape):
    inpt = layers.Input(input_shape)
    x = inpt

    x = layers.LSTM(128)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(128)(x)
    x = layers.ReLU()(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(1)(x)
    x = layers.Activation('sigmoid')(x)

    return Model(inpt, x)

