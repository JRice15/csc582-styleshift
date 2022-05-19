import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import load_preprocessed_sentence_data


max_sent_len = 100
embedding_dim = 50

X, Y, embedding_matrix = load_preprocessed_sent_data(max_sent_len, embedding_dim, 
                                drop_equal=True, target="label")

print("Building model...")

inpt = layers.Input((max_sent_len,), dtype=X.dtype)
x = inpt

# convert to embedded vectors
x = layers.Embedding(
    vectorizer.index_size,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)(x)

x = layers.LSTM(128)(x)

x = layers.BatchNormalization()(x)
x = layers.Dense(128)(x)
x = layers.ReLU()(x)

x = layers.BatchNormalization()(x)
x = layers.Dense(1)(x)
x = layers.Activation('sigmoid')(x)

model = Model(inpt, x)


print("Building and training...")
X, x_test, Y, y_test = train_test_split(X, Y, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)

model.summary()

callback_lst = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(patience=5, factor=0.1, verbose=1),
    callbacks.ModelCheckpoint("classifier.h5", save_best_only=True)
]

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=16,
    epochs=100,
    callbacks=callback_lst
)

print("\nEvaluating...")
print(model.evaluate(x_test, y_test))


