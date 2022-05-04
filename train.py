import numpy as np
import pandas as pd

from preprocess import TextEmbedder, TextTokenizer
from load_data import read_data

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks

from model import lstm1

def load_data():
    data = read_data("sentence")

    tokenizer = TextTokenizer()

    data["simple"] = data["simple"].apply(tokenizer.tokenize_sent)
    data["normal"] = data["normal"].apply(tokenizer.tokenize_sent)

    max_sent_len = 100
    print("max sent len:", max_sent_len)
    print("fraction of sentences truncated:", data["normal"].apply(lambda x: len(x) > max_sent_len).mean())

    vectorizer = TextEmbedder(max_sent_len, embedding_dim=50)

    normal_vec = vectorizer.convert_texts(data.normal)
    simple_vec = vectorizer.convert_texts(data.simple)

    X = np.concatenate([normal_vec, simple_vec], axis=0)
    Y = np.concatenate([np.ones((len(normal_vec),)), np.ones((len(simple_vec),))])

    return X, Y


def main():
    X, Y = load_data()

    X, x_test, Y, y_test = train_test_split(X, Y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1)

    model = lstm1(x_train[0].shape)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    model.summary()

    callback_lst = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.1, verbose=1),
    ]

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=16,
        epochs=2,
        callbacks=callback_lst
    )

    print(model.evaluate(x_test, y_test))




if __name__ == "__main__":
    main()
