import numpy as np
import pandas as pd

from preprocess import TextTokenizer
from load_data import read_data, load_glove_embeddings

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks


max_sent_len = 100
embedding_dim = 50

data = read_data("sentence")

tokenizer = TextTokenizer(max_sent_len)

data.simple = data.simple.apply(tokenizer.tokenize_sent)
data.normal = data.normal.apply(tokenizer.tokenize_sent)

print("max sent len:", max_sent_len)
print("fraction of sentences truncated:", data["normal"].apply(lambda x: len(x) > max_sent_len).mean())

data.simple = data.simple.apply(tokenizer.pad_and_truncate)
data.normal = data.normal.apply(tokenizer.pad_and_truncate)


X = np.concatenate([
        data.normal.to_list(), 
        data.simple.to_list()], axis=0, dtype=np.str_)
Y = np.concatenate([np.ones(len(data.normal)), np.zeros(len(data.simple))], axis=0)


embeddings = load_glove_embeddings(embedding_dim)

vocab = list(embeddings.keys())
vectorizer = tf.keras.layers.TextVectorization(
    standardize=None,
    split=None,
    output_mode='int',
    output_sequence_length=max_sent_len,
    vocabulary=vocab)

# Prepare embedding matrix
embedding_matrix = np.zeros((len(vocab)+2, embedding_dim))
for index,word in enumerate(vocab):
    embedding_vector = embeddings.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[index] = embedding_vector


inpt = layers.Input((max_sent_len,), dtype="string")
# convert to string to sequence of ints
x = vectorizer(inpt)

# convert to embedded vectors
x = layers.Embedding(
    len(vocab)+2,
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
]

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=16,
    epochs=100,
    callbacks=callback_lst
)

print(model.evaluate(x_test, y_test))




if __name__ == "__main__":
    main()
