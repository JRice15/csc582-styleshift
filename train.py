import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import TextTokenizer, TextVectorizer
from load_data import read_data, load_glove_embeddings


max_sent_len = 100
embedding_dim = 50

print("Loading data...")

data = read_data("sentence")

tokenizer = TextTokenizer(max_sent_len)

data.simple = data.simple.apply(tokenizer.tokenize_sent)
data.normal = data.normal.apply(tokenizer.tokenize_sent)

print("max sent len:", max_sent_len)
print("fraction of sentences truncated:", data["normal"].apply(lambda x: len(x) > max_sent_len).mean())

X_normal = pad_sequences(
            data.normal.to_list(),
            maxlen=max_sent_len,
            dtype=np.str_, truncating="post")
X_simple = pad_sequences(
            data.simple.to_list(),
            maxlen=max_sent_len,
            dtype=np.str_, truncating="post")

X = np.concatenate([X_normal, X_simple], axis=0, dtype=np.str_)
Y = np.concatenate([np.ones(len(data.normal)), np.zeros(len(data.simple))], axis=0)

print("Loading embeddings & vectorizing data...")
embeddings = load_glove_embeddings(embedding_dim)

vocab = list(embeddings.keys())
vectorizer = TextVectorizer(vocab)

# convert strings to ints
X = vectorizer.vectorize(X)

# Prepare embedding matrix
embedding_matrix = np.zeros((vectorizer.index_size, embedding_dim))
for word,index in vectorizer.word_index.items():
    embedding_vector = embeddings.get(word)
    embedding_matrix[index] = embedding_vector


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




if __name__ == "__main__":
    main()
