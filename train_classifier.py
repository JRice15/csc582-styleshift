import argparse
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import Model, callbacks, layers

from const import MAX_SENT_LEN
from load_data import load_preprocessed_sent_data, make_embedding_matrix
from tf_utils import MyModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize",type=int,default=64)
parser.add_argument("--embedding-dim",type=int,default=50)
parser.add_argument("--n-hidden-units",type=int,default=128)

parser.add_argument("--lr",type=float,default=1e-3)
parser.add_argument("--reducelr-epochs",type=int,default=5)
parser.add_argument("--reducelr-factor",type=float,default=0.1)
parser.add_argument("--earlystopping-epochs",type=int,default=12)
ARGS = parser.parse_args()

pprint(vars(ARGS))

dataset, vectorizer = load_preprocessed_sent_data(drop_equal=True, target="label")

embedding_matrix = make_embedding_matrix(ARGS.embedding_dim, vectorizer)

print("Building model...")

inpt = layers.Input((MAX_SENT_LEN,), dtype=tf.float32)
x = inpt

# convert to embedded vectors
x = layers.Embedding(
    vectorizer.vocab_size,
    ARGS.embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)(x)

x = layers.LSTM(ARGS.n_hidden_units)(x)

x = layers.BatchNormalization()(x)
x = layers.Dense(128)(x)
x = layers.ReLU()(x)

x = layers.BatchNormalization()(x)
x = layers.Dense(1)(x)
# no sigmoid when using from_logits=True in loss
# x = layers.Activation('sigmoid')(x)

model = Model(inpt, x)


print("Compiling and training...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["binary_accuracy"],
)

model.summary()

callback_lst = [
    callbacks.EarlyStopping(patience=ARGS.earlystopping_epochs, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(patience=ARGS.reducelr_epochs, factor=ARGS.reducelr_factor, verbose=1),
    MyModelCheckpoint("classifier.h5", epoch_per_save=5, save_best_only=True, verbose=1)
]

x_train, y_train, x_val, y_val, x_test, y_test = dataset


model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=ARGS.batchsize,
    epochs=200,
    callbacks=callback_lst
)

print("\nEvaluating...")
print(model.evaluate(x_test, y_test))


