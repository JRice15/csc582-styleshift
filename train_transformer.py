import logging
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability
from sklearn.model_selection import train_test_split

from const import MAX_SENT_LEN
from load_data import load_preprocessed_sent_data
from transformer import Transformer

# To keep this example small and relatively fast, the values for `num_layers, 
# d_model, dff` have been reduced. 
# The base model described in the [paper](https://arxiv.org/abs/1706.03762) used: 
# `num_layers=6, d_model=512, dff=2048`.

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize",type=int,default=128) 
parser.add_argument("--n-layers",type=int,default=4)
parser.add_argument("--d-model",type=int,default=128,help="hidden units in model")
parser.add_argument("--d-ff",type=int,default=512,help="hidden units in feedforward nets")
parser.add_argument("--n-heads",type=int,default=8,help="number of attention heads")
parser.add_argument("--dropout",type=int,default=0.1)
parser.add_argument("--epochs",type=int,default=100,help="max number of epochs (if early stopping doesn't occur")
parser.add_argument("--earlystopping-epochs",type=int,default=5)
ARGS = parser.parse_args()



# Use the Adam optimizer with a custom learning rate scheduler according to the 
# formula in the [paper](https://arxiv.org/abs/1706.03762).
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(ARGS.d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


### Loss and metrics

# Since the target sequences are padded, it is important to apply a padding mask 
# when calculating the loss.

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

@tf.function
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

@tf.function
def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)



dataset, vectorizer = load_preprocessed_sent_data(MAX_SENT_LEN, 50, target="simple", drop_equal=True)

x_train, y_train, x_val, y_val, x_test, y_test = dataset

### Training and checkpointing

model = Transformer(
    num_layers=ARGS.n_layers,
    d_model=ARGS.d_model,
    num_heads=ARGS.n_heads,
    dff=ARGS.d_ff,
    input_vocab_size=vectorizer.vocab_size,
    target_vocab_size=vectorizer.vocab_size,
    rate=ARGS.dropout)


# call with sample batch to build model shapes
sample_x = x_train[:ARGS.batchsize]
sample_y = y_train[:ARGS.batchsize, :-1]
print(sample_x.shape, sample_y.shape)
print(sample_x.dtype, sample_y.dtype)
print("Building...")
model([sample_x, sample_y])

print("Summary:")
model.summary()

print("Compiling...")
model.compile(
  optimizer=optimizer,
  loss=loss_function,
  metrics=[accuracy_function],
)

print("Training...")
callback_list = [
  tf.keras.callbacks.EarlyStopping(patience=ARGS.earlystopping_epochs),
  tf.keras.callbacks.ModelCheckpoint("transformer.h5", save_best_only=True)
]

model.fit(
  x_train, y_train,
  validation_data=(x_val, y_val),
  batch_size=ARGS.batchsize,
  epochs=ARGS.epochs,
  callbacks=callback_list,
)


