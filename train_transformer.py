import logging
import time
import argparse
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability
from sklearn.model_selection import train_test_split

from const import MAX_SENT_LEN
from load_data import load_preprocessed_sent_data, make_embedding_matrix
from transformer import Transformer
from tf_utils import MyModelCheckpoint

PRESETS = {
  "default": {
    "n_layers": 2,
    "d_model": 128,
    "d_ff": 512,
    "n_heads": 4,
    "d_key": 64,
    "dropout": 0.1,
  },
  "small": {
    "n_layers": 1,
    "d_model": 50,
    "d_ff": 64,
    "n_heads": 2,
    "d_key": 32,
    "dropout": 0.1,
  },
  # Baseline in Vaswani et al https://arxiv.org/pdf/1706.03762.pdf
  "orig": {
    "n_layers": 6,
    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 8,
    "d_key": 64,
    "dropout": 0.1,
    # "label_smoothing": 0.1
  }
}

parser = argparse.ArgumentParser()
# presets
parser.add_argument("--preset",default="default",choices=list(PRESETS.keys()))

# model params
parser.add_argument("--n-layers",type=int)
parser.add_argument("--d-model",type=int,help="dimension units in model")
parser.add_argument("--d-ff",type=int,help="hidden units in feedforward nets")
parser.add_argument("--n-heads",type=int,help="number of attention heads")
parser.add_argument("--d-key",type=int,help="dimension of key in attention")
parser.add_argument("--dropout",type=int)

# data params
parser.add_argument("--use-glove",action="store_true")
parser.add_argument("--max-vocab",type=int)
# parser.add_argument("--label-smoothing",type=float,default=0.1,help="e_ls in paper")

# training params
parser.add_argument("--batchsize",type=int,default=64) 
parser.add_argument("--epochs",type=int,default=100,help="max number of epochs (if early stopping doesn't occur")
parser.add_argument("--earlystopping-epochs",type=int,default=2)
parser.add_argument("--test",action="store_true",help="just run a small test version")

# misc
parser.add_argument("--save-path",default="transformer.tf",help="path tp save model to (must end with '.tf')")
ARGS = parser.parse_args()

# set preset values which haven't been overridden by cl args
for name,value in PRESETS[ARGS.preset].items():
  if getattr(ARGS, name) is None:
    setattr(ARGS, name, value)

assert ARGS.save_path.endswith(".tf")

pprint(vars(ARGS))


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
    lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    return lr

  def get_config(self):
    return {
      "d_model": int(self.d_model.numpy()),
      "warmup_steps": int(self.warmup_steps),
    }

lr_schedule = CustomSchedule(ARGS.d_model)
optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


### Loss and metrics
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# loss_spec = [
#     tf.TensorSpec(shape=(None,MAX_SENT_LEN-1), dtype=tf.int32),
#     tf.TensorSpec(shape=(None,MAX_SENT_LEN-1,vectorizer.vocab_size), dtype=tf.float32),    
# ]

# Since the target sequences are padded, it is important to apply a padding mask 
# when calculating the loss.

@tf.function
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

@tf.function
def accuracy_metric(real, pred):
  real = tf.cast(real, tf.int32)
  accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=tf.int32))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

accuracy_metric.__name__ = "my_acc"



### Dataset

dataset, vectorizer = load_preprocessed_sent_data(target="simple", drop_equal=True, 
                          start_end_tokens=True, max_vocab=ARGS.max_vocab)
x_train, y_train, x_val, y_val, x_test, y_test = dataset

if ARGS.use_glove:
  assert ARGS.d_model in [50, 100, 200, 300], "d_model must be equal to a glove embedding size"
  embedding_matrix = make_embedding_matrix(ARGS.d_model, vectorizer)


### Training and checkpointing

model = Transformer(
    num_layers=ARGS.n_layers,
    num_heads=ARGS.n_heads,
    d_model=ARGS.d_model,
    d_ff=ARGS.d_ff,
    d_key=ARGS.d_key,
    vocab_size=vectorizer.vocab_size,
    rate=ARGS.dropout,
    embedding_matrix=embedding_matrix if ARGS.use_glove else None,
)


# call with sample batch to build model shapes
sample_x = x_train[:ARGS.batchsize]
sample_y = y_train[:ARGS.batchsize, :-1]
print("Building...")
output, attn_weights = model([sample_x, sample_y])
print("batch shapes:")
print(" ", sample_x.shape, sample_y.shape)
print(" ", sample_x.dtype, sample_y.dtype)
print("output:", output.shape, output.dtype)

print("Summary:")
model.summary()


print("Compiling...")
model.compile(
  optimizer=optimizer,
  loss=loss_function,
  metrics=[accuracy_metric],
)

print("Training...")
callback_list = [
  tf.keras.callbacks.EarlyStopping(patience=ARGS.earlystopping_epochs, verbose=1),
  MyModelCheckpoint(ARGS.save_path + "transformer.tf", epochs_per_save=1, 
      save_best_only=True, verbose=1),
]

if ARGS.test:
  size = ARGS.batchsize * 20
  x_train = x_train[:size]
  y_train = y_train[:size]
  x_val = x_val[:size]
  y_val = y_val[:size]
  x_test = x_test[:size]
  y_test = y_test[:size]

model.fit(
  x_train, y_train,
  validation_data=(x_val, y_val),
  batch_size=ARGS.batchsize,
  epochs=ARGS.epochs,
  callbacks=callback_list,
)

