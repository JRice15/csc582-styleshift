import logging
import time
import json
import argparse
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from const import MAX_SENT_LEN, START_TOKEN, END_TOKEN, PADDING_TOKEN, SPECIAL_TOKENS


@tf.function
def _tf_greedy_predict(transformer, input_tokens, *, start, end, pad):
  # `tf.TensorArray` is required here (instead of a python list) so that the
  # dynamic-loop can be traced by `tf.function`.
  output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  output_array = output_array.write(0, start)
  for i in tf.range(1, MAX_SENT_LEN):
    output_array = output_array.write(i, pad)

  for i in tf.range(1, MAX_SENT_LEN):
    output = tf.transpose(output_array.stack())

    is_end = (output == end[0])
    if tf.reduce_all(tf.reduce_any(is_end, axis=-1), axis=0): # if every sentence contains at least one end token
      break

    output = output[:,:-1] # (batch_size, max_sent_len-1)
    predictions, _ = transformer([input_tokens, output], training=False)

    # select the last token from the seq_len dimension
    predictions = predictions[:, i-1, :]  # (batch_size, vocab_size)

    predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)

    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output_array = output_array.write(i, predicted_id)

    # if predicted_id == end:
    #   break

  output = tf.transpose(output_array.stack())
  # output.shape (batchsize, tokens)

  # `tf.function` prevents us from using the attention_weights that were
  # calculated on the last iteration of the loop. So recalculate them outside
  # the loop.
  _, auxiliary_outputs = transformer([input_tokens, output[:,:-1]], training=False)

  return output, auxiliary_outputs


def greedy_predict(transformer, input_tokens, vectorizer, attn_key, batchsize=32):
  """
  args:
    transformer: model
    input_tokens: vectorized inputs, array of ints, shape (n sentences, max sent len)
    vectorizer: TextVectorizer instance
    attn_key: key into auxiliary_outputs to get attention weights
    batchsize: size of batches to predict in
  returns:
    preds: shape (n sentences, MAX_SENT_LEN)
    attn: shape (n sentences, n attn heads, MAX_SENT_LEN-1, MAX_SENT_LEN)
  """
  if len(input_tokens.shape) == 1:
    input_tokens = input_tokens[tf.newaxis]

  all_preds = []
  all_attn = []
  for idx in tqdm(range(0, len(input_tokens), batchsize)):
    input_batch = input_tokens[idx:idx+batchsize]
    n_examples = len(input_batch) # may be less than batchsize on last batch
    # create start/end/padding batch tokens
    start = tf.cast(vectorizer.vectorize([START_TOKEN] * n_examples), tf.int32)
    end = tf.cast(vectorizer.vectorize([END_TOKEN] * n_examples), tf.int32)
    pad = tf.cast(vectorizer.vectorize([PADDING_TOKEN] * n_examples), tf.int32)
    # generate predictions
    pred, aux_outputs = _tf_greedy_predict(transformer, input_batch, start=start, end=end, pad=pad)
    all_preds.append(pred)
    all_attn.append(aux_outputs[attn_key])

  all_preds = tf.concat(all_preds, axis=0)
  all_attn = tf.concat(all_attn, axis=0)

  return all_preds, all_attn


def interpolate_OOV_predictions(preds, )
