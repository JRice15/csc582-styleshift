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
def _tf_predict_sents(transformer, input_tokens, *, start, end, pad):
  # `tf.TensorArray` is required here (instead of a python list) so that the
  # dynamic-loop can be traced by `tf.function`.
  output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  output_array = output_array.write(0, start)
  for i in tf.range(1, MAX_SENT_LEN):
    output_array = output_array.write(i, pad)

  for i in tf.range(1, MAX_SENT_LEN):
    output = tf.transpose(output_array.stack())
    output = output[:,:-1]

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

  return output


def predict_sentences(transformer, input_tokens, vectorizer, batchsize=32):
  """
  args:
    transformer: model
    input_tokens: vectorized inputs, array of ints, shape (n sentences, max sent len)
    vectorizer: TextVectorizer instance
    attn_key: key into auxiliary_outputs to get attention weights
    batchsize: size of batches to predict in
  """
  if len(input_tokens.shape) == 1:
    input_tokens = input_tokens[tf.newaxis]
  # batchsize = input_tokens.shape[0]

  start = tf.cast(vectorizer.vectorize([START_TOKEN] * batchsize), tf.int32)
  end = tf.cast(vectorizer.vectorize([END_TOKEN] * batchsize), tf.int32)
  pad = tf.cast(vectorizer.vectorize([PADDING_TOKEN] * batchsize), tf.int32)

  all_preds = []
  for idx in tqdm(range(0, len(input_tokens), batchsize)):
    input_batch = input_tokens[idx:idx+batchsize]
    pred, _ = _tf_predict_sents(transformer, input_batch, start=start, end=end, pad=pad)
    all_preds.append(pred)

  all_preds = tf.concatentate(all_preds, axis=0)

  # `tf.function` prevents us from using the attention_weights that were
  # calculated on the last iteration of the loop. So recalculate them outside
  # the loop.
  _, auxiliary_outputs = transformer([input_tokens, output[:,:-1]], training=False)

  return all_preds, auxiliary_outputs