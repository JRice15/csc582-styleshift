import logging
import time
import json
import argparse
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from const import MAX_SENT_LEN, START_TOKEN, END_TOKEN, PADDING_TOKEN, SPECIAL_TOKENS, OOV_TOKEN
from preprocess import get_vectorized_special

@tf.function
def _tf_greedy_predict(transformer, input_tokens, *, start, end, pad):
  # `tf.TensorArray` is required here (instead of a python list) so that the
  # dynamic-loop can be traced by `tf.function`.
  output_array = tf.TensorArray(dtype=tf.int32, size=MAX_SENT_LEN, dynamic_size=False)
  output_array = output_array.write(0, start)
  for i in tf.range(1, MAX_SENT_LEN):
    output_array = output_array.write(i, pad)

  for i in tf.range(1, MAX_SENT_LEN):
    output = tf.transpose(output_array.stack()) # (batch_size, max_sent_len)

    is_end = (output == end[0])
    if tf.reduce_all(tf.reduce_any(is_end, axis=-1), axis=0): # if every sentence contains at least one end token
      break

    predictions, _ = transformer([input_tokens, output[:,:-1]], training=False)

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


def greedy_predict(transformer, input_tokens, attn_key, batchsize=4):
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
    start = get_vectorized_special([START_TOKEN] * n_examples)
    end = get_vectorized_special([END_TOKEN] * n_examples)
    pad = get_vectorized_special([PADDING_TOKEN] * n_examples)
    # generate predictions
    pred, aux_outputs = _tf_greedy_predict(transformer, input_batch, start=start, end=end, pad=pad)
    all_preds.append(pred)
    all_attn.append(aux_outputs[attn_key])

  all_preds = tf.concat(all_preds, axis=0)
  all_attn = tf.concat(all_attn, axis=0)

  return all_preds, all_attn




def beam_search_predict(transformer, input_tokens, beam_size=8):
  assert input_tokens.shape[0] == 1 # batchsize 1 for now

  start = get_vectorized_special([START_TOKEN])
  end = get_vectorized_special([END_TOKEN])
  pad = get_vectorized_special([PADDING_TOKEN])

  beam = tf.TensorArray(dtype=tf.int32, size=MAX_SENT_LEN, dynamic_size=False)
  beam = beam.write(0, start)
  for i in tf.range(1, MAX_SENT_LEN):
    beam = beam.write(i, pad)

  beams = [beam]
  scores = [0]

  for i in tf.range(1, MAX_SENT_LEN):
    step_predictions = [] # list of prediction softmaxes
    for beam in beams:
      output = tf.transpose(beam.stack()) # (batch_size, max_sent_len)

      predictions, _ = transformer([input_tokens, output[:,:-1]], training=False)
      # select the last token from the seq_len dimension
      predictions = predictions[:, i-1, :]  # (batch_size, vocab_size)

      step_predictions.append(predictions)

    vocab_size = predictions[0].shape[-1]
    # concat to shape (n_beams, vocab_size)
    step_predictions = tf.concat(step_predictions, axis=0)

    # get top predictions
    probs, indicies = tf.math.top_k(step_predictions, k=beam_size)

    pred_tokens = indicies % vocab_size
    beam_ids = indicies // vocab_size

    # convert to log probs
    probs = tf.math.log(probs)

    new_beams = []
    new_scores = []
    for beam_id, pred_token, log_prob in zip(beam_ids, pred_tokens, probs):
      beam = beams[beam_id].write(i, pred_token)
      new_beams.append(beam)
      new_scores.append(scores[beam_id] + log_prob)
    
    beams = new_beams
    scores = new_scores




def interpolate_OOV_predictions(preds, x_raw, attn):
  """
  args:
    preds: np.array of str
    attn: np.array of attn weights
  """
  if len(attn.shape) > 3:
    # average over heads
    attn = np.mean(attn, axis=1)
  # attn.shape == (n_sentences, pred_len=99, inpt_len=100)
  
  # pred len is one less bc there is no attention for the first token, the START token
  n_sentences, pred_len, inpt_len = attn.shape
  # add row of all zeros to attn so the shapes work out
  zero_row = np.zeros((n_sentences, 1, inpt_len), attn.dtype)
  attn = np.concatenate([zero_row, attn], axis=1)

  interpolated = []
  for sent_idx in range(len(preds)):
    pred_i = preds[sent_idx]
    if OOV_TOKEN in pred_i:
      raw_i = x_raw[sent_idx]
      attn_i = attn[sent_idx]
      # get, for each sentence, for each word generated, what index in the input was paid the most attention
      top_attn_indicies = np.argmax(attn_i, axis=-1)
      # collect the actual strings from those indices
      top_attn = raw_i[top_attn_indicies]
      # combine
      oov_mask = (pred_i == OOV_TOKEN)
      result = np.where(oov_mask, top_attn, pred_i)
    else:
      result = pred_i
    interpolated.append(result)

  result = np.stack(interpolated, axis=0)
  return result


def to_final_sentences(sentences):
  """
  convert NP arrays, with <START> and <END>, to list of list of str with no special tokens
  """
  result = []
  for sent in sentences:
    sent = list(sent)
    if sent[0] == START_TOKEN:
      sent = sent[1:]
    if END_TOKEN in sent:
      sent = sent[:sent.index(END_TOKEN)]
    if any(x in sent for x in SPECIAL_TOKENS):
      print([x in sent for x in SPECIAL_TOKENS])
      print(sent)
      raise ValueError()
    result.append(sent)
  return result


