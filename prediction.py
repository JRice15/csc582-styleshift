import argparse
import json
import logging
import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from const import (END_TOKEN, MAX_SENT_LEN, NUMERIC_TOKEN, OOV_TOKEN,
                   PADDING_TOKEN, SPECIAL_TOKENS, START_TOKEN)
from preprocess import TextVectorizer, get_vectorized_special


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

        # select the predicted token from the seq_len dimension
        predictions = predictions[:, i-1, :] # (batch_size, vocab_size)

        predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output_array = output_array.write(i, predicted_id)

        # if predicted_id == end:
        #     break

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


PAD_VEC = get_vectorized_special(PADDING_TOKEN, as_tf=False)
END_VEC = get_vectorized_special(END_TOKEN, as_tf=False)

class Beam:
    """
    represents one possible decoding path
    attributes:
        tokens: list of str
    """

    def __init__(self, tokens, sum_log_probs, alpha, ngram_blocking, attn):
        self.tokens = tokens
        self.attn = attn
        self.sum_log_probs = sum_log_probs
        self.alpha = alpha
        self.ngram_blocking = ngram_blocking

        self.length = len(self.tokens)

        # from https://arxiv.org/pdf/1609.08144.pdf eq 14
        length_penalty = np.power((5 + self.length) / 6, self.alpha)
        self.score = sum_log_probs / length_penalty

    def __repr__(self):
        return f"Beam(score={self.score}, tokens={self.tokens})"

    def is_done(self):
        return self.tokens[-1] == END_VEC

    def as_np_array(self):
        """
        returns array, shape (1, MAX_SENT_LEN)
        """
        pad_list = [PAD_VEC] * (MAX_SENT_LEN - self.length)
        return np.array([self.tokens + pad_list])
        
    def extend(self, new_token, log_prob, attn):
        """
        make a new beam representing the extension with a new token
        returns
            is_valid: bool, whether this is a valid extension
            beam: new Beam, if is_valid
        """
        # ngram blocking
        if self.ngram_blocking is not None:
            n = self.ngram_blocking
            new_ngram = self.tokens[-(n-1):] + [new_token]
            # check each ngram in the list
            for i in range(len(self.tokens)):
                if self.tokens[i:i+n] == new_ngram:
                    return False, None
        return True, Beam(
            tokens=self.tokens + [new_token],
            attn=attn,
            sum_log_probs=self.sum_log_probs + log_prob,
            alpha=self.alpha,
            ngram_blocking=self.ngram_blocking,
        )



def beam_search_predict_one(transformer, input_tokens, *, attn_key, beam_size, alpha, ngram_blocking):
    """
    run beam search on one sentence. per https://arxiv.org/pdf/1609.08144.pdf
    args:
        input_tokens: shape (1, MAX_SENT_LEN)
    returns:
        prediction: shape (1, MAX_SENT_LEN)
        attn: shape (n_heads, MAX_SENT_LEN-1, MAX_SENT_LEN)
    """
    assert input_tokens.shape[0] == 1 # batchsize 1 for now

    start = get_vectorized_special(START_TOKEN, as_tf=False)
    end = get_vectorized_special(END_TOKEN, as_tf=False)
    pad = get_vectorized_special(PADDING_TOKEN, as_tf=False)

    input_tokens = input_tokens.numpy()
    if end in input_tokens:
        input_len = (input_tokens[0] == end).nonzero()[0][0]
    else:
        input_len = input_tokens.shape[-1]

    min_length = input_len // 5
    max_length = int(input_len * 2 + 1)

    complete_beams = []
    active_beams = [
        Beam([start], 0, alpha=alpha, ngram_blocking=ngram_blocking, attn=None),
    ]

    attn_shape = None

    min_score = np.NINF
    for i in range(1, max_length):
        # expand active beams
        n_active = beam_size - len(complete_beams)
        candidates = []
        # gather outputs generated so far in each beam
        outputs = np.concatenate([beam.as_np_array() for beam in active_beams], axis=0)
        outputs = tf.cast(outputs, dtype=tf.int32)
        tiled_input = tf.tile(input_tokens, [len(active_beams), 1])
        # run all beams as one batch
        predictions, aux_outputs = transformer([tiled_input, outputs[:,:-1]], training=False)
        predictions = predictions[:, i-1, :] # (n active beams, vocab_size)
        attns = aux_outputs[attn_key]
        if i == 1:
            attn_shape = attns[0].shape
        for beam, prediction, attn in zip(active_beams, predictions, attns):
            top_probs, top_tokens = tf.math.top_k(prediction, k=n_active)
            top_tokens = top_tokens.numpy()
            top_probs = np.log(top_probs.numpy()) # log probs
            min_prob = top_probs[0] - beam_size
            for token, prob in zip(top_tokens, top_probs):
                # pruning 1
                if prob > min_prob:
                    is_ok, candidate = beam.extend(token, prob, attn)
                    # check n-gram blocking
                    if is_ok:
                        # pruning 2
                        if candidate.score > min_score:
                            candidates.append(candidate)

        # get best candidates from expanded beams
        candidates = sorted(candidates, reverse=True, key=lambda x: x.score)
        candidates = candidates[:n_active]

        active_beams = []
        for candidate in candidates:
            if candidate.is_done():
                if candidate.length >= min_length:
                    complete_beams.append(candidate)
                    min_score = max(x.score for x in complete_beams) - beam_size
            else:
                active_beams.append(candidate)

        if not len(active_beams):
            break


    try:
        best_beam = max(complete_beams, key=lambda x: x.score)
        beam_arr = best_beam.as_np_array()
        # disallow copying the input
        if (beam_arr == input_tokens).all():
            complete_beams.remove(best_beam)
            best_beam = max(complete_beams, key=lambda x: x.score)
            beam_arr = best_beam.as_np_array()
        return beam_arr, best_beam.attn
    except ValueError: # max of empty sequence
        # default to copying the input if no good beams are found
        print("No good beam!")
        return input_tokens, np.zeros(attn_shape)



def beam_search_sentences(transformer, sentences, *, attn_key, beam_size=4, alpha=0.6, 
        ngram_blocking=3):
    """
    returns:
        preds: shape (n sents, MAX_SENT_LEN)
        attns: shape (n sents, n heads, MAX_SENT_LEN-1, MAX_SENT_LEN)
    """
    preds = []
    attns = []
    for sent in tqdm(sentences):
        pred, attn = beam_search_predict_one(transformer, sent[np.newaxis], beam_size=beam_size, 
                alpha=alpha, ngram_blocking=ngram_blocking, attn_key=attn_key)
        preds.append(pred)
        attns.append(attn)
    preds = np.concatenate(preds, axis=0)
    attns = np.stack(attns, axis=0)
    return preds, attns



def interpolate_OOV_predictions(preds, x_raw, attn):
    """
    fill in OOV and NUMERIC with the real tokens in the input that were paid attention to
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
            # no attention to start/end
            start_end_mask = (raw_i == START_TOKEN) | (raw_i == END_TOKEN)
            attn_i[:,start_end_mask] = 0
            # get, for each sentence, for each word generated, what index in the input was paid the most attention
            top_attn_indicies = np.argmax(attn_i, axis=-1)
            # collect the actual strings from those indices
            top_attn = raw_i[top_attn_indicies]
            # combine
            oov_mask = (pred_i == OOV_TOKEN) | (pred_i == NUMERIC_TOKEN)
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
            print("final sentence still contains special tokens?:", sent)
            # raise ValueError()
        result.append(sent)
    return result


