import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability


from const import MAX_SENT_LEN
from pointer_net import PointerNet

# logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

"""
From https://github.com/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb

My changes:
* using Keras built-in MultiHeadAttention instead of custom
* moving train_step inside Transformer to enable .fit() behavior
* add d_key that can be different from d_model, like in orig paper
* since we are translating english to english, I abstracted the two different
  embedding layers from the encoder and decoder, and use the same embedder for both
"""



def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]
  return tf.cast(pos_encoding, dtype=tf.float32)


### Masking

# Mask all the pad tokens in the batch of sequence. It ensures that the model does 
# not treat padding as the input. The mask indicates where pad value `0` is present

# JR CHANGE: 1 is for not padded, 0 is for padded
def create_padding_mask(seq):
  seq = tf.cast(tf.math.not_equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# The look-ahead mask is used to mask the future tokens in a sequence. In other 
# words, the mask indicates which entries should not be used.
# This means that to predict the third token, only the first and second token will 
# be used. Similarly to predict the fourth token, only the first, second and 
# the third tokens will be used and so on.
def create_look_ahead_mask(size):
    n = tf.cast((size * (size+1) / 2), tf.int32)
    mask = tensorflow_probability.math.fill_triangular(tf.ones((n,), dtype=tf.float32), upper=False)
    return mask

# output looks something like:
# [1 0 0]
# [1 1 0]
# [1 1 1]


### Point wise feed forward network

# Point wise feed forward network consists of two fully-connected layers with a 
# ReLU activation in between.
def point_wise_feed_forward_network(d_model, d_ff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(d_ff, activation='relu'),  # (batch_size, seq_len, d_ff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


### Encoder and decoder

class EncoderLayer(tf.keras.layers.Layer):

  def __init__(self, *, d_model, num_heads, d_key, d_ff, rate=0.1, **kwargs):
    super().__init__(**kwargs)
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_key = d_key
    self.d_ff = d_ff
    self.rate = rate

    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_key)
    self.ffn = point_wise_feed_forward_network(d_model, d_ff)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):
    attn_output = self.mha(x, x, attention_mask=mask, training=training)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    return out2

  def get_config(self):
    return {
      "d_model": self.d_model,
      "num_heads": self.num_heads,
      "d_key": self.d_key,
      "d_ff": self.d_ff,
      "rate": self.rate,
      **super().get_config(),
    }

### Decoder layer

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, *, d_model, num_heads, d_key, d_ff, rate=0.1, **kwargs):
    super().__init__(**kwargs)
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_key = d_key
    self.d_ff = d_ff
    self.rate = rate

    self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_key)
    self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_key)
    self.ffn = point_wise_feed_forward_network(d_model, d_ff)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    attn1, attn1_weights = self.mha1(x, x, attention_mask=look_ahead_mask, 
                              training=training, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn2_weights = self.mha2(out1, enc_output, attention_mask=padding_mask, 
                              training=training, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, out2, attn1_weights, attn2_weights

  def get_config(self):
    return {
      "d_model": self.d_model,
      "num_heads": self.num_heads,
      "d_key": self.d_key,
      "d_ff": self.d_ff,
      "rate": self.rate,
      **super().get_config(),
    }

### Encoder

class Encoder(tf.keras.layers.Layer):

  def __init__(self, *, num_layers, d_model, d_key, num_heads, d_ff, rate=0.1, **kwargs):
    super().__init__(**kwargs)
    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_key = d_key
    self.d_ff = d_ff
    self.rate = rate

    self.dropout = tf.keras.layers.Dropout(rate)
    self.enc_layers = [
        EncoderLayer(d_model=d_model, num_heads=num_heads, d_key=d_key, d_ff=d_ff, rate=rate)
        for _ in range(num_layers)]

  def call(self, x, training, mask):
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training=training, mask=mask)

    return x  # (batch_size, input_seq_len, d_model)

  def get_config(self):
    return {
      "num_layers": self.num_layers,
      "d_model": self.d_model,
      "num_heads": self.num_heads,
      "d_key": self.d_key,
      "d_ff": self.d_ff,
      "rate": self.rate,
      **super().get_config(),
    }

### Decoder

class Decoder(tf.keras.layers.Layer):

  def __init__(self, *, num_layers, d_model, d_key, num_heads, d_ff, rate=0.1, 
        **kwargs):
    super().__init__(**kwargs)
    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_key = d_key
    self.d_ff = d_ff
    self.rate = rate

    self.dropout = tf.keras.layers.Dropout(rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, d_key=d_key, rate=rate)
        for _ in range(num_layers)]

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    x = self.dropout(x, training=training)

    attention_weights = {}
    for i in range(self.num_layers):
      # only save and return last decoder state
      x, dec_state, attn1, attn2 = self.dec_layers[i](x, enc_output, training=training, 
            look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

      attention_weights[f"decoder_layer{i}_attn1_weights"] = attn1
      attention_weights[f"decoder_layer{i}_attn2_weights"] = attn2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, dec_state, attention_weights

  def get_config(self):
    return {
      "num_layers": self.num_layers,
      "d_model": self.d_model,
      "num_heads": self.num_heads,
      "d_key": self.d_key,
      "d_ff": self.d_ff,
      "rate": self.rate,
      **super().get_config(),
    }


### Create the transformer model

class Transformer(tf.keras.Model):
  """
  args:
    embedding_matrix: optional, pretrained embedding matrix for words, in which case we will use that as untrainable embedding weights
  
  call() returns:
    predicted tokens
    attn weights: dict
    pointer data: dict, containing pointer_logits and p_gen
  """

  def __init__(self, *, num_layers, num_heads, d_model, d_key, d_ff, vocab_size,
                rate=0.1, embedding_matrix=None, use_pointer_net=False, **kwargs):
    super().__init__(**kwargs)
    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_key = d_key
    self.d_ff = d_ff
    self.rate = rate
    self.vocab_size = vocab_size
    self.use_pointer_net = use_pointer_net

    # constant pos encoding
    self.pos_encoding = positional_encoding(MAX_SENT_LEN, d_model)

    # create embedding layer
    if embedding_matrix is not None:
      emb_vocab_size, emb_dim = embedding_matrix.shape
      assert emb_vocab_size == vocab_size
      self.embedding_layer = tf.keras.layers.Embedding(vocab_size, emb_dim, 
                                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                trainable=False,
                            )
    else:
      self.embedding_layer = tf.keras.layers.Embedding(vocab_size, d_model, trainable=True)

    # other layers
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, d_ff=d_ff, d_key=d_key,
                           rate=rate)
    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, d_ff=d_ff, d_key=d_key,
                           rate=rate)
    self.final_layer = tf.keras.layers.Dense(vocab_size)

    # pointer-generator
    if self.use_pointer_net:
      self.pointer_net = PointerNet(vocab_size=vocab_size)

  def call(self, inputs, training):
    # Keras models prefer if you pass all your inputs in the first argument
    inp_tokens, tar_tokens = inputs

    padding_mask, look_ahead_mask = self.create_masks(inp_tokens, tar_tokens)

    ### Encoder
    seq_len = tf.shape(inp_tokens)[1]
    inp_emb = self.embedding_layer(inp_tokens)  # (batch_size, input_seq_len, d_model)
    inp_emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    inp_emb += self.pos_encoding[:, :seq_len, :]

    enc_output = self.encoder(inp_emb, training=training, mask=padding_mask)
    # enc_output.shape: (batch_size, inp_seq_len, d_model)

    ### Decoder
    seq_len = tf.shape(tar_tokens)[1]
    tar_emb = self.embedding_layer(tar_tokens)  # (batch_size, tar_seq_len, d_model)
    tar_emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    tar_emb += self.pos_encoding[:, :seq_len, :]

    dec_output, dec_state, auxiliary_outputs = self.decoder(tar_emb, enc_output, 
            training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
    # dec_output.shape: (batch_size, tar_seq_len, d_model)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    ### Pointer-Generator mechanism
    if self.use_pointer_net:
      last_layer = self.num_layers - 1
      last_attn = auxiliary_outputs[f"decoder_layer{last_layer}_attn2_weights"]

      final_output, pointer_data = self.pointer_net(inp_tokens=inp_tokens, tar_embedded=tar_emb,
            generator_output=final_output, enc_output=enc_output, dec_state=dec_state, 
            attn_heads=last_attn)
      auxiliary_outputs.update(pointer_data)

    return final_output, auxiliary_outputs

  def create_masks(self, inp, tar):
    # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
    padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)

    # JR CHANGE: since masks are inverted, a zero (ignore that location) in either mask should result in a zero in the output
    look_ahead_mask = tf.minimum(dec_target_padding_mask, look_ahead_mask)

    return padding_mask, look_ahead_mask

  def train_step(self, data):
    inp, tar = data
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
      predictions, _ = self([inp, tar_inp], training=True)
      loss = self.compiled_loss(tar_real, predictions, regularization_losses=self.losses)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    self.compiled_metrics.update_state(tar_real, predictions)
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    # Unpack the data
    inp, tar = data
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    # Compute predictions
    predictions, auxiliary_outputs = self([inp, tar_inp], training=False)
    # Updates the metrics tracking the loss
    self.compiled_loss(tar_real, predictions, regularization_losses=self.losses)
    # Update the metrics.
    self.compiled_metrics.update_state(tar_real, predictions)
    # Return a dict mapping metric names to current value.
    # Note that it will include the loss (tracked in self.metrics).
    logs = {m.name: m.result() for m in self.metrics}
    if "p_gen" in auxiliary_outputs:
      logs["avg_p_gen"] = tf.reduce_mean(auxiliary_outputs["p_gen"])
    return logs

  def get_config(self):
    # if loading from config, we don't need to specify the embedding matrix 
    # initializer, because its weights have been saved already in the model
    return {
      "num_layers": self.num_layers,
      "d_model": self.d_model,
      "num_heads": self.num_heads,
      "d_key": self.d_key,
      "d_ff": self.d_ff,
      "rate": self.rate,
      "vocab_size": self.vocab_size,
      "use_pointer_net": self.use_pointer_net,
      **super().get_config(),
    }


# The target is divided into tar_inp and tar_real. tar_inp is passed as an input 
# to the decoder. `tar_real` is that same input shifted by 1: At each location in 
# `tar_input`, `tar_real` contains the  next token that should be predicted.
# 
# For example, `sentence = 'SOS A lion in the jungle is sleeping EOS'` becomes:
# 
# * `tar_inp =  'SOS A lion in the jungle is sleeping'`
# * `tar_real = 'A lion in the jungle is sleeping EOS'`
# 
# A transformer is an auto-regressive model: it makes predictions one part at a 
# time, and uses its output so far to decide what to do next.
# 
# During training this example uses teacher-forcing (like in the [text generation 
# tutorial](https://www.tensorflow.org/text/tutorials/text_generation)). Teacher 
# forcing is passing the true output to the next time step regardless of what the 
# model predicts at the current time step.
# 
# As the model predicts each token, *self-attention* allows it to look at the 
# previous tokens in the input sequence to better predict the next token.
# 
# To prevent the model from peeking at the expected output the model uses a 
# look-ahead mask.
