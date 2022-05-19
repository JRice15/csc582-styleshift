import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability

from transformer import Transformer


# To keep this example small and relatively fast, the values for `num_layers, 
# d_model, dff` have been reduced. 
# The base model described in the [paper](https://arxiv.org/abs/1706.03762) used: 
# `num_layers=6, d_model=512, dff=2048`.

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1


### Training and checkpointing

model = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    rate=dropout_rate)




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



# Use the Adam optimizer with a custom learning rate scheduler according to the 
# formula in the [paper](https://arxiv.org/abs/1706.03762).
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)
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


# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

model.compile(
  optimizer=optimizer,
  loss=loss_function,
  metrics=[accuracy_function],
)


callback_list = [
  tf.keras.callbacks.EarlyStopping(patience=5),
  tf.keras.callbacks.ModelCheckpoint("transformer.h5", save_best_only=True)
]


model.fit(
  x_train, x_test,
  validation_data=(x_val, y_val),
  batch_size=16,
  epochs=100,
  callbacks=callback_lst
)


EPOCHS = 20



for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(train_batches):
    train_step(inp, tar)

    if batch % 50 == 0:
      print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


### Run inference

# The following steps are used for inference:
# 
# * Encode the input sentence using the Portuguese tokenizer (`tokenizers.pt`). This is the encoder input.
# * The decoder input is initialized to the `[START]` token.
# * Calculate the padding masks and the look ahead masks.
# * The `decoder` then outputs the predictions by looking at the `encoder output` and its own output (self-attention).
# * Concatenate the predicted token to the decoder input and pass it to the decoder.
# * In this approach, the decoder predicts the next token based on the previous tokens it predicted.


# Note: The model is optimized for _efficient training_ and makes a next-token prediction for each token in the output simultaneously. This is redundant during inference, and only the last prediction is used.  This model can be made more efficient for inference if you only calculate the last prediction when running in inference mode (`training=False`).


class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=MAX_TOKENS):
    # input sentence is portuguese, hence adding the start and end token
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

    encoder_input = sentence

    # As the output language is english, initialize the output with the
    # english start token.
    start_end = self.tokenizers.en.tokenize([''])[0]
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    # `tf.TensorArray` is required here (instead of a python list) so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions, _ = self.transformer([encoder_input, output], training=False)

      # select the last token from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.argmax(predictions, axis=-1)

      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # output.shape (1, tokens)
    text = tokenizers.en.detokenize(output)[0]  # shape: ()

    tokens = tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop. So recalculate them outside
    # the loop.
    _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

    return text, tokens, attention_weights


# Note: This function uses an unrolled loop, not a dynamic loop. It generates 
# `MAX_TOKENS` on every call. Refer to [NMT with attention](nmt_with_attention.ipynb) 
# for an example implementation with a dynamic loop, which can be much more efficient.


# Create an instance of this `Translator` class, and try it out a few times:

translator = Translator(tokenizers, transformer)

def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Ground truth":15s}: {ground_truth}')


sentence = 'este é um problema que temos que resolver.'
ground_truth = 'this is a problem we have to solve .'

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)


sentence = 'os meus vizinhos ouviram sobre esta ideia.'
ground_truth = 'and my neighboring homes heard about this idea .'

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)


sentence = 'vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.'
ground_truth = "so i'll just share with you some stories very quickly of some magical things that have happened."

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)


# ## Attention plots


# The `Translator` class returns a dictionary of attention maps you can use to visualize the internal working of the model:


sentence = 'este é o primeiro livro que eu fiz.'
ground_truth = "this is the first book i've ever done."

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)


def plot_attention_head(in_tokens, translated_tokens, attention):
  # The plot is of the attention when a token was generated.
  # The model didn't generate `<START>` in the output. Skip it.
  translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention)
  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(translated_tokens)))

  labels = [label.decode('utf-8') for label in in_tokens.numpy()]
  ax.set_xticklabels(
      labels, rotation=90)

  labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
  ax.set_yticklabels(labels)


head = 0
# shape: (batch=1, num_heads, seq_len_q, seq_len_k)
attention_heads = tf.squeeze(
  attention_weights['decoder_layer4_block2'], 0)
attention = attention_heads[head]
attention.shape


in_tokens = tf.convert_to_tensor([sentence])
in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
in_tokens = tokenizers.pt.lookup(in_tokens)[0]
in_tokens


translated_tokens


plot_attention_head(in_tokens, translated_tokens, attention)


def plot_attention_weights(sentence, translated_tokens, attention_heads):
  in_tokens = tf.convert_to_tensor([sentence])
  in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
  in_tokens = tokenizers.pt.lookup(in_tokens)[0]

  fig = plt.figure(figsize=(16, 8))

  for h, head in enumerate(attention_heads):
    ax = fig.add_subplot(2, 4, h+1)

    plot_attention_head(in_tokens, translated_tokens, head)

    ax.set_xlabel(f'Head {h+1}')

  plt.tight_layout()
  plt.show()


plot_attention_weights(sentence, translated_tokens,
                       attention_weights['decoder_layer4_block2'][0])


# The model does okay on unfamiliar words. Neither "triceratops" or "encyclopedia" are in the input dataset and the model almost learns to transliterate them, even without a shared vocabulary:


sentence = 'Eu li sobre triceratops na enciclopédia.'
ground_truth = 'I read about triceratops in the encyclopedia.'

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

plot_attention_weights(sentence, translated_tokens,
                       attention_weights['decoder_layer4_block2'][0])


# ## Export


# That inference model is working, so next you'll export it as a `tf.saved_model`.
# 
# To do that, wrap it in yet another `tf.Module` sub-class, this time with a `tf.function` on the `__call__` method:


class ExportTranslator(tf.Module):
  def __init__(self, translator):
    self.translator = translator

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def __call__(self, sentence):
    (result,
     tokens,
     attention_weights) = self.translator(sentence, max_length=MAX_TOKENS)

    return result


# In the above `tf.function` only the output sentence is returned. Thanks to the [non-strict execution](https://tensorflow.org/guide/intro_to_graphs) in `tf.function` any unnecessary values are never computed.


translator = ExportTranslator(translator)


# Since the model is decoding the predictions using `tf.argmax` the predictions are deterministic. The original model and one reloaded from its `SavedModel` should give identical predictions:


translator('este é o primeiro livro que eu fiz.').numpy()


tf.saved_model.save(translator, export_dir='translator')


reloaded = tf.saved_model.load('translator')


reloaded('este é o primeiro livro que eu fiz.').numpy()


# ## Summary
# 
# In this tutorial you learned about:
# 
# * positional encoding
# * multi-head attention
# * the importance of masking 
# * and how to put it all together to build a transformer.
# 
# This implementation tried to stay close to the implementation of the 
# [original paper](https://arxiv.org/abs/1706.03762). If you want to practice 
# there are many things you could try with it. For example: 
# 
# * Using a different dataset to train the transformer.
# * Create the "Base Transformer" or "Transformer XL" configurations from the 
# original paper by changing the hyperparameters.
# * Use the layers defined here to create an implementation of 
# [BERT](https://arxiv.org/abs/1810.04805).
# * Implement beam search to get better predictions.


