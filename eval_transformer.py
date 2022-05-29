import logging
import time
import json
import argparse
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability
from sklearn.model_selection import train_test_split

from const import MAX_SENT_LEN, START_TOKEN, END_TOKEN, PADDING_TOKEN, SPECIAL_TOKENS
from load_data import load_preprocessed_sent_data
import transformer
from transformer_utils import CustomSchedule, loss_function, accuracy_metric
from pointer_net import PointerNet

parser = argparse.ArgumentParser()
parser.add_argument("--dir",required=True,help="dir to load model from (must end with '/')")
parser.add_argument("--batchsize",default=64,type=int,help="batchsize during eval")
parser.add_argument("--nsamples",default=5,type=int,help="number of sample predictions to show")
parser.add_argument("--samples-only",action="store_true",help="whether to only show samples, not eval on val/test data")
ARGS = parser.parse_args()

assert ARGS.dir.endswith("/")

pprint(vars(ARGS))

# load params from json
with open(ARGS.dir + "params.json", "r") as f:
  TRAIN_PARAMS = json.load(f)


custom_objs = {
    "Transformer": transformer.Transformer,
    "Encoder": transformer.Encoder,
    "Decoder": transformer.Decoder,
    "EncoderLayer": transformer.EncoderLayer,
    "DecoderLayer": transformer.DecoderLayer,
    "loss_function": loss_function,
    "accuracy_metric": accuracy_metric,
    "CustomSchedule": CustomSchedule,
    "PointerNet": PointerNet,
}

model = tf.keras.models.load_model(ARGS.dir + "model.tf", custom_objects=custom_objs)

model.summary()

# hacky way to compute vocab size of model
# vocab_size = model.final_layer.units - len(SPECIAL_TOKENS)
# print("vocab size:", vocab_size)
# get data
dataset, vectorizer = load_preprocessed_sent_data(target="simple", drop_equal=True, 
                          start_end_tokens=True, max_vocab=TRAIN_PARAMS["max_vocab"],
                          show_example=False)
x_train, y_train, x_val, y_val, x_test, y_test = dataset

# build
result = model([x_train[:ARGS.batchsize], y_train[:ARGS.batchsize, :-1]])


def predict_sentence(transformer, sentence):
  encoder_input = sentence[tf.newaxis]

  # As the output language is english, initialize the output with the
  # english start token.
  start = vectorizer.vectorize([START_TOKEN])
  end = vectorizer.vectorize([END_TOKEN])
  pad = vectorizer.vectorize([PADDING_TOKEN])

  # `tf.TensorArray` is required here (instead of a python list) so that the
  # dynamic-loop can be traced by `tf.function`.
  output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  output_array = output_array.write(0, start)
  for i in tf.range(1, MAX_SENT_LEN):
    output_array = output_array.write(i, pad)

  for i in tf.range(1, MAX_SENT_LEN):
    output = tf.transpose(output_array.stack())
    output = output[:,:-1]

    predictions, _ = transformer([encoder_input, output], training=False)

    # select the last token from the seq_len dimension
    predictions = predictions[:, i-1, :]  # (batch_size, vocab_size)

    predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)

    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output_array = output_array.write(i, predicted_id)

    if predicted_id == end:
      break

  output = tf.transpose(output_array.stack())
  # output.shape (1, tokens)
  text = vectorizer.unvectorize(output)[0]

  # `tf.function` prevents us from using the attention_weights that were
  # calculated on the last iteration of the loop. So recalculate them outside
  # the loop.
  _, auxiliary_outputs = transformer([encoder_input, output[:,:-1]], training=False)

  return text, auxiliary_outputs


def plot_attention_head(in_tokens, translated_tokens, attention):
  # The plot is of the attention when a token was generated.
  # The model didn't generate `<START>` in the output. Skip it.
  translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention.T)
  ax.set_xticks(range(len(translated_tokens)))
  ax.set_yticks(range(len(in_tokens)))

  ax.set_xticklabels([label for label in translated_tokens], rotation=90)
  ax.set_yticklabels([label for label in in_tokens])

  # plt.xlabel("Prediction")
  # plt.ylabel("Input")


def plot_attention_weights(in_tokens, translated_tokens, attention_heads, layer_name):
  fig = plt.figure(figsize=(16, 8))

  for h, head in enumerate(attention_heads):
    ax = fig.add_subplot(2, 4, h+1)

    plot_attention_head(in_tokens, translated_tokens, head)

    ax.set_xlabel(f'Head {h+1}')

  plt.suptitle(f"Attention weights for each head (layer {layer_name})")
  plt.tight_layout()
  plt.show()

  sum_head = attention_heads.sum(axis=0)
  plot_attention_head(in_tokens, translated_tokens, head)

  plt.suptitle(f"Sum of attention heads (layer {layer_name})")
  plt.tight_layout()
  plt.show()



print("Predictions on test set:")
for i in np.random.choice(len(x_test), size=ARGS.nsamples):
  inpt, target = x_test[i], y_test[i]
  pred, auxiliary_outputs = predict_sentence(model, inpt)

  inpt = vectorizer.unvectorize(inpt)
  target = vectorizer.unvectorize(target)
  results = {
    "inpt": " ".join(inpt).strip(),
    "targ": " ".join(target).strip(),
    "pred": " ".join(pred).strip(), 
  }
  print("Example", i)
  pprint(results)

  inpt_len = (inpt != PADDING_TOKEN).sum()
  pred_len = (pred != PADDING_TOKEN).sum()

  last_layer = TRAIN_PARAMS["n_layers"] - 1
  these_weights = tf.squeeze(auxiliary_outputs[f'decoder_layer{last_layer}_attn2_weights'], axis=0)

  plot_attention_weights(
    inpt[:inpt_len],
    pred[:pred_len],
    these_weights[:, :pred_len-1, :inpt_len].numpy(),
    layer_name=f"decoder_layer{last_layer}_attn2",
  )


if ARGS.samples_only:
  exit()


# monkey patch test step back onto the model bc it got lost somehow
def monkeypatched_test_step(*args, **kwargs):
    return transformer.Transformer.test_step(model, *args, **kwargs)
model.test_step = monkeypatched_test_step


def eval_on_dataset(x, y, name):
  print(f"Evaluating {name} data...")
  results = model.evaluate(
      x, y, 
      batch_size=ARGS.batchsize,
      return_dict=True
  )
  pprint(results)
  with open(ARGS.dir + name + "_results.json", "w") as f:
    json.dump(results, f, indent=2)

eval_on_dataset(x_val, y_val, "val")
eval_on_dataset(x_test, y_test, "test")

