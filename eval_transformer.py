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
import beam_search

parser = argparse.ArgumentParser()
parser.add_argument("--dir",required=True,help="dir to load model from (must end with '/')")
# parser.add_argument("--batchsize",default=64,type=int,help="batchsize during eval")
parser.add_argument("--nsamples",default=5,type=int,help="number of sample predictions to show")
parser.add_argument("--samples-only",action="store_true",help="whether to only show samples, not eval on val/test data")
parser.add_argument("--noplots",action="store_true",help="whether to not make plots")
ARGS = parser.parse_args()

assert ARGS.dir.endswith("/")

pprint(vars(ARGS))

# load params from json
with open(ARGS.dir + "params.json", "r") as f:
  TRAIN_PARAMS = json.load(f)

setattr(ARGS, "batchsize", TRAIN_PARAMS["batchsize"])


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
datasets, vectorizer = load_preprocessed_sent_data(target="simple", drop_equal=True, 
                          start_end_tokens=True, max_vocab=TRAIN_PARAMS["max_vocab"],
                          show_example=False)
x_train, y_train, x_val, y_val, x_test, y_test = datasets

# build
result = model([x_train[:ARGS.batchsize], y_train[:ARGS.batchsize, :-1]])


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
# for i in range(ARGS.nsamples):

for i in np.random.choice(len(x_test), size=ARGS.nsamples):
  inpt, target = x_test[i], y_test[i]
  pred, auxiliary_outputs = beam_search.predict_sentences(model, inpt, vectorizer)

  inpt = vectorizer.unvectorize(inpt)
  target = vectorizer.unvectorize(target)
  pred = vectorizer.unvectorize(pred)[0]
  results = {
    "inpt": " ".join(inpt).strip(),
    "targ": " ".join(target).strip(),
    "pred": " ".join(pred).strip(), 
    "copied input?": (pred == inpt).all(),
  }
  print("Example", i)
  pprint(results)

  # attention plots
  if not ARGS.noplots:
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

