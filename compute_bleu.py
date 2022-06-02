import logging
import time
import json
import argparse
from pprint import pprint
import os

import nltk
import matplotlib.pyplot as plt
from tqdm import tqdm
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
import prediction

parser = argparse.ArgumentParser()
parser.add_argument("--dir",required=True,help="dir to load model from (must end with '/')")
parser.add_argument("--method",default="greedy",choices=["greedy","beam"])
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

datasets, vectorizer, raw_test_data = load_preprocessed_sent_data(target="simple", drop_equal=True, 
                          start_end_tokens=True, max_vocab=TRAIN_PARAMS["max_vocab"],
                          show_example=False, return_raw_test=True)
_, _, _, _, x_test, y_test = datasets
x_test_raw, y_test_raw = raw_test_data


last_layer = TRAIN_PARAMS["n_layers"] - 1
if ARGS.method == "greedy":
  preds, attn = prediction.greedy_predict(
                          model, x_test, 
                          batchsize=ARGS.batchsize,
                          attn_key=f"decoder_layer{last_layer}_attn2_weights",
                        )
else:
  raise ValueError("Unknown method")

preds = vectorizer.unvectorize(preds)
attn = attn.numpy()

# turn OOV to real words
preds = prediction.interpolate_OOV_predictions(preds, x_test_raw, attn)

# strip start/end tokens, padding
preds = prediction.to_final_sentences(preds)
refs = prediction.to_final_sentences(y_test_raw)
refs = [[x] for x in refs] # nltk wants a list of refs for each pred

# compute our bleu
our_bleu = nltk.translate.bleu_score.corpus_bleu(refs, preds)
print("Our BLEU:", bleu)

# compute bleu of just copying the inputs
raw_inputs = prediction.to_final_sentences(x_test_raw)
inputs_bleu = nltk.translate.bleu_score.corpus_bleu(refs, raw_inputs)
print("Inputs BLEU:", inputs_bleu)

# initialize or update bleu score results
result_file = ARGS.dir + "bleu.json"
if os.path.exists(result_file):
  with open(result_file, "r") as f:
    results = json.load(f)
else:
  results = {}

results[ARGS.method] = our_bleu
results["inputs"] = inputs_bleu

with open(result_file, "w") as f:
  json.dump(results, f)

