import logging
import time
import json
import argparse
from pprint import pprint

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


x_test = x_test[:2]
x_test_raw = x_test_raw[:2]
y_test_raw = y_test_raw[:2]

last_layer = TRAIN_PARAMS["n_layers"] - 1
preds, attn = prediction.greedy_predict(model, x_test, vectorizer, 
                        batchsize=2, #ARGS.batchsize
                        attn_key=f"decoder_layer{last_layer}_attn2_weights",
                      )

print(preds.shape)
print(attn.shape)
