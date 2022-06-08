import argparse
import json
import logging
import os
import time
from pprint import pprint
import traceback

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tqdm import tqdm

import prediction
import transformer

from const import (END_TOKEN, MAX_SENT_LEN, PADDING_TOKEN, SPECIAL_TOKENS,
                   START_TOKEN)
from preprocess import load_preprocessed_sent_data, TextTokenizer, TextVectorizer, TextPadder
from pointer_net import PointerNet
from transformer_utils import CustomSchedule, accuracy_metric, loss_function

AVAILABLE_METHODS = ["greedy", "beam"]

parser = argparse.ArgumentParser()
parser.add_argument("--dir",required=True,help="dir to load model from (must end with '/')")
parser.add_argument("--method",default="beam",choices=AVAILABLE_METHODS)
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

# loading the model in the global scope helps prevent warnings for some reason?
model = tf.keras.models.load_model(ARGS.dir + "model.tf", custom_objects=custom_objs)
model.summary()

tokenizer = TextTokenizer(use_start_end_tokens=True)
padder = TextPadder()
vectorizer = TextVectorizer()

attn_key = f"decoder_layer{TRAIN_PARAMS['n_layers']-1}_attn2_weights"

print()
print("Automatic Simplification with a Pointer-Transformer")
print("Enter your sentence with spaces seperating each token. For example:")
print("  Hello , my legal name is `` Pointer - Transformer , '' but you can call me Tim .")
print()
while True:
    print("Enter your complex sentence: ", end="")
    raw = input()
    try:
        raw = raw.strip().lower()

        raw = tokenizer.tokenize_sent(raw)
        raw = padder.pad_sents([raw])
        tokens = vectorizer.vectorize(raw)

        print(vectorizer.unvectorize(tokens))

        results = prediction.compute_preds(model, vectorizer, method=ARGS.method, 
                    x_test=tokens, x_test_raw=raw, attn_key=attn_key)

        for name,sent in results.items():
            print(" ", name + ":")
            print("   ", " ".join(sent[0]))
            print()

    except Exception as e:
        print("Error!")
        traceback.print_exc()
        print("Anyway...\n")
    



