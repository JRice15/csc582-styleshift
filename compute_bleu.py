import argparse
import json
import logging
import os
import time
from pprint import pprint

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
from preprocess import load_preprocessed_sent_data
from pointer_net import PointerNet
from transformer_utils import CustomSchedule, accuracy_metric, loss_function

AVAILABLE_METHODS = ["greedy", "beam"]

parser = argparse.ArgumentParser()
parser.add_argument("--dir",required=True,help="dir to load model from (must end with '/')")
parser.add_argument("--subsample",type=int,default=20)
parser.add_argument("--method",default="greedy",choices=AVAILABLE_METHODS + ["all"])
ARGS = parser.parse_args()

assert ARGS.dir.endswith("/")

pprint(vars(ARGS))

# load params from json
with open(ARGS.dir + "params.json", "r") as f:
  TRAIN_PARAMS = json.load(f)

setattr(ARGS, "batchsize", TRAIN_PARAMS["batchsize"])

os.makedirs(ARGS.dir + "bleu/", exist_ok=True)


def sents_to_strings(sents):
    """
    list of list of str to list of str
    """
    return [" ".join(x) for x in sents]

def sents_from_strings(sents):
    return [str(x).split() for x in sents]



def compute_bleu(model, vectorizer, method, *, x_test, x_test_raw, y_test_raw):
    os.makedirs(ARGS.dir + "bleu/", exist_ok=True)

    # get caches CSV with results, or update it
    csv_file = ARGS.dir + f"bleu/preds_subsampled{ARGS.subsample}x.csv"
    if os.path.exists(csv_file):
        pred_df = pd.read_csv(csv_file)
        raw_inputs = sents_from_strings(pred_df["inputs"].to_list())
        refs = sents_from_strings(pred_df["refs"].to_list())
    else:
        raw_inputs = prediction.to_final_sentences(x_test_raw)
        refs = prediction.to_final_sentences(y_test_raw)
        pred_df = pd.DataFrame({
            "inputs": sents_to_strings(raw_inputs),
            "refs": sents_to_strings(refs),
        })

    last_layer = TRAIN_PARAMS["n_layers"]-1
    attn_key = f"decoder_layer{last_layer}_attn2_weights"
    # compute predictions if not present
    if method not in pred_df.columns:
        computed = prediction.compute_preds(model, vectorizer, method=method, 
                        x_test=x_test, x_test_raw=x_test_raw, attn_key=attn_key)
        for name,pred in computed.items():
            pred_df[name] = sents_to_strings(pred)
        pred_df.to_csv(csv_file, index=False)

    # initialize or update bleu score results
    result_file = ARGS.dir + f"bleu/bleu_subsampled{ARGS.subsample}x.json"
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            results = json.load(f)
    else:
        results = {
            "bleu": {},
            "pct_copy": {},
            "pct_correct": {},
        }

    results["total_examples"] = len(x_test_raw)

    # compute bleu for all methods
    nltk_refs = [[x] for x in refs] # nltk wants a list of refs for each pred
    for col in pred_df.columns:
        preds = sents_from_strings(pred_df[col].to_list())
        results["bleu"][col] = nltk.translate.bleu_score.corpus_bleu(nltk_refs, preds)
        # compute pct copies inputs
        pct_copy = np.mean([pred == inpt for pred,inpt in zip(preds, raw_inputs)])
        results["pct_copy"][col] = float(pct_copy)
        # compute pct exact correct
        pct_exact = np.mean([pred == targ for pred,targ in zip(preds, refs)])
        results["pct_correct"][col] = float(pct_exact)

    pprint(results)

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)


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


datasets, vectorizer, raw_test_data = load_preprocessed_sent_data(target="simple", drop_equal=True, 
                        start_end_tokens=True, min_word_freq=TRAIN_PARAMS["min_word_freq"],
                        show_example=False, return_raw_test=True)
_, _, _, _, x_test, y_test = datasets
x_test_raw, y_test_raw = raw_test_data

x_test_raw = x_test_raw[::ARGS.subsample]
y_test_raw = y_test_raw[::ARGS.subsample]
x_test = x_test[::ARGS.subsample]
y_test = y_test[::ARGS.subsample]

print(f"N examples with {ARGS.subsample}x subsampling:", len(x_test))

if ARGS.method == "all":
    for method in AVAILABLE_METHODS:
        compute_bleu(model, vectorizer, method, x_test=x_test, 
            x_test_raw=x_test_raw, y_test_raw=y_test_raw)
else:
    compute_bleu(model, vectorizer, ARGS.method, x_test=x_test, 
        x_test_raw=x_test_raw, y_test_raw=y_test_raw)


