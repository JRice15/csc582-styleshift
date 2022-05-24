import logging
import time
import argparse
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability
from sklearn.model_selection import train_test_split

from const import MAX_SENT_LEN
from load_data import load_preprocessed_sent_data
import transformer
from transformer_utils import CustomSchedule, loss_function, accuracy_metric


parser = argparse.ArgumentParser()
parser.add_argument("--path",required=True,help="path to model to load (must end with '.tf')")
ARGS = parser.parse_args()
pprint(vars(ARGS))


custom_objs = {
    "Transformer": transformer.Transformer,
    "Encoder": transformer.Encoder,
    "Decoder": transformer.Decoder,
    "EncoderLayer": transformer.EncoderLayer,
    "DecoderLayer": transformer.DecoderLayer,
    "loss_function": loss_function,
    "accuracy_metric": accuracy_metric,
    "CustomSchedule": CustomSchedule,
}

model = tf.keras.models.load_model(ARGS.path, custom_objects=custom_objs)

model.summary()
