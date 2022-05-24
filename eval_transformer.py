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

from const import MAX_SENT_LEN, START_TOKEN, END_TOKEN, PADDING_TOKEN, SPECIAL_TOKENS
from load_data import load_preprocessed_sent_data
import transformer
from transformer_utils import CustomSchedule, loss_function, accuracy_metric


parser = argparse.ArgumentParser()
parser.add_argument("--path",required=True,help="path to model to load (must end with '.tf')")
parser.add_argument("--batchsize",default=64,help="batchsize during eval")
ARGS = parser.parse_args()

assert ARGS.path.endswith(".tf")

pprint(vars(ARGS))

# load params from json
params_path = ARGS.path[:-3] + "_params.json"
with open(params_path, "r") as f:
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
}

model = tf.keras.models.load_model(ARGS.path, custom_objects=custom_objs)

model.summary()

# hacky way to compute vocab size of model
# vocab_size = model.final_layer.units - len(SPECIAL_TOKENS)
# print("vocab size:", vocab_size)
# get data
dataset, vectorizer = load_preprocessed_sent_data(target="simple", drop_equal=True, 
                          start_end_tokens=True, max_vocab=TRAIN_PARAMS.max_vocab)
x_train, y_train, x_val, y_val, x_test, y_test = dataset

# build
result, attn = model([x_train[:ARGS.batchsize], y_train[:ARGS.batchsize, :-1]])


# monkey patch test step back onto the model bc it got lost somehow
def monkeypatched_test_step(*args, **kwargs):
    return transformer.Transformer.test_step(model, *args, **kwargs)
model.test_step = monkeypatched_test_step

print("Evaluate val data...")
pprint(model.evaluate(
    x_test, y_test, 
    batch_size=ARGS.batchsize,
    return_dict=True
))

print("Evaluate test data...")
pprint(model.evaluate(
    x_test, y_test, 
    batch_size=ARGS.batchsize,
    return_dict=True
))


# @tf.function
def predict_sentence(transformer, sentence):
    encoder_input = sentence[tf.newaxis]        

    # initialize the output with the start token.
    start = vectorizer.vectorize([START_TOKEN])
    end = vectorizer.vectorize([END_TOKEN])
    pad = vectorizer.vectorize([PADDING_TOKEN])

    # `tf.TensorArray` is required here (instead of a python list) so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int32, size=MAX_SENT_LEN)
    for i in range(1, MAX_SENT_LEN):
        output_array = output_array.write(i, pad)
    output_array = output_array.write(0, start)

    for i in tf.range(1,MAX_SENT_LEN):
      output = tf.transpose(output_array.stack())[:,:-1]
      predictions, _ = transformer([encoder_input, output], training=False)

      # select the last token from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)

      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output_array = output_array.write(i, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # output.shape (1, tokens)
    text = vectorizer.unvectorize(output[0])

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop. So recalculate them outside
    # the loop.
    _, attention_weights = transformer([encoder_input, output[:,:-1]], training=False)

    return text, attention_weights


print("Predictions:")
for i in range(5):
    pred, attn_w = predict_sentence(model, x_test[0])

    inpt = vectorizer.unvectorize(x_test[0])
    target = vectorizer.unvectorize(y_test[0])
    print("input:", " ".join(inpt).strip())
    print("targ: ", " ".join(target).strip())
    print("pred: ", " ".join(pred).strip())

