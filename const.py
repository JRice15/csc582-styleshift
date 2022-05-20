import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

MAX_SENT_LEN = 100
MAX_WORD_LEN = 25

START_TOKEN = "<START>"
END_TOKEN = "<END>"
PADDING_TOKEN = ""
OOV_TOKEN = "<OOV>"

# padding must always be 0th index
SPECIAL_TOKENS = [PADDING_TOKEN, OOV_TOKEN, START_TOKEN, END_TOKEN]

# load vocab
with open("data/glove_6B/vocab.txt", "r") as f:
    VOCAB = f.readlines()
    VOCAB = [w.strip() for w in VOCAB]
    VOCAB = [w for w in VOCAB if len(w)]

# show some metrics
print("glove vocab size:", len(VOCAB))
_long_word_pct = np.mean([len(x) > MAX_WORD_LEN for x in VOCAB])
print("fraction of vocab words longer than", MAX_WORD_LEN, "letters:", _long_word_pct)