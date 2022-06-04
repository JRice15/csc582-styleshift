import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

MAX_SENT_LEN = 100 # in words
MAX_WORD_LEN = 25 # in letters

START_TOKEN = "<START>"
END_TOKEN = "<END>"
PADDING_TOKEN = ""
OOV_TOKEN = "<OOV>"
NUMERIC_TOKEN = "<NUM>"

# padding must always be 0th index
SPECIAL_TOKENS = [PADDING_TOKEN, OOV_TOKEN, NUMERIC_TOKEN, START_TOKEN, END_TOKEN]

