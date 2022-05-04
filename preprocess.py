import numpy as np
import pandas as pd
from collections import defaultdict

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

class TextTokenizer:
    """
    converts strings to list of individual tokens
    """

    # convert weird tokens in dataset to canonical form
    CUSTOM_TOKEN_TRANSLATIONS = {
        "-lrb-": "(",
        "-rrb-": ")"
    }

    def __init__(self, max_sent_len):
        self.max_sent_len = max_sent_len

    def tokenize_sent(self, sent):
        """
        convert sentence (string) to list of tokens
        """
        words = sent.lower().strip().split()
        words = [self.CUSTOM_TOKEN_TRANSLATIONS.get(w, w) for w in words] # first arg to get is key to find, second arg is the default to use if the key is not found
        return words

    def pad_and_truncate(self, words):
        """
        make a sents be the proper `max_sent_len`
        """
        if len(words) < self.max_sent_len:
            words += [""] * (self.max_sent_len - len(words))
        elif len(words) > self.max_sent_len:
            words = words[:self.max_sent_len]
        return words
