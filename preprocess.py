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


class TextVectorizer:
    """
    maps np.array of strings to np.array of ints
    attributes:
        word_index: dict mapping word to integer
        index_size: total number of index entries (highest index + 1)
    """

    def __init__(self, vocab, padding_token=""):
        if padding_token not in vocab:
            vocab = vocab + [padding_token]
        # +1 because 0th index is reserved for unknown tokens
        self.word_index = {word: index+1 for index,word in enumerate(sorted(vocab))}
        self.index_size = len(vocab) + 1

        mapper = lambda x: self.word_index.get(x, 0) # default to index 0 if not in vocab map
        self._word_mapper = np.vectorize(mapper)

    def vectorize(self, array):
        """
        convert string array to int array
        """
        return self._word_mapper(array)

