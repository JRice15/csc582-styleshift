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
        word_to_index: dict mapping word to integer
        index_to_word: inverse mapping
        vocab_size: total number of index entries (highest index + 1)
    """

    def __init__(self, padding_token=""):
        with open("data/glove_6B/vocab.txt", "r") as f:
            vocab = f.readlines()
            vocab = [w.strip() for w in vocab]
        if padding_token not in vocab:
            vocab = vocab + [padding_token]
        # +1 because 0th index is reserved for unknown tokens
        self.word_to_index = {word: index+1 for index,word in enumerate(sorted(vocab))}
        self.vocab_size = len(vocab) + 1

        mapper = lambda x: self.word_to_index.get(x, 0) # default to index 0 if not in vocab map
        self._vec_f = np.vectorize(mapper)

        self.index_to_word = {i:w for w,i in self.word_to_index.items()}
        unmapper = lambda x: self.index_to_word.get(x, " ")
        self._unvec_f = np.vectorize(unmapper)

        # sanity checks
        assert self._vec_f("THIS_TOKEN_DEFINITELY_DOES_NOT_APPEAR") == 0
        assert self._vec_f("word") != 0


    def vectorize(self, array):
        """
        convert string array to int array
        """
        return self._vec_f(array)

    def unvectorize(self, array):
        return self._unvec_f(array)

