from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from const import (END_TOKEN, MAX_SENT_LEN, OOV_TOKEN, PADDING_TOKEN,
                   SPECIAL_TOKENS, START_TOKEN, VOCAB)


class TextTokenizer:
    """
    converts strings to list of individual tokens
    """

    # convert weird tokens in dataset to canonical form
    CUSTOM_TOKEN_TRANSLATIONS = {
        "-lrb-": "(",
        "-rrb-": ")"
    }

    def __init__(self, use_start_end_tokens):
        self.use_start_end_tokens = use_start_end_tokens

    def tokenize_sent(self, sent):
        """
        convert sentence (string) to list of tokens
        """
        words = sent.lower().strip().split()
        words = [self.CUSTOM_TOKEN_TRANSLATIONS.get(w, w) for w in words] # first arg to get is key to find, second arg is the default to use if the key is not found
        if self.use_start_end_tokens:
            words = [START_TOKEN] + words + [END_TOKEN]
        return words


class TextVectorizer:
    """
    maps np.array of strings to np.array of ints
    attributes:
        word_to_index: dict mapping word to integer
        index_to_word: inverse mapping
        vocab_size: total number of index entries (highest index + n special tokens)
    """

    def __init__(self, max_vocab=None):
        # take top max_vocab tokens
        if max_vocab is None:
            max_vocab = len(VOCAB)
        vocab = VOCAB[:max_vocab]

        n_special = len(SPECIAL_TOKENS)
        # +N because those indexes are is reserved for unknown tokens
        self.word_to_index = {word: index+n_special for index,word in enumerate(vocab)}
        for i,tok in enumerate(SPECIAL_TOKENS):
            assert tok not in self.word_to_index
            assert i not in self.word_to_index.values()
            self.word_to_index[tok] = i
        self.vocab_size = len(self.word_to_index)

        oov_index = self.word_to_index[OOV_TOKEN]
        # default to OOV for words not in map
        self._vec_f = np.vectorize(lambda x: self.word_to_index.get(x, oov_index))

        self.index_to_word = {i:w for w,i in self.word_to_index.items()}
        self._unvec_f = np.vectorize(self.index_to_word.__getitem__, otypes=[np.str_])

        # sanity checks
        assert len(self.index_to_word) == len(self.word_to_index)
        assert self._vec_f("THIS_TOKEN_DEFINITELY_DOES_NOT_APPEAR") == oov_index
        assert self._vec_f(PADDING_TOKEN) == 0
        # assert self._vec_f("the") == n_special


    def vectorize(self, array):
        """
        convert string array to int array
        """
        return self._vec_f(array)

    def unvectorize(self, array):
        return self._unvec_f(array)


@np.vectorize
def _np_get_vec_special(tok):
    return SPECIAL_TOKENS.index(tok)

def get_vectorized_special(tok):
    """
    get the vectorization of a special token, as tf.int32
    """
    return tf.cast(_np_get_vec_special(tok), tf.int32)


