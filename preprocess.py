import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from const import (END_TOKEN, MAX_SENT_LEN, NUMERIC_TOKEN, OOV_TOKEN,
                   PADDING_TOKEN, SPECIAL_TOKENS, START_TOKEN, MAX_WORD_LEN)
from load_data import read_data, read_vocab


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

    def _tokenize_word(self, word):
        return self.CUSTOM_TOKEN_TRANSLATIONS.get(word, word) # first arg to get is key to find, second arg is the default to use if the key is not found

    def tokenize_sent(self, sent):
        """
        convert list of sentences (string), each string goes to a list of tokens
        """
        words = sent.lower().strip().split()
        words = [self._tokenize_word(w) for w in words]
        if self.use_start_end_tokens:
            words = [START_TOKEN] + words + [END_TOKEN]
        return words


class TextPadder:
    """
    convert lists of strings to array of str of uniform length, shape (n sents, max_sent_len)
    """

    def __init__(self, padding="post", truncating="post"):
        self.where_pad = padding
        self.where_truncate = truncating
        self.str_dtype = np.dtype("U" + str(MAX_WORD_LEN))

    def pad_sents(self, sents):
        return pad_sequences(
                sents,
                maxlen=MAX_SENT_LEN,
                dtype=self.str_dtype, # max 50 letters in a word 
                value=PADDING_TOKEN,
                padding=self.where_pad,
                truncating=self.where_truncate,
            )



class TextVectorizer:
    """
    maps np.array of strings to np.array of ints
    attributes:
        word_to_index: dict mapping word to integer
        index_to_word: inverse mapping
        vocab_size: total number of index entries (highest index + n special tokens)
    """

    # contains at least one number, and no letters
    IS_NUMERIC = re.compile(r"[^a-z]*\d[^a-z]*")

    def __init__(self, min_word_freq=3):
        vocab = read_vocab(min_count=min_word_freq)

        n_special = len(SPECIAL_TOKENS)
        # +N because those indexes are is reserved for unknown tokens
        self.word_to_index = {word: index+n_special for index,word in enumerate(vocab)}
        for i,tok in enumerate(SPECIAL_TOKENS):
            assert tok not in self.word_to_index
            assert i not in self.word_to_index.values()
            self.word_to_index[tok] = i
        self.vocab_size = len(self.word_to_index)
        self.index_to_word = {i:w for w,i in self.word_to_index.items()}

        oov_index = self.word_to_index[OOV_TOKEN]
        num_index = self.word_to_index[NUMERIC_TOKEN]

        def _vec_f(word):
            if re.fullmatch(self.IS_NUMERIC, word):
                return num_index
            # default to OOV for words not in map
            return self.word_to_index.get(word, oov_index)
            
        self._vec_f = np.vectorize(_vec_f, otypes=[np.int32])

        self._unvec_f = np.vectorize(self.index_to_word.__getitem__, otypes=[np.str_])

        # sanity checks
        assert len(self.index_to_word) == len(self.word_to_index)
        assert len(self.index_to_word) == max(self.index_to_word.keys()) + 1
        assert self._vec_f("THIS_TOKEN_DEFINITELY_DOES_NOT_APPEAR") == oov_index
        assert self._vec_f(PADDING_TOKEN) == 0
        assert self._vec_f("the") == n_special # most common word is first index after special tokens
        assert "testing" == self._unvec_f(self._vec_f("testing")) # make sure they are inverses

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

def get_vectorized_special(tok, as_tf=True):
    """
    get the vectorization of a special token, as tf.int32
    """
    result = _np_get_vec_special(tok)
    if as_tf:
        return tf.cast(result, tf.int32)
    if len(result.shape) == 0:
        return result.item()
    return result



"""
Full preprocessing pipeline
"""

def load_preprocessed_sent_data(target, drop_equal=False, start_end_tokens=False,
        min_word_freq=None, show_example=True, return_raw_test=False):
    """
    args:
        target: "label" or "simple"
        drop_equal: bool, whether to drop sentences that are the same
        start_end_tokens: bool, whether to include a start and end token at the beginning at end of each sentence
        return_raw_test: whether to return the raw test data (pre-OOV vectorization)
    returns:
        tuple(x_train, y_train, x_val, y_val, x_test, y_test)
        vectorizer
        (optional) raw test: tuple of (raw normal, raw simple), each is: list of list of str
    """
    print("Loading data...")
    data = read_data("sentence")
    print(len(data), "paired sentences found")

    tokenizer = TextTokenizer(use_start_end_tokens=start_end_tokens)

    if drop_equal:
        orig_len = len(data)
        data = data[data.simple != data.normal]
        print(orig_len - len(data), "examples dropped for being the same.", len(data), "remain")

    # tokenize
    data.simple = data.simple.apply(tokenizer.tokenize_sent)
    data.normal = data.normal.apply(tokenizer.tokenize_sent)

    orig_examples = len(data)
    data = data[(data.normal.apply(len) <= MAX_SENT_LEN) & (data.simple.apply(len) <= MAX_SENT_LEN)]
    print("fraction of sentences dropped for length:", (orig_examples - len(data)) / orig_examples)

    if show_example:
        print("example tokenized:")
        print(" ", data.normal.iloc[8192])

    padder = TextPadder()

    X_normal = padder.pad_sents(data.normal.to_list())
    X_simple = padder.pad_sents(data.simple.to_list())

    if return_raw_test:
        raw_data = (X_normal, X_simple)

    if show_example:
        print("example padded:")
        print(" ", X_normal[8192])

    vectorizer = TextVectorizer(min_word_freq=min_word_freq)

    X_normal = vectorizer.vectorize(X_normal)
    X_simple = vectorizer.vectorize(X_simple)

    if show_example:
        print("example vectorized:")
        print(" ", X_normal[8192])

    oov_vec_id = vectorizer.vectorize(OOV_TOKEN)
    padding_vec_id = vectorizer.vectorize(PADDING_TOKEN)
    oov_count = (X_normal == oov_vec_id).sum() + (X_simple == oov_vec_id).sum()
    total_count = (X_normal != padding_vec_id).sum() + (X_simple !=padding_vec_id).sum()
    print("Vectorizing OOV rate (fraction):", oov_count / total_count)
    print("Vocab size:", vectorizer.vocab_size)

    # 20% testing, ~10% validation
    train_inds, test_inds = train_test_split(np.arange(len(X_normal)), test_size=0.2, random_state=1)
    train_inds, val_inds = train_test_split(train_inds, test_size=0.1, random_state=1)

    set_names = ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]
    datasets = []

    for indexes in (train_inds, val_inds, test_inds):

        if target == "label":
            X = np.concatenate([X_normal[indexes], X_simple[indexes]], axis=0, dtype=np.str_)
            Y = np.concatenate([np.ones(len(indexes)), np.zeros(len(indexes))], axis=0)
            # convert strings to ints
            # X = vectorizer.vectorize(X)
        elif target == "simple":
            X = X_normal[indexes]
            Y = X_simple[indexes]
            # convert strings to ints
            # X = vectorizer.vectorize(X)
            # Y = vectorizer.vectorize(Y)
        else:
            raise ValueError(f"unknown target argument '{target}'")
        
        datasets += [X, Y]

    print("dataset sizes:")
    for name, ds in zip(set_names, datasets):
        print(" ", name, ds.shape)

    datasets = tuple([tf.constant(x, dtype=tf.int32) for x in datasets])

    if return_raw_test:
        raw_normal = raw_data[0][test_inds]
        raw_simple = raw_data[1][test_inds]
        raw = (raw_normal, raw_simple)
        return datasets, vectorizer, raw
    return datasets, vectorizer


