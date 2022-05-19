import numpy as np
import pandas as pd
from collections import defaultdict

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from load_data import read_data

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
        self.word_to_index = {word: index+1 for index,word in enumerate(sorted(vocab))}
        self.index_size = len(vocab) + 1

        mapper = lambda x: self.word_to_index.get(x, 0) # default to index 0 if not in vocab map
        self._vec_f = np.vectorize(mapper)

        self.index_to_word = {i:w for w,i in self.word_to_index.items()}
        unmapper = lambda x: self.index_to_word.get(x, " ")
        self._unvec_f = np.vectorize(unmapper)

        # sanity checks
        assert self._word_mapper("THIS_TOKEN_DEFINITELY_DOES_NOT_APPEAR") == 0
        assert self._word_mapper("word") != 0


    def vectorize(self, array):
        """
        convert string array to int array
        """
        return self._vec_f(array)

    def unvectorize(self, array):
        return self._unvec_f(array)


def load_preprocessed_sent_data(max_sent_len, embedding_dim, target="label", 
        drop_equal=False):
    """
    args:
        max_sent_len
        embedding_dim
        drop_equal: whether to drop sentences that are the same
        target: "label" or "simple"
    """
    print("Loading embeddings")
    embeddings = load_glove_embeddings(embedding_dim)
    vocab = list(embeddings.keys())

    vectorizer = TextVectorizer(vocab)

    # Prepare embedding matrix
    embedding_matrix = np.zeros((vectorizer.index_size, embedding_dim))
    for word,index in vectorizer.word_index.items():
        embedding_vector = embeddings.get(word)
        embedding_matrix[index] = embedding_vector

    print("Loading data...")
    data = read_data("sentence")

    tokenizer = TextTokenizer(max_sent_len)

    data.simple = data.simple.apply(tokenizer.tokenize_sent)
    data.normal = data.normal.apply(tokenizer.tokenize_sent)

    print("max sent len:", max_sent_len)
    print("fraction of sentences truncated:", data["normal"].apply(lambda x: len(x) > max_sent_len).mean())

    X_normal = pad_sequences(
                data.normal.to_list(),
                maxlen=max_sent_len,
                dtype=np.str_, truncating="post")
    X_simple = pad_sequences(
                data.simple.to_list(),
                maxlen=max_sent_len,
                dtype=np.str_, truncating="post")

    if target == "label":
        X = np.concatenate([X_normal, X_simple], axis=0, dtype=np.str_)
        Y = np.concatenate([np.ones(len(data.normal)), np.zeros(len(data.simple))], axis=0)
        # convert strings to ints
        X = vectorizer.vectorize(X)
    elif target == "simple":
        X = X_normal
        Y = X_simple
        # convert strings to ints
        X = vectorizer.vectorize(X)
        Y = vectorizer.vectorize(Y)
    else:
        raise ValueError(f"unknown target argument '{target}'")

    return X, Y, embedding_matrix
