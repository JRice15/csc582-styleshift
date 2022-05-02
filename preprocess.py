import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


class TextPreprocesser:
    """
    class that does the following to text input:
    - tokenise, by splitting and lowering
    - embed to vectors, using glove 6B embeddings
    """

    def __init__(self, sent_length, embedding_dim=100):
        """
        adapted from https://keras.io/examples/nlp/pretrained_word_embeddings/
        args:
            sent_length: length of sentences. 
            embedding_dim: size of vectors to use: 50, 100, 200, or 300
        """
        self.sent_length = sent_length
        self.embedding_dim = embedding_dim
        self.embeddings = self._load_embeddings(embedding_dim)
        self.zero_vec = np.zeros((embedding_dim,))


    def _load_embeddings(self, embedding_dim):
        # load glove embeddings
        embeddings = {}
        glove_path = "data/glove_6B/glove.6B.{}d.txt".format(embedding_dim)
        with open(glove_path, "r") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings[word] = coefs

        print("Glove embeddings: Found %s word vectors" % len(embeddings))
        return embeddings

    def convert_texts(self, texts):
        """
        convert list of sentences (strings) to an array of shape (num sentences, max sent length, embedding dim)
        """
        output = np.zeros((len(texts), self.sent_length, self.embedding_dim))
        for i,sent in enumerate(texts):
            # tokenize
            words = sent.lower().strip().split()
            # truncate too long sentences
            words = words[:self.sent_length]
            # convert known words to embeddings
            for j,word in enumerate(words):
                if word in self.embeddings:
                    output[i,j,:] = self.embeddings[word]

        return output

