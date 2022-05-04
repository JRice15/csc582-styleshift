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


class TextEmbedder:
    """
    class that embeds to vectors, using glove 6B embeddings
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
        # counters
        self.successful_embeds = 0
        self.embed_attempts = 0
        self.failed_embeds = defaultdict(lambda: 0)


    def _load_embeddings(self, embedding_dim):
        """
        load Glove embeddings. See https://nlp.stanford.edu/projects/glove/
        """
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
        convert list of sentences (each sent is a list of words) to an array of shape (num sentences, max sent length, embedding dim)
        """
        assert not isinstance(texts, str)
        output = np.zeros((len(texts), self.sent_length, self.embedding_dim))
        for i,words in tqdm(enumerate(texts), total=len(texts)):
            # convert known words to embeddings
            for j,word in enumerate(words[:self.sent_length]):
                self.embed_attempts += 1
                if word in self.embeddings:
                    output[i,j,:] = self.embeddings[word]
                    self.successful_embeds += 1
                # track failed embeddings
                else:
                    self.failed_embeds[word] += 1

        return output

    def success_rate(self):
        return self.successful_embeds / self.embed_attempts
