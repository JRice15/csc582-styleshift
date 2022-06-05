import os
import re

import numpy as np
import pandas as pd

"""
filters for bad examples
"""

BAD_TITLE_PREFIXES = [
    # wiki internals pages that should not be included
    "Template:",
    "Wikipedia:",
    "Category:",
    "MediaWiki:",
    "Help:",
]

@np.vectorize
def is_ok_title(title):
    return not any(title.startswith(x) for x in BAD_TITLE_PREFIXES)

BAD_TEXT_PATTERNS = [
    "jpg thumb",
    "JPG thumb",
    "gif thumb",
    "png thumb",
    "svg thumb",
]

@np.vectorize
def is_ok_text(text):
    return not any(x in text for x in BAD_TEXT_PATTERNS)

"""
reading/filtering raw data
"""

def read_data_tsv(filename, cols=("title", "para", "text")):
    df = pd.read_csv(filename, sep="\t", header=None)
    df.columns = cols
    return df

def read_data(kind="sentence"):
    """
    args:
        kind: "sentence"|"document"
    returns:    
        pandas df
    """
    simple = read_data_tsv("data/{}-aligned.v2/simple.aligned".format(kind), cols=("title", "para", "simple"))
    normal = read_data_tsv("data/{}-aligned.v2/normal.aligned".format(kind), cols=("title", "para", "normal"))
    normal = normal.drop(columns=["title", "para"])
    df = pd.concat([simple, normal], axis=1)
    # filter out bad pages
    df = df[is_ok_title(df["title"])]
    # filter out bad text
    all_text = df["normal"] + " " + df["simple"]
    df = df[is_ok_text(all_text)]
    return df.reset_index(drop=True)


"""
Vocab
"""

def read_vocab(min_count=None):
    """
    load the vocabulary
    returns:
        list of words, in order of frequency (most to least frequent)
    """
    if not os.path.exists("data/vocab.csv"):
        from make_vocab import make_vocab
        make_vocab()
    # don't interpret the valid vocab words 'nan' and 'null' as missing values
    df = pd.read_csv("data/vocab.csv", na_filter=False) 
    if min_count is not None:
        df = df[df["count"] >= min_count]
    return df["word"].to_list()
    


"""
Glove stuff
"""

def load_glove_embeddings(embedding_dim):
    embeddings = {}
    glove_path = "data/glove_6B/glove.6B.{}d.txt".format(embedding_dim)
    with open(glove_path, "r") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings[word] = coefs
    print("Glove embeddings: Found %s word vectors" % len(embeddings))
    return embeddings


def make_embedding_matrix(embedding_dim, vectorizer):
    embeddings = load_glove_embeddings(embedding_dim)

    # Prepare embedding matrix
    embedding_matrix = np.zeros((vectorizer.vocab_size, embedding_dim))
    for word,index in vectorizer.word_to_index.items():
        vec = embeddings.get(word)
        if vec is not None:
            embedding_matrix[index] = vec
    
    return embedding_matrix
