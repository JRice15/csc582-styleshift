import numpy as np
import pandas as pd

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
    return df


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


