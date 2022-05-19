import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import TextTokenizer, TextVectorizer


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


def make_embedding_matrix(embedding_dim, vectorizer):
    embeddings = load_glove_embeddings(embedding_dim)

    # Prepare embedding matrix
    embedding_matrix = np.zeros((vectorizer.vocab_size, embedding_dim))
    for word,index in vectorizer.word_to_index.items():
        embedding_vector = embeddings.get(word)
        embedding_matrix[index] = embedding_vector
    
    return embedding_matrix



def load_preprocessed_sent_data(max_sent_len, embedding_dim, target="label", 
        drop_equal=False):
    """
    args:
        max_sent_len
        embedding_dim
        drop_equal: whether to drop sentences that are the same
        target: "label" or "simple"
    returns:
        tuple(x_train, y_train, x_val, y_val, x_test, y_test)
        vectorizer
    """
    print("Loading data...")
    data = read_data("sentence")

    tokenizer = TextTokenizer()

    if drop_equal:
        orig_len = len(data)
        data = data[data.simple != data.normal]
        print(orig_len - len(data), "examples dropped for being the same.", len(data), "remain")

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

    vectorizer = TextVectorizer()

    # 20% testing, ~10% validation
    train_inds, test_inds = train_test_split(np.arange(len(X_normal)), test_size=0.2, random_state=1)
    train_inds, val_inds = train_test_split(train_inds, test_size=0.1, random_state=1)

    results = []

    for indexes in (train_inds, val_inds, test_inds):

        if target == "label":
            X = np.concatenate([X_normal[indexes], X_simple[indexes]], axis=0, dtype=np.str_)
            Y = np.concatenate([np.ones(len(indexes)), np.zeros(len(indexes))], axis=0)
            # convert strings to ints
            X = vectorizer.vectorize(X)
        elif target == "simple":
            X = X_normal[indexes]
            Y = X_simple[indexes]
            # convert strings to ints
            X = vectorizer.vectorize(X)
            Y = vectorizer.vectorize(Y)
        else:
            raise ValueError(f"unknown target argument '{target}'")
        
        results += [X, Y]

    results = [tf.constant(x, dtype=tf.int32) for x in results]

    return tuple(results), vectorizer

