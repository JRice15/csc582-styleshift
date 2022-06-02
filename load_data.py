import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import TextTokenizer, TextVectorizer
from const import MAX_SENT_LEN, PADDING_TOKEN, START_TOKEN, END_TOKEN, MAX_WORD_LEN, OOV_TOKEN

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
        vec = embeddings.get(word)
        if vec is not None:
            embedding_matrix[index] = vec
    
    return embedding_matrix



def load_preprocessed_sent_data(target="label", drop_equal=False, start_end_tokens=False,
        max_vocab=None, show_example=True, return_raw_test=False):
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

    str_dtype = np.dtype("U" + str(MAX_WORD_LEN))
    X_normal = pad_sequences(
                data.normal.to_list(),
                maxlen=MAX_SENT_LEN,
                dtype=str_dtype, # max 50 letters in a word 
                value=PADDING_TOKEN,
                padding="post",
                truncating="post",
            )
    X_simple = pad_sequences(
                data.simple.to_list(),
                maxlen=MAX_SENT_LEN,
                dtype=str_dtype,
                value=PADDING_TOKEN,
                padding="post",
                truncating="post",
            )
    if return_raw_test:
        raw_data = (X_normal, X_simple)

    if show_example:
        print("example padded:")
        print(" ", X_normal[8192])

    vectorizer = TextVectorizer(max_vocab=max_vocab)

    X_normal = vectorizer.vectorize(X_normal)
    X_simple = vectorizer.vectorize(X_simple)

    if show_example:
        print("example vectorized:")
        print(" ", [X_normal[8192]])

    oov_vec_id = vectorizer.vectorize(OOV_TOKEN)
    padding_vec_id = vectorizer.vectorize(PADDING_TOKEN)
    oov_count = (X_normal == oov_vec_id).sum() + (X_simple == oov_vec_id).sum()
    total_count = (X_normal != padding_vec_id).sum() + (X_simple !=padding_vec_id).sum()
    print("Vectorizing OOV rate (fraction):", oov_count / total_count)

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

