import re
import time

import numpy as np
import pandas as pd

from load_data import read_data
from preprocess import TextTokenizer
from const import NUMERIC_TOKEN

def make_vocab():
    print("Generating vocab...")
    t1 = time.perf_counter()

    data = read_data("sentence")

    tok = TextTokenizer(False)

    data.simple = data.simple.apply(tok.tokenize_sent)
    data.normal = data.normal.apply(tok.tokenize_sent)

    all_words = pd.concat([data.simple, data.normal]).explode()

    df = all_words.value_counts(sort=True, ascending=False)
    df = df.reset_index()
    df.columns = ["word", "count"]

    print("unique tokens:", len(df))

    # drop words that only appear once
    df = df[df["count"] > 1]
    print("dropping words with count == 1:", len(df), "remain")

    print("words with count >= 3:", len(df[df["count"] >= 3]))
    print("words with count >= 4:", len(df[df["count"] >= 4]))
    print("words with count >= 5:", len(df[df["count"] >= 5]))
    print("words with count >= 10:", len(df[df["count"] >= 10]))
    print("words with count >= 100:", len(df[df["count"] >= 100]))

    print(df)

    df.to_csv("data/vocab.csv", index=False)

    print("done:", time.perf_counter() - t1, "sec")

if __name__ == "__main__":
    make_vocab()