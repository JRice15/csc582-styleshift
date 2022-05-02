import numpy as np
import pandas as pd

def read_data_tsv(filename):
    df = pd.read_csv(filename, sep="\t", header=None)
    df.columns = ["title", "para", "text"]
    return df


def read_data(kind="sentence"):
    """
    args:
        kind: "sentence"|"document"
    returns:    
        pandas df: simple, normal
    """
    simple = read_data_tsv("data/{}-aligned.v2/simple.aligned".format(kind))
    normal = read_data_tsv("data/{}-aligned.v2/normal.aligned".format(kind))
    return simple, normal



