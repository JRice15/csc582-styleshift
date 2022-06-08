"""
convert csv with results to seperate txt files for each method
"""

import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dir",required=True,help="directory of model")
ARGS = parser.parse_args()


df = pd.read_csv(ARGS.dir + "bleu/preds_subsampled10x.csv")

os.makedirs(ARGS.dir + "seperated", exist_ok=True)

pd.set_option('display.max_colwidth', None)


for col in df.columns:
    path = ARGS.dir + "seperated/" + col + ".txt"
    data = df[col]
    with open(path, "w") as f:
        for sent in data:
            f.write(sent + "\n")
