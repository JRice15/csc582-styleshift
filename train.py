import numpy as np
import pandas as pd

from preprocess import TextPreprocesser
from load_data import read_data


# t = TextPreprocesser(10)
# print(t.convert_texts(texts).shape)

simple, normal = read_data("sentence")

print(simple)