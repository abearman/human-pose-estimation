import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FTRAIN = '/big1/231n-data/lsp_dataset/training.csv'

flip_indices= [
	(0, 10), (1, 11), (2, 8), (3, 9), (4, 6), (5, 7), (12, 22), (13, 23), (14, 20), (15, 21), (16, 18), (17, 19), (24, 24), (25, 25), (26, 26), (27, 27)
]

df = read_csv(os.path.expanduser(FTRAIN))
for i, j in flip_indices:
    print("# {} -> {}".format(df.columns[i], df.columns[j]))
