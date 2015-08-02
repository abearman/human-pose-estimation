import os
import numpy as np
import pandas
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import csv
import sys

FTRAIN = '/big1/231n-data/training.csv'
FTEST = '/big1/231n-data/test.csv'
DATA_PATH = '/big1/231n-data/'
FVAL = '/big1/231n-data/test-annotated.csv'

flip_indices= [(0, 10), (1, 11), (2, 8), (3, 9), (4, 6), (5, 7), (12, 22), (13, 23), (14, 20), (15, 21), (16, 18), (17, 19), (24, 24), (25, 25), (26, 26), (27, 27)
]

def load2d(test=False, cols=None, isTiny=False, isVal=False):
    csv.field_size_limit(sys.maxsize)
    X, y = load(test=test, isTiny=isTiny, isVal=isVal)
    X = X.reshape(-1, 3, 96, 96)
    return X, y

def load(test=False, cols=None, isTiny=False, isVal=False):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    if isVal: fname = FVAL

    if isTiny:
        fname = DATA_PATH + 'tiny-training.csv'
    df = pandas.read_csv(os.path.expanduser(fname), engine='python')

    print "Done reading csv"

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    #print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


