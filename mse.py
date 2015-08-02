from cnn import FlipBatchIterator
from cnn import AdjustVariable
import numpy as np
import pickle
import sys
from sklearn.metrics import mean_squared_error
from load import load2d

# First argument is the file we want to pickle
def MSE(argv):
	file = open(argv[0], 'r')
	net = pickle.load(file)

	X, y = load2d()  # load 2-d data
	print "MSE: ", mean_squared_error(net.predict(X), y)

if __name__ == "__main__":
	MSE(sys.argv[1:])                              

