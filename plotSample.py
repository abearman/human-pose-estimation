import matplotlib
matplotlib.use('Agg') # Or any other X11 back-end
from cnn import FlipBatchIterator
from cnn import AdjustVariable
from load import load, load2d
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
import sys

# First arg is file to de-pickle, second argument is "isTest"
def main(argv):
	file = open(argv[0], 'r')
	net = pickle.load(file)

	isTest = len(argv) > 1 and argv[1] == 'test'

	X, _ = load2d(test=isTest)
	y_pred = net.predict(X)

	fig = pyplot.figure(figsize=(6, 6))
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i in range(16):
		ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
		plot_sample(X[i], y_pred[i], ax)
	pyplot.savefig('samples')

def plot_sample(x, y, axis):
	img = x.reshape(96, 96, 3)
	axis.imshow(img)
	axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

if __name__ == "__main__":
	main(sys.argv[1:])
