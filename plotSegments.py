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

	print 'Loaded net'

	isTest = len(argv) > 1 and argv[1] == 'test'
	print 'isTest: ', isTest

	X, _ = load2d(test=isTest, isTiny=False)
	print 'Loaded data'
	print 'X shape: ', X.shape
	y_pred = net.predict(X)
	print "y_pred shape: ", y_pred.shape
	print "x shape: ", X.shape

	print 'Predicted data'

	fig = pyplot.figure(figsize=(6, 6))
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i in range(42):
		ax = fig.add_subplot(6, 7, i + 1, xticks=[], yticks=[])
		plot_sample(X[i], y_pred[i], ax)
	pyplot.savefig('segments')

def plot_sample(x, y, axis):
	img = x.reshape(96, 96, 3)
	axis.imshow(img)
	xs = y[0::2] * 48 + 48
	ys = y[1::2] * 48 + 48
	zipped = zip(xs, ys) 

	color_arr = ['#0949DB', '#00E900', '#FF0000', '#39FFFF', '#D900D9', '#FFE200', '#0949DB', '#00E900', '#FF0000', '#39FFFF', '#D900D9', '#FFE200', '#0949DB', '#00E900']
	pair_indices = [(5, 4), (0, 1), (3, 4), (1, 2), (2, 3), (12, 3), (12, 2), (10, 11), (9, 11), (9, 12), (12, 8), (7, 8), (6, 7), (12, 13)]
	i = 0
	for a, b in pair_indices:
		pt1 = zipped[a]
		pt2 = zipped[b]
		pyplot.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color_arr[i], linestyle='-', linewidth=2)
		i += 1

if __name__ == "__main__":
	main(sys.argv[1:])
