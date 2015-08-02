import matplotlib
matplotlib.use('Agg') # Or any other X11 back-end
from cnn import FlipBatchIterator
from cnn import AdjustVariable
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
import sys

# First argument is the file we want to pickle
def plotLoss(argv):
	file = open(argv[0], 'r')
	net = pickle.load(file)
	
	train_loss = np.array([i["train_loss"] for i in net.train_history_])
	#valid_loss = np.array([i["valid_loss"] for i in net.train_history_])

	pyplot.plot(train_loss, linewidth=3) 
	#pyplot.plot(valid_loss, linewidth=3) 
	pyplot.grid()
	pyplot.legend()
	pyplot.xlabel("epoch")
	pyplot.ylabel("loss")
	#pyplot.ylim(1e-3, 1e-2)
	pyplot.yscale("log")
	#pyplot.show()

	print "Saving figure"
	pyplot.savefig('loss')

if __name__ == "__main__": 
	plotLoss(sys.argv[1:])
