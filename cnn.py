import csv
import sys
from load import load, load2d
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
import signal
import sys
import cPickle as pickle
import sys
import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano
from matplotlib import pyplot
import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer

def float32(k):
	return np.cast['float32'](k)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

class AdjustVariable(object):
	def __init__(self, name, start=0.03, stop=0.001):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None

	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

		epoch = train_history[-1]['epoch']
		new_value = float32(self.ls[epoch - 1])
		getattr(nn, self.name).set_value(new_value)

class FlipBatchIterator(BatchIterator):
   flip_indices= [
      (0, 10), (1, 11), (2, 8), (3, 9), 
      (4, 6), (5, 7), (12, 22), (13, 23), 
      (14, 20), (15, 21), (16, 18), (17, 19), 
      (24, 24), (25, 25), (26, 26), (27, 27)
   ]

   def transform(self, Xb, yb):
      Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
      
      # Flip half of the images in this batch at random:
      bs = Xb.shape[0]
      indices = np.random.choice(bs, bs / 2, replace=False)
      Xb[indices] = Xb[indices, :, :, ::-1]

      if yb is not None:
         # Horizontal flip of all x coordinates:
         yb[indices, ::2] = yb[indices, ::2] * -1
   
         # Swap places, e.g. left_eye_center_x -> right_eye_center_x
         for a, b in self.flip_indices:
            yb[indices, a], yb[indices, b] = (
               yb[indices, b], yb[indices, a])

      return Xb, yb

##############################################

def main():
	csv.field_size_limit(sys.maxsize)
	sys.setrecursionlimit(10000)
	X, y = load2d()  # load 2-d data

	print "Loaded data"

	#net_pretrain = None
	#with open('kfkd-net3.pickle', 'rb') as f:
	#	net_pretrain = pickle.load(f)

	#from sklearn.base import clone
	#net3 = clone(net_pretrain)
	#set number of epochs relative to number of training examples:
	#net3.max_epochs = int(1e7 / y.shape[0])

	net3.fit(X, y)

	with open('net3.pickle', 'wb') as f:
		pickle.dump(net3, f, -1)

	print "MSE: ", mean_squared_error(net3.predict(X), y)	
	

#####################################

net3 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('conv2', Conv2DLayer),
        ('pool2', MaxPool2DLayer),
        ('conv3', Conv2DLayer),
        ('pool3', MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('dropout5', layers.DropoutLayer),
		  ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 96, 96),
    conv1_num_filters=96, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
    conv3_num_filters=256, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
    hidden4_num_units=1000,
    dropout4_p=0.6,
    hidden5_num_units=1000,
    dropout5_p=0.6,
	 output_num_units=28, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200),
        ],
    max_epochs=3000,
    verbose=1,
    )


if __name__ == "__main__": main()
