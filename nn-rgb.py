import os
import numpy as np
import pandas
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg') # Or any other X11 back-end
import matplotlib.pyplot as pyplot
from load import load
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

FTRAIN = '/big1/231n-data/lsp_dataset/train.csv'
FTEST = '/big1/231n-data/lsp_dataset/test.csv'

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 27648),  # 96x96x3 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=28,  # 28 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.005,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
    )

X, y = load()
print "Done loading"
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))
net1.fit(X, y)

# Training for 400 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
import cPickle as pickle
with open('net1.pickle', 'wb') as f:
    pickle.dump(net1, f, -1)

train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
fig = pyplot.figure()
pyplot.plot(train_loss, linewidth=3, label="train")
pyplot.plot(valid_loss, linewidth=3, label="valid")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
#pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
#pyplot.show()
fig.savefig('loss.png')

#def plot_sample(x, y, axis):
 #   print "x shape: ", x.shape
  #  img = x.reshape(96, 96)
   # axis.imshow(img, cmap='gray')
   # axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

#X, _ = load(test=True)
#y_pred = net1.predict(X)

#fig = pyplot.figure(figsize=(6, 6))
#fig.subplots_adjust(
 #   left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

#for i in range(16):
 #   ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
  #  plot_sample(X[i], y_pred[i], ax)

#pyplot.show()
