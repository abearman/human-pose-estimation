import matplotlib
matplotlib.use('Agg') # Or any other X11 back-end
from cnn import FlipBatchIterator
from cnn import AdjustVariable
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
import sys
from load import load2d
import math

# First arg is the file we want to de-pickle
# Second arg is the value of threshold 
# Third arg is 'isVal'
def main(argv):
	file = open(argv[0], 'r')
	net = pickle.load(file)

	isVal = (len(argv) > 2) and (argv[2] == 'val')

	X, y_gt = load2d(isVal=isVal) # for training
	print "y_gt shape: ", y_gt.shape
	y_pred = net.predict(X)

	total_limbs = 14 # 2x ankle-knee, 2x knee-hip, hip-hip, 2x hip-shoulder, neck-head, 2x shoulder-neck, 2x shoulder-elbow, 2x elbow-wrist
	pair_indices = [(5, 4), (0, 1), (3, 4), (1, 2), (2, 3), (9, 3), (8, 2), (10, 11), (9, 10), (9, 12), (12, 8), (7, 8), (6, 7), (12, 13)]

	#pair_indices = [(0, 1), (1, 2), (4, 5), (3, 4), (6, 7), (7, 8), (10, 11), (9, 10), (12, 13)]
	#pair_indices = [(9, 12), (8, 12), (9, 3), (2, 3), (2, 8)] # torso 

	pcp = [0] * len(pair_indices)
	num_examples = y_gt.shape[0]
	num_joints = y_gt.shape[1] 
	
	# Iterates over each example
	for i in xrange(num_examples):
		y_gt_tup = []
		y_pred_tup = []

		# Builds tuple vector of joints (x, y)
		for j in xrange(0, num_joints, 2):
			y_gt_tup.append((y_gt[i, j], y_gt[i, j+1]))
			y_pred_tup.append((y_pred[i, j], y_pred[i, j+1]))

		# Iterates over each joint pairing
		#for a,b in pair_indices:
		for x in range(len(pair_indices)):
			a, b = pair_indices[x]
			gt_distance = tup_distance(y_gt_tup[a], y_gt_tup[b])
			error_radius = 0.5 * gt_distance

			if withinThreshold(y_pred_tup[a], y_gt_tup[a], error_radius) or withinThreshold(y_pred_tup[b], y_gt_tup[b], error_radius):
				pcp[x] += 1

	pcp = [x / float(num_examples) for x in pcp]
	print "PCP: ", pcp

def withinThreshold(point1, point2, threshold):
	return (tup_distance(point1, point2) <= threshold)

def tup_distance(tup1, tup2):
	return math.sqrt((tup1[0] - tup2[0])**2 + (tup1[1] - tup2[1])**2)

if __name__ == "__main__": main(sys.argv[1:])

