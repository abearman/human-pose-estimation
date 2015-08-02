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
# Third arg is "isVal" 
def main(argv):
	file = open(argv[0], 'r')
	fraction = float(argv[1])
	net = pickle.load(file)

	isVal = (len(argv) > 2) and (argv[2] == 'val')
	
	X, y_gt = load2d(isVal=isVal) # for training
	y_pred = net.predict(X)

	# 0-based indices for torso
	shoulder_l = 9
	hip_r = 2

	num_examples = y_gt.shape[0]
	num_joint_coords = y_gt.shape[1] 
	pdj = [0] * 14 

	# Iterates over each example
	for i in xrange(num_examples):
		num_correct_joints = 0
		y_gt_tup = []
		y_pred_tup = []

		# Builds tuple vector of joints (x, y) 
		for j in xrange(0, num_joint_coords, 2):
			y_gt_tup.append((y_gt[i, j], y_gt[i, j+1]))
			y_pred_tup.append((y_pred[i, j], y_pred[i, j+1]))

		torso_diameter = tup_distance(y_gt_tup[shoulder_l], y_gt_tup[hip_r])
		error_radius = fraction * torso_diameter

		# Iterates over each joint
		for j in xrange(len(y_gt_tup)):
			if withinThreshold(y_gt_tup[j], y_pred_tup[j], error_radius): 
				pdj[j] += 1

	print "num examples: ", num_examples	
	pdj = [x / float(num_examples) for x in pdj]
	print "PDJ: ", pdj 
	overall = sum(pdj) / len(pdj)
	print "PDJ: ", overall

	head = (pdj[12] + pdj[13]) / 2
	torso = (pdj[12] + pdj[3] + pdj[2] + pdj[8] + pdj[9]) / 5
	arms = (pdj[11] + pdj[10] + pdj[9] + pdj[8] + pdj[7] + pdj[6]) / 6
	legs = (pdj[5] + pdj[4] + pdj[3] + pdj[2] + pdj[1] + pdj[0]) / 6
	print "PDJ head, torso, arms, legs: ", [head, torso, arms, legs]

def withinThreshold(point1, point2, threshold):
	return (tup_distance(point1, point2) <= threshold)

def tup_distance(tup1, tup2):
	return math.sqrt((tup1[0] - tup2[0])**2 + (tup1[1] - tup2[1])**2)

if __name__ == "__main__": main(sys.argv[1:])

