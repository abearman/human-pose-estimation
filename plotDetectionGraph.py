import matplotlib
matplotlib.use('Agg') # Or any other X11 back-end
from cnn import FlipBatchIterator
from cnn import AdjustVariable
from load import load, load2d
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
import sys
import math
import pdj

# First arg is file to de-pickle, second arg is "isTest"
def main(argv):
	file = open(argv[0], 'r')
	net = pickle.load(file)
	print 'Loaded net'

	isVal = len(argv) > 1 and argv[1] == 'val'
	print 'isVal: ', isVal

	X, y_gt = load2d(isVal=isVal, isTiny=False)
	print 'Loaded data'
	y_pred = net.predict(X)
	print 'Predicted data'

	pdj = getPDJ(y_gt, y_pred)
	fig = pyplot.figure(figsize=(6, 6))
	axes = pyplot.gca()
	pyplot.grid()
	axes.set_ylim([0.0, 1.0])
	axes.set_xlim([0.0, 0.5])
	pyplot.xlabel('Normalized distance to true joint')
	pyplot.ylabel('Detection rate')
	pyplot.title('PDJ for different limbs over various thresholds')
	
	fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] 
	pyplot.plot(fractions, pdj[:,0], linewidth=2, label='head', color='#FC474C') 
	pyplot.plot(fractions, pdj[:,1], linewidth=2, label='torso', color='#8DE047')
	pyplot.plot(fractions, pdj[:,2], linewidth=2, label='arms', color='#FFDD50')
	pyplot.plot(fractions, pdj[:,3], linewidth=2, label='legs', color='#53A3D7')

	pyplot.legend(loc='upper left', shadow=True, fontsize='medium')

	pyplot.savefig('detections')	

def getPDJ(y_gt, y_pred):
	# 4 "limbs:" arms, head, torso, legs
	pdj = np.zeros((6, 14)) # from 0 to 0.5

	# 0-based indices for torso
	shoulder_l = 9
	hip_r = 2
	
	num_examples = y_gt.shape[0]
	num_joint_coords = y_gt.shape[1]

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

		index = 0		
		for fraction in drange(0.0, 0.6, 0.1):
			error_radius = fraction * torso_diameter
		  
			# Iterates over each joint
			for j in xrange(len(y_gt_tup)):
				if withinThreshold(y_gt_tup[j], y_pred_tup[j], error_radius): 
					pdj[index][j] += 1			
			
			index += 1			

	for i in xrange(6):
		pdj[i,:] = [x / float(num_examples) for x in pdj[i,:]] 
	
	pdjCombined = np.zeros((6, 4))
	# Iterate over each fraction
	for i in xrange(6):
		# Head
		pdjCombined[i, 0] = (pdj[i, 12] + pdj[i, 13]) / 2	
		# Torso
		pdjCombined[i, 1] = (pdj[i, 12] + pdj[i, 3] + pdj[i, 2] + pdj[i, 8] + pdj[i, 9]) / 5
		# Arms
		pdjCombined[i, 2] = (pdj[i,11] + pdj[i,10] + pdj[i,9] + pdj[i,8] + pdj[i,7] + pdj[i,6]) / 6
		# Legs
		pdjCombined[i, 3] = (pdj[i,5] + pdj[i,4] + pdj[i,3] + pdj[i,2] + pdj[i,1] + pdj[i,0]) / 6

	print "PDJ shape: ", pdjCombined.shape
	print pdjCombined
	return pdjCombined

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def withinThreshold(point1, point2, threshold):
	return (tup_distance(point1, point2) <= threshold)

def tup_distance(tup1, tup2):
	return math.sqrt((tup1[0] - tup2[0])**2 + (tup1[1] - tup2[1])**2)

if __name__ == "__main__":
	main(sys.argv[1:])
