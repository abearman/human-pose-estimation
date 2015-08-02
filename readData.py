import scipy.io
import csv
import glob
import os
from PIL import Image
import sys
import numpy as np

# First argument is isTiny
def main(argv):
	datapath1 = '/big1/231n-data/lspet_dataset/'
	datapath2 = '/big1/231n-data/lsp_dataset/' 
	joints1 = scipy.io.loadmat(datapath1 + 'joints.mat')
	joints1 = joints1['joints']
	# We have to transpose because the extended dataset has joints.mat as shape (14, 3, 10000)
	joints1 = joints1.transpose(1, 0, 2)

	joints2 = scipy.io.loadmat(datapath2 + 'joints.mat')
	joints2 = joints2['joints']
	print "joints2 shape: ", joints2.shape

	trainfile = ""
	isTiny = len(argv) > 0 and argv[0] == 'tiny'
	if isTiny:
		trainfile = '/big1/231n-data/tiny-training.csv'
	else:
		trainfile = '/big1/231n-data/training.csv'
	valfile = '/big1/231n-data/test.csv'

	t = open(trainfile, 'w')
	v = open(valfile, 'w')

	# Iterate over all examples
	
	# Writes the header for training file. 
	t.write("right_ankle_x,right_ankle_y,right_knee_x,right_knee_y,right_hip_x,right_hip_y,left_hip_x,left_hip_y,left_knee_x,left_knee_y,left_ankle_x,left_ankle_y,right_wrist_x,right_wrist_y,right_elbow_x,right_elbow_y,right_shoulder_x,right_shoulder_y,left_shoulder_x,left_shoulder_y,left_elbow_x,left_elbow_y,left_wrist_x,left_wrist_y,neck_x,neck_y,head_top_x,head_top_y,Image\n")

	# Writes header for val file
	v.write("ImageId,Image\n")

	if isTiny:
		writeFolder(datapath2, joints2, t, isTiny=True)
	else:
		writeFolder(datapath1, joints1, t)
		writeFolder(datapath2, joints2, t, v, test=True)
	t.close()
	v.close()

# We only need to write to v if test is True
def writeFolder(datapath, joints, t, v=None, test=False, isTiny=False):
	filelist = glob.glob(os.path.join(datapath + 'images/', '*.jpg'))
	sorted_unscaled_filelist = sorted(glob.glob(os.path.join(datapath + 'images-unscaled/', '*.jpg')))
	numfiles = len(filelist)
	print "numfiles: ", numfiles
	i = 0

	for infile in sorted(filelist):
		unscaledImg = Image.open(sorted_unscaled_filelist[i])
		width, height = unscaledImg.size

		jointsX = joints[0,:,i]
		jointsY = joints[1,:,i]

		rescaledXs = jointsX / width * 96;
		rescaledYs = jointsY / height * 96;

		zipped = [val for pair in zip(rescaledXs, rescaledYs) for val in pair]
		xys = ','.join(str(e) for e in zipped)

		im = Image.open(infile, 'r')
		pix_val = list(im.getdata())

		if all(isinstance(x, tuple) for x in pix_val):
			pix_val = [x for sets in pix_val for x in sets]
		pxs = ' '.join(str(e) for e in pix_val)

		# Write to validation file (if we're halfway through the non-extended set)
		if test and i >= 1000:
			v.write(str(i) + "," + pxs + "\n")
		# Write to training file
		else:
			t.write(xys + ",")
			t.write(pxs + "\n")

		print i

		if isTiny and i > 40: break
		i += 1

if __name__ == "__main__": 
	main(sys.argv[1:])
