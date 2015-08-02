import scipy.io
import csv
import glob
import os
from PIL import Image
import sys
import numpy as np

def main(argv):
	datapath = '/big1/231n-data/lsp_dataset/' 
	joints = scipy.io.loadmat(datapath + 'joints.mat')
	joints = joints['joints']
	# We have to transpose because the extended dataset has joints.mat as shape (14, 3, 10000)

	valfile = '/big1/231n-data/test-annotated.csv'
	v = open(valfile, 'w')

	# Iterate over all examples
	
	# Writes the header for training file. 
	v.write("right_ankle_x,right_ankle_y,right_knee_x,right_knee_y,right_hip_x,right_hip_y,left_hip_x,left_hip_y,left_knee_x,left_knee_y,left_ankle_x,left_ankle_y,right_wrist_x,right_wrist_y,right_elbow_x,right_elbow_y,right_shoulder_x,right_shoulder_y,left_shoulder_x,left_shoulder_y,left_elbow_x,left_elbow_y,left_wrist_x,left_wrist_y,neck_x,neck_y,head_top_x,head_top_y,Image\n")

	writeFolder(datapath, joints, v) 
	v.close()

# We only need to write to v if test is True
def writeFolder(datapath, joints, v):
	filelist = glob.glob(os.path.join(datapath + 'images/', '*.jpg'))
	sorted_unscaled_filelist = sorted(glob.glob(os.path.join(datapath + 'images-unscaled/', '*.jpg')))
	numfiles = len(filelist)
	print "numfiles: ", numfiles
	i = 0

	# Only look at last 1000
	for infile in sorted(filelist):
		if (i >= 1000):
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

			v.write(xys + ",")
			v.write(pxs + "\n")

		print i
		i += 1

if __name__ == "__main__": 
	main(sys.argv[1:])
