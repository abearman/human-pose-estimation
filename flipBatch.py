import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

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
