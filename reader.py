import os
import numpy as np

def readdata(path):
	f = open(path, 'r')
	names = []
	labels = []
	data = []
	max_value = -1000
	for line in f:
		classes = [0.0]*4
		splits = line.split()
		vector = [float(x) for x in splits[2:]]
		names.append(splits[0])
		if max_value < max(vector):
			max_value = max(vector)

		data.append(vector)
		classes[int(splits[1])/90] = 1
		labels.append(classes)

	# # print max_value
	# max_value = 1
	return np.array(data) / max_value, np.array(labels), np.array(names)
