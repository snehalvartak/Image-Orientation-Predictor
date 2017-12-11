
import numpy as np
from scipy.spatial import distance
import operator
from sys import stdout
from collections import Counter
import pandas as pd

def readData(path,data):
	vectors={}
	f=open(path,'r')
	for line in f:
		line=line.strip()
		if data=='train':
			identity=line.split(' ')[0]+"_"+line.split(' ')[1]
		else:
			identity=line.split(' ')[0]
		vectors[identity]=np.array(map(int,line.split(' ')[1:]))
	return vectors

# Use Euclidean distance to get the most common <--> nearest neighbor
def getNearestNeighbor(trainVectors,vectors):
	k = 1
	dist = {key:[distance.euclidean(trainVectors[key][1:],vectors),trainVectors[key][0]] for key in trainVectors}
	topk = map(list,zip(*sorted(dist.iteritems(),key=operator.itemgetter(1),reverse=False)[:k])[1])
	topk = np.asarray(zip(*topk)[1])
	return Counter(topk).most_common(1)[0][0]

def dumpPredictions(dump):
	with open('nearest_output.txt','wt') as f:
		for lines in dump:
			f.write(' '.join(str(line) for line in lines))
			f.write('\n')

def calcNeighbours(trainVectors,testVectors):
	actualValues=[]
	predictedValues=[]
	dump=[]
	
	n=len(testVectors)
	i=1
	
	for key in testVectors.keys():
		stdout.write("\rCalculating nearest neighbor for %d/%d" % (i,n))
		stdout.flush()
		actualValues.append(testVectors[key][0])
		predicted=getNearestNeighbor(trainVectors,testVectors[key][1:])
		predictedValues.append(predicted)
		dump.append([key,predicted])
		i+=1
	dumpPredictions(dump)
	
	actual = pd.Series(actualValues, name='Actual')
	predicted = pd.Series(predictedValues, name='Predicted')
	df_confusion = pd.crosstab(actual,predicted)
	confMatrix=df_confusion.as_matrix()
	print ("\n")
	print ("Accuracy : "+str(sum(np.diag(confMatrix)) * 100 / float(np.sum(confMatrix))))

	print ('\n')
	print (df_confusion)