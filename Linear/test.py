import numpy as np
from LinerClassifier import *
from loadCIFAR import *
import sys 



image , imageLabel = load('../Dataset/test_batch')
Iteration = 0
cifar = LinearClassifier()
cifar.load(sys.argv[1])
loss = 50
valid = []
for I in range(100):
	rangeImage = I*100
	trainImage = image[rangeImage:rangeImage+100]
	crctLabel = imageLabel[rangeImage:rangeImage+100]
	trainImage = cifar.dataCentering(trainImage)
	score = cifar.score(trainImage)
	valid.append(cifar.validate(score,crctLabel))
print np.mean(valid)


