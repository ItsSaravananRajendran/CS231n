import numpy as np
from NeuralNet import *
from loadCIFAR import *
from random import *


fileLabel = [1,2,3,4,5]
image , imageLabel = load('../Dataset/data_batch_4')
Iteration = 0
cifar = NeuralNet()
loss = 50
validImageFull,validImageLabelFull = load('../Dataset/test_batch')
trainImage = image[100:200]
crctLabel = imageLabel[100:200]
trainImage = cifar.dataCentering(trainImage)
loss = 5
minLoss = 50
count = 0
#cifar.load('W1.npy','W2.npy')
prevLoss=0
mapping = np.random.randint(10000, size=100)
while loss>.5:
	mapping = np.random.randint(10000, size=100)
	trainImage = image[mapping]
	crctLabel = imageLabel[mapping]
	trainImage = cifar.dataCentering(trainImage)
	score = cifar.forwardPass(trainImage)
	prevLoss = loss
	loss, r1,r2 ,zeroCount= cifar.backPropagation(score,crctLabel,trainImage)
	if count%100 == 0:
		validImage = validImageFull[0:100]
		validImageLabel = validImageLabelFull[0:100]
		validImage = cifar.dataCentering(validImage)
		score = cifar.forwardPass(validImage)
		valid = cifar.validate(score,validImageLabel)
		print loss,r1,r2,cifar.step,zeroCount,np.mean(valid)
		image,imageLabel = load('../Dataset/data_batch_'+str(sample(fileLabel,  1)[0]))
		mapping = np.random.randint(10000, size=100)
		trainImage = image[mapping]
		crctLabel = imageLabel[mapping]
		trainImage = cifar.dataCentering(trainImage)
		if minLoss > loss:
			minLoss = loss
			cifar.save('W1.npy','W2.npy')
	count +=1 
	
