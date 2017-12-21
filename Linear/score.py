import numpy as np
from LinerClassifier import *
from loadCIFAR import *
from random import *


fileLabel = [1,2,3,4,5]
image , imageLabel = load('../Dataset/data_batch_4')
Iteration = 0
cifar = LinearClassifier()
cifar.load('OverFitting.npy')
loss = 50    
trainImage = image[100:200]
crctLabel = imageLabel[100:200]
trainImage = cifar.dataCentering(trainImage)
minloss = 99999
regLoss = 0

while loss > 0.01 :
	score = cifar.score(trainImage)
	loss,regLoss = cifar.SVMloss(score,crctLabel,trainImage)
	Iteration += 1
	if minloss > loss:
		minloss = loss
		cifar.save('OverFitting.npy')
	print "loss = "+str(loss) +" regLoss = "+ str(regLoss) + " epoch = "+ str(Iteration/100)+" Validation" +str(cifar.validate(score,crctLabel))


print loss,regLoss
