from loadCIFAR import *
import numpy as np 

class NeuralNet:

	def __init__(self):
		self.W1 = np.random.randn(3073,100) * np.sqrt(2.0/3073)
		self.W2 = np.random.randn(101,10) * np.sqrt(2.0/101)
		self.lam = 0.05
		self.delta = 1
		self.step = -1*(10**-8)
		self.mean = np.zeros((3072,),dtype='uint8')
		for I in range(1,6):
			image,_ = load('../Dataset/data_batch_'+str(I))
			mean =np.mean(image,axis=0)
			self.mean = np.add(self.mean,mean)
		self.mean /= 5.0


	def dataCentering(self,trainImage):
		data = np.matrix(trainImage)
		data = np.subtract(data,self.mean)
		one = np.ones((100,1))
		data = np.concatenate((data,one), axis=1)
		return data


	def forwardPass(self,trainImage):
		#y1 = x.w1 + b1
		self.y1 = trainImage.dot(self.W1)

		#Relu
		self.y2 = self.y1
		self.y2[self.y1<0] = 0
		self.y2 = np.concatenate((self.y2,np.ones((100,1))),axis=1)
		#print self.y2.shape
		#y3 = y2.w2 + b2
		y3 = self.y2.dot(self.W2)

		return y3
		
	def backPropagation(self,score,crctLabel,trainImage):
		Sy =(np.array([[score[I,crctLabel[I]]] for I in xrange(len(crctLabel))]))
		LossVector = np.subtract(score,Sy)
		LossVector += self.delta

		#clamp negative value to zero
		LossVector[LossVector < 0] = 0
		countZero =307300 -  np.count_nonzero(LossVector)
		DL_DY3 = LossVector
		#print "DL_DY3 = "+str(DL_DY3.shape)

		#Y != J
		LossVector[LossVector == self.delta] = 0
		DL_DY3[DL_DY3==self.delta] = -1

		regW2Loss = self.lam*self.L2(self.W2)
		loss = np.sum(LossVector)/len(LossVector)
		DL_DY3[DL_DY3>0] = 1

		DL_DW2 = (self.y2.T.dot(DL_DY3)) + 0.5*self.lam*self.W2
		#print "DL_DW2 = "+str(DL_DW2.shape)
		#print "W2 = "+str(self.W2.shape)

		DL_DY2 = DL_DY3.dot(self.W2.T)
		DL_DY2 = np.delete(DL_DY2,100,1)

		#print "DL_DY2 ="+str(DL_DY2.shape)
		DL_DY1 = DL_DY2

		DL_DY1[self.y1<0] = 0

		regW1Loss = self.lam*self.L2(self.W1)

		DL_DW1 = trainImage.T.dot(DL_DY1) + 0.5*self.lam*self.W1

		self.W2 += self.step*DL_DW2
		self.W1 += self.step*3*DL_DW1

		return loss,regW2Loss,regW1Loss,countZero




	def L2(self,W):
		w = np.square(W)
		return np.sum(w)



	def load(self,W1,W2):
		self.W1 = np.load(W1)
		self.W2 = np.load(W2)


	def save(self,W1,W2):
		np.save(W1,self.W1)
		np.save(W2,self.W2)



	def validate(self,score,crctLabel):
        	count = 0
        	scoreList = np.argmax(score,axis=1)
        	for I in range(100):
            		if scoreList[I] == crctLabel[I]:
                		count+= 1
        	return count/100.0


