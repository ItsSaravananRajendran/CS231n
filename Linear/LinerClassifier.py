import numpy as np
from loadCIFAR import *

class LinearClassifier:
    
    def __init__(self):
        self.W = 0.001* np.random.randn(3073,10)
        self.delta = 1
        self.lam = 0.1
        self.step = -1*(10**-10)
        self.mean = np.zeros((3072,),dtype='uint8')
        for I in range(1,6):
            image,_ = load('../Dataset/data_batch_'+str(I))
            mean =np.mean(image,axis=0)
            self.mean = np.add(self.mean,mean)
        self.mean /= 5.0
        np.save('mean.npy',self.mean)

    def load(self,name):
        self.W = np.load(name)

    def dataCentering(self,trainImg):
        data = np.matrix(trainImg)
        data = np.subtract(data,self.mean)
        one = np.ones((100,1))
        data = np.concatenate((data,one), axis=1)
        return data

    def score(self,trainImg):     
        scoreVector = trainImg.dot(self.W)
        return scoreVector

    def SVMloss(self,scoreVector,trainLabel,trainImg):
        Sy =(np.array([[scoreVector[I,trainLabel[I]]] for I in xrange(len(trainLabel))]))
        lossVector = np.subtract(scoreVector,Sy)
        lossVector += self.delta
        lossVector[lossVector<0] = 0
        copyLossVector = lossVector
        lossVector[lossVector == self.delta] = 0
        copyLossVector[copyLossVector==self.delta] = -1
        regularization = self.L2()
        regLoss = self.lam*regularization
        loss = np.sum(lossVector)/len(lossVector) 
        copyLossVector[copyLossVector>0] = 1
        gradient = trainImg.T.dot(copyLossVector)
        self.W += self.step*gradient
        return loss,regLoss

    def L2(self):
        w = np.square(self.W)
        return np.sum(w)

    def singleImage(self,image):
        score = self.W.T.dot(image.T)
        print np.argmax(score)

    def save(self,name="W.npy"):
        np.save(name,self.W)

    def validate(self,score,crctLabel):
        count = 0
        scoreList = np.argmax(score,axis=1)
        for I in range(100):
            if scoreList[I] == crctLabel[I]:
                count+= 1
        return count/100.0


