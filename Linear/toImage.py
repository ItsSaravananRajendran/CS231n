import cv2
import cPickle
import numpy as np

dict = None


with open("./data_batch_1","r") as input:
    dict = cPickle.load(input)

cv2.namedWindow("fram",cv2.WINDOW_NORMAL )

#raw = dict.T[1] + 127
raw = dict['data'][20]
img = np.transpose(np.reshape(raw,(3, 32,32)), (1,2,0))


print img
cv2.imshow("fram",img)
k = cv2.waitKey(0) & 0xFF 

