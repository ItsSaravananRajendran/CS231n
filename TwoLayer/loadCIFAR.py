import cPickle
import numpy as np

def load(name):
	dict = {}
	dictLabel = {}


	with open(name,"r") as input:
	    dict = cPickle.load(input)



	x= dict['data']
	orLabel= dict['labels']

	return np.array(x),np.array(orLabel)

