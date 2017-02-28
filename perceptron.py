import numpy as np
import pdb
import math
from logic_generators import LogicGenerator

def sigmoid(x):
	return 1 / (1+math.exp(-x)) 

class Perceptron():

	def __init__(self,decision_function=sigmoid,eta0=1.0):
		# pass along whatever you please (e.g., sigmoid, linear, tanh) as long as it
		# takes a scalar and produces a scalar
		self.fun = decision_function
		self.eta0 = eta0


	def fit(self,X,y, coef_init = None, iterations = 30):
		# inputs: 	X, n by m numpy array (array of arrays)
		#				n = number of training examples
		#				m = the number of features
		# 			y, length n numpy array of training labels

		# add an extra term for the bias to each training example
		X = np.c_[np.ones(X.shape[0]),X]

		if coef_init:
			self.coef_ = coef_init
		else:
			self.coef_ = np.zeros(X.shape[1])
		
		f_x = np.array([0] * len(y))


		for i in range(iterations):
			# pdb.set_trace()
			# get the predictions for this round
			f_x = np.apply_along_axis(self.activation_function,1,X) 

			# save a copy of the previous weights
			previous_coef_ = np.copy(self.coef_)

			# update the weights
			self.coef_ += np.dot(X.transpose(),(y-f_x)*self.eta0) 

			
			# check for convergence
			if np.array_equal(previous_coef_,self.coef_):
				break


	def activation_function(self,example):
		# pdb.set_trace()
		if self.fun(np.dot(example,self.coef_)) > 0.5:
			return 1
		else:
			return 0

	def predict(self,X):
		X = np.c_[np.ones(X.shape[0]),X]
		return np.dot(X,self.coef_)



if __name__ == "__main__":
	perc = Perceptron(eta0=0.2)
	lg = LogicGenerator(0.3)
	X_train, y = lg.XOR(40)
	# pdb.set_trace()
	x1 = [0,1,0,1]
	x2 = [0,0,1,1]

	X_test = np.array([x1,x2]).transpose()

	perc.fit(X_train,y)
	print perc.coef_
	print perc.predict(X_test)
