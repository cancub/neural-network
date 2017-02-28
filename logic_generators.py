import numpy as np
import random
from datetime import datetime
import pdb

class LogicGenerator():
	"""
	example: 

	range is given to be 0.2 during initialization

	LogicGenerator.NAND() is called with a count for the number of
	output rows to generate for a NAND

	1 or 0 is selected randomly for each input and
	a random value is generated for both within 0.2 of their values
	but within the range [0,1]

	x1 = 1 -> 0.8943
	x2 = 0 -> 0.1231

	the expected output is generated for these values
	y = 1 NAND 0 = 1

	what is returned is a 2-tuple of numpy matrix (array of arrays) as the first entry
	with the first column = x1 and second column = x2 
	and an array of outputs for these columns as the second entry
	[array([[x1,x2],[x1,x2]]),array([y,y])]
	in this istance:
	[[0.8943,0.1231], [1.]]

	and this can then be fed into a perceptron or a neural network
	"""

	def __init__(self, randomized = True, deviation = 0.25):
		if deviation >= 0.5:
			raise ValueError('Logic values cannot be further than 0.5 from their expected value')

		self.deviation = deviation
		self.randomized = randomized

		random.seed(datetime.now())

	def get_random_signal(self, x):
		x = x.astype(float)
		for i in range(len(x)):
			if x[i]:
				# get the randomized "1"
				np.put(x,[i], [1 - random.random() * self.deviation])
			else:
				# get the randomized "0"
				np.put(x,[i], [random.random() * self.deviation])

		return x

	def get_deterministic_signal(self, x):
		x = x.astype(float)
		for i in range(len(x)):
			if x[i]:
				# get the deviated "1"
				np.put(x,[i], [1 - self.deviation])
			else:
				# get the deviated "0"
				np.put(x,[i], [self.deviation])

		return x


	def OR(self, count = 1):
		x1 = np.random.randint(2,size = count)
		x2 = np.random.randint(2,size = count)

		y = np.logical_or(x1,x2).astype(int)

		if self.randomized:
			func = self.get_random_signal
		else:
			func = self.get_deterministic_signal
		

		x1 = np.apply_along_axis(func,0,x1)
		x2 = np.apply_along_axis(func,0,x2)


		return [np.column_stack((x1,x2)), y]

	def NOR(self, count = 1):
		# just call OR and invert the y
		ret = self.OR(count)

		return [ret[0],np.logical_not(ret[1]).astype(int)]

	def AND(self, count = 1):
		x1 = np.random.randint(2,size = count)
		x2 = np.random.randint(2,size = count)

		y = np.logical_and(x1,x2).astype(int)

		if self.randomized:
			func = self.get_random_signal
		else:
			func = self.get_deterministic_signal
		

		x1 = np.apply_along_axis(func,0,x1)
		x2 = np.apply_along_axis(func,0,x2)

		return [np.column_stack((x1,x2)), y]

	def NAND(self, count = 1):
		# just call AND and invert the y

		ret = self.AND(count)

		return [ret[0],np.logical_not(ret[1]).astype(int)]

	def XOR(self, count = 1):
		# XOR = (x1 OR x2) AND NOT (x1 AND x2)
		# pdb.set_trace()

		x1 = np.random.randint(2,size = count)
		x2 = np.random.randint(2,size = count)

		y = np.logical_and(np.logical_or(x1,x2),np.logical_not(np.logical_and(x1,x2))).astype(int)

		if self.randomized:
			func = self.get_random_signal
		else:
			func = self.get_deterministic_signal
		

		x1 = np.apply_along_axis(func,0,x1)
		x2 = np.apply_along_axis(func,0,x2)

		return [np.column_stack((x1,x2)), y]


if __name__ == "__main__":
	lg = LogicGenerator(False, 0.3)
	print "OR: \n{}\n".format(lg.OR(3))
	print "AND: \n{}\n".format(lg.AND(3))
	print "NOR: \n{}\n".format(lg.NOR(3))
	print "NAND: \n{}\n".format(lg.NAND(3))
	print "XOR: \n{}\n".format(lg.XOR(3))