import numpy as np
import logic_generators
import pdb
import math

def logistic(a):
	#pdb.set_trace()
	return np.array([(1 / (1+math.exp(-item))) for item in a])

def identity(a):
	return a

def tanh(a):
	return np.tanh(a)


class NeuralNetwork():
	def __init__(self,hidden_layer_sizes = (2,), activation = 'logistic', max_iter = 200, shuffle = True,
		tol = 1e-4, learning_rate_init = 0.5, power_t = 0.5, verbose = False, momentum = 0.9,
		early_stopping = False, validation_fraction = 0.1):

		self.loss_ = 0
		self.n_outputs_ = 1

		# we don't know how many inputs there will be so we can't say how many weights will be needed,
		# but we can start the list anyways 
		self.coefs_ = [None] 
		# randomly initiate the coeficients to be used for the outputs of each of the hidden layers
		for i in range(len(hidden_layer_sizes)):
			if i == len(hidden_layer_sizes) - 1:
				# this is the output layer, and we assume at this point that the output is just one value
				# self.coefs_.append(np.random.random(size=(self.n_outputs_,hidden_layer_sizes[i])) * 0.01)
				self.coefs_.append(np.ones((self.n_outputs_,hidden_layer_sizes[i])) * 0.01)
			else:
				# self.coefs_.append(np.random.random(size=(hidden_layer_sizes[i+1],hidden_layer_sizes[i])) * 0.01)
				self.coefs_.append(np.ones((hidden_layer_sizes[i+1],hidden_layer_sizes[i])) * 0.01)

		self.intercepts_ = [np.ones(size) for size in list(hidden_layer_sizes)]
		self.intercepts_.append(np.ones(1))
		self.n_iter_ = 0
		self.n_layers_ = len(hidden_layer_sizes) + 2
		if activation == 'logistic':
			self.activation_ = logistic
		elif activation == 'identity':
			self.activation_ = identity
		elif activation == 'tanh':
			self.activation_ == tanh

		self.out_activation_ = self.activation_
		self.max_iter = max_iter
		self.shuffle = shuffle
		self.tol = tol
		self.learning_rate_init = learning_rate_init
		self.power_t = power_t
		self.verbose = verbose
		self.momentum = momentum
		self.early_stopping = early_stopping
		self.validation_fraction = validation_fraction
		self.hidden_layer_sizes = list(hidden_layer_sizes)


		# 

	def fit(self,X,y):
		#pdb.set_trace()

		input_size = X.shape[1]

		# look at the number of features and randomize the corresponding coefficients
		# self.coefs_[0] = np.random.random((self.hidden_layer_sizes[0],input_size))*0.01
		self.coefs_[0] = np.ones((self.hidden_layer_sizes[0],input_size))*0.01

		deck = np.column_stack((X,y))
		if self.shuffle:
			# combine the inputs and outputs into one array
			np.random.shuffle(deck)

		# split into validation and training sets
		validation, train = np.vsplit(deck,np.array([int(deck.shape[0] * self.validation_fraction)])) 

		X_val = validation[:,:-1]
		y_val = validation[:,-1]
		X_train = train[:,:-1]
		y_train = train[:,-1]

		# iterate many times over the same train data
		for iteration in range(self.max_iter):
			# train using each of the training data 
			for i in range(len(y_train)):
				o, layer_hs = self.feedforward_detailed(X_train[i])

				# with a final output and the outputs of each of the layers, we can alter the weights using back
				# propogation

				self.backpropagate(o,layer_hs,X_train[i],y_train[i])


			# check the training error after this training iteration
			train_error = self.check_error(X_train,y_train)
			# check the validation error after this training iteration
			validation_error = self.check_error(X_val,y_val)

			print "Round {0}:\nTraining error: {1}\nValidation error: {2}\n".format(iteration,
				train_error,validation_error)

			self.learning_rate_init /= 2



	def feedforward_detailed(self,x):
		#pdb.set_trace()
		l_input = x
		l_outputs = []

		# move through each of the layers
		for i in range(len(self.hidden_layer_sizes)):
			if i > 0:
				# the outputs of the previous layer are the inputs to the next layer
				l_input = l_outputs[-1]

			# take the dot product of the inputs with the weights corresponding to each of the output nodes
			# for example for the input layer with two input features and two hidden nodes in the next layer,
			# this is effectively doing [[[w11,w12],[w21,w22]] dot [x1,x2] + bias1]
			l_ax = self.intercepts_[i] + np.dot(self.coefs_[i],l_input)

			# store the outputs of the activation function from the previous round (h(a(x)))
			# for use in backpropogation
			l_outputs.append(self.activation_(l_ax))

		# at this point we have stored in l_outputs the h(a(x)) for all the hidden layers

		# now we have the input to the output layer, so obtain the input to the output activation
		# function via the same process as with the hidden layers
		if self.n_outputs_ == 1:
			a = self.intercepts_[-1] + np.dot(self.coefs_[-1][0],l_outputs[-1])
			output = self.out_activation_(a)[0]
		else:
			a = self.intercepts_[-1] + np.dot(self.coefs_[-1],l_outputs[-1])
			output = self.out_activation_(a)

		return [output,l_outputs]

	def feedforward(self,x):
		#pdb.set_trace()
		l_input = x

		# move through each of the layers
		for i in range(len(self.hidden_layer_sizes)):
			if i > 0:
				# run the outputs of the previous round (a(x)) through the activation function
				l_input = self.activation_(l_ax)

			# take the dot product of the inputs with the weights corresponding to each of the output nodes
			# for example for the input layer with two input features and two hidden nodes in the next layer,
			# this is effectively doing [[[w11,w12],[w21,w22]] dot [x1,x2] + bias1]
			l_ax = np.dot(self.coefs_[i],l_input) + self.intercepts_[i]

		hx = self.activation_(l_ax)

		if self.n_outputs_ == 1:
			a = self.intercepts_[-1] + np.dot(self.coefs_[-1][0],hx)
			# now we have the input to the output layer, so run it though the output activation functio
			return self.out_activation_(a)[0]
		else:
			a = self.intercepts_[-1] + np.dot(self.coefs_[-1],hx)
			return self.out_activation_(a)

	def backpropagate(self, output,layer_os, x,y):

		#pdb.set_trace()
		# get the delta value which relates the output to the prediction based on the current weights
		delta = (y-output)*output*(1-output)

		flat_coefs = np.reshape(self.coefs_[0],np.prod(self.coefs_[0].shape))
		repeated_next_layer_output = np.repeat(layer_os[0],self.coefs_[0].shape[0])
		repeated_next_layer_weights = np.repeat(self.coefs_[1],self.coefs_[0].shape[0])
		tiled_inputs = np.tile(x,self.coefs_[0].shape[0])

		layer_coefs = flat_coefs \
						- self.learning_rate_init*-delta* \
						np.multiply(np.multiply(repeated_next_layer_output,1-repeated_next_layer_output), 
						np.multiply(repeated_next_layer_weights, 
						tiled_inputs))

		self.coefs_[0] = np.reshape(layer_coefs, self.coefs_[0].shape)

		# start with the input layer weights
		# W1 = self.coefs_[0] - self.learning_rate_init*-delta*h1*(1-h1)*x
		# W2 = self.coefs_[0] - self.learning_rate_init*-delta*h2*(1-h2)*x

		self.coefs_[1] = self.coefs_[1] - self.learning_rate_init*-delta*layer_os[0]

	def check_error(self,X,y):

		error = 0.

		for i in range(len(y)):
			output = self.feedforward(X[i])
			error += math.pow(output-y[i],2)

		return (error / len(y))


	def predict(self,X):
		pass


if __name__ == "__main__":
	lg = logic_generators.LogicGenerator(False, 0)
	X_train, y = lg.XOR(2000)

	nn = NeuralNetwork()

	nn.fit(X_train,y)

	#pdb.set_trace()