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
        self.n_outputs_ = 0

        # we don't know how many inputs there will be so we can't say how many weights will be needed,
        # but we can start the list anyways 
        self.coefs_ = [np.array([[0.15,0.2],[0.25,0.3]]),np.array([0.4,0.45])]
        self.coefs_ = [np.array([[0.15,0.2],[0.25,0.3]]),np.array([[0.4,0.45],[.5,0.55]])]

        self.intercepts_ = np.array([0.35,0.60])
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
        # pdb.set_trace()
        pdb.set_trace()
        trial = True

        if not trial:
            input_size = X.shape[1]

            # look at the number of features and randomize the corresponding coefficients
            # self.coefs_[0] = np.random.random((self.hidden_layer_sizes[0],input_size))*0.01
            # self.coefs_[0] = np.ones((self.hidden_layer_sizes[0],input_size))*0.01

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

            if len(y_train.shape) > 1:
                num_examples = y_train.shape[0]
                # self.n_outputs_ = y_train.shape[1]
                self.n_outputs_ = y_train.shape[1]

        else:
            X_train = X
            y_train = y
            num_examples = 1
            self.n_outputs_ = len(y_train)
            

        # iterate many times over the same train data
        for iteration in range(self.max_iter):
            # print iteration
            # train using each of the training data
            # pdb.set_trace()
            for i in range(num_examples):
                if trial:
                    o, layer_hs = self.feedforward(X_train)
                    y_example = y_train
                else:
                    o, layer_hs = self.feedforward(X_train[i])
                    y_example = [y_train[i]]

                self.loss_ = 0

                for j in range (len(o)):
                    self.loss_ += math.pow(y_example[j] - o[j],2)/2 

                print self.loss_

                # with a final output and the outputs of each of the layers, we can alter the weights using back
                # propogation

                if trial:
                    self.coefs_ =  self.backpropagate(o,layer_hs,X_train,y_example)
                else:
                    self.coefs_ =  self.backpropagate(o,layer_hs,X_train[i],y_example)


            # check the training error after this training iteration
            # train_error = self.check_error(X_train,y_train)
            # train_error = self.check_error(X,y)
            # check the validation error after this training iteration
            # validation_error = self.check_error(X_val,y_val)
            # validation_error = self.check_error(X,y)

            # print "Round {0}:\nTraining error: {1}\nValidation error: {2}\n".format(iteration,
            #   train_error,validation_error)

            # self.learning_rate_init /= 2



    def feedforward(self,x,layer_inputs = True):
        # pdb.set_trace()
        # here's where any sort of modications will be made to the initial input
        # to get the input to the first hidden layer
        l_inputs = [x]

        # move through each of the layers
        for i in range(len(self.hidden_layer_sizes)):

            # take the dot product of the inputs with the weights corresponding to each of the output nodes
            # for example for the input layer with two input features and two hidden nodes in the next layer,
            # this is effectively doing [[[w11,w12],[w21,w22]] dot [x1,x2] + bias1]
            # where w12 is the weight on the second input to the first node in the next layer
            l_ax = self.intercepts_[i] + np.dot(self.coefs_[i],l_inputs[-1])

            # store the outputs of the activation function from the previous round (h(a(x)))
            # for use in backpropogation
            l_inputs.append(self.activation_(l_ax))

        # at this point we have stored in l_outputs the h(a(x)) for all the hidden layers

        # now we have the input to the output layer, so obtain the input to the output activation
        # function via the same process as with the hidden layers
        if self.n_outputs_ == 1:
            a = self.intercepts_[-1] + np.dot(self.coefs_[-1],l_inputs[-1])
            output = self.out_activation_(np.array([a]))
        else:
            a = self.intercepts_[-1] + np.dot(self.coefs_[-1],l_inputs[-1])
            output = self.out_activation_(a)

        if layer_inputs:
            return [output,l_inputs]
        else:
            return output

    # def feedforward(self,x):
    #   #pdb.set_trace()
    #   l_input = x

    #   # move through each of the layers
    #   for i in range(len(self.hidden_layer_sizes) + 1):
    #       if i > 0:
    #           # run the outputs of the previous round (a(x)) through the activation function
    #           l_input = self.activation_(l_ax)

    #       # take the dot product of the inputs with the weights corresponding to each of the output nodes
    #       # for example for the input layer with two input features and two hidden nodes in the next layer,
    #       # this is effectively doing [[[w11,w12],[w21,w22]] dot [x1,x2] + bias1]
    #       # where w12 is the weight on the second input to the first node in the next layer
    #       l_ax = np.dot(self.coefs_[i],l_input) + self.intercepts_[i]

    #   hx = self.activation_(l_ax)

    #   if self.n_outputs_ == 1:
    #       a = self.intercepts_[-1] + np.dot(self.coefs_[-1][0],hx)
    #       # now we have the input to the output layer, so run it though the output activation functio
    #       return self.out_activation_(a)[0]
    #   else:
    #       a = self.intercepts_[-1] + np.dot(self.coefs_[-1],hx)
    #       return self.out_activation_(a)

    def backpropagate(self, output,layer_inputs, x,y):

        new_coefs = [None]* len(self.coefs_)

        # pdb.set_trace()
        # get the delta value which relates the output to the prediction based on the current weights
        d_E_total_wrt_output = - (y - output)
        d_outputs_wrt_all_layer_inputs = np.multiply(output,1-output)
        output_deltas = np.multiply(d_E_total_wrt_output,d_outputs_wrt_all_layer_inputs)
        d_E_total_wrt_layer_weights = np.multiply(
                                        np.repeat(output_deltas,len(layer_inputs[-1])),
                                        np.tile(layer_inputs[-1],len(output_deltas))
                                        )

        try:
            flat_coefs = np.reshape(self.coefs_[-1],np.prod(self.coefs_[-1].shape))
        except:
            flat_coefs = np.array(self.coefs_[-1])
        # repeated_next_layer_weights = np.repeat(self.coefs_[1],self.coefs_[0].shape[0])
        # tiled_inputs = np.tile(x,self.coefs_[0].shape[0])

        layer_coefs = flat_coefs - self.learning_rate_init*d_E_total_wrt_layer_weights

        new_coefs[-1] = np.reshape(layer_coefs, self.coefs_[-1].shape)

        # start with the input layer weights
        # W1 = self.coefs_[0] - self.learning_rate_init*-delta*h1*(1-h1)*x
        # W2 = self.coefs_[0] - self.learning_rate_init*-delta*h2*(1-h2)*x

        # get the element-wise product of all the outputs

        flat_coefs = np.reshape(self.coefs_[0],np.prod(self.coefs_[0].shape))
        if len(output_deltas) > 1:
            sum_d_E_total_wrt_layer_output = np.dot(np.transpose(self.coefs_[-1]),output_deltas)
        else:
            sum_d_E_total_wrt_layer_output = np.dot(self.coefs_[-1],np.repeat(output_deltas,len(self.coefs_[-1])))

        d_layer_outputs_wrt_all_layer_inputs = np.multiply(layer_inputs[-1],1-layer_inputs[-1])
        layer_deltas = np.multiply(sum_d_E_total_wrt_layer_output,d_layer_outputs_wrt_all_layer_inputs)

        d_E_total_wrt_layer_weights = np.multiply(
                                        np.repeat(layer_deltas,len(layer_inputs[-2])),
                                        np.tile(layer_inputs[-2],len(layer_deltas))
                                        )


        layer_coefs = flat_coefs - self.learning_rate_init*d_E_total_wrt_layer_weights
        new_coefs[-2] = np.reshape(layer_coefs, self.coefs_[-2].shape)

        return new_coefs

    def check_error(self,X,y):

        error = 0.

        for i in range(len(y)):
            output = self.feedforward(X[i], False)
            error += math.pow(output-y[i],2)

        return (error / len(y))


    def predict(self,X):
        pass


if __name__ == "__main__":
    lg = logic_generators.LogicGenerator(False, 0)
    X_train = np.array([0.05,0.1])
    y = np.array([0.01,0.99])

    # X_train, y = lg.XOR(2000)

    nn = NeuralNetwork(max_iter = 10000)

    pdb.set_trace()
    

    nn.fit(X_train,y)
