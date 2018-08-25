import numpy as np 

class Network:
	def __init__(self,sizes) :
		self.num_layers = len(sizes)
		self.sizes = sizes 
		self.bias = [np.random.rand(y,1) for y in self.sizes[0:]]
		self.weights = [np.random.rand(y,x)
										for y,x in zip(sizes[:-1],sizes[1:])]

	
	def feedforward (self,a):
		"""
			Return the output of the network if 'a' is a input 
		"""
		for b,w in zip(self.bias,self.weights) :
			a = sigmoid(np.dot(w,a)+b)
		return a 

	def SGD (self,Trainning_Data,epochs,mini_batch_size,eta,test_data=None):
		"""
			Train the neural network using mini-batch stochastic
			gradient descent. The "training_data" is a list of tuples
			"(x, y)" representing the training inputs and the desired
			outputs. The other non-optional parameters are
			self-explanatory. If "test_data" is provided then the
			network will be evaluated against the test data after each
			epoch, and partial progress printed out. This is useful for
			tracking progress, but slows things down substantially.
		"""
		if test_data :
			n_test = len(data) 
		n = len(Trainning_Data)
		for j in range(epochs) :
			random.shuffle(Trainning_Data)
			mini_batchs = [Trainning_Data[k:k+mini_batch_size]
											for k in range(0,n,mini_batch_size)]
			for mini_batch in mini_batchs :
				self.update_mini_batch(mini_batch,eta) 
			if test_data:
				print ("epoch {0}: {1}/{2}".format(j,self.evaluate(test_data),n_test))
			else :
				print ("epoch {0} complete".format(j))

	def update_mini_batch (self,mini_batch,eta) :
		"""
			Update the network's weights and biases by applying
			gradient descent using backpropagation to a single mini batch.
			The "mini_batch" is a list of tuples "(x, y)", and "eta"
			is the learning rate.
		"""

		nabla_b = [np.zeros(b.shape) for b in self.bias]
		nabla_w = [np.zeros(w.shape for w in  self.weights)]

		for x ,y in mini_batch : 
			delta_nabla_b ,delta_nabla_w = self.backprop(x,y) 
			nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
			nabla_b = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
		self.weights = [w - (eta/len(mini_batch))*nw 
											for w,nw in zip (self.weights,nabla_w)]

		self.bias = [b - (eta/len(mini_batch))*nb 
											for b,nb in zip (self.bias,nabla_b)]
		
	def backprop (self,x,y) :
		nabla_b = [np.zeros(b.shape) for b in self.bias]
		nabla_w = [np.zeros(w.shape for w in  self.weights)]
		activation = x 
		activations = [x] 
		zs = []
		for b,w in zip(self.bias,self.weights) :
			z = np.dot(w,activation) +b 
			zs.append(z) 
			activation = sigmoid(z) 	
			activations.append(activation)
		delta = self.cost_derivative (activations[-1],y) * sigmoid_prime(zs[-1])

		nabla_b[-1] =delta 
		nabla_w[-1] = np.dot(delta, activations[-2].T)

		for l in range(2,self.num_layers) :
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].T,delta ) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta,activations[-l-1].T)
		return (nabla_b,nabla_w)

	def evaluate(self,test_data) :
		test_result = [(np.argmax(self.feedforward(x)),y) 
										for (x,y) in  test_data ]
		return sum(int(x==y) for (x,y) in test_result)

	def cost_derivative(self,output_activations,y) :
		return output_activations - y 

def sigmoid (z):
		return 1.0/(1.0+np.exp(-z))

def sigmoid_prime (z) :
		return sigmoid(z)*(1-sigmoid(z))