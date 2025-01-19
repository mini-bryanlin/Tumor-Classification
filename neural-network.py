import numpy as np
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input):
        pass
    def backward(self, output_gradient, learning_rate):
        pass

class Dense(Layer):
    def __init__(self,input_size,output_size):
        self.weights = np.random.rand(output_size,input_size)
        self.bias = np.random.rand(output_size,1)
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights,self.input) + self.bias
    def backward(self, output_gradient, learning_rate):
        weights_gardient = np.dot(output_gradient,self.input.T)
        self.weights -= learning_rate*weights_gardient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)

class Activation(Layer):
    def __init__(self,activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient,self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1- np.tanh(x) ** 2
        super().__init__(tanh,tanh_prime)
class MSE(Activation):
    def __init(self):
        mse = lambda x: 