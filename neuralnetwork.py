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
def mse(y_hat, y_true):
    return (np.mean(np.power((y_hat- y_true),2)))/2

def mse_prime(y_true,y_pred):
    return (y_pred-y_true)/np.size(y_true)

def predict(network,input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output
def train(network,alpha,epochs,loss, loss_prime,x,y,prin = True):
    for time in range(epochs):
        #calculate loss
        error = 0 
        for x, y in zip(x,y):
            prediction = predict(network,x)
            error += loss(y,prediction)

            #gradient descent
            gradient = loss_prime(y, prediction)
            for layer in network[::-1]:
                gradient = layer.backwards(gradient,alpha)
            
        error /= len(x)

        if prin:
            print(f"{e + 1}/{epochs}, error={error}")

        
            

