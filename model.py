from neuralnetwork import Layer
from neuralnetwork import Dense, mse, mse_prime, predict, train, Tanh
import numpy as np
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4,2,1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


layer1 = Dense(2,3,"one")
act1 = Tanh("act1")
layer2 = Dense(3,1,"two")
act2 = Tanh("act2")
network = [
    layer1,
    act1,
    layer2,
    act2
    
]

train(network,mse,mse_prime,X,Y, epochs=10000)
print(predict(network, X[1]))


# zeroes = np.zeros((2,3))
# ones = np.ones((3,1))
# np.dot(zeroes,ones)

