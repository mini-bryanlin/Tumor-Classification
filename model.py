from neuralnetwork import Layer
from neuralnetwork import Dense, mse, mse_prime, predict, train
import numpy as np
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4,2,1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


