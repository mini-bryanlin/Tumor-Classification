from neuralnetwork import Layer
from neuralnetwork import Dense, mse, mse_prime, predict, train, Tanh, bce, bce_prime, Sigmoid, ReLU
import numpy as np
import pandas as pd
from Saving_Weights import save
from loading_weights import load_weights
from load_data import load_data
features =["radius_mean", "perimeter_mean", "area_mean", "compactness_mean","concave points_mean", "concavity_mean","fractal_dimension_se","radius_se","radius_worst","perimeter_worst","area_worst","concave points_worst",]
train_x,train_y,test_x,test_y, length = load_data("~/Tumor-Classification/breast-cancer.csv",features)


X = np.reshape(train_x, (length,12,1))
Y = np.reshape(train_y, (length, 1, 1))
print(Y)

layer1 = Dense(12,24,"one")
act1 = ReLU("act1")
layer2 = Dense(24,16,"two")
act2 = ReLU("act2")
layer3 = Dense(16,8,"three")
act3 = ReLU("act3")
layer4 = Dense(8,1,"four")
finalact = Sigmoid("finalact")
network = [
    layer1,
    act1,
    layer2,
    act2
    ,layer3,
    act3,
    layer4,
    finalact
]
print(predict(network,X[1]))
# (network,alpha,epochs,loss, loss_prime,x,y,prin = True)
train(network,0.1,500,bce,bce_prime,X,Y,True)
# save(network)
print(predict(network,X[1]))


# zeroes = np.ze00ros((2,3))
# ones = np.ones((3,1))
# np.dot(zeroes,ones)

