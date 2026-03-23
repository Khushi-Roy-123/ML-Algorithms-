import numpy as np

################ One Feature ####################

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([35,42,47,53,55,64,68,75,82,90])

m = len(x)

################# GD ####################
b0, b1 = 1, 1
alpha = 0.01
epochs = 1000
lamda= 0.1

for i in range(epochs):
    y_pred = b0 + b1*x
    error = y_pred - y

    db0 = (1/m) * np.sum(error)  
    db1 = (1/m) * np.sum(error * x) + lamda * np.sign(b1)

    b0 = b0 - alpha * db0
    b1 = b1 - alpha * db1

print(b0, b1)

############## Multiple Feature ###################

import pandas as pd

data = pd.read_csv('data/data.csv')


X = np.c_[np.ones(len(data)), data['bedrooms'], data['bathrooms']]
y = np.array(data['price'])

m, n = X.shape

# theta vector for multiple features
theta = np.ones(n)
alpha = 0.01
epochs = 5000
lamda = 0.1

for i in range(epochs):
    y_pred = X @ theta
    error = y_pred - y

    gradient = (1/m) * (X.T @ error)

    reg = lamda * np.sign(theta)
    reg[0] = 0  
    theta = theta - alpha * (gradient + reg)

print(theta)