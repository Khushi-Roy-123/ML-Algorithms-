import numpy as np

#single feature linear regression

######## Solving Normal Equation #############

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) #number of hours studied
y = np.array([35, 42, 47, 53, 55, 64, 68, 75, 82, 90]) #corresponding scores

m = len(x)

matrix_x = np.c_[np.ones(x.shape[0]), x] #adding a column of ones for the intercept term


xtx = matrix_x.T @ matrix_x
inv = np.linalg.inv(xtx)

xty = matrix_x.T @ y

theta = inv @ xty

print(theta)



########## Gradient Decent ############

b0 = 1
b1 = 1
epochs = 100000
alpha = 0.05

for i in range(epochs):
    ypred = b0 + b1*x

    dbo = (1/m) * sum(ypred-y)
    db1 = (1/m) * sum((ypred-y)*x)

    b0 = b0-alpha*dbo
    b1 = b1-alpha*db1


# Multi feature linear Regression

##################### solving Normal Equation ###################

import pandas as pd

data = pd.read_csv('data/data.csv')
X_features = np.array([np.ones(data['bedrooms'].shape[0]), np.array(data['bedrooms'][:]), np.array(data['bathrooms'][:])]).T

y = np.array(data['price'])
n = len(y)

xtx = np.linalg.inv((X_features.T @ X_features))
xty = X_features.T @ y

theta = xtx @ xty
print(theta)


################# Gradient Descent ####################
b0 = 1
b1 = 1
b2 = 1
epochs = 5000
alpha = 0.05

for i in range(epochs):
    yprediction = b0 + b1*np.array(data['bedrooms'])+ b2*np.array(data['bathrooms'])

    db0 = (1/n) * sum(yprediction - y)
    db1 = (1/n) * sum((yprediction - y) * np.array(data['bedrooms']))
    db2 = (1/n) * sum((yprediction - y) * np.array(data['bathrooms']))

    b0 = b0 - alpha * db0
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2

print(b0, b1, b2)
 