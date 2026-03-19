import numpy as np
import pandas as pd

data = pd.read_csv('data/data.csv')

x1 = np.array(data['bedrooms'])
x2 = np.array(data['bathrooms'])
y = np.array(data['price'])

n = len(y)
################# Normal Equation #############

X = np.array([np.ones(n),x1,x2,x1**2,x2**2,x1 * x2]).T
theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
print("Normal Equation:", theta)

################## Gradient Descent #################
b0 = b1 = b2 = b3 = b4 = b5 = 1

alpha = 0.00001
epochs = 5000

for i in range(epochs):
    ypred = (b0 + b1*x1 + b2*x2 + b3*(x1**2) + b4*(x2**2) + b5*(x1*x2))

    error = ypred - y

    db0 = (1/n) * sum(error)
    db1 = (1/n) * sum(error * x1)
    db2 = (1/n) * sum(error * x2)
    db3 = (1/n) * sum(error * (x1**2))
    db4 = (1/n) * sum(error * (x2**2))
    db5 = (1/n) * sum(error * (x1*x2))

    b0 -= alpha * db0
    b1 -= alpha * db1
    b2 -= alpha * db2
    b3 -= alpha * db3
    b4 -= alpha * db4
    b5 -= alpha * db5

print("Gradient Descent:", b0, b1, b2, b3, b4, b5)