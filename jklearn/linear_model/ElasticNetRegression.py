import numpy as np
import pandas as pd

data = pd.read_csv('data/data.csv')

X = np.c_[np.ones(len(data)), data['bedrooms'], data['bathrooms']]
y = np.array(data['price'])

m, n = X.shape
theta = np.zeros(n)

alpha = 0.01
epochs = 5000

lambda1 = 0.1   # Lasso
lambda2 = 0.1   # Ridge

for i in range(epochs):

    y_pred = X @ theta
    error = y_pred - y

    gradient = (1/m) * (X.T @ error)

    l1 = lambda1 * np.sign(theta)
    l2 = lambda2 * theta
    l1[0] = 0
    l2[0] = 0

    theta = theta - alpha * (gradient + l1 + l2)

print(theta)



################## Coordinate Descent #######################

lambda_ = 10

for _ in range(epochs):
    for j in range(n):

        # skip bias (no regularization)
        if j == 0:
            theta[j] = np.sum(y - (X @ theta) + theta[j]*X[:, j]) / np.sum(X[:, j]**2)
            continue
        y_pred = X @ theta
        r_j = y - y_pred + theta[j]*X[:, j]

        rho = np.sum(X[:, j] * r_j)
        z = np.sum(X[:, j]**2)

        if rho > lambda_:
            theta[j] = (rho - lambda_) / z
        elif rho < -lambda_:
            theta[j] = (rho + lambda_) / z
        else:
            theta[j] = 0

print(theta)