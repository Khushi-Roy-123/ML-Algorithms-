import numpy as np
import pandas as pd

################## SINGLE FEATURE ##########################

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([35,42,47,53,55,64,68,75,82,90])

m = len(x)

# Feature matrix
X = np.c_[np.ones(m), x]

# -------- Normal Equation (Linear) --------
theta_linear = np.linalg.inv(X.T @ X) @ (X.T @ y)
print("Linear Theta (Normal):", theta_linear)

################# Normal Equation ########################
lambda_ = 0.1
I = np.eye(X.shape[1])
I[0,0] = 0  # don't regularize bias

theta_ridge = np.linalg.inv(X.T @ X + lambda_ * I) @ (X.T @ y)
print("Ridge Theta (Normal):", theta_ridge)


#################Gradient Descent #########################
b0, b1 = 1, 1
alpha = 0.05
epochs = 100000

for i in range(epochs):
    y_pred = b0 + b1*x
    error = y_pred - y

    db0 = (1/m) * np.sum(error)                # no regularization
    db1 = (1/m) * np.sum(error * x) + lambda_ * b1

    b0 -= alpha * db0
    b1 -= alpha * db1

print("Ridge (GD):", b0, b1)



################ MULTI-FEATURE ###############


data = pd.read_csv('data/data.csv')

bed = np.array(data['bedrooms'])
bath = np.array(data['bathrooms'])
y = np.array(data['price'])

n = len(y)

# Feature matrix
X_features = np.c_[np.ones(n), bed, bath]

################# Normal Equation ################
theta_linear_multi = np.linalg.inv(X_features.T @ X_features) @ (X_features.T @ y)
print("Linear Theta Multi:", theta_linear_multi)

######################## Normal Equation ###########################
lambda_ = 0.1
I = np.eye(X_features.shape[1])
I[0,0] = 0

theta_ridge_multi = np.linalg.inv(X_features.T @ X_features + lambda_ * I) @ (X_features.T @ y)
print("Ridge Theta Multi:", theta_ridge_multi)


############ Gradient Descent ##################
b0, b1, b2 = 1, 1, 1
alpha = 0.05
epochs = 5000

for i in range(epochs):
    y_pred = b0 + b1*bed + b2*bath
    error = y_pred - y

    db0 = (1/n) * np.sum(error)
    db1 = (1/n) * np.sum(error * bed) + lambda_ * b1
    db2 = (1/n) * np.sum(error * bath) + lambda_ * b2

    b0 -= alpha * db0
    b1 -= alpha * db1
    b2 -= alpha * db2

print( b0, b1, b2)