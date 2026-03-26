import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('data/data.csv')

X_train=data[['bedrooms','bathrooms']].values 
y_train=data['city'].values

m,n=X_train.shape

plt.scatter(X_train[y_train == 0,0],X_train[y_train ==0,1],color ='tab:blue',label ='class 0', s=20)
plt.scatter(X_train[y_train == 1,0],X_train[y_train ==1,1],color ='tab:red',label ='class 1', s=20)
plt.xlabel('bedrooms')
plt.ylabel('bathrooms')
plt.legend()
plt.show()