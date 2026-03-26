import numpy as np
from collections import Counter
from jklearn.utils.distance import euclidean_distance

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_idx]
        return Counter(k_labels).most_common(1)[0][0]