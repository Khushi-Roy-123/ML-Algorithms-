import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)

        for _ in range(self.epochs):
            y_pred = self.sigmoid(X @ self.theta)
            gradient = (1/m) * (X.T @ (y_pred - y))
            self.theta -= self.lr * gradient

    def predict(self, X):
        probs = self.sigmoid(X @ self.theta)
        return (probs >= 0.5).astype(int)