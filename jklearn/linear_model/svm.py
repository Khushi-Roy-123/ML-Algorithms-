import numpy as np


class SVM:
    def __init__(self, lr=0.001, c=1.0, ep=1000):
        self.lr = lr
        self.c = c
        self.ep = ep
        self.w = None
        self.b = 0.0
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        t = np.where(y <= 0, -1.0, 1.0)
        m, n = x.shape
        self.w = np.zeros(n, dtype=float)
        self.b = 0.0

        for _ in range(self.ep):
            for i in range(m):
                xi = x[i]
                yi = t[i]
                z = yi * (np.dot(xi, self.w) + self.b)
                if z >= 1:
                    dw = self.w
                    db = 0.0
                else:
                    dw = self.w - self.c * yi * xi
                    db = -self.c * yi

                self.w -= self.lr * dw
                self.b -= self.lr * db

        self.coef_ = self.w.copy()
        self.intercept_ = float(self.b)
        return self

    def decision_function(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x @ self.w + self.b

    def predict(self, x):
        s = self.decision_function(x)
        return (s >= 0).astype(int)