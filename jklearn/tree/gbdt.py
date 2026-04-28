import numpy as np

from .decision_tree import DecisionTreeRegressor

class GBDTRegressor:
    def __init__(self, n=100, lr=0.1, md=3, ms=2, rs=None):
        self.n = n
        self.lr = lr
        self.md = md
        self.ms = ms
        self.rs = rs
        self.b_ = None
        self.ts_ = []

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same sample count")
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")

        _ = np.random.default_rng(self.rs)
        self.b_ = float(np.mean(y))
        p = np.full(y.shape[0], self.b_, dtype=float)
        self.ts_ = []

        for _ in range(self.n):
            r = y - p
            t = DecisionTreeRegressor(max_depth=self.md, min_samples_split=self.ms)
            t.fit(x, r)
            p = p + self.lr * t.predict(x)
            self.ts_.append(t)

        return self

    def predict(self, x):
        if self.b_ is None or not self.ts_:
            raise ValueError("Model is not fitted yet. Call fit(x, y) first.")

        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        p = np.full(x.shape[0], self.b_, dtype=float)
        for t in self.ts_:
            p = p + self.lr * t.predict(x)
        return p