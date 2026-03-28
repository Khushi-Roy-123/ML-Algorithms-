import numpy as np


class LinearRegression:
    """Ordinary Least Squares Linear Regression with optional gradient descent training."""

    def __init__(self, fit_intercept=True, solver="normal", lr=0.01, epochs=1000):
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.lr = lr
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = 0.0

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_design = self._add_intercept(X)

        if self.solver == "normal":
            theta = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ y)
        elif self.solver == "gd":
            m, n = X_design.shape
            theta = np.zeros(n, dtype=float)

            for _ in range(self.epochs):
                y_pred = X_design @ theta
                gradient = (1 / m) * (X_design.T @ (y_pred - y))
                theta -= self.lr * gradient
        else:
            raise ValueError("solver must be either 'normal' or 'gd'")

        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.coef_ + self.intercept_
