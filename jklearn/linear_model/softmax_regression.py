import numpy as np

class SoftmaxRegression:
	

	def __init__(self, lr=0.1, epochs=1000):
		self.lr = lr
		self.epochs = epochs
		self.weights_ = None
		self.bias_ = None
		self.classes_ = None

	def _softmax(self, z):
		z = z - np.max(z, axis=1, keepdims=True)
		ez = np.exp(z)
		return ez / np.sum(ez, axis=1, keepdims=True)

	def _one_hot(self, y_idx, n_classes):
		y_oh = np.zeros((y_idx.size, n_classes), dtype=float)
		y_oh[np.arange(y_idx.size), y_idx] = 1.0
		return y_oh

	def fit(self, X, y):
		X = np.asarray(X, dtype=float)
		y = np.asarray(y)

		if X.ndim == 1:
			X = X.reshape(-1, 1)

		self.classes_, y_idx = np.unique(y, return_inverse=True)
		m, n = X.shape
		k = self.classes_.size

		self.weights_ = np.zeros((n, k), dtype=float)
		self.bias_ = np.zeros(k, dtype=float)
		y_oh = self._one_hot(y_idx, k)

		for _ in range(self.epochs):
			scores = X @ self.weights_ + self.bias_
			probs = self._softmax(scores)

			grad_w = (X.T @ (probs - y_oh)) / m
			grad_b = np.mean(probs - y_oh, axis=0)

			self.weights_ -= self.lr * grad_w
			self.bias_ -= self.lr * grad_b

		return self

	def predict_proba(self, X):
		if self.weights_ is None or self.bias_ is None:
			raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

		X = np.asarray(X, dtype=float)
		if X.ndim == 1:
			X = X.reshape(-1, 1)

		scores = X @ self.weights_ + self.bias_
		return self._softmax(scores)

	def predict(self, X):
		probs = self.predict_proba(X)
		idx = np.argmax(probs, axis=1)
		return self.classes_[idx]

