import numpy as np


class GaussianNB:
    """Gaussian Naive Bayes classifier for continuous features."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None
        self.var_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        self.classes_, y_indices = np.unique(y, return_inverse=True)
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features), dtype=float)
        self.var_ = np.zeros((n_classes, n_features), dtype=float)
        self.class_prior_ = np.zeros(n_classes, dtype=float)

        for class_idx in range(n_classes):
            X_class = X[y_indices == class_idx]
            self.theta_[class_idx, :] = X_class.mean(axis=0)
            self.var_[class_idx, :] = X_class.var(axis=0)
            self.class_prior_[class_idx] = X_class.shape[0] / X.shape[0]

        epsilon = self.var_smoothing * np.max(self.var_)
        self.var_ += epsilon

        return self

    def _joint_log_likelihood(self, X):
        if self.classes_ is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        n_classes = self.classes_.shape[0]
        jll = np.zeros((n_samples, n_classes), dtype=float)

        for class_idx in range(n_classes):
            mean = self.theta_[class_idx]
            var = self.var_[class_idx]
            log_prior = np.log(self.class_prior_[class_idx])

            # log P(x|y) for Gaussian features plus class log prior.
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var,
                axis=1,
            )
            jll[:, class_idx] = log_prior + log_likelihood

        return jll

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)

        # Numerical stability: subtract max before exponentiating.
        log_prob_x = jll - np.max(jll, axis=1, keepdims=True)
        prob = np.exp(log_prob_x)
        prob_sum = np.sum(prob, axis=1, keepdims=True)
        return prob / prob_sum

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        class_indices = np.argmax(jll, axis=1)
        return self.classes_[class_indices]
