"""AdaBoost (Adaptive Boosting) classifier implementation."""

import numpy as np

from .decision_tree import DecisionTreeClassifier


class AdaBoostClassifier:
    """AdaBoost classifier that boosts weak learners by focusing on misclassified samples.
    
    AdaBoost iteratively trains weak learners (decision stumps by default) on weighted
    versions of the dataset, with weights adjusted to emphasize misclassified samples.
    The final prediction is a weighted combination of all weak learners.
    
    Parameters
    ----------
    n_estimators : int, default=50
        Number of weak learners (stumps) to train.
    learning_rate : float, default=1.0
        Learning rate that shrinks the contribution of each classifier.
    max_depth : int, default=1
        Maximum depth of the weak learners (decision stumps).
    random_state : int or None, default=None
        Random state for reproducibility.
    """

    def __init__(self, n_estimators=50, learning_rate=1.0, max_depth=1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.estimators_ = []
        self.alphas_ = []
        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        """Fit the AdaBoost classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        self.classes_, y_indices = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        
        if self.n_classes_ != 2:
            raise ValueError("AdaBoost currently supports binary classification only")

        # Initialize uniform sample weights
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples

        self.estimators_ = []
        self.alphas_ = []
        rng = np.random.default_rng(self.random_state)

        for iteration in range(self.n_estimators):
            # Train weak learner on weighted data
            weak_learner = DecisionTreeClassifier(max_depth=self.max_depth)
            weak_learner.fit(X, y)
            
            # Get predictions
            predictions = weak_learner.predict(X)
            
            # Calculate weighted error rate
            misclassified = predictions != y
            weighted_error = np.sum(sample_weights[misclassified])
            
            # Prevent division by zero and ensure alpha is reasonable
            if weighted_error <= 0:
                weighted_error = 1e-10
            if weighted_error >= 1 - 1e-10:
                # This learner is not better than random guessing
                break
            
            # Calculate learner weight (alpha)
            alpha = self.learning_rate * 0.5 * np.log((1 - weighted_error) / weighted_error)
            
            self.estimators_.append(weak_learner)
            self.alphas_.append(alpha)
            
            # Update sample weights
            # Increase weights for misclassified samples, decrease for correct ones
            exponents = np.where(misclassified, 1, -1)
            sample_weights *= np.exp(alpha * exponents)
            sample_weights /= np.sum(sample_weights)

        return self

    def predict(self, X):
        """Predict class for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y : array of shape (n_samples,)
            Predicted class labels.
        """
        if not self.estimators_:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get predictions from all weak learners
        predictions = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        
        for estimator, alpha in zip(self.estimators_, self.alphas_):
            weak_pred = estimator.predict(X)
            # Convert to class indices
            class_indices = np.searchsorted(self.classes_, weak_pred)
            # Add weighted predictions
            predictions[np.arange(X.shape[0]), class_indices] += alpha

        # Return class with maximum weighted vote
        class_indices = np.argmax(predictions, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X):
        """Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        if not self.estimators_:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get predictions from all weak learners
        scores = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        
        for estimator, alpha in zip(self.estimators_, self.alphas_):
            weak_pred = estimator.predict(X)
            class_indices = np.searchsorted(self.classes_, weak_pred)
            scores[np.arange(X.shape[0]), class_indices] += alpha

        # Normalize to probabilities
        # Use softmax-like normalization
        scores = scores - np.min(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return proba
