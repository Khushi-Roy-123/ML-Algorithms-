from __future__ import annotations

import math

import numpy as np

from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


def _resolve_max_features(max_features, n_features):
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(math.log2(n_features)))
        raise ValueError("max_features must be None, an int, a float, 'sqrt', or 'log2'")
    if isinstance(max_features, float):
        if not 0.0 < max_features <= 1.0:
            raise ValueError("float max_features must be in the range (0, 1]")
        return max(1, int(math.ceil(n_features * max_features)))
    if isinstance(max_features, int):
        if max_features < 1:
            raise ValueError("int max_features must be at least 1")
        return min(n_features, max_features)
    raise ValueError("Unsupported max_features value")


class _ExtraTreeClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0, max_features=None, n_thresholds=10, random_state=None):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split, min_impurity_decrease=min_impurity_decrease)
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_impurity = self._gini(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        feature_count = _resolve_max_features(self.max_features, n_features)
        candidate_features = self._rng.choice(n_features, size=feature_count, replace=False)

        for feature_idx in candidate_features:
            column = X[:, feature_idx]
            column_min = float(np.min(column))
            column_max = float(np.max(column))
            if column_min == column_max:
                continue

            thresholds = self._rng.uniform(column_min, column_max, size=self.n_thresholds)
            for threshold in thresholds:
                left_mask = column <= threshold
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue

                left_impurity = self._gini(y[left_mask])
                right_impurity = self._gini(y[right_mask])
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                child_impurity = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = float(threshold)

        if best_feature is None:
            return None

        return {"feature_index": best_feature, "threshold": best_threshold, "gain": best_gain}


class _ExtraTreeRegressor(DecisionTreeRegressor):
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0, max_features=None, n_thresholds=10, random_state=None):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split, min_impurity_decrease=min_impurity_decrease)
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_mse = self._mse(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        feature_count = _resolve_max_features(self.max_features, n_features)
        candidate_features = self._rng.choice(n_features, size=feature_count, replace=False)

        for feature_idx in candidate_features:
            column = X[:, feature_idx]
            column_min = float(np.min(column))
            column_max = float(np.max(column))
            if column_min == column_max:
                continue

            thresholds = self._rng.uniform(column_min, column_max, size=self.n_thresholds)
            for threshold in thresholds:
                left_mask = column <= threshold
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue

                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                child_mse = (n_left / n_samples) * left_mse + (n_right / n_samples) * right_mse
                gain = parent_mse - child_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = float(threshold)

        if best_feature is None:
            return None

        return {"feature_index": best_feature, "threshold": best_threshold, "gain": best_gain}


class ExtraTreesClassifier:
    """Extremely randomized tree ensemble for classification."""

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0, max_features="sqrt", n_thresholds=10, bootstrap=False, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.classes_ = None
        self.n_classes_ = None
        self.estimators_ = []

    def fit(self, X, y):
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

        self.classes_, y_indices = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            tree_random_state = int(rng.integers(0, np.iinfo(np.uint32).max))
            tree = _ExtraTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                n_thresholds=self.n_thresholds,
                random_state=tree_random_state,
            )
            tree.classes_ = self.classes_
            tree.n_classes_ = self.n_classes_

            if self.bootstrap:
                sample_indices = rng.integers(0, X.shape[0], size=X.shape[0])
                sample_X = X[sample_indices]
                sample_y = y_indices[sample_indices]
            else:
                sample_X = X
                sample_y = y_indices

            tree.tree_ = tree._grow_tree(sample_X, sample_y, depth=0)
            self.estimators_.append(tree)

        return self

    def predict_proba(self, X):
        if not self.estimators_:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probabilities = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for estimator in self.estimators_:
            probabilities += estimator.predict_proba(X)
        probabilities /= len(self.estimators_)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]


class ExtraTreesRegressor:
    """Extremely randomized tree ensemble for regression."""

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0, max_features=1.0, n_thresholds=10, bootstrap=False, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of target values")
        if X.shape[0] != y.shape[0] or self.n_estimators < 1:
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must contain the same number of samples")
            raise ValueError("n_estimators must be at least 1")

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            tree_random_state = int(rng.integers(0, np.iinfo(np.uint32).max))
            tree = _ExtraTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                n_thresholds=self.n_thresholds,
                random_state=tree_random_state,
            )

            if self.bootstrap:
                sample_indices = rng.integers(0, X.shape[0], size=X.shape[0])
                sample_X = X[sample_indices]
                sample_y = y[sample_indices]
            else:
                sample_X = X
                sample_y = y

            tree.tree_ = tree._grow_tree(sample_X, sample_y, depth=0)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        if not self.estimators_:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = np.zeros((len(self.estimators_), X.shape[0]), dtype=float)
        for idx, estimator in enumerate(self.estimators_):
            predictions[idx] = estimator.predict(X)
        return np.mean(predictions, axis=0)