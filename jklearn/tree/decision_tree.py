from dataclasses import dataclass

import numpy as np

@dataclass
class _Node:
    feature_index: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    value: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None

class DecisionTreeClassifier:
    

    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.classes_ = None
        self.n_classes_ = None
        self.tree_ = None

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
        self.n_classes_ = self.classes_.shape[0]
        self.tree_ = self._grow_tree(X, y_indices, depth=0)
        return self

    def predict(self, X):
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X):
        if self.tree_ is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probs = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for row_idx, row in enumerate(X):
            leaf = self._traverse_tree(row, self.tree_)
            probs[row_idx] = leaf.value
        return probs

    def _grow_tree(self, X, y, depth):
        n_samples, _ = X.shape
        num_labels = np.unique(y).shape[0]

        if (
            num_labels == 1
            or n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return _Node(value=self._leaf_value(y))

        best_split = self._best_split(X, y)
        if best_split is None or best_split["gain"] <= self.min_impurity_decrease:
            return _Node(value=self._leaf_value(y))

        left_mask = X[:, best_split["feature_index"]] <= best_split["threshold"]
        right_mask = ~left_mask

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left_child,
            right=right_child,
        )

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_impurity = self._gini(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            values = np.unique(X[:, feature_idx])
            if values.shape[0] < 2:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
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
                    best_threshold = threshold

        if best_feature is None:
            return None

        return {"feature_index": best_feature, "threshold": best_threshold, "gain": best_gain}

    def _gini(self, y):
        if y.size == 0:
            return 0.0

        counts = np.bincount(y, minlength=self.n_classes_)
        probabilities = counts / y.size
        return 1.0 - np.sum(probabilities ** 2)

    def _leaf_value(self, y):
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        total = counts.sum()
        if total == 0:
            return np.full(self.n_classes_, 1.0 / self.n_classes_)
        return counts / total

    def _traverse_tree(self, x, node):
        if node.is_leaf:
            return node

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class DecisionTreeRegressor:
    

    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree_ = None

    def fit(self, X, y):
        
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of target values")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X):
        
        if self.tree_ is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = np.zeros(X.shape[0], dtype=float)
        for row_idx, row in enumerate(X):
            leaf = self._traverse_tree(row, self.tree_)
            predictions[row_idx] = leaf.value.item()
        return predictions

    def _grow_tree(self, X, y, depth):
        
        n_samples, _ = X.shape

        # Stopping criteria
        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return _Node(value=np.array([np.mean(y)]))

        best_split = self._best_split(X, y)
        if best_split is None or best_split["gain"] <= self.min_impurity_decrease:
            return _Node(value=np.array([np.mean(y)]))

        left_mask = X[:, best_split["feature_index"]] <= best_split["threshold"]
        right_mask = ~left_mask

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left_child,
            right=right_child,
        )

    def _best_split(self, X, y):
        
        n_samples, n_features = X.shape
        parent_mse = self._mse(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            values = np.unique(X[:, feature_idx])
            if values.shape[0] < 2:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue

                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                
                # Weighted MSE of children
                child_mse = (n_left / n_samples) * left_mse + (n_right / n_samples) * right_mse
                
                # Information gain
                gain = parent_mse - child_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        if best_feature is None:
            return None

        return {"feature_index": best_feature, "threshold": best_threshold, "gain": best_gain}

    def _mse(self, y):
        
        if y.size == 0:
            return 0.0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _traverse_tree(self, x, node):
        
        if node.is_leaf:
            return node

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)