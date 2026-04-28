import numpy as np

from .decision_tree import DecisionTreeRegressor

class CatBoostRegressor:
    

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        l2_leaf_reg=3.0,
        cat_features=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.l2_leaf_reg = l2_leaf_reg
        self.cat_features = cat_features
        self.random_state = random_state

        self.base_score_ = None
        self.trees_ = []
        self.cat_feature_idx_ = None
        self.cat_mappings_ = None
        self.global_mean_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same sample count")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if self.l2_leaf_reg < 0:
            raise ValueError("l2_leaf_reg must be >= 0")

        self.n_features_in_ = X.shape[1]
        self.global_mean_ = float(np.mean(y))
        self.cat_feature_idx_ = self._resolve_cat_feature_idx(X)

        rng = np.random.default_rng(self.random_state)
        X_fit, self.cat_mappings_ = self._fit_transform_categorical(X, y, rng)

        self.base_score_ = self.global_mean_
        pred = np.full(y.shape[0], self.base_score_, dtype=float)
        self.trees_ = []

        shrink = 1.0 / (1.0 + self.l2_leaf_reg)
        for _ in range(self.n_estimators):
            residual = y - pred
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_fit, residual)
            update = tree.predict(X_fit)
            pred = pred + self.learning_rate * shrink * update
            self.trees_.append(tree)

        return self

    def predict(self, X):
        if self.base_score_ is None or not self.trees_:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")

        X_pred = self._transform_categorical(X)

        pred = np.full(X_pred.shape[0], self.base_score_, dtype=float)
        shrink = 1.0 / (1.0 + self.l2_leaf_reg)
        for tree in self.trees_:
            pred = pred + self.learning_rate * shrink * tree.predict(X_pred)
        return pred

    def _resolve_cat_feature_idx(self, X):
        n_features = X.shape[1]

        if self.cat_features is None:
            idx = []
            for j in range(n_features):
                try:
                    X[:, j].astype(float)
                except (TypeError, ValueError):
                    idx.append(j)
            return idx

        if isinstance(self.cat_features, (list, tuple, np.ndarray)):
            if len(self.cat_features) == 0:
                return []

            if all(isinstance(v, (bool, np.bool_)) for v in self.cat_features):
                if len(self.cat_features) != n_features:
                    raise ValueError("Boolean cat_features mask must match number of features")
                return [i for i, is_cat in enumerate(self.cat_features) if is_cat]

            idx = [int(v) for v in self.cat_features]
            for j in idx:
                if j < 0 or j >= n_features:
                    raise ValueError("cat_features index out of range")
            return sorted(set(idx))

        raise ValueError("cat_features must be None, an index list, or a boolean mask")

    def _fit_transform_categorical(self, X, y, rng):
        X_out = np.zeros((X.shape[0], X.shape[1]), dtype=float)
        mappings = {}

        cat_idx_set = set(self.cat_feature_idx_)
        for j in range(X.shape[1]):
            if j not in cat_idx_set:
                X_out[:, j] = X[:, j].astype(float)
                continue

            col = X[:, j]
            perm = rng.permutation(X.shape[0])
            encoded_perm = np.zeros(X.shape[0], dtype=float)

            prior_weight = 1.0
            running_sum = {}
            running_count = {}

            for i in perm:
                key = col[i]
                s = running_sum.get(key, 0.0)
                c = running_count.get(key, 0)
                encoded_perm[i] = (s + prior_weight * self.global_mean_) / (c + prior_weight)
                running_sum[key] = s + y[i]
                running_count[key] = c + 1

            X_out[:, j] = encoded_perm

            final_map = {}
            for key, s in running_sum.items():
                c = running_count[key]
                final_map[key] = (s + prior_weight * self.global_mean_) / (c + prior_weight)
            mappings[j] = final_map

        return X_out, mappings

    def _transform_categorical(self, X):
        X_out = np.zeros((X.shape[0], X.shape[1]), dtype=float)
        cat_idx_set = set(self.cat_feature_idx_)

        for j in range(X.shape[1]):
            if j not in cat_idx_set:
                X_out[:, j] = X[:, j].astype(float)
                continue

            mapping = self.cat_mappings_.get(j, {})
            col = X[:, j]
            X_out[:, j] = np.array(
                [mapping.get(value, self.global_mean_) for value in col],
                dtype=float,
            )

        return X_out