import numpy as np

class LightGBMRegressor:
    

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=31,
        min_data_in_leaf=20,
        max_bins=64,
        reg_lambda=1.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.max_bins = max_bins
        self.reg_lambda = reg_lambda
        self.random_state = random_state

        self.base_score_ = None
        self.trees_ = []
        self.bin_edges_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
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
        if self.max_bins < 2:
            raise ValueError("max_bins must be at least 2")
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be at least 2")
        if self.min_data_in_leaf < 1:
            raise ValueError("min_data_in_leaf must be at least 1")

        _ = np.random.default_rng(self.random_state)

        self.bin_edges_ = self._fit_binner(X)
        X_bin = self._bin_data(X)

        self.base_score_ = float(np.mean(y))
        pred = np.full(y.shape[0], self.base_score_, dtype=float)
        self.trees_ = []

        for _ in range(self.n_estimators):
            grad = pred - y
            hess = np.ones_like(grad)

            all_idx = np.arange(X_bin.shape[0], dtype=int)
            leaf_count = [1]
            tree = self._build_tree(X_bin, grad, hess, all_idx, depth=0, leaf_count=leaf_count)

            update = self._predict_tree_batch(X_bin, tree)
            pred = pred + self.learning_rate * update
            self.trees_.append(tree)

        return self

    def predict(self, X):
        if self.base_score_ is None or self.bin_edges_ is None or not self.trees_:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_bin = self._bin_data(X)
        pred = np.full(X.shape[0], self.base_score_, dtype=float)
        for tree in self.trees_:
            pred = pred + self.learning_rate * self._predict_tree_batch(X_bin, tree)
        return pred

    def _fit_binner(self, X):
        edges = []
        quantiles = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        for j in range(X.shape[1]):
            feature = X[:, j]
            q = np.quantile(feature, quantiles)
            e = np.unique(q)
            edges.append(e.astype(float))
        return edges

    def _bin_data(self, X):
        X_bin = np.zeros_like(X, dtype=int)
        for j, e in enumerate(self.bin_edges_):
            if e.size == 0:
                X_bin[:, j] = 0
            else:
                X_bin[:, j] = np.digitize(X[:, j], e, right=False)
        return X_bin

    def _leaf_value(self, grad, hess, idx):
        g = np.sum(grad[idx])
        h = np.sum(hess[idx])
        return float(-g / (h + self.reg_lambda))

    def _best_split(self, X_bin, grad, hess, idx):
        n_features = X_bin.shape[1]
        g_total = np.sum(grad[idx])
        h_total = np.sum(hess[idx])

        best = None
        best_gain = 0.0

        for j in range(n_features):
            bins = X_bin[idx, j]
            n_bins = int(np.max(bins)) + 1
            g_hist = np.bincount(bins, weights=grad[idx], minlength=n_bins)
            h_hist = np.bincount(bins, weights=hess[idx], minlength=n_bins)
            c_hist = np.bincount(bins, minlength=n_bins)

            g_left = 0.0
            h_left = 0.0
            c_left = 0
            for b in range(n_bins - 1):
                g_left += g_hist[b]
                h_left += h_hist[b]
                c_left += int(c_hist[b])

                c_right = idx.size - c_left
                if c_left < self.min_data_in_leaf or c_right < self.min_data_in_leaf:
                    continue

                g_right = g_total - g_left
                h_right = h_total - h_left

                gain = (
                    (g_left * g_left) / (h_left + self.reg_lambda)
                    + (g_right * g_right) / (h_right + self.reg_lambda)
                    - (g_total * g_total) / (h_total + self.reg_lambda)
                )

                if gain > best_gain:
                    best_gain = gain
                    best = {"feature": j, "bin_threshold": b}

        if best is None:
            return None

        best["gain"] = float(best_gain)
        return best

    def _build_tree(self, X_bin, grad, hess, idx, depth, leaf_count):
        if (
            idx.size < 2 * self.min_data_in_leaf
            or depth >= self.max_depth
            or leaf_count[0] >= self.num_leaves
        ):
            return {"leaf_value": self._leaf_value(grad, hess, idx)}

        split = self._best_split(X_bin, grad, hess, idx)
        if split is None or split["gain"] <= 0.0:
            return {"leaf_value": self._leaf_value(grad, hess, idx)}

        feature = split["feature"]
        threshold = split["bin_threshold"]
        bins = X_bin[idx, feature]
        left_idx = idx[bins <= threshold]
        right_idx = idx[bins > threshold]

        if left_idx.size < self.min_data_in_leaf or right_idx.size < self.min_data_in_leaf:
            return {"leaf_value": self._leaf_value(grad, hess, idx)}

        leaf_count[0] += 1
        left_node = self._build_tree(X_bin, grad, hess, left_idx, depth + 1, leaf_count)
        right_node = self._build_tree(X_bin, grad, hess, right_idx, depth + 1, leaf_count)

        return {
            "feature": feature,
            "bin_threshold": threshold,
            "left": left_node,
            "right": right_node,
        }

    def _predict_tree_one(self, x_bin, node):
        if "leaf_value" in node:
            return node["leaf_value"]

        if x_bin[node["feature"]] <= node["bin_threshold"]:
            return self._predict_tree_one(x_bin, node["left"])
        return self._predict_tree_one(x_bin, node["right"])

    def _predict_tree_batch(self, X_bin, tree):
        out = np.zeros(X_bin.shape[0], dtype=float)
        for i in range(X_bin.shape[0]):
            out[i] = self._predict_tree_one(X_bin[i], tree)
        return out