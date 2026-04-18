import copy

import numpy as np

from jklearn.linear_model import LinearRegression, LogisticRegression


class _BaseStacking:
    def __init__(self, estimators, final_estimator=None, n_folds=5, shuffle=True, random_state=42, passthrough=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.passthrough = passthrough

        self.estimators_ = []
        self.final_estimator_ = None

    def _validate_estimators(self):
        if not isinstance(self.estimators, (list, tuple)) or len(self.estimators) == 0:
            raise ValueError("estimators must be a non-empty list of (name, estimator) tuples")

        validated = []
        seen_names = set()
        for item in self.estimators:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("each estimator must be a (name, estimator) tuple")
            name, estimator = item
            if not isinstance(name, str) or not name:
                raise ValueError("each estimator name must be a non-empty string")
            if name in seen_names:
                raise ValueError("estimator names must be unique")
            if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
                raise ValueError(f"estimator '{name}' must implement fit and predict")
            seen_names.add(name)
            validated.append((name, estimator))

        return validated

    def _make_kfolds(self, X):
        n_samples = X.shape[0]
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.n_folds > n_samples:
            raise ValueError("n_folds cannot be greater than the number of samples")

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
        folds = np.array_split(indices, self.n_folds)
        return [fold.astype(int) for fold in folds if fold.size > 0]

    def _make_stratified_folds(self, y):
        y = np.asarray(y)
        n_samples = y.shape[0]
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.n_folds > n_samples:
            raise ValueError("n_folds cannot be greater than the number of samples")

        rng = np.random.default_rng(self.random_state)
        fold_lists = [[] for _ in range(self.n_folds)]
        class_counts = np.bincount(np.searchsorted(np.unique(y), y))
        if class_counts.size > 0 and np.min(class_counts) < self.n_folds:
            return self._make_kfolds(np.empty((n_samples, 1)))

        for class_label in np.unique(y):
            class_indices = np.flatnonzero(y == class_label)
            if self.shuffle:
                rng.shuffle(class_indices)
            class_folds = np.array_split(class_indices, self.n_folds)
            for fold_idx, class_fold in enumerate(class_folds):
                if class_fold.size > 0:
                    fold_lists[fold_idx].extend(class_fold.tolist())

        folds = [np.array(sorted(fold), dtype=int) for fold in fold_lists if len(fold) > 0]
        if len(folds) < self.n_folds:
            return self._make_kfolds(np.empty((n_samples, 1)))
        return folds

    def _stack_predictions(self, estimator, X):
        if hasattr(estimator, "predict_proba"):
            return np.asarray(estimator.predict_proba(X), dtype=float)
        return np.asarray(estimator.predict(X), dtype=float).reshape(-1, 1)

    def _fit_base_estimators(self, X, y):
        fitted_estimators = []
        for _, estimator in self._validated_estimators:
            fitted = copy.deepcopy(estimator)
            fitted.fit(X, y)
            fitted_estimators.append(fitted)
        self.estimators_ = fitted_estimators

    def _build_meta_features(self, X, y=None, stratified=False):
        folds = self._make_stratified_folds(y) if stratified else self._make_kfolds(X)
        meta_chunks = []

        for _, estimator in self._validated_estimators:
            estimator_features = []
            for fold_indices in folds:
                train_mask = np.ones(X.shape[0], dtype=bool)
                train_mask[fold_indices] = False
                X_train = X[train_mask]
                X_valid = X[fold_indices]

                fold_estimator = copy.deepcopy(estimator)
                if y is None:
                    fold_estimator.fit(X_train)
                else:
                    fold_estimator.fit(X_train, y[train_mask])

                estimator_features.append((fold_indices, self._stack_predictions(fold_estimator, X_valid)))

            column_count = estimator_features[0][1].shape[1]
            stacked = np.zeros((X.shape[0], column_count), dtype=float)
            for fold_indices, predictions in estimator_features:
                stacked[fold_indices] = predictions
            meta_chunks.append(stacked)

        meta_features = np.hstack(meta_chunks)
        if self.passthrough:
            meta_features = np.hstack([meta_features, X])
        return meta_features


class StackingRegressor(_BaseStacking):
    def __init__(self, estimators, final_estimator=None, n_folds=5, shuffle=True, random_state=42, passthrough=False):
        super().__init__(estimators, final_estimator, n_folds, shuffle, random_state, passthrough)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        self._validated_estimators = self._validate_estimators()
        meta_features = self._build_meta_features(X, y, stratified=False)

        if self.final_estimator is None:
            self.final_estimator_ = LinearRegression()
        else:
            self.final_estimator_ = copy.deepcopy(self.final_estimator)

        self.final_estimator_.fit(meta_features, y)
        self._fit_base_estimators(X, y)
        return self

    def predict(self, X):
        if self.final_estimator_ is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        meta_features = np.hstack([self._stack_predictions(estimator, X) for estimator in self.estimators_])
        if self.passthrough:
            meta_features = np.hstack([meta_features, X])
        return self.final_estimator_.predict(meta_features)


class StackingClassifier(_BaseStacking):
    def __init__(self, estimators, final_estimator=None, n_folds=5, shuffle=True, random_state=42, passthrough=False):
        super().__init__(estimators, final_estimator, n_folds, shuffle, random_state, passthrough)
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        self.classes_ = np.unique(y)
        self._validated_estimators = self._validate_estimators()
        meta_features = self._build_meta_features(X, y, stratified=True)

        if self.final_estimator is None:
            self.final_estimator_ = LogisticRegression()
        else:
            self.final_estimator_ = copy.deepcopy(self.final_estimator)

        self.final_estimator_.fit(meta_features, y)
        self._fit_base_estimators(X, y)
        return self

    def predict(self, X):
        if self.final_estimator_ is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        meta_features = np.hstack([self._stack_predictions(estimator, X) for estimator in self.estimators_])
        if self.passthrough:
            meta_features = np.hstack([meta_features, X])
        return self.final_estimator_.predict(meta_features)

    def predict_proba(self, X):
        if self.final_estimator_ is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")
        if not hasattr(self.final_estimator_, "predict_proba"):
            raise AttributeError("final_estimator does not implement predict_proba")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        meta_features = np.hstack([self._stack_predictions(estimator, X) for estimator in self.estimators_])
        if self.passthrough:
            meta_features = np.hstack([meta_features, X])
        return self.final_estimator_.predict_proba(meta_features)