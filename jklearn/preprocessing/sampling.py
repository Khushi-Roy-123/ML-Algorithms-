import numpy as np

class UnderSampling:
    def __init__(self, seed=42):
        self.seed = seed

    def fit_resample(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same number of samples")

        cls, cnt = np.unique(y, return_counts=True)
        n = int(np.min(cnt))
        r = np.random.default_rng(self.seed)

        ids = []
        for c in cls:
            ci = np.where(y == c)[0]
            si = r.choice(ci, size=n, replace=False)
            ids.append(si)

        ids = np.concatenate(ids)
        r.shuffle(ids)
        return x[ids], y[ids]

class OverSampling:
    def __init__(self, seed=42):
        self.seed = seed

    def fit_resample(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same number of samples")

        cls, cnt = np.unique(y, return_counts=True)
        n = int(np.max(cnt))
        r = np.random.default_rng(self.seed)

        ids = []
        for c in cls:
            ci = np.where(y == c)[0]
            si = r.choice(ci, size=n, replace=True)
            ids.append(si)

        ids = np.concatenate(ids)
        r.shuffle(ids)
        return x[ids], y[ids]

class SMOTE:
    def __init__(self, k=5, seed=42):
        self.k = k
        self.seed = seed

    def _knn(self, x):
        d = x[:, None, :] - x[None, :, :]
        d = np.sum(d * d, axis=2)
        np.fill_diagonal(d, np.inf)
        k = min(self.k, x.shape[0] - 1)
        if k < 1:
            return None
        return np.argsort(d, axis=1)[:, :k]

    def fit_resample(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same number of samples")
        if self.k < 1:
            raise ValueError("k must be at least 1")

        cls, cnt = np.unique(y, return_counts=True)
        n_max = int(np.max(cnt))
        r = np.random.default_rng(self.seed)

        xs = [x]
        ys = [y]

        for c, n in zip(cls, cnt):
            add = n_max - int(n)
            if add <= 0:
                continue

            xc = x[y == c]
            if xc.shape[0] == 1:
                x_new = np.repeat(xc, add, axis=0)
                y_new = np.full(add, c, dtype=y.dtype)
                xs.append(x_new)
                ys.append(y_new)
                continue

            nn = self._knn(xc)
            x_new = np.empty((add, x.shape[1]), dtype=float)
            for i in range(add):
                a = r.integers(0, xc.shape[0])
                b = int(r.choice(nn[a]))
                lam = r.random()
                x_new[i] = xc[a] + lam * (xc[b] - xc[a])

            y_new = np.full(add, c, dtype=y.dtype)
            xs.append(x_new)
            ys.append(y_new)

        x_out = np.vstack(xs)
        y_out = np.concatenate(ys)

        idx = np.arange(y_out.shape[0])
        r.shuffle(idx)
        return x_out[idx], y_out[idx]
