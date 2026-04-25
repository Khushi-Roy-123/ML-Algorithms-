import numpy as np


class KMeans:
    def __init__(self, k=8, ep=300, tol=1e-4, n_init=10, seed=42):
        self.k = k
        self.ep = ep
        self.tol = tol
        self.n_init = n_init
        self.seed = seed
        self.cen = None
        self.lab = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _chk(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("X must be a 2D array")
        return x

    def _dis(self, x, c):
        d = x[:, None, :] - c[None, :, :]
        return np.sum(d * d, axis=2)

    def _ini(self, x, r):
        idx = r.choice(x.shape[0], size=self.k, replace=False)
        return x[idx].copy()

    def _mov(self, x, l, r):
        c = np.empty((self.k, x.shape[1]), dtype=float)
        for i in range(self.k):
            m = l == i
            if np.any(m):
                c[i] = np.mean(x[m], axis=0)
            else:
                c[i] = x[r.integers(0, x.shape[0])]
        return c

    def _run(self, x, r):
        c = self._ini(x, r)
        n_it = 0
        for i in range(1, self.ep + 1):
            d = self._dis(x, c)
            l = np.argmin(d, axis=1)
            n_c = self._mov(x, l, r)
            n_it = i
            if np.max(np.abs(n_c - c)) <= self.tol:
                c = n_c
                break
            c = n_c

        d = self._dis(x, c)
        l = np.argmin(d, axis=1)
        iner = float(np.sum(d[np.arange(x.shape[0]), l]))
        return c, l, iner, n_it

    def fit(self, x):
        x = self._chk(x)
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.k > x.shape[0]:
            raise ValueError("k must not exceed the number of samples")
        if self.ep < 1:
            raise ValueError("ep must be at least 1")
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1")

        r = np.random.default_rng(self.seed)
        best = None
        best_iner = np.inf

        for _ in range(self.n_init):
            c, l, iner, n_it = self._run(x, r)
            if iner < best_iner:
                best_iner = iner
                best = (c.copy(), l.copy(), n_it)

        self.cen, self.lab, self.n_iter_ = best
        self.inertia_ = float(best_iner)
        return self

    def predict(self, x):
        if self.cen is None:
            raise ValueError("Model is not fitted yet. Call fit(x) first.")
        x = self._chk(x)
        d = self._dis(x, self.cen)
        return np.argmin(d, axis=1)

    def fit_predict(self, x):
        return self.fit(x).lab

    def elbow(self, x, k_min=1, k_max=10):
        x = self._chk(x)
        if k_min < 1:
            raise ValueError("k_min must be at least 1")
        if k_max < k_min:
            raise ValueError("k_max must be greater than or equal to k_min")
        if k_max > x.shape[0]:
            raise ValueError("k_max must not exceed the number of samples")

        ks = np.arange(k_min, k_max + 1)
        ins = np.empty(ks.shape[0], dtype=float)

        for i, k in enumerate(ks):
            m = KMeans(
                k=int(k),
                ep=self.ep,
                tol=self.tol,
                n_init=self.n_init,
                seed=self.seed,
            )
            m.fit(x)
            ins[i] = m.inertia_

        return ks, ins
