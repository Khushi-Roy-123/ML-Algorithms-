import numpy as np


class SVM:
    def __init__(
        self,
        lr=0.001,
        c=1.0,
        ep=1000,
        ker="linear",
        deg=3,
        gam=None,
        r=1.0,
        tol=1e-3,
        eps=1e-5,
        seed=42,
    ):
        self.lr = lr
        self.c = c
        self.ep = ep
        self.ker = ker
        self.deg = deg
        self.gam = gam
        self.r = r
        self.tol = tol
        self.eps = eps
        self.seed = seed
        self.w = None
        self.b = 0.0
        self.a = None
        self.xs = None
        self.ys = None
        self.g = None
        self.coef_ = None
        self.intercept_ = 0.0

    def _kernel_name(self):
        k = str(self.ker).strip().lower()
        if k in {"linear", "lin"}:
            return "linear"
        if k in {"rbf", "gaussian"}:
            return "rbf"
        if k in {"poly", "polynomial"}:
            return "poly"
        raise ValueError("ker must be one of: linear, rbf, poly")

    def _resolve_gamma(self, x):
        n_features = x.shape[1]
        if self.gam is None:
            return 1.0 / n_features

        if isinstance(self.gam, str):
            g = self.gam.strip().lower()
            if g == "scale":
                v = float(np.var(x))
                if v <= 0:
                    return 1.0 / n_features
                return 1.0 / (n_features * v)
            if g == "auto":
                return 1.0 / n_features
            raise ValueError("gam must be float, None, 'scale', or 'auto'")

        return float(self.gam)

    def _k(self, x1, x2):
        k = self._kernel_name()
        if k == "linear":
            return float(np.dot(x1, x2))
        if k == "rbf":
            d = x1 - x2
            return float(np.exp(-self.g * np.dot(d, d)))
        if k == "poly":
            return float((self.g * np.dot(x1, x2) + self.r) ** self.deg)
        raise ValueError("ker must be one of: linear, rbf, poly")

    def _f1(self, x):
        if self.xs is None:
            return self.b
        if self._kernel_name() == "linear" and self.w is not None:
            return float(np.dot(self.w, x) + self.b)
        s = 0.0
        for i in range(self.xs.shape[0]):
            s += self.a[i] * self.ys[i] * self._k(self.xs[i], x)
        return float(s + self.b)

    def _f(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        out = np.zeros(x.shape[0], dtype=float)
        for i in range(x.shape[0]):
            out[i] = self._f1(x[i])
        return out

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        self.ker = self._kernel_name()

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        t = np.where(y <= 0, -1.0, 1.0).astype(float)
        m, n = x.shape
        self.g = self._resolve_gamma(x)

        a = np.zeros(m, dtype=float)
        b = 0.0
        rng = np.random.default_rng(self.seed)
        pas = 0

        for _ in range(self.ep):
            ch = 0
            for i in range(m):
                xi = x[i]
                yi = t[i]
                fi = b
                for k in range(m):
                    if a[k] > 0:
                        fi += a[k] * t[k] * self._k(x[k], xi)
                ei = fi - yi

                c1 = yi * ei < -self.tol and a[i] < self.c
                c2 = yi * ei > self.tol and a[i] > 0
                if not (c1 or c2):
                    continue

                j = rng.integers(0, m - 1)
                if j >= i:
                    j += 1
                xj = x[j]
                yj = t[j]

                fj = b
                for k in range(m):
                    if a[k] > 0:
                        fj += a[k] * t[k] * self._k(x[k], xj)
                ej = fj - yj

                ai0 = a[i]
                aj0 = a[j]

                if yi != yj:
                    lo = max(0.0, aj0 - ai0)
                    hi = min(self.c, self.c + aj0 - ai0)
                else:
                    lo = max(0.0, ai0 + aj0 - self.c)
                    hi = min(self.c, ai0 + aj0)

                if lo == hi:
                    continue

                kii = self._k(xi, xi)
                kjj = self._k(xj, xj)
                kij = self._k(xi, xj)
                eta = 2.0 * kij - kii - kjj
                if eta >= 0:
                    continue

                aj = aj0 - yj * (ei - ej) / eta
                aj = np.clip(aj, lo, hi)
                if abs(aj - aj0) < self.eps:
                    continue

                ai = ai0 + yi * yj * (aj0 - aj)

                b1 = b - ei - yi * (ai - ai0) * kii - yj * (aj - aj0) * kij
                b2 = b - ej - yi * (ai - ai0) * kij - yj * (aj - aj0) * kjj

                if 0 < ai < self.c:
                    b = b1
                elif 0 < aj < self.c:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)

                a[i] = ai
                a[j] = aj
                ch += 1

            if ch == 0:
                pas += 1
            else:
                pas = 0

            if pas >= 5:
                break

        msk = a > self.eps
        self.a = a[msk]
        self.xs = x[msk]
        self.ys = t[msk]
        self.b = 0.0

        if self.xs.shape[0] == 0:
            self.b = float(b)
        else:
            bb = []
            for i in range(self.xs.shape[0]):
                s = 0.0
                for j in range(self.xs.shape[0]):
                    s += self.a[j] * self.ys[j] * self._k(self.xs[j], self.xs[i])
                bb.append(self.ys[i] - s)
            self.b = float(np.mean(bb))

        if self.ker == "linear" and self.xs.shape[0] > 0:
            self.w = (self.a * self.ys) @ self.xs
            self.coef_ = self.w.copy()
        elif self.ker == "linear":
            self.w = np.zeros(n, dtype=float)
            self.coef_ = self.w.copy()
        else:
            self.w = None
            self.coef_ = None

        self.intercept_ = float(self.b)
        return self

    def decision_function(self, x):
        return self._f(x)

    def predict(self, x):
        s = self.decision_function(x)
        return (s >= 0).astype(int)