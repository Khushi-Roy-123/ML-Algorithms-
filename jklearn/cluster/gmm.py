import numpy as np


class GaussianMixture:
    def __init__(
        self,
        k=2,
        ep=100,
        tol=1e-4,
        reg_covar=1e-6,
        n_init=1,
        seed=42,
    ):
        self.k = k
        self.ep = ep
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.seed = seed

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.lab = None
        self.lower_bound_ = -np.inf
        self.n_iter_ = 0
        self.converged_ = False

    def _chk(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("X must be a 2D array")
        return x

    def _empirical_cov(self, x):
        n_features = x.shape[1]
        if x.shape[0] <= 1:
            c = np.eye(n_features, dtype=float)
        else:
            c = np.cov(x, rowvar=False, bias=True)
            c = np.atleast_2d(c).astype(float)
        c.flat[:: n_features + 1] += self.reg_covar
        return c

    def _init_params(self, x, rng):
        n_samples, n_features = x.shape
        idx = rng.choice(n_samples, size=self.k, replace=False)

        self.weights_ = np.full(self.k, 1.0 / self.k, dtype=float)
        self.means_ = x[idx].copy()

        base_cov = self._empirical_cov(x)
        self.covariances_ = np.repeat(base_cov[None, :, :], self.k, axis=0)

    def _estimate_log_gaussian_prob(self, x):
        n_samples, n_features = x.shape
        log_prob = np.empty((n_samples, self.k), dtype=float)

        for i in range(self.k):
            mean = self.means_[i]
            cov = self.covariances_[i]

            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError("Covariance matrix is not positive definite")

            diff = x - mean
            sol = np.linalg.solve(cov, diff.T).T
            quad = np.sum(diff * sol, axis=1)
            log_prob[:, i] = -0.5 * (
                n_features * np.log(2.0 * np.pi) + log_det + quad
            )

        return log_prob

    def _logsumexp(self, a):
        a_max = np.max(a, axis=1, keepdims=True)
        return a_max + np.log(np.sum(np.exp(a - a_max), axis=1, keepdims=True))

    def _e_step(self, x):
        weighted_log_prob = self._estimate_log_gaussian_prob(x) + np.log(self.weights_)
        log_prob_norm = self._logsumexp(weighted_log_prob)
        log_resp = weighted_log_prob - log_prob_norm
        resp = np.exp(log_resp)
        lower_bound = float(np.mean(log_prob_norm))
        return resp, lower_bound

    def _m_step(self, x, resp):
        n_samples, n_features = x.shape
        nk = resp.sum(axis=0) + 10.0 * np.finfo(float).eps

        self.weights_ = nk / n_samples
        self.means_ = (resp.T @ x) / nk[:, None]

        covariances = np.empty((self.k, n_features, n_features), dtype=float)
        for i in range(self.k):
            diff = x - self.means_[i]
            cov = (resp[:, i][:, None] * diff).T @ diff / nk[i]
            cov.flat[:: n_features + 1] += self.reg_covar
            covariances[i] = cov

        self.covariances_ = covariances

    def _fit_once(self, x, rng):
        self._init_params(x, rng)
        prev_lower_bound = -np.inf

        converged = False
        n_it = 0
        for i in range(1, self.ep + 1):
            resp, lower_bound = self._e_step(x)
            self._m_step(x, resp)
            n_it = i

            if abs(lower_bound - prev_lower_bound) <= self.tol:
                converged = True
                break
            prev_lower_bound = lower_bound

        resp, lower_bound = self._e_step(x)
        labels = np.argmax(resp, axis=1)
        return {
            "weights": self.weights_.copy(),
            "means": self.means_.copy(),
            "covariances": self.covariances_.copy(),
            "labels": labels,
            "lower_bound": float(lower_bound),
            "n_iter": n_it,
            "converged": converged,
        }

    def fit(self, x):
        x = self._chk(x)
        n_samples = x.shape[0]

        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.k > n_samples:
            raise ValueError("k must not exceed the number of samples")
        if self.ep < 1:
            raise ValueError("ep must be at least 1")
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1")
        if self.reg_covar <= 0:
            raise ValueError("reg_covar must be > 0")

        rng = np.random.default_rng(self.seed)

        best = None
        for _ in range(self.n_init):
            result = self._fit_once(x, rng)
            if best is None or result["lower_bound"] > best["lower_bound"]:
                best = result

        self.weights_ = best["weights"]
        self.means_ = best["means"]
        self.covariances_ = best["covariances"]
        self.lab = best["labels"]
        self.lower_bound_ = best["lower_bound"]
        self.n_iter_ = best["n_iter"]
        self.converged_ = best["converged"]
        return self

    def predict_proba(self, x):
        if self.means_ is None:
            raise ValueError("Model is not fitted yet. Call fit(x) first.")
        x = self._chk(x)
        weighted_log_prob = self._estimate_log_gaussian_prob(x) + np.log(self.weights_)
        log_prob_norm = self._logsumexp(weighted_log_prob)
        return np.exp(weighted_log_prob - log_prob_norm)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def fit_predict(self, x):
        return self.fit(x).lab
