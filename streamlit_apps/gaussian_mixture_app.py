import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_blobs

from jklearn.cluster import GaussianMixture


class A:
    def __init__(self):
        self.t = "Gaussian Mixture Models (GMM)"

    def d(self, n, c, s):
        x, y = make_blobs(
            n_samples=n,
            n_features=2,
            centers=c,
            cluster_std=s,
            random_state=12,
        )
        return x, y

    def g(self, x):
        u = np.arange(x[:, 0].min() - 1, x[:, 0].max() + 1, 0.06)
        v = np.arange(x[:, 1].min() - 1, x[:, 1].max() + 1, 0.06)
        xx, yy = np.meshgrid(u, v)
        z = np.c_[xx.ravel(), yy.ravel()]
        return xx, yy, z

    def p(self, x, y, m):
        xx, yy, z = self.g(x)
        yp = m.predict(z).reshape(xx.shape)
        f, a = plt.subplots(figsize=(7, 4))
        a.contourf(xx, yy, yp, alpha=0.35)
        a.scatter(x[:, 0], x[:, 1], c=y, s=22)
        a.scatter(m.means_[:, 0], m.means_[:, 1], c="black", s=90, marker="x")
        a.set_xlabel("x1")
        a.set_ylabel("x2")
        st.pyplot(f)

    def r(self):
        st.title(self.t)
        st.caption("Probabilistic clustering using a mixture of Gaussian distributions.")

        n = st.sidebar.slider("n", 120, 1000, 360)
        c = st.sidebar.slider("components", 2, 6, 3)
        s = st.sidebar.slider("cluster_std", 0.2, 2.0, 0.9)
        ep = st.sidebar.slider("ep", 20, 400, 120)
        n_init = st.sidebar.slider("n_init", 1, 10, 3)
        tol = st.sidebar.slider("tol", 1e-6, 1e-2, 1e-4, format="%.6f")
        reg = st.sidebar.slider("reg_covar", 1e-8, 1e-2, 1e-6, format="%.8f")

        x, y = self.d(n, c, s)
        m = GaussianMixture(k=c, ep=ep, tol=tol, reg_covar=reg, n_init=n_init, seed=42)
        m.fit(x)
        yp = m.predict(x)

        st.write("lower_bound", float(m.lower_bound_))
        st.write("converged", bool(m.converged_))
        st.write("n_iter", int(m.n_iter_))
        st.write("weights", np.round(m.weights_, 4))
        st.write("means", np.round(m.means_, 4))
        st.write("cluster counts", np.bincount(yp, minlength=c))

        self.p(x, y, m)


if __name__ == "__main__":
    A().r()
