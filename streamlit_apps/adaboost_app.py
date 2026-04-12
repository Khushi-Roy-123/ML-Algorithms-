import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from jklearn.tree import AdaBoostClassifier


class A:
    def __init__(self):
        self.t = "AdaBoost"

    def d(self, n):
        x, y = make_blobs(n_samples=n, n_features=2, centers=2, cluster_std=1.6, random_state=13)
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
        a.scatter(x[:, 0], x[:, 1], c=y, s=20)
        a.set_xlabel("x1")
        a.set_ylabel("x2")
        st.pyplot(f)

    def r(self):
        st.title(self.t)
        n = st.sidebar.slider("n", 80, 600, 260)
        ne = st.sidebar.slider("n_est", 2, 80, 20)
        lr = st.sidebar.slider("lr", 0.05, 2.0, 1.0)
        md = st.sidebar.slider("md", 1, 3, 1)
        x, y = self.d(n)
        xt, xv, yt, yv = train_test_split(x, y, test_size=0.25, random_state=42)
        m = AdaBoostClassifier(n_estimators=ne, learning_rate=lr, max_depth=md, random_state=42)
        m.fit(xt, yt)
        ypv = m.predict(xv)
        st.write("acc", float(accuracy_score(yv, ypv)))
        self.p(x, y, m)


if __name__ == "__main__":
    A().r()
