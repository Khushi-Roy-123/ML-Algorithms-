import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class M:
    def __init__(self, a=1.0):
        self.a = a
        self.w = None

    def fit(self, x, y):
        n = x.shape[1]
        i = np.eye(n)
        i[0, 0] = 0
        self.w = np.linalg.pinv(x.T @ x + self.a * i) @ (x.T @ y)
        return self

    def predict(self, x):
        return x @ self.w


class A:
    def __init__(self):
        self.t = "Ridge Regression"

    def d(self, n, s):
        x, y = make_regression(n_samples=n, n_features=1, noise=s, random_state=24)
        o = np.ones((x.shape[0], 1))
        z = np.c_[o, x]
        return x, z, y

    def p(self, x, y, yp):
        i = np.argsort(x[:, 0])
        f, a = plt.subplots(figsize=(7, 4))
        a.scatter(x[:, 0], y, s=20)
        a.plot(x[i, 0], yp[i], linewidth=2)
        a.set_xlabel("x")
        a.set_ylabel("y")
        st.pyplot(f)

    def r(self):
        st.title(self.t)
        n = st.sidebar.slider("n", 60, 600, 220)
        s = st.sidebar.slider("noise", 1, 50, 14)
        a = st.sidebar.slider("lam", 0.0, 20.0, 1.0)
        x, z, y = self.d(n, s)
        zt, zv, yt, yv = train_test_split(z, y, test_size=0.25, random_state=42)
        m = M(a=a)
        m.fit(zt, yt)
        yp = m.predict(z)
        ypv = m.predict(zv)
        st.write("mse", float(mean_squared_error(yv, ypv)))
        st.write("r2", float(r2_score(yv, ypv)))
        self.p(x, y, yp)


if __name__ == "__main__":
    A().r()
