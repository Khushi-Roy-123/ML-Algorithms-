import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from jklearn.linear_model import LinearRegression


class A:
    def __init__(self):
        self.t = "Linear Regression"

    def d(self, n, s):
        x, y = make_regression(n_samples=n, n_features=1, noise=s, random_state=42)
        return x, y

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
        n = st.sidebar.slider("n", 50, 500, 180)
        s = st.sidebar.slider("noise", 1, 40, 12)
        sv = st.sidebar.selectbox("solver", ["normal", "gd"])
        lr = st.sidebar.slider("lr", 0.0001, 0.1, 0.01)
        ep = st.sidebar.slider("ep", 200, 5000, 1200)
        x, y = self.d(n, s)
        xt, xv, yt, yv = train_test_split(x, y, test_size=0.2, random_state=42)
        m = LinearRegression(solver=sv, lr=lr, epochs=ep)
        m.fit(xt, yt)
        yp = m.predict(x)
        ypv = m.predict(xv)
        st.write("mse", float(mean_squared_error(yv, ypv)))
        st.write("r2", float(r2_score(yv, ypv)))
        self.p(x, y, yp)


if __name__ == "__main__":
    A().r()
