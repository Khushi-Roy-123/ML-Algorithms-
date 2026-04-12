import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from jklearn.linear_model import LinearRegression


class A:
    def __init__(self):
        self.t = "Polynomial Regression"

    def d(self, n, s):
        r = np.random.default_rng(21)
        x = r.uniform(-3, 3, size=(n, 1))
        y = 4 + 2 * x[:, 0] - 1.4 * (x[:, 0] ** 2) + 0.7 * (x[:, 0] ** 3) + r.normal(0, s, n)
        return x, y

    def f(self, x, p):
        z = [x[:, 0] ** i for i in range(1, p + 1)]
        return np.column_stack(z)

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
        n = st.sidebar.slider("n", 80, 700, 250)
        s = st.sidebar.slider("noise", 0.1, 3.0, 1.0)
        p = st.sidebar.slider("deg", 2, 6, 3)
        sv = st.sidebar.selectbox("solver", ["normal", "gd"])
        lr = st.sidebar.slider("lr", 0.0001, 0.1, 0.01)
        ep = st.sidebar.slider("ep", 300, 6000, 1500)
        x, y = self.d(n, s)
        z = self.f(x, p)
        xt, xv, yt, yv = train_test_split(z, y, test_size=0.25, random_state=42)
        m = LinearRegression(solver=sv, lr=lr, epochs=ep)
        m.fit(xt, yt)
        yp = m.predict(z)
        ypv = m.predict(xv)
        st.write("mse", float(mean_squared_error(yv, ypv)))
        st.write("r2", float(r2_score(yv, ypv)))
        self.p(x, y, yp)


if __name__ == "__main__":
    A().r()
