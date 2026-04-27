import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_moons, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from jklearn.ensemble import StackingClassifier, StackingRegressor
from jklearn.linear_model import LinearRegression
from jklearn.naive_bayes import GaussianNB
from jklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class A:
    def __init__(self):
        self.t = "Stacking"

    def _plot_classifier_boundary(self, x, y, model):
        x0_min, x0_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
        x1_min, x1_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

        grid_x0 = np.arange(x0_min, x0_max, 0.03)
        grid_x1 = np.arange(x1_min, x1_max, 0.03)
        xx, yy = np.meshgrid(grid_x0, grid_x1)
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = model.predict(grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.contourf(xx, yy, zz, alpha=0.35)
        ax.scatter(x[:, 0], x[:, 1], c=y, s=18)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Stacking Classifier Decision Surface")
        st.pyplot(fig)

    def _run_classifier_demo(self):
        st.subheader("Classifier")
        n_samples = st.sidebar.slider("classifier samples", 120, 1500, 600)
        noise = st.sidebar.slider("classifier noise", 0.05, 0.5, 0.2)
        folds_cls = st.sidebar.slider("classifier folds", 2, 8, 5)

        x, y = make_moons(n_samples=n_samples, noise=noise, random_state=7)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

        model = StackingClassifier(
            estimators=[
                ("tree", DecisionTreeClassifier(max_depth=4, random_state=13)),
                ("nb", GaussianNB()),
            ],
            final_estimator=DecisionTreeClassifier(max_depth=5, random_state=21),
            n_folds=folds_cls,
            random_state=17,
        )

        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test, pred)

        st.write("accuracy", float(acc))
        self._plot_classifier_boundary(x, y, model)

    def _run_regressor_demo(self):
        st.subheader("Regressor")
        n_samples = st.sidebar.slider("regressor samples", 120, 2000, 700)
        noise = st.sidebar.slider("regressor noise", 2.0, 30.0, 10.0)
        folds_reg = st.sidebar.slider("regressor folds", 2, 8, 5)

        x, y = make_regression(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            noise=noise,
            random_state=19,
        )

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

        model = StackingRegressor(
            estimators=[
                ("lin", LinearRegression()),
                ("tree", DecisionTreeRegressor(max_depth=5, random_state=31)),
            ],
            final_estimator=LinearRegression(),
            n_folds=folds_reg,
            random_state=29,
        )

        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        st.write("rmse", float(rmse))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(y_test, pred, s=16, alpha=0.7)
        line_min = min(float(np.min(y_test)), float(np.min(pred)))
        line_max = max(float(np.max(y_test)), float(np.max(pred)))
        ax.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=1.5)
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
        ax.set_title("Stacking Regressor: Predicted vs Actual")
        st.pyplot(fig)

    def r(self):
        st.title(self.t)
        mode = st.sidebar.selectbox("mode", ["classifier", "regressor"])
        if mode == "classifier":
            self._run_classifier_demo()
        else:
            self._run_regressor_demo()


if __name__ == "__main__":
    A().r()
