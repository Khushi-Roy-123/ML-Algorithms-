import streamlit as st

from adaboost_app import A as AB
from decision_tree_classifier_app import A as DTC
from decision_tree_regressor_app import A as DTR
from elastic_net_regression_app import A as ENR
from extra_trees_classifier_app import A as ETC
from extra_trees_regressor_app import A as ETR
from gaussian_nb_app import A as GNB
from gbdt_regressor_app import A as GBDT
from knn_app import A as KNN
from lasso_regression_app import A as LAR
from linear_regression_app import A as LR
from logistic_regression_app import A as LOG
from polynomial_regression_app import A as PR
from ridge_regression_app import A as RIR
from svm_app import A as SVM


class H:
    def __init__(self):
        self.m = {
            "Linear Regression": LR,
            "Polynomial Regression": PR,
            "Ridge Regression": RIR,
            "Lasso Regression": LAR,
            "Elastic Net Regression": ENR,
            "Logistic Regression": LOG,
            "SVM": SVM,
            "KNN": KNN,
            "Gaussian NB": GNB,
            "Decision Tree Classifier": DTC,
            "Decision Tree Regressor": DTR,
            "AdaBoost": AB,
            "Extra Trees Classifier": ETC,
            "Extra Trees Regressor": ETR,
            "GBDT Regressor": GBDT,
        }

    def r(self):
        st.sidebar.title("Algorithms")
        k = st.sidebar.selectbox("model", list(self.m.keys()))
        self.m[k]().r()


if __name__ == "__main__":
    H().r()
