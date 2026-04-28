"""Top-level package exports for jklearn."""

from .linear_model import (
    LinearRegression,
    LogisticRegression,
    SoftmaxRegression,
    SVM,
)
from .ensemble import StackingClassifier, StackingRegressor
from .cluster import KMeans, GaussianMixture
from .naive_bayes import GaussianNB
from .neighbors import KNN
from .preprocessing import UnderSampling, OverSampling, SMOTE
from .tree import (
    AdaBoostClassifier,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GBDTRegressor,
    LightGBMRegressor,
    CatBoostRegressor,
)

__all__ = [
    # Linear models
    "LinearRegression",
    "LogisticRegression",
    "SoftmaxRegression",
    "SVM",
    # Clustering
    "KMeans",
    "GaussianMixture",
    # Neighbors
    "KNN",
    # Naive Bayes
    "GaussianNB",
    # Tree-based models
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "AdaBoostClassifier",
    "GBDTRegressor",
    "LightGBMRegressor",
    "CatBoostRegressor",
    # Ensemble
    "StackingClassifier",
    "StackingRegressor",
    # Preprocessing
    "UnderSampling",
    "OverSampling",
    "SMOTE",
]

__version__ = "0.1.0"
