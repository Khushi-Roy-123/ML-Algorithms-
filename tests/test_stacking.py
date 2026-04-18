import numpy as np

from jklearn.ensemble import StackingClassifier, StackingRegressor
from jklearn.linear_model import LinearRegression
from jklearn.naive_bayes import GaussianNB
from jklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def test_stacking_classifier_fit_predict():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.1],
            [0.1, 0.9],
            [0.9, 0.2],
            [0.8, 0.9],
        ]
    )
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

    model = StackingClassifier(
        estimators=[
            ("tree", DecisionTreeClassifier(max_depth=2)),
            ("nb", GaussianNB()),
        ],
        final_estimator=DecisionTreeClassifier(max_depth=2),
        n_folds=4,
        random_state=7,
    )
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1})


def test_stacking_regressor_fit_predict():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0, 11.0, 13.0])

    model = StackingRegressor(
        estimators=[
            ("lin1", LinearRegression()),
            ("lin2", DecisionTreeRegressor(max_depth=2)),
        ],
        final_estimator=LinearRegression(),
        n_folds=3,
        random_state=11,
    )
    model.fit(X, y)
    preds = model.predict(np.array([[7.0], [8.0]]))

    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))