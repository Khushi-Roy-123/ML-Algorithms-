import numpy as np

from jklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def test_decision_tree_classifier_fit_and_predict():
    X = np.array(
        [
            [1.0, 1.0],
            [1.2, 1.1],
            [1.1, 0.9],
            [3.9, 4.0],
            [4.1, 4.2],
            [4.0, 3.8],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    m = DecisionTreeClassifier(max_depth=2)
    m.fit(X, y)
    p = m.predict(np.array([[1.05, 1.0], [4.05, 4.1]]))

    assert np.array_equal(p, np.array([0, 1]))


def test_decision_tree_classifier_predict_proba_sums_to_one():
    X = np.array(
        [
            [0.0],
            [0.1],
            [1.0],
            [1.1],
        ]
    )
    y = np.array([0, 0, 1, 1])

    m = DecisionTreeClassifier(max_depth=1)
    m.fit(X, y)
    pr = m.predict_proba(np.array([[0.05], [1.05]]))

    assert pr.shape == (2, 2)
    assert np.allclose(np.sum(pr, axis=1), 1.0)


def test_decision_tree_regressor_fit_and_predict():
    """Test basic regression tree fit and predict."""
    X = np.array(
        [
            [1.0],
            [1.5],
            [2.0],
            [4.0],
            [4.5],
            [5.0],
        ]
    )
    y = np.array([1.0, 1.2, 1.1, 4.0, 4.1, 3.9])

    m = DecisionTreeRegressor(max_depth=2)
    m.fit(X, y)
    preds = model.predict(np.array([[1.2], [4.3]]))

    assert preds.shape == (2,)
    assert isinstance(preds[0], (float, np.floating))
    # First prediction should be close to ~1.0, second close to ~4.0
    assert 0.8 < preds[0] < 1.5
    assert 3.8 < preds[1] < 4.5


def test_decision_tree_regressor_single_sample():
    """Test regression tree with single sample prediction."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X, y)
    pred = model.predict(np.array([[2.5]]))

    assert pred.shape == (1,)
    assert 2.0 <= pred[0] <= 3.0


def test_decision_tree_regressor_fits_simple_data():
    """Test regression tree fits simple linear data well."""
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = 2 * X.ravel() + 1

    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X, y)
    preds = model.predict(X)

    # Predictions should be very close to actual values on training data
    assert np.allclose(preds, y, atol=0.5)