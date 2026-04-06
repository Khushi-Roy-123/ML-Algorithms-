import numpy as np

from jklearn.tree import ExtraTreesClassifier, ExtraTreesRegressor


def test_extra_trees_classifier_fit_and_predict():
    X = np.array(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [1.2, 1.2],
            [4.0, 4.1],
            [4.2, 3.9],
            [3.9, 4.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = ExtraTreesClassifier(n_estimators=25, max_depth=3, random_state=0)
    model.fit(X, y)

    preds = model.predict(np.array([[1.05, 1.0], [4.05, 4.0]]))

    assert np.array_equal(preds, np.array([0, 1]))


def test_extra_trees_classifier_predict_proba_sums_to_one():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([0, 0, 1, 1])

    model = ExtraTreesClassifier(n_estimators=20, max_depth=2, random_state=1)
    model.fit(X, y)
    probs = model.predict_proba(np.array([[0.05], [1.05]]))

    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_extra_trees_regressor_fit_and_predict():
    X = np.array([[1.0], [1.5], [2.0], [4.0], [4.5], [5.0]])
    y = np.array([1.0, 1.2, 1.1, 4.0, 4.1, 3.9])

    model = ExtraTreesRegressor(n_estimators=30, max_depth=4, random_state=0)
    model.fit(X, y)
    preds = model.predict(np.array([[1.2], [4.3]]))

    assert preds.shape == (2,)
    assert isinstance(preds[0], (float, np.floating))
    assert 0.8 < preds[0] < 1.5
    assert 3.8 < preds[1] < 4.5