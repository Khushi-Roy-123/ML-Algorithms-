import numpy as np

from jklearn.tree import DecisionTreeClassifier


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

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)
    preds = model.predict(np.array([[1.05, 1.0], [4.05, 4.1]]))

    assert np.array_equal(preds, np.array([0, 1]))


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

    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X, y)
    probs = model.predict_proba(np.array([[0.05], [1.05]]))

    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)