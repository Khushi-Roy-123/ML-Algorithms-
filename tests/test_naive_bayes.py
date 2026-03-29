import numpy as np

from jklearn.naive_bayes import GaussianNB


def test_gaussian_nb_fit_and_predict():
    X = np.array(
        [
            [1.0, 2.0],
            [1.1, 1.9],
            [4.0, 5.0],
            [4.2, 5.1],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = GaussianNB()
    model.fit(X, y)
    preds = model.predict(np.array([[1.05, 2.0], [4.1, 5.0]]))

    assert np.array_equal(preds, np.array([0, 1]))


def test_gaussian_nb_predict_proba_sums_to_one():
    X = np.array(
        [
            [0.0],
            [0.2],
            [1.0],
            [1.2],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = GaussianNB()
    model.fit(X, y)
    probs = model.predict_proba(np.array([[0.1], [1.1]]))

    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
