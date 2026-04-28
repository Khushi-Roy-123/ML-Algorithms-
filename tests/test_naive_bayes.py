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

    m = GaussianNB()
    m.fit(X, y)
    p = m.predict(np.array([[1.05, 2.0], [4.1, 5.0]]))

    assert np.array_equal(p, np.array([0, 1]))


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

    m = GaussianNB()
    m.fit(X, y)
    pr = m.predict_proba(np.array([[0.1], [1.1]]))

    assert pr.shape == (2, 2)
    assert np.allclose(np.sum(pr, axis=1), 1.0)
