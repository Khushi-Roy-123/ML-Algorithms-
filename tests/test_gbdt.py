import numpy as np

from jklearn.tree import GBDTRegressor


def test_gbdt_fit_predict_shape_and_error():
    x = np.linspace(-2.0, 2.0, 120).reshape(-1, 1)
    y = x.ravel() ** 2 + 0.2 * x.ravel() + 1.0

    m = GBDTRegressor(n=60, lr=0.1, md=2, ms=2)
    m.fit(x, y)
    p = m.predict(x)

    e = np.mean((y - p) ** 2)
    assert p.shape == y.shape
    assert e < 0.08


def test_gbdt_invalid_n_raises():
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])

    m = GBDTRegressor(n=0)
    try:
        m.fit(x, y)
        assert False
    except ValueError as ex:
        assert "n must be at least 1" in str(ex)