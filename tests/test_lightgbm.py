import numpy as np

from jklearn.tree import LightGBMRegressor


def test_lightgbm_fit_predict_shape_and_error():
    x = np.linspace(-3.0, 3.0, 180).reshape(-1, 1)
    y = np.sin(x.ravel()) + 0.1 * x.ravel()

    m = LightGBMRegressor(
        n_estimators=80,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=16,
        min_data_in_leaf=5,
        max_bins=32,
        reg_lambda=1.0,
        random_state=7,
    )
    m.fit(x, y)
    p = m.predict(x)

    e = np.mean((y - p) ** 2)
    assert p.shape == y.shape
    assert e < 0.04


def test_lightgbm_invalid_n_estimators_raises():
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])

    m = LightGBMRegressor(n_estimators=0)
    try:
        m.fit(x, y)
        assert False
    except ValueError as ex:
        assert "n_estimators must be at least 1" in str(ex)


def test_lightgbm_predict_before_fit_raises():
    m = LightGBMRegressor()
    try:
        m.predict(np.array([[1.0], [2.0]]))
        assert False
    except ValueError as ex:
        assert "Model is not fitted yet" in str(ex)