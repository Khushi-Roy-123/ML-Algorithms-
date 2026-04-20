import numpy as np

from jklearn.tree import CatBoostRegressor


def test_catboost_fit_predict_shape_and_error():
    rng = np.random.default_rng(42)
    n = 220

    num = rng.normal(0.0, 1.0, size=n)
    cats = rng.choice(np.array(["a", "b", "c"], dtype=object), size=n)

    cat_effect = {"a": 1.2, "b": -0.5, "c": 0.3}
    y = 1.8 * num + np.array([cat_effect[c] for c in cats]) + 0.05 * rng.normal(size=n)

    X = np.column_stack([num.astype(object), cats])

    m = CatBoostRegressor(
        n_estimators=120,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=4,
        l2_leaf_reg=2.0,
        cat_features=[1],
        random_state=0,
    )
    m.fit(X, y)
    p = m.predict(X)

    e = np.mean((y - p) ** 2)
    assert p.shape == y.shape
    assert e < 0.18


def test_catboost_unknown_category_predicts():
    X_train = np.array(
        [
            [0.0, "x"],
            [1.0, "y"],
            [2.0, "x"],
            [3.0, "z"],
        ],
        dtype=object,
    )
    y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

    m = CatBoostRegressor(n_estimators=40, cat_features=[1], random_state=1)
    m.fit(X_train, y_train)

    X_test = np.array([[1.5, "never_seen"]], dtype=object)
    p = m.predict(X_test)

    assert p.shape == (1,)
    assert np.isfinite(p[0])


def test_catboost_invalid_n_estimators_raises():
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])

    m = CatBoostRegressor(n_estimators=0)
    try:
        m.fit(x, y)
        assert False
    except ValueError as ex:
        assert "n_estimators must be at least 1" in str(ex)


def test_catboost_predict_before_fit_raises():
    m = CatBoostRegressor()
    try:
        m.predict(np.array([[1.0, "a"]], dtype=object))
        assert False
    except ValueError as ex:
        assert "Model is not fitted yet" in str(ex)