import numpy as np

from jklearn.linear_model import LinearRegression, LogisticRegression, SoftmaxRegression, SVM


def test_linear_regression_normal_equation_fit_predict():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegression(solver="normal")
    model.fit(X, y)
    preds = model.predict(np.array([[5.0], [6.0]]))

    assert np.allclose(model.coef_, [2.0], atol=1e-8)
    assert np.isclose(model.intercept_, 1.0, atol=1e-8)
    assert np.allclose(preds, [11.0, 13.0], atol=1e-8)


def test_linear_regression_gradient_descent_fit():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])

    model = LinearRegression(solver="gd", lr=0.05, epochs=5000)
    model.fit(X, y)

    assert np.allclose(model.coef_, [2.0], atol=1e-2)
    assert np.isclose(model.intercept_, 1.0, atol=1e-2)


def test_logistic_regression_binary_prediction_shape():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    y = np.array([0, 1, 1, 0])

    model = LogisticRegression(lr=0.1, epochs=2000)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1})


def test_softmax_regression_multiclass_fit_predict():
    X = np.array(
        [
            [2.0, 2.0],
            [2.2, 1.8],
            [1.8, 2.1],
            [-2.0, -2.0],
            [-2.1, -1.8],
            [-1.8, -2.2],
            [2.0, -2.0],
            [2.2, -1.8],
            [1.8, -2.1],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    model = SoftmaxRegression(lr=0.2, epochs=2500)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1, 2})
    assert np.mean(preds == y) >= 0.95


def test_softmax_regression_predict_proba_rows_sum_to_one():
    X = np.array(
        [
            [1.0, 1.0],
            [1.2, 0.8],
            [-1.0, -1.0],
            [-1.2, -0.8],
            [1.0, -1.0],
            [1.2, -0.9],
        ]
    )
    y = np.array([0, 0, 1, 1, 2, 2])

    model = SoftmaxRegression(lr=0.15, epochs=2000).fit(X, y)
    probs = model.predict_proba(X)

    assert probs.shape == (X.shape[0], 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)


def test_svm_linear_separable_fit_predict():
    x = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -1.0],
            [-1.0, -1.5],
            [1.0, 1.5],
            [1.5, 1.0],
            [2.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    m = SVM(lr=0.001, c=1.0, ep=4000)
    m.fit(x, y)
    p = m.predict(x)
    s = m.decision_function(x)
    t = np.where(y <= 0, -1, 1)

    assert np.array_equal(p, y)
    assert np.all(t * s > 0)


def test_svm_rbf_xor_fit_predict():
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 1, 1, 0])

    m = SVM(c=10.0, ep=3000, ker="rbf", gam=2.0)
    m.fit(x, y)
    p = m.predict(x)

    assert np.array_equal(p, y)


def test_svm_poly_xor_fit_predict():
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 1, 1, 0])

    m = SVM(c=10.0, ep=3000, ker="poly", deg=2, gam=1.0, r=1.0)
    m.fit(x, y)
    p = m.predict(x)

    assert np.array_equal(p, y)
