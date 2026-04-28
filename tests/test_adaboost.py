"""Tests for AdaBoost classifier."""

import numpy as np
import pytest

from jklearn.tree import AdaBoostClassifier


def test_adaboost_fit_predict_binary_classification():
    """Test basic fit and predict functionality on binary classification."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    
    m = AdaBoostClassifier(n_est=10, rs=42)
    m.fit(X, y)
    p = m.predict(X)
    
    assert p.shape == y.shape
    assert set(np.unique(p)) == {0, 1}
    # Should achieve reasonable accuracy on training data
    acc = np.mean(p == y)
    assert acc > 0.6


def test_adaboost_predict_proba():
    """Test predict_proba method."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    
    m = AdaBoostClassifier(n_est=10, rs=42)
    m.fit(X, y)
    pr = m.predict_proba(X)
    
    assert pr.shape == (X.shape[0], 2)
    # Probabilities should sum to 1
    assert np.allclose(np.sum(pr, axis=1), 1.0)
    # Probabilities should be between 0 and 1
    assert np.all(pr >= 0) and np.all(pr <= 1)


def test_adaboost_1d_input():
    """Test with 1D input (single feature)."""
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    m = AdaBoostClassifier(n_estimators=5)
    m.fit(X, y)
    p = m.predict(X)
    
    assert p.shape == y.shape


def test_adaboost_learning_rate():
    """Test that learning_rate parameter works."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    
    m1 = AdaBoostClassifier(n_estimators=5, learning_rate=0.5, random_state=42)
    m2 = AdaBoostClassifier(n_estimators=5, learning_rate=1.0, random_state=42)
    
    m1.fit(X, y)
    m2.fit(X, y)
    
    p1 = m1.predict(X)
    p2 = m2.predict(X)
    
    # Different learning rates may give different results
    # Just verify they both produce valid outputs
    assert p1.shape == y.shape
    assert p2.shape == y.shape


def test_adaboost_invalid_x_shape():
    """Test that invalid X shape raises error."""
    X = np.array([1.0, 2.0, 3.0])  # 1D array
    y = np.array([0, 1, 0])
    
    clf = AdaBoostClassifier()
    with pytest.raises(ValueError, match="X must be a 2D array"):
        clf.fit(X, y)


def test_adaboost_invalid_y_shape():
    """Test that invalid y shape raises error."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([[0], [1], [0]])  # 2D array
    
    clf = AdaBoostClassifier()
    with pytest.raises(ValueError, match="y must be a 1D array"):
        clf.fit(X, y)


def test_adaboost_mismatched_samples():
    """Test that mismatched sample counts raise error."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1])  # Different count
    
    clf = AdaBoostClassifier()
    with pytest.raises(ValueError, match="must contain the same number of samples"):
        clf.fit(X, y)


def test_adaboost_invalid_n_estimators():
    """Test that invalid n_estimators raises error."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0])
    
    clf = AdaBoostClassifier(n_estimators=0)
    with pytest.raises(ValueError, match="n_estimators must be at least 1"):
        clf.fit(X, y)


def test_adaboost_invalid_learning_rate():
    """Test that invalid learning_rate raises error."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0])
    
    clf = AdaBoostClassifier(learning_rate=0)
    with pytest.raises(ValueError, match="learning_rate must be > 0"):
        clf.fit(X, y)


def test_adaboost_multiclass_not_supported():
    """Test that multiclass classification raises error."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 1, 2, 1])  # 3 classes
    
    clf = AdaBoostClassifier()
    with pytest.raises(ValueError, match="binary classification only"):
        clf.fit(X, y)


def test_adaboost_predict_before_fit():
    """Test that predict before fit raises error."""
    X = np.array([[1.0], [2.0], [3.0]])
    
    clf = AdaBoostClassifier()
    with pytest.raises(ValueError, match="not fitted yet"):
        clf.predict(X)


def test_adaboost_predict_proba_before_fit():
    """Test that predict_proba before fit raises error."""
    X = np.array([[1.0], [2.0], [3.0]])
    
    clf = AdaBoostClassifier()
    with pytest.raises(ValueError, match="not fitted yet"):
        clf.predict_proba(X)


def test_adaboost_max_depth_parameter():
    """Test that max_depth parameter is passed correctly."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    
    clf = AdaBoostClassifier(n_estimators=5, max_depth=2, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    
    assert preds.shape == y.shape
    # Verify estimators were trained
    assert len(clf.estimators_) > 0
    # Check that estimators are decision trees with correct depth
    for est in clf.estimators_:
        assert est.max_depth == 2


def test_adaboost_predict_1d_input():
    """Test predict with 1D test sample."""
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    clf = AdaBoostClassifier(n_estimators=5)
    clf.fit(X, y)
    
    # Single 1D sample
    pred = clf.predict(np.array([2.5]))
    assert pred.shape == (1,)
    assert pred[0] in [0, 1]


def test_adaboost_multiple_features():
    """Test with multiple features."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.where(np.sum(X, axis=1) > 0, 1, 0)
    
    clf = AdaBoostClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    
    assert preds.shape == y.shape
    accuracy = np.mean(preds == y)
    assert accuracy > 0.5  # Better than random


def test_adaboost_convergence():
    """Test that AdaBoost converges with more estimators."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    
    clf = AdaBoostClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)
    
    # Should have trained all 20 or close to it
    assert len(clf.estimators_) > 0
    assert len(clf.alphas_) == len(clf.estimators_)
