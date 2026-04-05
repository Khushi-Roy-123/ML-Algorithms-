"""Example demonstrating Decision Tree Regressor."""

import numpy as np
from jklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def example_basic_regression():
    """Basic example: fitting a regression tree to synthetic data."""
    print("=" * 60)
    print("Example 1: Basic Regression Tree")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(X).ravel() + np.random.randn(80) * 0.15
    
    # Create and fit model
    model = DecisionTreeRegressor(max_depth=4, min_samples_split=5)
    model.fit(X, y)
    
    # Make predictions
    X_test = np.linspace(0, 5, 100).reshape(-1, 1)
    y_pred = model.predict(X_test)
    
    print(f"Training set size: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Max depth: {model.max_depth}")
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print()


def example_multifeature_regression():
    """Example: regression with multiple features."""
    print("=" * 60)
    print("Example 2: Multifeature Regression")
    print("=" * 60)
    
    # Generate data with multiple features
    np.random.seed(42)
    X = np.random.rand(100, 3) * 10
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.5
    
    # Train-test split (simple)
    split_idx = 80
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and fit model
    model = DecisionTreeRegressor(max_depth=5, min_samples_split=3)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mse = np.mean((train_pred - y_train) ** 2)
    test_mse = np.mean((test_pred - y_test) ** 2)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print()


def example_max_depth_effect():
    """Example: comparing different max_depth values."""
    print("=" * 60)
    print("Example 3: Effect of Max Depth Parameter")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    X = np.sort(10 * np.random.rand(50, 1), axis=0)
    y = (X.ravel() ** 2) / 10 + np.random.randn(50) * 0.5
    
    depths = [1, 3, 5, 10]
    
    print(f"Dataset size: {X.shape[0]}")
    print(f"\nComparing different max_depth values:")
    print("-" * 50)
    print(f"{'Max Depth':<15} {'Train MSE':<15} {'Num Leaves (est)':<15}")
    print("-" * 50)
    
    for depth in depths:
        model = DecisionTreeRegressor(max_depth=depth)
        model.fit(X, y)
        
        train_pred = model.predict(X)
        train_mse = np.mean((train_pred - y) ** 2)
        
        # Estimate number of leaves (roughly 2^depth for full tree)
        est_leaves = min(2 ** depth, X.shape[0])
        
        print(f"{depth:<15} {train_mse:<15.6f} {est_leaves:<15}")
    
    print()


def example_simple_linear_fit():
    """Example: fitting a simple linear relationship."""
    print("=" * 60)
    print("Example 4: Fitting Simple Linear Relationship")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.5
    
    # Fit model
    model = DecisionTreeRegressor(max_depth=4)
    model.fit(X, y)
    
    # Make predictions
    test_sample = np.array([[5.0, 3.0]])
    pred = model.predict(test_sample)
    expected = 3 * 5.0 + 2 * 3.0  # = 21.0
    
    print(f"Test sample features: {test_sample[0]}")
    print(f"Expected value (approx): {expected:.2f}")
    print(f"Model prediction: {pred[0]:.6f}")
    print(f"Error: {abs(pred[0] - expected):.6f}")
    print()


if __name__ == "__main__":
    example_basic_regression()
    example_multifeature_regression()
    example_max_depth_effect()
    example_simple_linear_fit()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
