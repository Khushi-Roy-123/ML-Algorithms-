# Regression Tree Implementation

## Overview

The `DecisionTreeRegressor` is a machine learning algorithm that builds a decision tree to predict continuous target values (regression). It uses **Mean Squared Error (MSE)** as the criterion to recursively split features at each node, creating a binary tree structure where leaf nodes predict the mean of target values in their partition.

## How It Works

### Algorithm Steps

1. **Initialization**: Start with all training data at the root node
2. **Feature Selection**: For each feature, find the threshold that minimizes MSE of resulting splits
3. **Recursive Splitting**: Split the data into left (≤ threshold) and right (> threshold) partitions
4. **Termination**: Stop when:
   - Maximum depth is reached
   - Node has fewer samples than `min_samples_split`
   - Information gain is below `min_impurity_decrease`
5. **Prediction**: For a new sample, traverse the tree from root to leaf and return the mean value stored in that leaf

### Key Formula: MSE Gain

For a split, the information gain is calculated as:
$$\text{Gain} = \text{MSE}_{\text{parent}} - \left(\frac{n_{\text{left}}}{n} \cdot \text{MSE}_{\text{left}} + \frac{n_{\text{right}}}{n} \cdot \text{MSE}_{\text{right}}\right)$$

Where:
- $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$
- $\bar{y}$ is the mean of target values in the partition

## API Reference

### DecisionTreeRegressor

```python
from jklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    max_depth=None,              # Maximum tree depth (None = unlimited)
    min_samples_split=2,         # Minimum samples to split a node
    min_impurity_decrease=0.0    # Minimum information gain to split
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | int or None | None | Maximum depth of the tree. Often set to control overfitting |
| `min_samples_split` | int | 2 | Minimum number of samples required to split an internal node |
| `min_impurity_decrease` | float | 0.0 | Minimum information gain required to split a node |

#### Methods

**fit(X, y)**
- Builds the decision tree using training data
- `X`: 2D array of shape (n_samples, n_features)
- `y`: 1D array of shape (n_samples,) containing target values
- Returns: self (for method chaining)

**predict(X)**
- Makes predictions on new data
- `X`: 2D array of shape (n_samples, n_features) or 1D array for single sample
- Returns: 1D array of predicted values

## Usage Examples

### Basic Regression

```python
import numpy as np
from jklearn.tree import DecisionTreeRegressor

# Generate sample data
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

# Create and fit the model
model = DecisionTreeRegressor(max_depth=3)
model.fit(X, y)

# Make predictions
predictions = model.predict(np.array([[2.5], [4.5]]))
print(predictions)  # Output: [~5.0, ~9.0]
```

### Multi-feature Regression

```python
# Data with multiple features
X = np.random.rand(100, 3) * 10
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100) * 0.5

# Split into train/test
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Train model
model = DecisionTreeRegressor(max_depth=5, min_samples_split=3)
model.fit(X_train, y_train)

# Evaluate
train_pred = model.predict(X_train)
train_mse = np.mean((train_pred - y_train) ** 2)
print(f"Training MSE: {train_mse}")
```

### Controlling Overfitting

```python
# Use max_depth to prevent overfitting
model = DecisionTreeRegressor(
    max_depth=4,              # Limit tree depth
    min_samples_split=5,      # Require at least 5 samples to split
    min_impurity_decrease=0.01 # Require minimum gain to split
)
model.fit(X_train, y_train)
```

## Key Characteristics

### Advantages
- ✅ No feature scaling required (tree-based)
- ✅ Captures non-linear relationships well
- ✅ Fast predictions (O(log n) traversals)
- ✅ Interprets feature importance (see which features are used for splits)
- ✅ No assumptions about data distribution

### Disadvantages
- ❌ Prone to overfitting without proper depth/sample constraints
- ❌ Greedy algorithm (may not find globally optimal tree)
- ❌ Unstable (small data changes can result in very different trees)
- ❌ Generally worse than ensemble methods (Random Forests, Gradient Boosting)

## Hyperparameter Tuning Guide

| Hyperparameter | Effect of Increase | When to Increase |
|---|---|---|
| **max_depth** | More complex tree, higher variance | When model underfits |
| **min_samples_split** | Simpler tree, lower variance | When model overfits |
| **min_impurity_decrease** | Simpler tree, lower variance | When model overfits |

### Example: Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
}

# Note: This example shows the pattern; you may need to adapt based on your chosen framework
best_params = {'max_depth': 5, 'min_samples_split': 5}
model = DecisionTreeRegressor(**best_params)
```

## Performance Considerations

### Training Complexity
- **Time**: O(n × m × log n) where n = samples, m = features
- **Space**: O(log n) for recursion depth

### Prediction Complexity
- **Time**: O(log n) to traverse tree
- **Space**: O(1)

## Testing

All functionality is tested with:

```bash
python -m pytest tests/test_decision_tree.py -v
```

Tests cover:
- Basic fit and predict functionality
- Single sample predictions
- Fitting simple synthetic data
- Handling edge cases

## See Also

- Decision Tree Classifier: `jklearn.tree.DecisionTreeClassifier`
- Other regression methods: `LinearRegression`, `PolynomialRegression`, `LassoRegression`, `RidgeRegression`
- Ensemble methods: RandomForestRegressor, GradientBoostingRegressor (coming soon)

## File Structure

```
jklearn/
└── tree/
    ├── __init__.py                 # Module exports
    └── decision_tree.py            # DecisionTreeRegressor implementation
examples/
└── regression_tree_example.py      # Usage examples
tests/
└── test_decision_tree.py           # Unit tests
```

## References

1. Breiman et al. "Classification and Regression Trees" (CART algorithm)
2. Friedman, J. H. (2001). "Greedy function approximation: A gradient boosting machine"
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"
