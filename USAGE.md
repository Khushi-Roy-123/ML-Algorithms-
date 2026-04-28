# Using jklearn as a Library

## Installation

### Option 1: Install from local directory (development mode)
```bash
cd path/to/Algorithms
pip install -e .
```

### Option 2: Direct import (add to PYTHONPATH)
Add the Algorithms folder to your Python path, then import jklearn.

---

## Quick Start

### Import algorithms
```python
from jklearn import (
    LinearRegression,
    LogisticRegression,
    KNN,
    KMeans,
    DecisionTreeClassifier,
    SVM,
    GaussianNB,
)
```

### Linear Models
```python
from jklearn import LinearRegression, RidgeRegression, LassoRegression

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Ridge Regression (L2 regularization)
ridge = RidgeRegression(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

# Lasso Regression (L1 regularization)
lasso = LassoRegression(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
```

### Classification
```python
from jklearn import LogisticRegression, KNN, SVM, GaussianNB

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# K-Nearest Neighbors
knn = KNN(k=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Support Vector Machine
svm = SVM(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
```

### Clustering
```python
from jklearn import KMeans, GaussianMixture

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X)

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
labels = gmm.predict(X)
```

### Tree-Based Models
```python
from jklearn import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreesClassifier,
    GBDTRegressor,
    AdaBoostClassifier,
)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Extra Trees
et = ExtraTreesClassifier(n_trees=10)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)

# Gradient Boosting
gbdt = GBDTRegressor(n_trees=100, learning_rate=0.1)
gbdt.fit(X_train, y_train)
y_pred = gbdt.predict(X_test)

# AdaBoost
ada = AdaBoostClassifier(n_estimators=50)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
```

### Preprocessing
```python
from jklearn import UnderSampling, OverSampling, SMOTE

# Handle imbalanced datasets
under = UnderSampling()
X_balanced, y_balanced = under.fit_resample(X, y)

over = OverSampling()
X_balanced, y_balanced = over.fit_resample(X, y)

smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X, y)
```

---

## Example: Complete ML Pipeline

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from jklearn import LogisticRegression, KNN, SVM

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN (k=3)': KNN(k=3),
    'SVM': SVM(kernel='rbf'),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}: {score:.2%}")
```

---

## Project Structure

```
jklearn/
├── __init__.py              # Main package exports
├── linear_model/            # Linear regression/classification
│   ├── linear_regression.py
│   ├── LogisticRegression.py
│   ├── RidgeRegression.py
│   ├── LassoRegression.py
│   ├── ElasticNetRegression.py
│   ├── PolynomialRegression.py
│   ├── softmax_regression.py
│   └── svm.py
├── tree/                    # Tree-based models
│   ├── decision_tree.py
│   ├── extra_trees.py
│   ├── adaboost.py
│   ├── gbdt.py
│   ├── catboost.py
│   ├── lightgbm.py
│   └── random_forest.py
├── cluster/                 # Clustering algorithms
│   ├── kmeans.py
│   └── gmm.py
├── neighbors/               # Nearest neighbors
│   └── knn.py
├── naive_bayes/             # Naive Bayes classifiers
│   └── gaussian_naive_bayes.py
├── ensemble/                # Ensemble methods
│   └── stacking.py
├── preprocessing/           # Data preprocessing
│   ├── scaler.py
│   └── sampling.py
└── utils/                   # Utility functions
    ├── distance.py
    └── metrics.py
```

---

## Requirements

- Python >= 3.8
- NumPy >= 1.23
- (Optional) Matplotlib >= 3.7 for visualizations
- (Optional) Scikit-learn >= 1.3 for utilities
- (Optional) Streamlit >= 1.32 for web apps

---

## Development

To install in development mode with all extras:

```bash
pip install -e ".[dev,streamlit]"
```

---

## Notes

- All algorithms are implemented from scratch for educational purposes
- Some algorithms may differ slightly from scikit-learn in interface or performance optimization
- See individual module docstrings for detailed algorithm documentation
