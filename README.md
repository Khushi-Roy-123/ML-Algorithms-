# jklearn - Machine Learning Algorithms Library

> A **from-scratch** machine learning algorithms library with 21+ classifiers, regressors, and clustering algorithms implemented in pure Python for educational purposes.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.23%2B-green)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

**jklearn** is a comprehensive machine learning library built from the ground up for learning and understanding ML algorithms. Each algorithm is implemented without relying on high-level ML libraries, giving you complete insight into how each method works.

**Perfect for:**
- Students learning ML fundamentals
- Understanding algorithm internals
- Educational projects and assignments
- Experimenting with different ML techniques

## Features

### Implemented Algorithms (21)

#### Linear Models (4)
- **Linear Regression** - Basic linear regression using normal equation
- **Logistic Regression** - Binary and multi-class classification
- **Softmax Regression** - Multinomial logistic regression
- **Support Vector Machine (SVM)** - Linear and kernel SVM

#### Tree-Based Models (8)
- **Decision Tree** - Classification and regression trees
- **Extra Trees** - Extremely randomized trees with multiple trees
- **AdaBoost** - Adaptive boosting classifier
- **Gradient Boosting (GBDT)** - Sequential tree boosting
- **CatBoost** - Categorical boosting
- **LightGBM** - Light gradient boosting machines
- **Random Forest** - Ensemble of decision trees (via Extra Trees)
- **Stacking** - Meta-learner ensemble method

#### Instance-Based Learning (1)
- **K-Nearest Neighbors (KNN)** - Classification and regression

#### Probabilistic Models (2)
- **Gaussian Naive Bayes** - Probabilistic classifier with Gaussian distribution
- **Gaussian Mixture Model (GMM)** - Unsupervised clustering with EM algorithm

#### Clustering (1)
- **K-Means** - Centroid-based clustering algorithm

#### Data Preprocessing (3)
- **UnderSampling** - Reduce majority class samples
- **OverSampling** - Increase minority class samples
- **SMOTE** - Synthetic Minority Oversampling Technique

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Khushi-Roy-123/jklearn.git
cd jklearn

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from jklearn import LinearRegression, KNN, DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a classifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.2%}")
```

---

## Documentation

### Example: Multi-Algorithm Comparison

```python
import numpy as np
from jklearn import LogisticRegression, KNN, SVM, GaussianNB
from sklearn.datasets import make_classification

# Create sample data
X, y = make_classification(n_samples=200, n_features=10, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN (k=5)': KNN(k=5),
    'SVM (linear)': SVM(kernel='linear'),
    'Gaussian NB': GaussianNB(),
}

# Train and evaluate
for name, model in models.items():
    model.fit(X[:150], y[:150])
    score = (model.predict(X[150:]) == y[150:]).mean()
    print(f"{name}: {score:.2%}")
```

### Example: Clustering

```python
from jklearn import KMeans
import numpy as np

# Generate sample data
X = np.random.randn(100, 2)

# Fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.predict(X)
print(f"Cluster assignments: {labels}")
```

### Example: Data Preprocessing

```python
from jklearn import SMOTE
from sklearn.datasets import make_classification

# Imbalanced dataset
X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)

# Apply SMOTE
smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X, y)

print(f"Original distribution: {np.bincount(y)}")
print(f"Balanced distribution: {np.bincount(y_balanced)}")
```

---

## Project Structure

```
jklearn/
├── __init__.py                  # Main package exports
├── linear_model/                # Linear regression & classification
│   ├── linear_regression.py     # Linear Regression
│   ├── LogisticRegression.py    # Logistic Regression
│   ├── softmax_regression.py    # Softmax Regression
│   └── svm.py                   # Support Vector Machine
├── tree/                        # Tree-based algorithms
│   ├── decision_tree.py         # Decision Tree
│   ├── extra_trees.py           # Extra Trees
│   ├── adaboost.py              # AdaBoost
│   ├── gbdt.py                  # Gradient Boosting
│   ├── catboost.py              # CatBoost
│   └── lightgbm.py              # LightGBM
├── cluster/                     # Clustering algorithms
│   ├── kmeans.py                # K-Means
│   └── gmm.py                   # Gaussian Mixture Model
├── neighbors/                   # Instance-based learning
│   └── knn.py                   # K-Nearest Neighbors
├── naive_bayes/                 # Probabilistic classifiers
│   └── gaussian_naive_bayes.py  # Gaussian Naive Bayes
├── ensemble/                    # Ensemble methods
│   └── stacking.py              # Stacking
├── preprocessing/               # Data preprocessing
│   └── sampling.py              # Resampling techniques
└── utils/                       # Utility functions
    ├── distance.py              # Distance metrics
    └── metrics.py               # Evaluation metrics
```

---

## Requirements

- **Python** >= 3.8
- **NumPy** >= 1.23

### Optional Dependencies

For examples and additional features:
- **Matplotlib** >= 3.7 - Data visualization
- **Scikit-learn** >= 1.3 - Data utilities and benchmarking
- **Streamlit** >= 1.32 - Interactive web apps

---

## Learning Path

This repository is organized for learning. We recommend:

1. **Start with basics:** Linear Regression → Logistic Regression → KNN
2. **Move to trees:** Decision Trees → Ensemble Methods
3. **Explore clustering:** K-Means → GMM
4. **Understand preprocessing:** Resampling techniques
5. **Study implementations:** Read the algorithm code, understand math

For each algorithm:
- Read the implementation in `jklearn/`
- Check the docstrings for mathematical formulation
- Run examples in `examples/` or `streamlit_apps/`
- Study the test cases in `tests/`

---

## Testing

Run the test suite:

```bash
pytest tests/
```

Or test specific algorithms:

```bash
pytest tests/test_linear_models.py -v
pytest tests/test_decision_tree.py -v
```

---

## Web Applications

Interactive Streamlit apps for visualizing algorithms:

```bash
# Run main app
streamlit run streamlit_apps/Home.py

# Run specific algorithm
streamlit run streamlit_apps/linear_regression_app.py
```

Available apps:
- Linear Regression
- Decision Trees
- KNN Classification
- Clustering (K-Means, GMM)
- Ensemble Methods
- And more...

---

## Algorithm Implementation Details

Each algorithm includes:
- Pure Python/NumPy implementation
- Detailed docstrings with math
- Fit and predict methods (sklearn-style API)
- Educational comments explaining steps
- Example usage in docstrings

### Example: Reading Algorithm Source

All algorithms follow a consistent structure:

```python
class MyAlgorithm:
    """
    Algorithm Name
    
    Mathematical formulation and explanation...
    
    Attributes:
        param1: Description
        param2: Description
    """
    
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X, y):
        """Train the model"""
        pass
    
    def predict(self, X):
        """Make predictions"""
        pass
```

---

## Educational Value

This library is designed for:
- **Understanding** how ML algorithms work at the code level
- **Learning** the mathematical foundations
- **Experimenting** with different hyperparameters
- **Comparing** algorithms on your own datasets
- **Extending** with your own implementations

---

## Important Notes

- **For production use:** Use scikit-learn, XGBoost, or other production-grade libraries
- **For learning:** This library prioritizes clarity over performance
- **Some optimizations:** Intentionally omitted for educational clarity
- **Not optimized:** For very large datasets, use production libraries

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NumPy Documentation](https://numpy.org/)
- [100 Days ML Challenge](./100_DAYS.md) - Original challenge structure
- [Usage Guide](./USAGE.md) - Detailed usage examples

---

## Contributing

Contributions are welcome! Feel free to:
- Fix bugs
- Add documentation
- Improve implementations
- Add new algorithms
- Create examples

---

## Support

For questions or issues:
1. Check existing examples in `examples/` and `streamlit_apps/`
2. Read algorithm docstrings
3. Review test cases in `tests/`
4. Open an issue on GitHub
