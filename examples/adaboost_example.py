"""Example demonstrating AdaBoost classifier usage."""

import numpy as np
import matplotlib.pyplot as plt
from jklearn.tree import AdaBoostClassifier


def generate_binary_classification_data(n_samples=200, random_state=42):
    """Generate a simple binary classification dataset."""
    np.random.seed(random_state)
    
    # Class 0: centered around (-1, -1)
    X_class0 = np.random.randn(n_samples // 2, 2) + np.array([-1, -1])
    y_class0 = np.zeros(n_samples // 2)
    
    # Class 1: centered around (1, 1)
    X_class1 = np.random.randn(n_samples // 2, 2) + np.array([1, 1])
    y_class1 = np.ones(n_samples // 2)
    
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def main():
    print("AdaBoost Classifier Example")
    print("=" * 50)
    
    # Generate data
    X, y = generate_binary_classification_data(n_samples=200, random_state=42)
    
    # Split into train and test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Train AdaBoost with different configurations
    configs = [
        {"n_estimators": 5, "learning_rate": 1.0, "name": "5 estimators (LR=1.0)"},
        {"n_estimators": 10, "learning_rate": 1.0, "name": "10 estimators (LR=1.0)"},
        {"n_estimators": 10, "learning_rate": 0.5, "name": "10 estimators (LR=0.5)"},
    ]
    
    for config in configs:
        name = config.pop("name")
        
        # Train
        clf = AdaBoostClassifier(random_state=42, **config)
        clf.fit(X_train, y_train)
        
        # Evaluate
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        
        train_acc = np.mean(train_pred == y_train)
        test_acc = np.mean(test_pred == y_test)
        
        print(f"Configuration: {name}")
        print(f"  Number of weak learners trained: {len(clf.estimators_)}")
        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Test accuracy: {test_acc:.4f}")
        print()
    
    # Demonstrate predict_proba
    print("Probability predictions for test samples:")
    print("-" * 50)
    clf = AdaBoostClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test[:5])
    
    for i in range(5):
        pred_class = np.argmax(proba[i])
        print(f"Sample {i}: Class 0 prob={proba[i, 0]:.4f}, Class 1 prob={proba[i, 1]:.4f}")
        print(f"          Predicted class: {pred_class}")
    
    print()
    print("AdaBoost successfully demonstrates adaptive boosting:")
    print("  - Focuses training on misclassified samples from previous rounds")
    print("  - Combines weak learners (decision stumps) into a strong classifier")
    print("  - Achieved competitive accuracy on the test set")


if __name__ == "__main__":
    main()
