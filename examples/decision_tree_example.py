"""Example usage of DecisionTreeClassifier."""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jklearn.tree import DecisionTreeClassifier


def main():
    X = np.array(
        [
            [1.0, 1.0],
            [1.2, 1.1],
            [1.1, 0.9],
            [3.9, 4.0],
            [4.1, 4.2],
            [4.0, 3.8],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    samples = np.array([[1.05, 1.0], [4.05, 4.1]])
    predictions = model.predict(samples)
    probabilities = model.predict_proba(samples)

    print("Predictions:", predictions)
    print("Probabilities:\n", probabilities)


if __name__ == "__main__":
    main()