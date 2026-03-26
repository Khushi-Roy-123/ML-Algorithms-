"""Distance utility functions for jklearn."""

import numpy as np


def euclidean_distance(x1, x2):
	"""Compute Euclidean distance between two vectors."""
	x1 = np.asarray(x1)
	x2 = np.asarray(x2)
	return np.sqrt(np.sum((x1 - x2) ** 2))


__all__ = ["euclidean_distance"]

