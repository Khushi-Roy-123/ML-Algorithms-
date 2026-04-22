import numpy as np
import pytest

from jklearn import HierarchicalClustering


def _two_blob_data():
    return np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [5.0, 5.0],
            [5.2, 4.9],
            [4.8, 5.1],
        ]
    )


def _assert_two_group_labels(labels):
    assert len(np.unique(labels[:3])) == 1
    assert len(np.unique(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_agglomerative_dendrogram_and_labels():
    x = _two_blob_data()
    m = HierarchicalClustering(n_clusters=2, method="agglomerative", linkage="average")

    labels = m.fit_predict(x)
    z = m.dendrogram()

    assert labels.shape == (x.shape[0],)
    _assert_two_group_labels(labels)
    assert z.shape == (x.shape[0] - 1, 4)
    assert np.all(z[:, 3] >= 2)


def test_divisive_dendrogram_and_labels():
    x = _two_blob_data()
    m = HierarchicalClustering(n_clusters=2, method="divisive", linkage="average")

    labels = m.fit_predict(x)
    z = m.dendrogram()

    assert labels.shape == (x.shape[0],)
    _assert_two_group_labels(labels)
    assert z.shape == (x.shape[0] - 1, 4)


def test_invalid_method_raises():
    x = _two_blob_data()
    with pytest.raises(ValueError, match="method"):
        HierarchicalClustering(method="unknown").fit(x)
