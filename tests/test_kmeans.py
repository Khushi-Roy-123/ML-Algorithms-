import numpy as np

from jklearn import KMeans


def test_kmeans_two_groups():
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [5.0, 5.0],
            [5.2, 4.9],
            [4.8, 5.1],
        ]
    )

    m = KMeans(k=2, ep=100, n_init=5, seed=0)
    m.fit(x)
    p = m.predict(x)

    assert p.shape == (x.shape[0],)
    assert len(np.unique(p[:3])) == 1
    assert len(np.unique(p[3:])) == 1
    assert p[0] != p[3]
    assert m.cen.shape == (2, 2)
    assert m.inertia_ >= 0


def test_kmeans_elbow_range_and_shape():
    x = np.array(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.1, 0.2],
            [5.0, 5.0],
            [5.2, 4.8],
            [4.8, 5.1],
            [10.0, 0.0],
            [10.2, 0.2],
            [9.8, -0.2],
        ]
    )

    m = KMeans(ep=150, n_init=8, seed=1)
    ks, ins = m.elbow(x, k_min=1, k_max=5)

    assert np.array_equal(ks, np.array([1, 2, 3, 4, 5]))
    assert ins.shape == (5,)
    assert np.all(ins >= 0)
    assert np.all(ins[:-1] >= ins[1:] - 1e-10)
