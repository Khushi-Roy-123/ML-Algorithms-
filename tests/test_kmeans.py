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
