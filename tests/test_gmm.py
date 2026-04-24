import numpy as np

from jklearn import GaussianMixture


def test_gmm_two_groups_probabilities_and_labels():
    x = np.array(
        [
            [0.0, 0.1],
            [0.2, -0.1],
            [-0.1, 0.0],
            [5.1, 4.9],
            [4.8, 5.2],
            [5.0, 5.1],
        ]
    )

    m = GaussianMixture(k=2, ep=100, n_init=3, seed=0)
    m.fit(x)

    p = m.predict(x)
    proba = m.predict_proba(x)

    assert p.shape == (x.shape[0],)
    assert proba.shape == (x.shape[0], 2)
    assert np.allclose(np.sum(proba, axis=1), 1.0)

    assert len(np.unique(p[:3])) == 1
    assert len(np.unique(p[3:])) == 1
    assert p[0] != p[3]

    assert m.means_.shape == (2, 2)
    assert m.covariances_.shape == (2, 2, 2)
    assert m.weights_.shape == (2,)
    assert np.isfinite(m.lower_bound_)


def test_gmm_validation_and_not_fitted_errors():
    x = np.array([[0.0], [1.0], [2.0]])

    with np.testing.assert_raises(ValueError):
        GaussianMixture(k=0).fit(x)

    with np.testing.assert_raises(ValueError):
        GaussianMixture(k=4).fit(x)

    with np.testing.assert_raises(ValueError):
        GaussianMixture(reg_covar=0.0).fit(x)

    m = GaussianMixture(k=2)
    with np.testing.assert_raises(ValueError):
        m.predict(x)
