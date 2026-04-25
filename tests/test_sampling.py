import numpy as np

from jklearn import OverSampling, SMOTE, UnderSampling


def _cnt(y):
    c, n = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(c, n)}


def test_undersampling_balances_to_min():
    x0 = np.array([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]])
    x1 = np.array([[1.0], [1.1]])
    x = np.vstack([x0, x1])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1])

    m = UnderSampling(seed=0)
    xr, yr = m.fit_resample(x, y)
    d = _cnt(yr)

    assert xr.shape[0] == 4
    assert d[0] == 2
    assert d[1] == 2


def test_oversampling_balances_to_max():
    x0 = np.array([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]])
    x1 = np.array([[1.0], [1.1]])
    x = np.vstack([x0, x1])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1])

    m = OverSampling(seed=0)
    xr, yr = m.fit_resample(x, y)
    d = _cnt(yr)

    assert xr.shape[0] == 12
    assert d[0] == 6
    assert d[1] == 6


def test_smote_balances_to_max():
    x0 = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [-0.1, 0.2],
            [0.1, -0.2],
            [0.3, 0.0],
            [0.0, 0.3],
        ]
    )
    x1 = np.array([[3.0, 3.0], [3.2, 3.1], [2.9, 2.8]])
    x = np.vstack([x0, x1])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])

    m = SMOTE(k=2, seed=0)
    xr, yr = m.fit_resample(x, y)
    d = _cnt(yr)

    assert xr.shape[0] == 12
    assert d[0] == 6
    assert d[1] == 6
