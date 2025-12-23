import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import toeplitz

from tick.simulation import (
    features_normal_cov_toeplitz,
    features_normal_cov_uniform,
    weights_sparse_exp,
    weights_sparse_gauss,
)


def test_features_normal_cov_uniform_reproducible():
    np.random.seed(42)
    features = features_normal_cov_uniform(n_samples=4, n_features=3)

    np.random.seed(42)
    C = np.random.uniform(size=(3, 3))
    np.fill_diagonal(C, 1.0)
    cov = 0.5 * (C + C.T)
    expected = np.random.multivariate_normal(np.zeros(3), cov, size=4)

    assert_allclose(features, expected)


def test_features_normal_cov_toeplitz_dtype_and_structure():
    np.random.seed(0)
    features = features_normal_cov_toeplitz(
        n_samples=5, n_features=4, cov_corr=0.3, dtype="float32"
    )

    assert features.shape == (5, 4)
    assert features.dtype == np.float32

    np.random.seed(0)
    cov = toeplitz(0.3 ** np.arange(0, 4))
    expected = np.random.multivariate_normal(np.zeros(4), cov, size=5).astype(
        "float32"
    )
    assert_allclose(features, expected)


def test_weights_sparse_gauss_and_exp_signatures():
    np.random.seed(123)
    w_gauss = weights_sparse_gauss(n_weights=6, nnz=3, std=0.5, dtype="float32")
    assert w_gauss.shape == (6,)
    assert w_gauss.dtype == np.float32
    assert np.count_nonzero(w_gauss) == 3

    w_exp = weights_sparse_exp(n_weigths=6, nnz=4, scale=2.0, dtype="float64")
    assert_allclose(
        w_exp, np.array([-1.0, 0.60653066, -0.36787944, 0.22313016, 0.0, 0.0])
    )
    assert w_exp.dtype == np.float64
