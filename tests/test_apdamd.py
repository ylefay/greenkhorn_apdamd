import numpy as np
from ot.apdamd.apdamd import OT
from ot.ot import penalised_cost, Round


def test_apdamd(eps=1.0):
    np.random.seed(1)

    def Gaussian_OT(Gaussian_1, Gaussian_2):
        m1, cov1 = Gaussian_1
        m2, cov2 = Gaussian_2
        sqrt_cov1 = np.linalg.cholesky(cov2)
        invsqrt_cov1 = np.linalg.inv(sqrt_cov1)
        cov = invsqrt_cov1 @ np.linalg.cholesky(sqrt_cov1 @ cov2 @ sqrt_cov1.T) @ invsqrt_cov1

        def T(x):
            return m2 + cov @ (x - m1)

        return T

    def euclidean_wasserstein(xs, ys, order=2):
        return np.mean(np.linalg.norm(xs - ys, axis=-1, ord=order) ** order) ** (1 / order)

    def pdf(x, m, cov):
        return np.exp(-(x - m).T @ np.linalg.inv(cov) @ (x - m) / 2)

    n = 5
    m1 = np.random.randn(n)
    m2 = np.random.randn(n)
    cov1 = np.random.randn(n, n)
    cov1 = cov1 @ cov1.T
    cov2 = np.random.randn(n, n)
    cov2 = cov2 @ cov2.T
    Gaussian_1 = (m1, cov1)
    Gaussian_2 = (m2, cov2)
    T = Gaussian_OT(Gaussian_1, Gaussian_2)
    N_samples = 6
    samples_1 = np.random.multivariate_normal(m1, cov1, N_samples)
    samples_2 = np.random.multivariate_normal(m2, cov2, N_samples)
    C = np.array([[np.linalg.norm(sample_1 - sample_2, ord=2) for sample_1 in samples_1] for sample_2 in samples_2])
    r = np.vectorize(pdf, signature='(n),(m),(l, r)->()')(samples_1, m1, cov1)
    c = np.vectorize(pdf, signature='(n),(m),(l, r)->()')(samples_2, m2, cov2)
    r = r / np.sum(r)
    c = c / np.sum(c)
    n = c.shape[0]
    X = np.random.rand(n ** 2).reshape((n, n))
    X /= np.sum(X)
    # tp, _, _ = OT(X, C, r, c, eps)
    # print(tp)
    r = np.array([0.5, 0.3, 0.2])
    c = np.array([0.5, 0.3, 0.2])
    C = np.array([[0., 0.7, 0.3], [0.7, 0., 0.3], [0.3, 0.3, 0.4]])
    n = r.shape[0]
    X = np.random.rand(n ** 2).reshape((n, n))
    X = Round(X, r, c)
    X /= np.sum(X)
    eta = 0.05
    eps = 4 * np.log(2) * eta
    tp, n_iter, _ = OT(X, C, r, c, eps)
    print(tp)
    assert np.allclose(tp @ np.ones(n, ), r)
    assert np.allclose(tp.T @ np.ones(n, ), c)
    print(penalised_cost(C, tp, eta))
    print(tp)
    print(n_iter)


test_apdamd(eps=0.5)
