import numpy as np
from ot.ot import Round
from ot.gaussian_ot import pdf


def sample_problem(n=2):
    C = np.random.rand(n ** 2).reshape((n, n))
    X = np.random.rand(n ** 2).reshape((n, n))
    X /= np.sum(X)
    r = np.random.rand(n)
    r /= np.sum(r)
    c = np.random.rand(n)
    c /= np.sum(c)
    X = Round(X, r, c)
    return X, C, r, c


def simple_problem():
    n = 2
    r = np.array([0.5, 0.5])
    c = np.array([0.5, 0.5])
    C = np.array([[0., 1.], [1.0, 0.]])
    X = np.random.rand(n ** 2).reshape((n, n))
    X = Round(X, r, c)
    return X, C, r, c


def sample_gaussian_OT(n, N_samples, Gaussians):
    if Gaussians is None:
        m1 = np.random.randn(n)
        m2 = np.random.randn(n)
        cov1 = np.random.randn(n, n)
        cov1 = cov1 @ cov1.T
        cov2 = np.random.randn(n, n)
        cov2 = cov2 @ cov2.T
    else:
        m1, cov1 = Gaussians[0]
        m2, cov2 = Gaussians[1]

    samples_1 = np.random.multivariate_normal(m1, cov1, N_samples)
    samples_2 = np.random.multivariate_normal(m2, cov2, N_samples)
    C = np.array([[np.linalg.norm(sample_1 - sample_2, ord=2) for sample_1 in samples_1] for sample_2 in samples_2])
    # r = np.vectorize(pdf, signature='(n),(m),(l, r)->()')(samples_1, m1, cov1)
    # c = np.vectorize(pdf, signature='(n),(m),(l, r)->()')(samples_2, m2, cov2)
    r = pdf(samples_1, m1, cov1)
    c = pdf(samples_2, m2, cov2)
    X = np.random.rand(n ** 2).reshape((n, n))
    X = Round(X, r, c)
    X /= np.sum(X)
    return X, C, r, c
