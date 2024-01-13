import numpy as np
from optimal_transport.ot import Round
from optimal_transport.gaussian_ot import pdf
import jax

jax.config.update("jax_enable_x64", True)


def sample_problem(n=2):
    C = np.random.rand(n ** 2).reshape((n, n))
    X = np.random.rand(n ** 2).reshape((n, n))
    X /= np.sum(X)
    r = np.random.rand(n)
    r /= np.sum(r)
    c = np.random.rand(n)
    c /= np.sum(c)
    X = Round(X, r, c)
    return X.astype(np.float64), C.astype(np.float64), r.astype(np.float64), c.astype(np.float64)


def simple_problem():
    n = 2
    r = np.array([0.5, 0.5])
    c = np.array([0.5, 0.5])
    C = np.array([[0., 1.], [1.0, 0.]])
    X = np.random.rand(n ** 2).reshape((n, n))
    X = Round(X, r, c)
    return X.astype(np.float64), C.astype(np.float64), r.astype(np.float64), c.astype(np.float64)


def sample_gaussian_OT(n, N_samples, Gaussians=None):
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
    X = np.random.rand(N_samples ** 2).reshape((N_samples, N_samples))
    X = Round(X, r, c)
    X /= np.sum(X)
    return X.astype(np.float64), C.astype(np.float64), r.astype(np.float64), c.astype(np.float64)


def sample_gaussian_OT_exact(N, n, Gaussians=None):
    np.random.seed(0)
    vmin = 0.01
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
    sigma1 = np.linalg.cholesky(cov1)
    eig1 = np.linalg.eig(sigma1)[1]
    sigma2 = np.linalg.cholesky(cov2)
    eig2 = np.linalg.eig(sigma2)[1]
    t1 = np.linspace(m1 - 2 * np.max(eig1, axis=-1), m1 + 2 * np.max(eig1, axis=-1), N)
    t2 = np.linspace(m2 - 2 * np.max(eig2, axis=-1), m2 + 2 * np.max(eig2, axis=-1), N)
    r = pdf(t1, m1, cov1)
    r += np.max(r) * vmin
    c = pdf(t2, m2, cov2)
    c += np.max(c) * vmin
    r /= np.sum(r)
    c /= np.sum(c)
    t12 = np.sum(t1 ** 2, axis=1)
    t22 = np.sum(t2 ** 2, axis=1)
    t1xt2 = np.dot(t1, t2.T)
    t12 = t12.reshape(-1, 1)
    C = (t12 + t22 - 2 * t1xt2)

    OT = np.random.rand(N ** 2).reshape((N, N))
    OT = Round(OT, r, c)
    OT /= np.sum(OT)
    return OT.astype(np.float64), C.astype(np.float64), r.astype(np.float64), c.astype(np.float64)
