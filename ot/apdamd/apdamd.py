import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from ot.greenkhorn.greenkhorn import Round


def bregman_divergence(phi):
    def divergence(x, y):
        return phi(x) - phi(y) - jnp.dot(jax.grad(phi)(y), x - y)

    return divergence


def apdamd(phi, A, b, eps_p, delta=1e-3, z_fun=None, iter_max=100):
    alpha_bar = 0
    alpha = 0
    L = 1
    n = b.shape[0] / 2
    z, mu, Lambda = jnp.zeros(n, )
    x = 0
    bregman = bregman_divergence(phi)
    if z_fun is None:
        def objective(z, z_p, mu, alpha):
            return jnp.dot(jax.grad(phi)(mu), z - mu) + bregman(z, z_p) / alpha

        def z_fun(z, mu, alpha):
            return minimize(lambda _z: objective(_z, z, mu, alpha), z, method='BFGS').x

    def criterion(inps):
        n_iter, x, *_ = inps
        return (n_iter < iter_max) & (jnp.linalg.norm(A @ x - b, ord=1) > eps_p)

    def criterion_iter(inps):
        n_iter, M, _, _, mu, _, Lambda = inps
        return (n_iter < iter_max) & (bregman(Lambda, mu) > M / 2 * jnp.linalg.norm(Lambda - mu, ord=jnp.inf) ** 2)

    def iter_fun(inps):
        n_iter, x, alpha_bar, alpha, L, z, mu, Lambda = inps
        M = L / 2
        inps = (0, x, alpha_bar, alpha, L, z, mu, Lambda)
        _, M, alpha_bar, alpha, mu, z, Lambda = jax.lax.while_loop(iter_iter_fun, criterion_iter, inps)
        L = M / 2
        return n_iter + 1, x, alpha_bar, alpha, L, z, mu, Lambda

    def iter_iter_fun(inps):
        n_iter, M, alpha_bar, alpha, mu, z, Lambda = inps
        M = 2 * M
        alpha = (1 + jnp.sqrt(1 + 4 * delta * M * alpha_bar)) / (2 * delta * M)
        alpha_bar = alpha_bar + alpha
        mu = (alpha * z + alpha_bar * Lambda) / alpha_bar
        z = z_fun(z, mu, alpha)
        Lambda = (alpha * z + alpha_bar * Lambda) / alpha_bar
        return n_iter + 1, M, alpha_bar, alpha, mu, z, Lambda

    inps = (0, x, alpha_bar, alpha, L, z, mu, Lambda)
    _, x, *_ = jax.lax.while_loop(iter_fun, criterion, inps)
    X = jnp.reshape(x, (n, n), order='F')
    return X


def OT(X, C, r, c, eps, phi=None, delta=None, z_fun=None, iter_max=1000):
    n = C.shape[0]
    if phi is None:
        def phi(x):
            return 1 / (2 * n) * jnp.linalg.norm(x, ord=2) ** 2

        z_fun = None

        if delta is None:
            delta = n

    # eta = eps / (4 * jnp.log(n))
    # phi depends upon eta
    eps_p = eps / (8 * jnp.linalg.norm(C, ord=jnp.inf))
    r_tilde = (1 - eps_p / 8) * r + eps_p / (8 * n) * jnp.ones(n, )
    c_tilde = (1 - eps_p / 8) * c + eps_p / (8 * n) * jnp.ones(n, )
    b = jnp.vstack([r_tilde, c_tilde])
    vecX = jnp.reshape(X, (n ** 2,), order='F')
    A = jnp.vstack([r, c]) @ vecX.T @ jnp.linalg.pinv(vecX @ vecX.T)
    X_tilde = apdamd(phi, A, b, eps_p / 2, delta=delta, z_fun=z_fun, iter_max=iter_max)
    X_hat = Round(X_tilde, r, c)
    return X_hat
