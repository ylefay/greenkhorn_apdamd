import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from ot.ot import Round, penality

jax.config.update("jax_enable_x64", True)


def bregman_divergence(phi):
    def bregman(x, y):
        return phi(x) - phi(y) - (x - y).T @ jax.grad(phi)(y)

    return bregman


def apdamd(f, bregman, phi, x, A, b, eps_p, delta=1e-3, z_fun=None, iter_max=100):
    alpha_bar = 0
    alpha = 0
    L = 1

    n = A.shape[-1]
    sqn = int(jnp.sqrt(n))
    m = A.shape[0]
    common_value = jnp.ones(m, )
    z, mu, Lambda = common_value, common_value, common_value
    if bregman is None:
        bregman = bregman_divergence(phi)
    if z_fun is None:  # Fallback to scipy optimizer for computing z_fun given phi.
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

    def fun_x(Lambda, x_init):
        return minimize(lambda _x: f(_x) + A.T @ Lambda @ _x.T, x_init, method='BFGS').x

    def iter_iter_fun(inps):
        n_iter, M, alpha_bar, alpha, mu, z, Lambda = inps
        old_alpha_bar = alpha_bar
        M = 2 * M
        alpha = (1 + jnp.sqrt(1 + 4 * delta * M * alpha_bar)) / (2 * delta * M)
        alpha_bar = alpha_bar + alpha
        mu = (alpha * z + alpha_bar * Lambda) / alpha_bar
        z = z_fun(z, mu, alpha)
        Lambda = (alpha * z + old_alpha_bar * Lambda) / alpha_bar
        return n_iter + 1, M, alpha_bar, alpha, mu, z, Lambda

    def iter_fun(inps):
        n_iter, x, alpha_bar, alpha, L, z, mu, Lambda = inps
        M = L / 2
        inps = (0, M, alpha_bar, alpha, mu, z, Lambda)

        while criterion_iter(inps):
            inps = iter_iter_fun(inps)
        _, M, alpha_bar, alpha, mu, z, Lambda = inps
        # _, M, alpha_bar, alpha, mu, z, Lambda = jax.lax.while_loop(criterion_iter, iter_iter_fun, inps)
        x = (alpha * fun_x(mu, x) + alpha_bar * x) / alpha_bar
        L = M / 2
        return n_iter + 1, x, alpha_bar, alpha, L, z, mu, Lambda

    inps = (0, x, alpha_bar, alpha, L, z, mu, Lambda)
    while criterion(inps):
        inps = iter_fun(inps)
    n_iter, x, *_ = inps
    # n_iter, x, *_ = jax.lax.while_loop(criterion, iter_fun, inps)
    X = jnp.reshape(x, (sqn, sqn), order='F')
    error = jnp.linalg.norm(A @ x - b, ord=1)
    return X, n_iter, error


def OT(X, C, r, c, eps, phi=None, bregman=None, delta=None, z_fun=None, iter_max=1000):
    n = C.shape[0]
    if phi is None:
        # Fallback to phi(z) = 1/(2n) ||z||_2^2
        # Thus, B_phi(z, z') = 1/ n ||z - z'||_2^2
        # And, first-order condition for z is z = n alpha grad(phi)(mu) + z' = alpha * mu + z'

        def phi(x):
            return 1 / (2 * n) * jnp.linalg.norm(x, ord=2) ** 2

        def z_fun(z, mu, alpha):
            return alpha * mu + z

        def bregman(z, z_p):
            return 1 / n * jnp.linalg.norm(z - z_p, ord=2) ** 2

        if delta is None:
            delta = n

    eta = eps / (4 * jnp.log(n))
    eps_p = eps / (8 * jnp.linalg.norm(C, ord=jnp.inf))
    r_tilde = (1 - eps_p / 8) * r + eps_p / (8 * n) * jnp.ones(n, )
    c_tilde = (1 - eps_p / 8) * c + eps_p / (8 * n) * jnp.ones(n, )
    b = jnp.hstack([r_tilde, c_tilde])
    vecX = jnp.reshape(X, (n ** 2,), order='F')
    A = jnp.hstack([X @ jnp.ones(n, ), X.T @ jnp.ones(n, )]).reshape(2 * n, 1) @ vecX.T.reshape(1, n ** 2) * 1 / (
            vecX @ vecX.T)

    def f(x):  # x = vecX
        return jnp.reshape(C, -1, order='F').T @ x - penality(x, eta)

    X_tilde, n_iter, error = apdamd(f, bregman, phi, vecX, A, b, eps_p / 2, delta=delta, z_fun=z_fun, iter_max=iter_max)
    X_hat = Round(X_tilde, r, c)
    return X_hat, n_iter, error
