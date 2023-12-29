import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


def bregman_divergence(phi):
    def bregman(x, y):
        return phi(x) - phi(y) - (x - y).T @ jax.grad(phi)(y)

    return bregman


def apdamd(f, bregman, phi, x, A, b, eps_p, delta, x_fun, z_fun, iter_max):
    n = A.shape[-1]
    sqn = int(jnp.sqrt(n))
    m = A.shape[0]
    alpha_bar = 0
    alpha = 0
    L = 1
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
        return ((n_iter < iter_max) & (
                bregman(Lambda, mu) > M / 2 * jnp.linalg.norm(Lambda - mu, ord=jnp.inf) ** 2)) | (n_iter == 0)

    if x_fun is None:  # Fallback to scipy optimizer for computing x_fun given f.
        def x_fun(Lambda, x_init):
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

        _, M, alpha_bar, alpha, mu, z, Lambda = jax.lax.while_loop(criterion_iter, iter_iter_fun, inps)
        x = (alpha * x_fun(mu, x) + alpha_bar * x) / alpha_bar
        L = M / 2
        return n_iter + 1, x, alpha_bar, alpha, L, z, mu, Lambda

    inps = (0, x, alpha_bar, alpha, L, z, mu, Lambda)
    n_iter, x, *_ = jax.lax.while_loop(criterion, iter_fun, inps)
    X = jnp.reshape(x, (sqn, sqn), order='F')
    return X, n_iter
