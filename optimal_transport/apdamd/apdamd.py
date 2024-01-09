import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


def bregman_divergence(phi):
    def bregman(x, y):
        return phi(x) - phi(y) - (x - y).T @ jax.grad(phi)(y)

    return bregman


def apdamd(varphi, bregman_varphi, x, A, b, eps_p, f, bregman_phi=None, phi=None, delta=None, x_fun=None, z_fun=None,
           iter_max=None):
    n = A.shape[-1]
    m = A.shape[0]
    alpha_bar = 0
    alpha = 0
    L = 1
    common_value = jnp.zeros(m, )
    z, mu, Lambd = common_value, common_value, common_value

    if phi is None:
        def phi(x):
            return 1 / n * jnp.linalg.norm(x, ord=2) ** 2

        def bregman_phi(x, x_p):
            return 1 / n * jnp.linalg.norm(x - x_p, ord=2) ** 2
    if bregman_phi is None:
        bregman_phi = bregman_divergence(phi)
    if bregman_varphi is None:
        bregman_varphi = bregman_divergence(varphi)

    if z_fun is None:  # Fallback to scipy optimizer for computing z_fun given phi.
        def objective(z, z_p, mu, alpha):
            return jnp.dot(jax.grad(varphi)(mu), z - mu) + bregman_phi(z, z_p) / alpha

        def z_fun(z, mu, alpha):
            return minimize(lambda _z: objective(_z, z, mu, alpha), z, method='BFGS').x

    def criterion(inps):
        n_iter, x, *_ = inps
        return ((n_iter < iter_max) & (jnp.linalg.norm(A @ x - b, ord=1) > eps_p)) | (n_iter == 0)

    def criterion_iter(inps):
        n_iter, M, _, _, mu, _, Lambd = inps
        return ((n_iter < jnp.inf) & (
                bregman_varphi(Lambd, mu) > M / 2 * jnp.linalg.norm(Lambd - mu, ord=jnp.inf) ** 2)) | (n_iter == 0)

    if x_fun is None:  # Fallback to scipy optimizer for computing x_fun given f.
        def x_fun(Lambd, x_init):
            return minimize(lambda _x: f(_x) + A.T @ Lambd @ _x.T, x_init, method='BFGS').x

    def iter_iter_fun(inps):
        n_iter, M, alpha_bar, alpha, mu, z, Lambd = inps
        old_alpha_bar = alpha_bar
        # M = jax.lax.cond(2 * M <= 2e260, lambda M: 2 * M, lambda M: M, M)
        M = 2 * M
        alpha = (1 + jnp.sqrt(1 + 4 * delta * M * alpha_bar)) / (2 * delta * M)
        alpha_bar = alpha_bar + alpha
        mu = (alpha * z + old_alpha_bar * Lambd) / alpha_bar
        z = z_fun(z, mu, alpha)
        Lambd = (alpha * z + old_alpha_bar * Lambd) / alpha_bar
        return n_iter + 1, M, alpha_bar, alpha, mu, z, Lambd

    def iter_fun(inps):
        n_iter, x, alpha_bar, alpha, L, z, mu, Lambd = inps
        M = L / 2
        old_alpha_bar = alpha_bar
        inps = (0, M, alpha_bar, alpha, mu, z, Lambd)
        while criterion_iter(inps):
            inps = iter_iter_fun(inps)
        _, M, alpha_bar, alpha, mu, z, Lambd = inps
        # _, M, alpha_bar, alpha, mu, z, Lambd = jax.lax.while_loop(criterion_iter, iter_iter_fun, inps)
        x = (x_fun(mu, x) * alpha + old_alpha_bar * x) / alpha_bar
        # L = jax.lax.cond(M >= 2e-298, lambda L: L / 2, lambda L: L, L.astype(jnp.float64))
        L = L / 2
        return n_iter + 1, x, alpha_bar, alpha, L, z, mu, Lambd

    inps = (0, x, alpha_bar, alpha, L, z, mu, Lambd)
    # n_iter, x, *_ = jax.lax.while_loop(criterion, iter_fun, inps)
    while criterion(inps):
        inps = iter_fun(inps)
    n_iter, x, *_ = inps
    return x, n_iter
